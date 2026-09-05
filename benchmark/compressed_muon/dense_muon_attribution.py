"""Low-cost, serial dense-Muon checksum attribution runner.

The CUDA execution path intentionally delegates model construction, data,
training steps, and checksum calculation to ``benchmark_arc_2x2``.  This file
only selects one diagnostic mode and archives its metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist

from benchmark.compressed_muon import benchmark_arc_2x2 as benchmark


MODE_SPECS: dict[str, dict[str, Any]] = {
    "rank_local_custom_hook": {
        "distributed_mesh": None,
        "register_custom_ddp_hook": True,
        "use_triton": None,
    },
    "process_group_custom_hook": {
        "distributed_mesh": "ddp_process_group",
        "register_custom_ddp_hook": True,
        "use_triton": None,
    },
    "rank_local_default_reducer": {
        "distributed_mesh": None,
        "register_custom_ddp_hook": False,
        "use_triton": None,
    },
    "rank_local_no_triton": {
        "distributed_mesh": None,
        "register_custom_ddp_hook": True,
        "use_triton": False,
    },
    "upstream_ddp": {
        "distributed_mesh": "ddp_process_group",
        "register_custom_ddp_hook": False,
        "use_triton": None,
    },
}


def mode_spec(mode: str) -> dict[str, Any]:
    """Return a defensive copy of the exact isolation switches for ``mode``."""
    try:
        return dict(MODE_SPECS[mode])
    except KeyError as exc:
        raise ValueError(f"unknown attribution mode: {mode!r}") from exc


@dataclass
class DiagnosticConfig:
    mode: str
    model: str = "gpt130m"
    warmup_steps: int = 2
    measure_steps: int = 12
    seed: int = 42
    world_size: int = 4
    local_batch: int = 1
    sequence_length: int = 256

    def __post_init__(self) -> None:
        mode_spec(self.mode)
        if self.model not in benchmark.MODEL_PRESETS:
            raise ValueError(f"unknown model preset: {self.model}")
        if self.warmup_steps < 0 or self.measure_steps < 1:
            raise ValueError("warmup_steps must be non-negative and measure_steps positive")
        if self.world_size < 1 or self.local_batch < 1 or self.sequence_length < 1:
            raise ValueError("world_size, local_batch, and sequence_length must be positive")


def diagnostic_experiment_id(config: DiagnosticConfig) -> str:
    return (
        f"CM-muon-dense-{config.model}-ddp-ws{config.world_size}-s{config.seed}"
        f"-{config.mode}"
    )


def diagnostic_benchmark_config(
    config: DiagnosticConfig, *, output: str | None = None
) -> benchmark.BenchmarkConfig:
    """Build the existing benchmark config with only diagnostic switches changed."""
    switches = mode_spec(config.mode)
    return benchmark.BenchmarkConfig(
        experiment_id=diagnostic_experiment_id(config),
        optimizer="muon",
        sync="dense",
        model=config.model,
        warmup_steps=config.warmup_steps,
        measure_steps=config.measure_steps,
        seed=config.seed,
        output=output,
        profile_output=None,
        compile_model=False,
        world_size=config.world_size,
        local_batch=config.local_batch,
        sequence_length=config.sequence_length,
        formal=False,
        muon_distributed_mesh=("ddp_process_group"
                               if switches["distributed_mesh"] == "ddp_process_group"
                               else "none"),
        muon_use_triton=switches["use_triton"],
        register_custom_ddp_hook=switches["register_custom_ddp_hook"],
        include_rank_details=True,
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_diagnostic(config: DiagnosticConfig, output: str) -> dict[str, Any]:
    """Run one CUDA diagnostic cell and write its enriched result on rank 0."""
    benchmark_config = diagnostic_benchmark_config(config, output=output)
    result = benchmark.run_benchmark(benchmark_config)
    rank = int(os.environ.get("RANK", "0"))
    backend = dist.get_backend() if dist.is_available() and dist.is_initialized() else "none"
    switches = mode_spec(config.mode)
    result["mode"] = config.mode
    result["diagnostic"] = {
        "mode": config.mode,
        "distributed_mesh": switches["distributed_mesh"],
        "register_custom_ddp_hook": switches["register_custom_ddp_hook"],
        "use_triton": result["optimizer_config"].get("use_triton")
        if switches["use_triton"] is None else switches["use_triton"],
        "backend": backend,
        "git_commit": _git_commit(),
        "world_size": config.world_size,
        "seed": config.seed,
        "model": config.model,
    }
    result["environment"]["backend"] = backend
    result["environment"]["git_commit"] = result["diagnostic"]["git_commit"]
    if rank == 0:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def summarize_artifacts(root: str | Path) -> dict[str, Any]:
    """Summarize completed and failed cells without requiring CUDA."""
    root = Path(root)
    cells = []
    for cell_dir in sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []:
        status_path = cell_dir / "status.json"
        result_path = cell_dir / "result.json"
        status_data = json.loads(status_path.read_text()) if status_path.exists() else {
            "status": "unknown", "exit_code": None
        }
        result_data = json.loads(result_path.read_text()) if result_path.exists() else None
        correctness = (result_data or {}).get("correctness") or {}
        cells.append({
            "mode": cell_dir.name,
            "status": status_data.get("status", "unknown"),
            "exit_code": status_data.get("exit_code"),
            "result": result_data,
            "rank_checksum_pairs": correctness.get("rank_checksum_pairs"),
            "parameter_checksum_agreement": correctness.get("parameter_checksum_agreement"),
            "finite_loss": correctness.get("finite_loss"),
            "finite_parameters": correctness.get("finite_parameters"),
        })
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return {"schema_version": 1, "status_counts": counts, "cells": cells}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_SPECS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt130m", choices=tuple(benchmark.MODEL_PRESETS))
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--local-batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = DiagnosticConfig(
        mode=args.mode, model=args.model, warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps, seed=args.seed, world_size=args.world_size,
        local_batch=args.local_batch, sequence_length=args.sequence_length,
    )
    try:
        run_diagnostic(config, args.output)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
