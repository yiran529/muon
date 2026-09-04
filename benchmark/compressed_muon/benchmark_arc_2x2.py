"""A small, reproducible DDP benchmark for dense and ARC-TopK optimizers.

The benchmark deliberately keeps configuration and result construction usable on
CPU.  The actual runner requires CUDA because its measurements use CUDA events.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function

from dion.adamw_arctopk import ArcTopKAdamW
from dion.arc_topk_sync import ArcTopKLogicalBytes, ArcTopKSyncConfig, estimate_arc_logical_bytes, group_parameters_by_shape_dtype
from dion.muon import Muon
from dion.muon_arctopk import ArcTopKMuon
from models.gpt_model import GPT, GPTConfig

MODEL_PRESETS = {
    "gpt60m": dict(model_dim=512, n_layer=4, n_head=8),
    "gpt130m": dict(model_dim=768, n_layer=8, n_head=12),
    "gpt350m": dict(model_dim=1024, n_layer=20, n_head=16),
    "gpt1b": dict(model_dim=1536, n_layer=30, n_head=24),
}


@dataclass
class BenchmarkConfig:
    experiment_id: str
    optimizer: str
    sync: str
    model: str
    warmup_steps: int
    measure_steps: int
    seed: int
    output: Optional[str]
    profile_output: Optional[str]
    compile_model: bool = False
    world_size: int = 4
    local_batch: int = 1
    sequence_length: int = 256
    gradient_accumulation: int = 1
    formal: bool = True
    transport: str = "normal"
    ratio: float = 0.2
    projection_rank: int = 4
    eta: float = 0.1
    start_compress_step: int = 0


def validate_config(config: BenchmarkConfig) -> BenchmarkConfig:
    if config.optimizer not in {"adamw", "muon"}:
        raise ValueError("optimizer must be one of: adamw, muon")
    if config.sync not in {"dense", "arc"}:
        raise ValueError("sync must be one of: dense, arc")
    if config.model not in MODEL_PRESETS:
        raise ValueError(f"unknown model preset: {config.model}")
    if config.sync == "arc" and config.optimizer not in {"adamw", "muon"}:
        raise ValueError(f"ARC sync is unsupported for optimizer {config.optimizer!r}")
    if config.formal and config.warmup_steps < 20:
        raise ValueError("formal benchmark requires warmup_steps >= 20")
    if config.formal and config.measure_steps < 100:
        raise ValueError("formal benchmark requires measure_steps >= 100")
    if config.formal and config.world_size != 4:
        raise ValueError("formal benchmark requires world_size == 4")
    if config.local_batch < 1 or config.sequence_length < 1:
        raise ValueError("local_batch and sequence_length must be positive")
    expected = re.compile(
        rf"^CM[^-]+-{re.escape(config.optimizer)}-{re.escape(config.sync)}-"
        rf"{re.escape(config.model)}-ddp-ws{config.world_size}-s{config.seed}(?:$|-.*)"
    )
    if not expected.match(config.experiment_id):
        raise ValueError(
            "experiment_id must encode optimizer, sync, model, world size and seed "
            f"(expected *-{config.optimizer}-{config.sync}-{config.model}-ddp-"
            f"ws{config.world_size}-s{config.seed})"
        )
    return config


def parse_args(argv: Optional[list[str]] = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), required=True)
    parser.add_argument("--sync", choices=("dense", "arc"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--measure-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-output")
    parser.add_argument("--compile-model", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile-model", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=False)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--local-batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="relax formal minimums")
    parser.add_argument("--transport", choices=("normal", "p2p_disabled"), default="normal")
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        experiment_id=args.experiment_id, optimizer=args.optimizer, sync=args.sync,
        model=args.model, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps,
        seed=args.seed, output=args.output, profile_output=args.profile_output,
        compile_model=args.compile_model, world_size=args.world_size,
        local_batch=args.local_batch, sequence_length=args.sequence_length,
        gradient_accumulation=args.gradient_accumulation,
        formal=not args.smoke, transport=args.transport,
    )
    return validate_config(config)


def _parameter_count(model_name: str) -> int:
    p = MODEL_PRESETS[model_name]
    d, l = p["model_dim"], p["n_layer"]
    # GPT ties neither embedding nor lm_head; this avoids constructing 1B models
    return 2 * 50304 * d + 12 * l * d * d


def build_result_skeleton(config: BenchmarkConfig) -> dict[str, Any]:
    validate_config(config)
    preset = MODEL_PRESETS[config.model]
    optimizer_config = (
        {"lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01}
        if config.optimizer == "adamw"
        else {"lr": 0.02, "mu": 0.95, "weight_decay": 0.01}
    )
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "optimizer": config.optimizer,
        "sync_mode": config.sync,
        "transport": config.transport,
        "model": {"label": config.model, "dim": preset["model_dim"],
                  "layers": preset["n_layer"], "heads": preset["n_head"],
                  "parameters": _parameter_count(config.model)},
        "workload": {"world_size": config.world_size, "local_batch": config.local_batch,
                      "sequence_length": config.sequence_length,
                      "gradient_accumulation": config.gradient_accumulation, "dtype": "bfloat16"},
        "arc": {"ratio": config.ratio, "projection_rank": config.projection_rank,
                "eta": config.eta, "seed": config.seed,
                "start_compress_step": config.start_compress_step},
        "optimizer_config": optimizer_config,
        "timing_ms": {"step_samples": [], "fwd_bwd_samples": [],
                      "optimizer_samples": [], "step_mean": 0.0},
        "throughput": {"tokens_per_second": 0.0},
        "memory": {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0},
        "communication": {"dense_gradient_bytes": 0, "arc_seed_bytes": 0,
                          "arc_sketch_bytes": 0, "arc_selected_values_bytes": 0,
                          "uncompressed_bytes": 0},
        "profiler": {"trace_path": None, "nccl_kernel_time_ms": None, "collectives": []},
        "environment": {"git_commit": "", "torch": torch.__version__,
                         "cuda": torch.version.cuda or "", "nccl": "", "gpu_names": []},
    }


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_model(config: BenchmarkConfig, device: torch.device) -> GPT:
    torch.manual_seed(42)
    model = GPT(GPTConfig(sequence_len=config.sequence_length, **{
        "n_embd": MODEL_PRESETS[config.model]["model_dim"],
        "n_layer": MODEL_PRESETS[config.model]["n_layer"],
        "n_head": MODEL_PRESETS[config.model]["n_head"],
    }))
    model.init_weights()
    model.to(device=device, dtype=torch.bfloat16)
    if config.compile_model:
        model.compile()
    return model


def _optimizer_parameters(model: GPT) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    compressed = list(model.transformer.h.parameters())
    uncompressed = [*model.transformer.wte.parameters(), *model.lm_head.parameters()]
    return compressed, uncompressed


def build_optimizer(config: BenchmarkConfig, model: GPT, process_group=None):
    compressed, uncompressed = _optimizer_parameters(model)
    if config.optimizer == "adamw":
        if config.sync == "arc":
            return ArcTopKAdamW(
                [{"params": compressed, "arc_compress": True}, {"params": uncompressed}],
                process_group=process_group, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
                arc_topk_ratio=config.ratio, arc_projection_rank=config.projection_rank,
                arc_eta=config.eta, arc_seed=config.seed,
                arc_start_compress_step=config.start_compress_step,
            )
        return torch.optim.AdamW([*compressed, *uncompressed], lr=1e-3,
                                 betas=(0.9, 0.999), weight_decay=0.01)
    groups = [{"params": compressed, "algorithm": "muon"},
              {"params": uncompressed, "algorithm": "adamw"}]
    cls = ArcTopKMuon if config.sync == "arc" else Muon
    kwargs = dict(lr=0.02, mu=0.95, weight_decay=0.01, adjust_lr="spectral_norm")
    if cls is ArcTopKMuon:
        kwargs.update(arc_topk_ratio=config.ratio, arc_projection_rank=config.projection_rank,
                      arc_eta=config.eta, arc_seed=config.seed,
                      arc_start_compress_step=config.start_compress_step)
    return cls(groups, distributed_mesh=process_group, **kwargs)


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    names = []
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"git_commit": commit, "torch": torch.__version__, "cuda": torch.version.cuda or "",
            "nccl": getattr(torch.cuda, "nccl", None) and torch.cuda.nccl.version(),
            "gpu_names": names}


def _run_steps(model, optimizer, batches, config, ddp_model=None, measure=False):
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark execution requires CUDA; configuration helpers are CPU-safe")
    measured_events = []
    fwd_samples, opt_samples, step_samples = [], [], []
    for tokens, targets in batches:
        step_start = torch.cuda.Event(enable_timing=True)
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        opt_start = torch.cuda.Event(enable_timing=True)
        opt_end = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record(); fwd_start.record()
        context = ddp_model.no_sync() if config.sync == "arc" and ddp_model is not None else _nullcontext()
        with context:
            with record_function("benchmark/forward_backward"):
                loss = model(tokens, targets=targets)
                loss.backward()
        fwd_end.record(); opt_start.record()
        with record_function("benchmark/optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        opt_end.record(); step_end.record()
        if measure: measured_events.append((fwd_start, fwd_end, opt_start, opt_end, step_start, step_end))
    torch.cuda.synchronize()
    for fwd_start, fwd_end, opt_start, opt_end, step_start, step_end in measured_events:
        fwd_samples.append(fwd_start.elapsed_time(fwd_end))
        opt_samples.append(opt_start.elapsed_time(opt_end))
        step_samples.append(step_start.elapsed_time(step_end))
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 and measure:
        # Materialize all local event samples first, then report the slowest rank.
        for samples in (fwd_samples, opt_samples, step_samples):
            sample_tensor = torch.tensor(samples, device=torch.cuda.current_device(), dtype=torch.float64)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.MAX)
            samples[:] = sample_tensor.cpu().tolist()
    return fwd_samples, opt_samples, step_samples


def run_profiler(model, optimizer, batches, config, *, ddp_model=None, output=None):
    """Capture exactly 3 wait, 3 warmup, and 5 active profiler steps."""
    if not torch.cuda.is_available():
        raise RuntimeError("profiler execution requires CUDA")
    output = output or config.profile_output
    if not output:
        raise ValueError("a profile output path is required")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    trace_path = str(output) if rank == 0 else None
    def on_trace_ready(prof):
        if trace_path:
            prof.export_chrome_trace(trace_path)
    schedule = torch.profiler.schedule(wait=3, warmup=3, active=5, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule, on_trace_ready=on_trace_ready, record_shapes=False,
    ) as prof:
        for tokens, targets in batches[:11]:
            context = ddp_model.no_sync() if config.sync == "arc" and ddp_model is not None else _nullcontext()
            with context:
                with record_function("benchmark/forward_backward"):
                    loss = model(tokens, targets=targets)
                    loss.backward()
            with record_function("benchmark/optimizer"):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            prof.step()
    return trace_path


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run a configured CUDA benchmark and optionally write its JSON result."""
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark execution")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if config.world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    group = dist.group.WORLD if dist.is_initialized() else None
    random.seed(config.seed + rank); torch.manual_seed(config.seed + rank)
    model = build_model(config, device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if group else model
    optimizer = build_optimizer(config, model, group)
    # Rank-local data are deterministic and independent of model initialization.
    random.seed(config.seed + rank); torch.manual_seed(config.seed + rank)
    batches = [(torch.randint(0, 50304, (config.local_batch, config.sequence_length), device=device),
                torch.randint(0, 50304, (config.local_batch, config.sequence_length), device=device))
               for _ in range(config.warmup_steps + config.measure_steps)]
    # Compile-trigger calls are intentionally isolated from measured optimizer state.
    _run_steps(ddp_model, optimizer, batches[:config.warmup_steps], config, ddp_model)
    model = build_model(config, device); ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if group else model
    optimizer = build_optimizer(config, model, group)
    _run_steps(ddp_model, optimizer, batches[:config.warmup_steps], config, ddp_model)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    fwd, opt, step = _run_steps(ddp_model, optimizer, batches[config.warmup_steps:], config, ddp_model, True)
    result = build_result_skeleton(config)
    result["timing_ms"].update(step_samples=step, fwd_bwd_samples=fwd, optimizer_samples=opt,
                                step_mean=statistics.mean(step) if step else 0.0)
    result["throughput"]["tokens_per_second"] = (config.local_batch * config.sequence_length * config.world_size / (statistics.mean(step) / 1000)) if step else 0.0
    result["memory"] = {"peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
    raw = _raw_model(ddp_model); compressed, uncompressed = _optimizer_parameters(raw)
    if config.sync == "dense":
        arc_bytes = ArcTopKLogicalBytes(
            dense_gradient=sum(p.numel() * p.element_size() for p in [*compressed, *uncompressed]),
            arc_seed=0, arc_sketch=0, arc_selected_values=0, uncompressed=0,
        )
    else:
        arc_bytes = estimate_arc_logical_bytes(
            compressed_batches=group_parameters_by_shape_dtype(compressed),
            uncompressed_params=uncompressed,
            config=ArcTopKSyncConfig(config.ratio, config.projection_rank, config.eta, config.seed, config.start_compress_step),
            step=config.start_compress_step + 1,
        )
    result["communication"] = asdict(arc_bytes)
    result["environment"] = _environment()
    if config.profile_output:
        run_profiler(ddp_model, optimizer, batches[config.warmup_steps:], config,
                     ddp_model=ddp_model, output=config.profile_output)
        result["profiler"]["trace_path"] = config.profile_output if rank == 0 else None
    if rank == 0 and config.output:
        Path(config.output).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    config = parse_args(argv)
    run_benchmark(config)


if __name__ == "__main__":
    main()
