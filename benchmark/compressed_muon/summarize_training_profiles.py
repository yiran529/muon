"""Summarize per-rank targeted training traces under one artifact root."""

from __future__ import annotations

import argparse
import json
import statistics
import re

from pathlib import Path

from benchmark.compressed_muon.profiler_trace import summarize_training_trace


def _mean_mapping(cells, key):
    categories = sorted({category for cell in cells for category in cell[key]})
    return {
        category: statistics.mean(cell[key].get(category, 0.0) for cell in cells)
        for category in categories
    }


def _max_optional(values):
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _mean_optional(values):
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def _validate_plan_completeness(artifact_root: Path, cells: list[dict]) -> None:
    plan_path = artifact_root / "plan.json"
    if not plan_path.is_file():
        raise SystemExit("plan.json is required for a complete experiment summary")
    plan = json.loads(plan_path.read_text())
    expected_cells = set(plan["cells"])
    actual_cells = {cell["cell"] for cell in cells}
    if actual_cells != expected_cells:
        raise SystemExit(
            f"cell set mismatch: expected {sorted(expected_cells)}, got {sorted(actual_cells)}"
        )
    expected_ranks = set(range(int(plan["world_size"])))
    for cell in cells:
        cell_dir = artifact_root / cell["cell"]
        exit_path = cell_dir / "exit_code.txt"
        if not exit_path.is_file() or exit_path.read_text().strip() != "0":
            raise SystemExit(f"nonzero or missing exit code for {cell['cell']}")
        logs = "\n".join(
            path.read_text(errors="replace")
            for path in (cell_dir / "stdout.log", cell_dir / "stderr.log")
            if path.is_file()
        )
        if re.search(r"\b(OOM|out of memory|timeout|timed out|traceback)\b", logs, re.I):
            raise SystemExit(f"failure marker found in logs for {cell['cell']}")
        if plan.get("require_final_timing") and "step_avg:" not in logs:
            raise SystemExit(f"missing final timing for {cell['cell']}")
        actual_ranks = {rank["rank"] for rank in cell["ranks"]}
        if actual_ranks != expected_ranks:
            raise SystemExit(
                f"rank set mismatch for {cell['cell']}: "
                f"expected {sorted(expected_ranks)}, got {sorted(actual_ranks)}"
            )
        signatures = []
        for rank in cell["ranks"]:
            signature = (rank["bucket_count"],) + tuple(
                (
                    item["category"],
                    item["operation"],
                    item["message_bytes"],
                )
                for item in rank["collective_launches"]
                if item["category"].startswith(("arc_hook_", "greedylore_hook_"))
            )
            signatures.append(signature)
        if len(set(signatures)) != 1:
            raise SystemExit(f"rank-divergent hook signature for {cell['cell']}")
        if cell["mode"].startswith("arc") and any(
            item["category"] == "arc_seed"
            for rank in cell["ranks"]
            for item in rank["collectives"]
        ):
            raise SystemExit(f"seed collective found in ARC cell {cell['cell']}")


def summarize_profile_root(artifact_root: Path, *, require_plan: bool = False) -> dict:
    cells = []
    for cell_dir in sorted(artifact_root.glob("*-r*")):
        traces = []
        for trace_path in sorted((cell_dir / "profiler").glob("rank-*.json")):
            item = summarize_training_trace(trace_path)
            item["rank"] = int(trace_path.stem.split("-")[-1])
            traces.append(item)
        if not traces:
            continue
        categories = sorted({k for item in traces for k in item["cpu_ranges_ms"]})
        rank_max_ranges = {
            category: max(item["cpu_ranges_ms"].get(category, 0.0) for item in traces)
            for category in categories
        }
        gpu_categories = sorted({k for item in traces for k in item["gpu_ranges_ms"]})
        rank_max_gpu_ranges = {
            category: max(item["gpu_ranges_ms"].get(category, 0.0) for item in traces)
            for category in gpu_categories
        }
        collective_categories = sorted({
            item["category"] for trace in traces for item in trace["collectives"]
        })
        rank_max_collectives = {
            category: max(
                next((item["duration_ms"] for item in trace["collectives"] if item["category"] == category), 0.0)
                for trace in traces
            )
            for category in collective_categories
        }
        cells.append({
            "cell": cell_dir.name,
            "mode": re.sub(r"-r\d+$", "", cell_dir.name),
            "ranks": traces,
            "rank_max_cpu_ranges_ms": rank_max_ranges,
            "rank_max_gpu_ranges_ms": rank_max_gpu_ranges,
            "rank_max_collective_time_ms": rank_max_collectives,
            "rank_max_nccl_union_time_ms": max(item["nccl_union_time_ms"] for item in traces),
            "rank_max_exposed_nccl_time_ms": max(item["exposed_nccl_time_ms"] for item in traces),
            "rank_max_arc_collective_backward_compute_overlap_ms": max(
                item["arc_collective_backward_compute_overlap_ms"] for item in traces
            ),
            "rank_max_exposed_gradient_sync_tail_ms": max(
                item["exposed_gradient_sync_tail_ms"] for item in traces
            ),
            "rank_max_compressor_critical_path_tail_ms": _max_optional(
                item["compressor_critical_path_tail_ms"] for item in traces
            ),
            "max_unattributed_nccl_fraction": max(item["unattributed_nccl_fraction"] for item in traces),
        })
    if not cells:
        raise SystemExit("no rank traces found")
    if require_plan:
        _validate_plan_completeness(artifact_root, cells)
    payload = {"schema_version": 1, "cells": cells}
    for mode in sorted({cell["mode"] for cell in cells}):
        mode_cells = [cell for cell in cells if cell["mode"] == mode]
        values = [cell["rank_max_cpu_ranges_ms"].get("profile_window", 0.0) for cell in mode_cells]
        payload.setdefault("by_mode", {})[mode] = {
            "n": len(values),
            "mean_rank_max_profile_window_ms": statistics.mean(values),
            "mean_rank_max_nccl_union_time_ms": statistics.mean(
                cell["rank_max_nccl_union_time_ms"] for cell in mode_cells
            ),
            "mean_rank_max_exposed_nccl_time_ms": statistics.mean(
                cell["rank_max_exposed_nccl_time_ms"] for cell in mode_cells
            ),
            "mean_rank_max_arc_collective_backward_compute_overlap_ms": statistics.mean(
                cell["rank_max_arc_collective_backward_compute_overlap_ms"]
                for cell in mode_cells
            ),
            "mean_rank_max_exposed_gradient_sync_tail_ms": statistics.mean(
                cell["rank_max_exposed_gradient_sync_tail_ms"]
                for cell in mode_cells
            ),
            "mean_rank_max_compressor_critical_path_tail_ms": _mean_optional(
                cell["rank_max_compressor_critical_path_tail_ms"]
                for cell in mode_cells
            ),
            "mean_rank_max_cpu_ranges_ms": _mean_mapping(mode_cells, "rank_max_cpu_ranges_ms"),
            "mean_rank_max_gpu_ranges_ms": _mean_mapping(mode_cells, "rank_max_gpu_ranges_ms"),
            "mean_rank_max_collective_time_ms": _mean_mapping(mode_cells, "rank_max_collective_time_ms"),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-plan", action="store_true")
    args = parser.parse_args()
    payload = summarize_profile_root(args.artifact_root, require_plan=args.require_plan)
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
