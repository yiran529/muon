"""Summarize per-rank targeted training traces under one artifact root."""

from __future__ import annotations

import argparse
import json
import statistics

from pathlib import Path

from benchmark.compressed_muon.profiler_trace import summarize_training_trace


def _mean_mapping(cells, key):
    categories = sorted({category for cell in cells for category in cell[key]})
    return {
        category: statistics.mean(cell[key].get(category, 0.0) for cell in cells)
        for category in categories
    }


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
        actual_ranks = {rank["rank"] for rank in cell["ranks"]}
        if actual_ranks != expected_ranks:
            raise SystemExit(
                f"rank set mismatch for {cell['cell']}: "
                f"expected {sorted(expected_ranks)}, got {sorted(actual_ranks)}"
            )


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
            "mode": cell_dir.name.split("-r", 1)[0],
            "ranks": traces,
            "rank_max_cpu_ranges_ms": rank_max_ranges,
            "rank_max_collective_time_ms": rank_max_collectives,
            "rank_max_nccl_union_time_ms": max(item["nccl_union_time_ms"] for item in traces),
            "rank_max_exposed_nccl_time_ms": max(item["exposed_nccl_time_ms"] for item in traces),
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
            "mean_rank_max_cpu_ranges_ms": _mean_mapping(mode_cells, "rank_max_cpu_ranges_ms"),
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
