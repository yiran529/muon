"""Summarize per-rank targeted training traces under one artifact root."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics

from pathlib import Path

from benchmark.compressed_muon.profiler_trace import summarize_training_trace


_FINAL_TIMING_RE = re.compile(r"step_avg:(?P<value>[^\s]*)")
_TIMING_CELL_RE = re.compile(r"^(?P<mode>.+)-timing-r(?P<repeat>[0-9]+)$")
_LOSS_RE = re.compile(r"val_loss:(?P<value>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))")
_MEMORY_RE = re.compile(
    r"Peak memory consumption: (?P<value>[0-9]+) MiB\b"
)


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


def _cell_logs(cell_dir: Path) -> str:
    return "\n".join(
        path.read_text(errors="replace")
        for path in (cell_dir / "stdout.log", cell_dir / "stderr.log")
        if path.is_file()
    )


def _final_timing(logs: str, cell: str) -> float:
    matches = list(_FINAL_TIMING_RE.finditer(logs))
    if not matches:
        raise SystemExit(f"missing final timing for {cell}")
    final_token = matches[-1].group("value")
    if not final_token.endswith("ms"):
        raise SystemExit(f"invalid final timing for {cell}")
    try:
        final_timing = float(final_token[:-2])
    except ValueError as exc:
        raise SystemExit(f"invalid final timing for {cell}") from exc
    if not math.isfinite(final_timing) or final_timing <= 0:
        raise SystemExit(f"invalid final timing for {cell}")
    return final_timing


def _timing_cell_record(artifact_root: Path, cell_name: str) -> dict:
    cell_dir = artifact_root / cell_name
    if not cell_dir.is_dir():
        raise SystemExit(f"missing timing cell directory for {cell_name}")
    command_path = cell_dir / "command.txt"
    if not command_path.is_file() or not command_path.read_text().strip():
        raise SystemExit(f"missing required command for timing cell {cell_name}")
    exit_path = cell_dir / "exit_code.txt"
    if not exit_path.is_file() or exit_path.read_text().strip() != "0":
        raise SystemExit(f"nonzero or missing exit code for timing cell {cell_name}")
    finished_path = cell_dir / "finished_at.txt"
    if not finished_path.is_file() or not finished_path.read_text().strip():
        raise SystemExit(f"missing completion timestamp for timing cell {cell_name}")
    logs = _cell_logs(cell_dir)
    if re.search(
        r"\b(OOM|out of memory|timeout|timed out|traceback)\b", logs, re.I
    ):
        raise SystemExit(f"failure marker found in logs for timing cell {cell_name}")
    step_avg_ms = _final_timing(logs, cell_name)
    loss_values = [
        float(match.group("value")) for match in _LOSS_RE.finditer(logs)
    ]
    memory_values = [
        int(match.group("value")) for match in _MEMORY_RE.finditer(logs)
    ]
    match = _TIMING_CELL_RE.fullmatch(cell_name)
    if match is None:
        raise SystemExit(f"invalid timing cell name {cell_name}")
    record = {
        "mode": match.group("mode"),
        "repeat": int(match.group("repeat")),
        "step_avg_ms": step_avg_ms,
        "exit_code": 0,
    }
    if loss_values and math.isfinite(loss_values[-1]):
        record["final_validation_loss"] = loss_values[-1]
    if memory_values:
        record["peak_allocated_mib"] = memory_values[-1]
    return record


def _timing_summary_from_records(plan: dict, records: list[dict]) -> dict:
    tokens_per_update = int(plan.get("global_batch_size", 0)) * int(
        plan.get("sequence_length", 0)
    )
    modes = {}
    for mode in sorted({record["mode"] for record in records}):
        mode_records = sorted(
            (record for record in records if record["mode"] == mode),
            key=lambda record: record["repeat"],
        )
        values = [record["step_avg_ms"] for record in mode_records]
        mean_step_ms = statistics.mean(values)
        sample_stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        mode_summary = {
            "cells": [
                {key: value for key, value in record.items() if key != "mode"}
                for record in mode_records
            ],
            "mean_step_ms": mean_step_ms,
            "median_step_ms": statistics.median(values),
            "sample_stdev_step_ms": sample_stdev,
            "cv_percent": 100 * sample_stdev / mean_step_ms,
            "peak_allocated_mib": max(
                (
                    record["peak_allocated_mib"]
                    for record in mode_records
                    if "peak_allocated_mib" in record
                ),
                default=None,
            ),
        }
        if tokens_per_update:
            mode_summary["mean_throughput_tokens_per_second"] = statistics.mean(
                tokens_per_update * 1000 / value for value in values
            )
        modes[mode] = mode_summary

    payload = {
        "schema_version": 1,
        "tokens_per_update": tokens_per_update,
        "timing_scope": "profiler-off timing cells",
        "modes": modes,
    }
    if plan.get("timing_num_iterations") is not None:
        measured_updates = int(plan["timing_num_iterations"]) - int(
            plan.get("timing_warmup_steps", 0)
        )
        interval = int(plan.get("greedy_lore", {}).get("update_interval", 0))
        if measured_updates > 0 and interval > 0 and measured_updates % interval == 0:
            periods = measured_updates // interval
            period_label = "period" if periods == 1 else "periods"
            count_label = "one" if periods == 1 else str(periods)
            payload["timing_scope"] = (
                "profiler-off, "
                f"{measured_updates} measured updates = {count_label} complete "
                f"interval-{interval} {period_label}"
            )

    def percentile(values, fraction):
        values = sorted(values)
        return values[round(fraction * (len(values) - 1))]

    def bootstrap_interval(values, *, samples=100_000, seed=44):
        rng = random.Random(seed)
        means = [
            statistics.mean(rng.choice(values) for _ in values)
            for _ in range(samples)
        ]
        return [percentile(means, 0.025), percentile(means, 0.975)]

    mode_by_repeat = {
        mode: {
            row["repeat"]: row["step_avg_ms"]
            for row in mode_summary["cells"]
        }
        for mode, mode_summary in modes.items()
    }
    preferred_pairs = [
        (
            "greedylore_local_svd_full",
            "dense",
            "full_isolation_vs_dense",
        ),
        (
            "greedylore_local_svd_partial",
            "dense",
            "partial_isolation_vs_dense",
        ),
        (
            "greedylore_local_svd_full",
            "greedylore_local_svd_partial",
            "full_vs_partial_isolation",
        ),
        ("greedylore_local_svd", "dense", "local_svd_vs_dense"),
        ("greedylore_broadcast", "dense", "broadcast_vs_dense"),
        (
            "greedylore_local_svd",
            "greedylore_broadcast",
            "local_svd_vs_broadcast",
        ),
    ]
    paired = {}
    for left, right, pair_name in preferred_pairs:
        if left not in mode_by_repeat or right not in mode_by_repeat:
            continue
        repeats = sorted(set(mode_by_repeat[left]) & set(mode_by_repeat[right]))
        if not repeats:
            continue
        differences = [
            mode_by_repeat[left][index] - mode_by_repeat[right][index]
            for index in repeats
        ]
        ratios = [
            mode_by_repeat[left][index] / mode_by_repeat[right][index]
            for index in repeats
        ]
        paired[pair_name] = {
            "left": left,
            "right": right,
            "paired_differences_ms": differences,
            "mean_difference_ms": statistics.mean(differences),
            "paired_ratios": ratios,
            "mean_ratio": statistics.mean(ratios),
            "mean_percent_change": 100 * (statistics.mean(ratios) - 1),
            "bootstrap_95pct_mean_difference_ms": bootstrap_interval(differences),
        }
    if paired:
        payload["paired"] = paired
    return payload


def _validate_timing_summary(payload: object, timing_cells: list[str]) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("modes"), dict):
        raise SystemExit("timing-summary.json must contain a modes object")
    expected = {}
    for cell_name in timing_cells:
        match = _TIMING_CELL_RE.fullmatch(cell_name)
        if match is None:
            raise SystemExit(f"invalid timing cell name {cell_name}")
        expected.setdefault(match.group("mode"), set()).add(int(match.group("repeat")))
    modes = payload["modes"]
    if set(modes) != set(expected):
        raise SystemExit(
            f"timing summary mode set mismatch: expected {sorted(expected)}, "
            f"got {sorted(modes)}"
        )
    for mode, repeats in expected.items():
        mode_summary = modes[mode]
        rows = mode_summary.get("cells") if isinstance(mode_summary, dict) else None
        if not isinstance(rows, list):
            raise SystemExit(f"timing summary cells missing for {mode}")
        actual_repeats = {row.get("repeat") for row in rows if isinstance(row, dict)}
        if actual_repeats != repeats or len(rows) != len(repeats):
            raise SystemExit(f"timing summary repeat set mismatch for {mode}")
        for row in rows:
            if not isinstance(row, dict) or row.get("exit_code") != 0:
                raise SystemExit(f"timing summary contains nonzero exit for {mode}")
            value = row.get("step_avg_ms")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise SystemExit(f"timing summary contains invalid step timing for {mode}")


def _write_or_validate_timing_summary(
    artifact_root: Path, plan: dict, timing_cells: list[str], records: list[dict]
) -> None:
    timing_path = artifact_root / "timing-summary.json"
    if timing_path.is_file():
        try:
            existing = json.loads(timing_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid timing-summary.json: {exc}") from exc
        _validate_timing_summary(existing, timing_cells)
    payload = _timing_summary_from_records(plan, records)
    _validate_timing_summary(payload, timing_cells)
    timing_path.write_text(json.dumps(payload, indent=2) + "\n")


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
        logs = _cell_logs(cell_dir)
        if re.search(r"\b(OOM|out of memory|timeout|timed out|traceback)\b", logs, re.I):
            raise SystemExit(f"failure marker found in logs for {cell['cell']}")
        if plan.get("require_final_timing"):
            _final_timing(logs, cell["cell"])
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

    timing_cells = list(plan.get("timing_cells", []))
    if timing_cells:
        records = [_timing_cell_record(artifact_root, cell) for cell in timing_cells]
        _write_or_validate_timing_summary(artifact_root, plan, timing_cells, records)


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
            "rank_max_greedylore_local_gpu_union_ms": max(
                item["greedylore_local_gpu_union_ms"] for item in traces
            ),
            "rank_max_greedylore_local_backward_compute_overlap_ms": max(
                item["greedylore_local_backward_compute_overlap_ms"]
                for item in traces
            ),
            "rank_max_exposed_greedylore_local_gpu_ms": max(
                item["exposed_greedylore_local_gpu_ms"] for item in traces
            ),
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
    if require_plan:
        _validate_plan_completeness(artifact_root, cells)
    elif not cells:
        raise SystemExit("no rank traces found")
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
            "mean_rank_max_greedylore_local_gpu_union_ms": statistics.mean(
                cell["rank_max_greedylore_local_gpu_union_ms"]
                for cell in mode_cells
            ),
            "mean_rank_max_greedylore_local_backward_compute_overlap_ms": statistics.mean(
                cell["rank_max_greedylore_local_backward_compute_overlap_ms"]
                for cell in mode_cells
            ),
            "mean_rank_max_exposed_greedylore_local_gpu_ms": statistics.mean(
                cell["rank_max_exposed_greedylore_local_gpu_ms"]
                for cell in mode_cells
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
