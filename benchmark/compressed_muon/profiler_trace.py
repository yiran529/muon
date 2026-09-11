"""Offline Chrome trace attribution for benchmark communication ranges."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_CATEGORY_NAMES = {
    "DDP bucket All-Reduce": "ddp_gradient",
    "benchmark/forward_backward": "compute",
    "benchmark/optimizer": "optimizer",
    "arc/projection": "arc_projection",
    "arc/seed": "arc_seed",
    "arc/topk": "arc_topk",
    "arc/selected_values": "arc_selected_values",
    "arc/dense_uncompressed": "arc_dense_uncompressed",
    "arc/ef21m": "arc_ef21m",
    "muon/newton_schulz": "muon_newton_schulz",
    "muon/result_collective": "muon_result",
    "train/profile_window": "profile_window",
    "train/final_microstep": "final_microstep",
    "train/final_forward": "final_forward",
    "train/final_backward": "final_backward",
    "train/gradient_norm": "gradient_norm",
    "train/optimizer": "optimizer",
    "arc/state_stack": "arc_state_stack",
    "arc/sketch_compute": "arc_sketch_compute",
    "arc/sketch_wait": "arc_sketch_wait",
    "arc/gather": "arc_gather",
    "arc/selected_values_wait": "arc_selected_values_wait",
    "arc/scatter": "arc_scatter",
    "arc/state_copy": "arc_state_copy",
    "muon/result_collective_wait": "muon_result_wait",
    "arc_hook/local_prepare": "arc_hook_local_prepare",
    "arc_hook/dense": "arc_hook_dense",
    "arc_hook/sketch": "arc_hook_sketch",
    "arc_hook/topk": "arc_hook_topk",
    "arc_hook/selected_values": "arc_hook_selected_values",
    "arc_hook/finalize": "arc_hook_finalize",
    "arc_hook/bucket_ready": "arc_hook_bucket_ready",
    "arc_hook/future_complete": "arc_hook_future_complete",
    "greedylore_hook/bucket_ready": "greedylore_hook_bucket_ready",
    "greedylore_hook/dense": "greedylore_hook_dense",
    "greedylore_hook/local_svd": "greedylore_hook_local_svd",
    "greedylore_hook/basis_broadcast": "greedylore_hook_basis_broadcast",
    "greedylore_hook/score": "greedylore_hook_score",
    "greedylore_hook/score_plus_aux_allreduce": "greedylore_hook_score_plus_aux_allreduce",
    "greedylore_hook/topr": "greedylore_hook_topr",
    "greedylore_hook/factor": "greedylore_hook_factor",
    "greedylore_hook/factor_allreduce": "greedylore_hook_factor_allreduce",
    "greedylore_hook/error": "greedylore_hook_error",
    "greedylore_hook/reconstruction": "greedylore_hook_reconstruction",
    "greedylore_hook/future_complete": "greedylore_hook_future_complete",
}

_OPERATION_BY_CATEGORY = {
    "arc_seed": "broadcast",
    "greedylore_hook_basis_broadcast": "broadcast",
}

_GREEDY_LORE_COLLECTIVES = {
    "greedylore_hook_dense",
    "greedylore_hook_basis_broadcast",
    "greedylore_hook_score_plus_aux_allreduce",
    "greedylore_hook_factor_allreduce",
}

_COLLECTIVE_CATEGORIES = {
    "ddp_gradient", "arc_seed", "arc_sketch", "arc_selected_values",
    "arc_dense_uncompressed", "muon_result", "arc_hook_dense",
    "arc_hook_sketch", "arc_hook_selected_values", *_GREEDY_LORE_COLLECTIVES,
}

_ARC_HOOK_COLLECTIVES = {
    "arc_hook_dense", "arc_hook_sketch", "arc_hook_selected_values",
}

_GRADIENT_COLLECTIVES = {
    "ddp_gradient", "arc_sketch", "arc_selected_values",
    "arc_dense_uncompressed", *_ARC_HOOK_COLLECTIVES, *_GREEDY_LORE_COLLECTIVES,
}

_GREEDY_LORE_LOCAL_CATEGORIES = {
    "greedylore_hook_local_svd",
    "greedylore_hook_score",
    "greedylore_hook_topr",
    "greedylore_hook_factor",
    "greedylore_hook_error",
    "greedylore_hook_reconstruction",
}

_HOOK_LOCAL_CATEGORIES = {
    "arc_hook_local_prepare", "arc_hook_topk", "arc_hook_finalize",
    *_GREEDY_LORE_LOCAL_CATEGORIES,
}


def _events(trace: Any) -> list[dict[str, Any]]:
    if isinstance(trace, (str, Path)):
        with Path(trace).open() as f:
            trace = json.load(f)
    if isinstance(trace, dict):
        trace = trace.get("traceEvents", trace.get("events", []))
    if not isinstance(trace, list):
        raise ValueError("Chrome trace must contain a traceEvents list")
    return [e for e in trace if isinstance(e, dict)]


def _args(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args", {})
    return args if isinstance(args, dict) else {}


def _correlation(event: dict[str, Any]):
    args = _args(event)
    # Kineto's GPU kernels carry both a CUDA launch correlation and the CPU
    # record's External id. The latter is what joins record_param_comms to the
    # kernel; synthetic/older traces may only provide correlation.
    for key in ("External id", "external_id", "external id", "correlation", "correlation_id"):
        if key in args:
            return args[key]
        if key in event:
            return event[key]
    return None


def _duration(event: dict[str, Any]) -> float:
    return float(event.get("dur", 0.0))


def _interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    total = 0.0
    start = end = None
    for a, b in ordered:
        if start is None:
            start, end = a, b
        elif a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    if start is not None:
        total += end - start
    return total


def _overlap(a: Iterable[tuple[float, float]], b: Iterable[tuple[float, float]]) -> float:
    aa = _merge_intervals(a); bb = _merge_intervals(b); i = j = 0; total = 0.0
    while i < len(aa) and j < len(bb):
        left, right = max(aa[i][0], bb[j][0]), min(aa[i][1], bb[j][1])
        if right > left: total += right - left
        if aa[i][1] < bb[j][1]: i += 1
        else: j += 1
    return total


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _range_category(name: str) -> str | None:
    if name in _CATEGORY_NAMES: return _CATEGORY_NAMES[name]
    for range_name, category in _CATEGORY_NAMES.items():
        if name.startswith(range_name + " "):
            return category
        if name.startswith(range_name + "/"):
            return category
    lower = name.lower().replace("-", "_")
    if "ddp" in lower and ("reduce" in lower or "bucket" in lower): return "ddp_gradient"
    if "arc" in lower and "sketch" in lower: return "arc_sketch"
    if "selected" in lower and ("value" in lower or "arc" in lower): return "arc_selected_values"
    if "result" in lower and "collective" in lower: return "muon_result"
    return None


def _numeric_arg(event: dict[str, Any], key: str) -> int:
    args = _args(event)
    if key in args:
        return int(args[key])
    for value in args.values():
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.startswith("{"):
                continue
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if key in decoded:
                return int(decoded[key])
    match = re.search(rf"(?:^|[ /]){re.escape(key)}=(\d+)(?:$| )", str(event.get("name", "")))
    if match:
        return int(match.group(1))
    return 0


def _is_nccl(event: dict[str, Any]) -> bool:
    """Return true only for GPU NCCL kernels, not CPU/GPU annotations."""
    name = str(event.get("name", "")).lower()
    category = str(event.get("cat", "")).lower()
    if "nccl" not in name:
        return False
    return category in {"kernel", "cuda_kernel", "gpu"} or "ncclkernel" in name or "nccldevkernel" in name


def _is_cpu_annotation(event: dict[str, Any]) -> bool:
    """Exclude Kineto's GPU mirror of a record_function annotation."""
    return str(event.get("cat", "")).lower() != "gpu_user_annotation"


def attribute_trace(trace: Any) -> dict[str, Any]:
    """Attribute NCCL kernels to the nearest named launch range.

    Chrome timestamps are microseconds; all public durations are milliseconds.
    Correlation IDs are preferred, while temporal containment is a fallback for
    traces exported by older PyTorch profiler versions.
    """
    events = _events(trace)
    ranges = []
    for event in events:
        if event.get("ph") not in {"X", "B"}: continue
        category = _range_category(str(event.get("name", "")))
        if category:
            start = float(event.get("ts", 0)); end = start + _duration(event)
            ranges.append((category, start, end, _correlation(event), event))
    # PyTorch profiler commonly inserts a c10d launch/op node between a user
    # range and the GPU kernel. Propagate the user's category through that node
    # and its correlation/external id.
    correlation_categories = {
        str(r[3]): r[0] for r in ranges
        if r[3] is not None and r[0] in _COLLECTIVE_CATEGORIES
    }
    correlation_ranges = {
        str(r[3]): r[4] for r in ranges
        if r[3] is not None and r[0] in _COLLECTIVE_CATEGORIES
    }
    for event in events:
        name = str(event.get("name", "")).lower()
        corr = _correlation(event)
        if corr is None or ("c10d" not in name and "record_param_comms" not in name
                            and "allreduce" not in name and "allgather" not in name
                            and "broadcast" not in name):
            continue
        parent = _args(event).get("parent_correlation", _args(event).get("parent_external_id"))
        if parent is not None and str(parent) in correlation_categories:
            correlation_categories[str(corr)] = correlation_categories[str(parent)]
            correlation_ranges[str(corr)] = correlation_ranges.get(str(parent))
        else:
            start = float(event.get("ts", 0)); end = start + _duration(event)
            nested = [r for r in ranges if r[0] in _COLLECTIVE_CATEGORIES
                      and r[4].get("pid") == event.get("pid")
                      and r[4].get("tid") == event.get("tid")
                      and r[1] <= start and end <= r[2]]
            if nested:
                correlation_categories[str(corr)] = min(nested, key=lambda r: r[2] - r[1])[0]
                correlation_ranges[str(corr)] = min(nested, key=lambda r: r[2] - r[1])[4]
            elif "record_param_comms" in name and str(_args(event).get("Collective name", "")).lower() in {"allreduce", "all_reduce"}:
                backward = [
                    r for r in ranges
                    if r[0] == "final_backward"
                    and r[4].get("pid") == event.get("pid")
                    and r[1] <= start and end <= r[2]
                ]
                if backward:
                    correlation_categories[str(corr)] = "ddp_gradient"
                    correlation_ranges[str(corr)] = min(
                        backward, key=lambda r: r[2] - r[1]
                    )[4]
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"kernel_count": 0, "duration_us": 0.0, "message_bytes": 0})
    nccl_intervals = []; compute_intervals = []; counted_payloads = set()
    category_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    backward_ranges = [r for r in ranges if r[0] == "final_backward"]
    hook_local_ranges = [r for r in ranges if r[0] in _HOOK_LOCAL_CATEGORIES]
    cpu_events_by_correlation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cpu_event in events:
        cpu_correlation = _correlation(cpu_event)
        if cpu_correlation is None or _is_nccl(cpu_event):
            continue
        if "kernel" in str(cpu_event.get("cat", "")).lower():
            continue
        cpu_events_by_correlation[str(cpu_correlation)].append(cpu_event)
    for event in events:
        if event.get("ph") not in {"X", "B"}: continue
        start = float(event.get("ts", 0)); end = start + _duration(event)
        if _is_nccl(event):
            corr = _correlation(event)
            candidates = [r for r in ranges if r[0] in _COLLECTIVE_CATEGORIES
                          and corr is not None and str(r[3]) == str(corr)]
            if not candidates:
                candidates = [r for r in ranges if r[0] in _COLLECTIVE_CATEGORIES
                              and r[1] <= start and end <= r[2]]
            category = (correlation_categories.get(str(corr)) if corr is not None else None) or (candidates[-1][0] if candidates else None)
            if category is None:
                backward_ranges = [
                    r for r in ranges
                    if r[0] == "final_backward" and r[1] <= start and end <= r[2]
                ]
                category = "ddp_gradient" if backward_ranges else "unattributed"
            item = groups[category]
            item["kernel_count"] += 1; item["duration_us"] += _duration(event)
            nccl_intervals.append((start, end))
            category_intervals[category].append((start, end))
            mapped_range = correlation_ranges.get(str(corr)) if corr is not None else None
            metadata_ranges = candidates or ([next((r for r in ranges if r[4] is mapped_range), None)] if mapped_range is not None else [])
            metadata_ranges = [r for r in metadata_ranges if r is not None]
            payload_key = (category, str(corr)) if corr is not None else (
                category, id(metadata_ranges[-1][4]) if metadata_ranges else id(event)
            )
            if payload_key in counted_payloads:
                continue
            counted_payloads.add(payload_key)
            if metadata_ranges:
                metadata = _args(metadata_ranges[-1][4])
                item["message_bytes"] += _numeric_arg(metadata_ranges[-1][4], "bytes") or int(next((metadata.get(k) for k in ("bytes", "message_bytes", "size_bytes", "collective_bytes") if metadata.get(k) is not None), 0) or 0)
            else:
                metadata = _args(event)
                item["message_bytes"] += int(next((metadata.get(k) for k in ("bytes", "message_bytes", "size_bytes", "collective_bytes") if metadata.get(k) is not None), 0) or 0)
        elif ("kernel" in str(event.get("cat", "")).lower()
              or str(event.get("cat", "")).lower() in {"gpu", "cuda_kernel"}
              or ("stream" in event and "compute" in str(event.get("name", "")).lower())):
            corr = _correlation(event)
            correlated_cpu_events = (
                cpu_events_by_correlation.get(str(corr), [])
                if corr is not None
                else []
            )
            local_candidates = [
                local for local in hook_local_ranges
                if local[1] <= start and end <= local[2]
                or (
                    corr is not None
                    and local[3] is not None
                    and str(corr) == str(local[3])
                )
            ]
            for cpu_event in correlated_cpu_events:
                cpu_start = float(cpu_event.get("ts", 0.0))
                cpu_end = cpu_start + _duration(cpu_event)
                local_candidates.extend(
                    local for local in hook_local_ranges
                    if local[1] <= cpu_start and cpu_end <= local[2]
                )
            local_category = (
                min(local_candidates, key=lambda local: local[2] - local[1])[0]
                if local_candidates
                else None
            )
            if local_category is not None:
                category_intervals[local_category].append((start, end))
            is_backward_compute = any(
                any(
                    backward[1] <= float(cpu_event.get("ts", 0.0))
                    and float(cpu_event.get("ts", 0.0)) + _duration(cpu_event) <= backward[2]
                    for backward in backward_ranges
                )
                for cpu_event in correlated_cpu_events
            )
            if is_backward_compute and local_category is None:
                compute_intervals.append((start, end))
    launch_metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "message_bytes": 0}
    )
    collective_launches = []
    for category, _start, _end, _corr, range_event in ranges:
        if category not in _COLLECTIVE_CATEGORIES:
            continue
        if not _is_cpu_annotation(range_event):
            continue
        if "/payload " not in str(range_event.get("name", "")):
            continue
        message_bytes = _numeric_arg(range_event, "bytes")
        launch_metrics[category]["count"] += 1
        launch_metrics[category]["message_bytes"] += message_bytes
        collective_launches.append({
            "category": category,
            "operation": _OPERATION_BY_CATEGORY.get(category, "all_reduce"),
            "message_bytes": message_bytes,
            "start_us": _start,
        })
    collective_launches.sort(key=lambda item: item["start_us"])

    collectives = []
    for category in sorted(groups):
        item = groups[category]
        launches = launch_metrics.get(category)
        collectives.append({"category": category, "kernel_count": item["kernel_count"],
                            "operation": _OPERATION_BY_CATEGORY.get(category, "all_reduce"),
                            "launch_count": (launches["count"] if launches else item["kernel_count"]),
                            "duration_ms": item["duration_us"] / 1000.0,
                            "message_bytes": (launches["message_bytes"] if launches else item["message_bytes"])})
    union_us = _interval_union(nccl_intervals)
    overlap_us = _overlap(nccl_intervals, compute_intervals)
    arc_intervals = [
        interval
        for category in _ARC_HOOK_COLLECTIVES
        for interval in category_intervals.get(category, [])
    ]
    gradient_intervals = [
        interval
        for category in _GRADIENT_COLLECTIVES
        for interval in category_intervals.get(category, [])
    ]
    greedylore_collective_intervals = [
        interval
        for category in _GREEDY_LORE_COLLECTIVES
        for interval in category_intervals.get(category, [])
    ]
    greedylore_local_intervals = [
        interval
        for category in _GREEDY_LORE_LOCAL_CATEGORIES
        for interval in category_intervals.get(category, [])
    ]
    greedylore_future_ends = [
        end for category, _start, end, _corr, _event in ranges
        if category == "greedylore_hook_future_complete"
    ]
    last_compute_end = max((end for _start, end in compute_intervals), default=None)
    exposed_gradient_tail_us = (
        _interval_union(
            (max(start, last_compute_end), end)
            for start, end in gradient_intervals
            if end > last_compute_end
        )
        if last_compute_end is not None
        else _interval_union(gradient_intervals)
    )
    backward_start = min((item[1] for item in backward_ranges), default=None)
    backward_end = max((item[2] for item in backward_ranges), default=None)
    first_arc_start = min((start for start, _end in arc_intervals), default=None)
    last_arc_end = max((end for _start, end in arc_intervals), default=None)
    compressor_end = max(
        [
            end for _start, end in greedylore_collective_intervals
        ] + [
            end for _start, end in greedylore_local_intervals
        ] + greedylore_future_ends,
        default=None,
    )
    return {
        "nccl_kernel_time_ms": sum(item["duration_ms"] for item in collectives),
        "raw_nccl_total_ms": sum(item["duration_ms"] for item in collectives),
        "nccl_union_time_ms": union_us / 1000.0,
        "compute_overlap_ms": overlap_us / 1000.0,
        "nccl_compute_overlap_ms": overlap_us / 1000.0,
        "exposed_nccl_time_ms": max(0.0, union_us - overlap_us) / 1000.0,
        "exposed_time_ms": max(0.0, union_us - overlap_us) / 1000.0,
        "exposed_is_trace_estimate": True,
        "arc_collective_backward_compute_overlap_ms": (
            _overlap(arc_intervals, compute_intervals) / 1000.0
        ),
        "exposed_gradient_sync_tail_ms": exposed_gradient_tail_us / 1000.0,
        "compressor_critical_path_tail_ms": (
            max(0.0, compressor_end - last_compute_end) / 1000.0
            if last_compute_end is not None and compressor_end is not None
            else None if last_compute_end is None else 0.0
        ),
        "gpu_ranges_ms": {
            category: _interval_union(category_intervals.get(category, [])) / 1000.0
            for category in sorted(_HOOK_LOCAL_CATEGORIES)
            if category_intervals.get(category)
        },
        "first_arc_collective_from_backward_start_ms": (
            (first_arc_start - backward_start) / 1000.0
            if first_arc_start is not None and backward_start is not None
            else None
        ),
        "last_arc_completion_from_backward_end_ms": (
            (last_arc_end - backward_end) / 1000.0
            if last_arc_end is not None and backward_end is not None
            else None
        ),
        "collectives": collectives,
        "collective_launches": collective_launches,
    }


def summarize_training_trace(trace: Any) -> dict[str, Any]:
    """Add named CPU-range accounting to the existing NCCL attribution."""
    events = _events(trace)
    windows = [
        event for event in events
        if event.get("ph") == "X"
        and event.get("name") == "train/profile_window"
        and _is_cpu_annotation(event)
    ]
    if windows:
        window_start = min(float(event.get("ts", 0.0)) for event in windows)
        window_end = max(
            float(event.get("ts", 0.0)) + _duration(event) for event in windows
        )
        events = [
            event for event in events
            if float(event.get("ts", 0.0)) >= window_start
            and float(event.get("ts", 0.0)) + _duration(event) <= window_end
        ]
    result = attribute_trace(events)
    cpu_ranges: dict[str, float] = defaultdict(float)
    for event in events:
        if event.get("ph") != "X":
            continue
        if not _is_cpu_annotation(event):
            continue
        if "/payload " in str(event.get("name", "")):
            continue
        category = _range_category(str(event.get("name", "")))
        if category is not None:
            cpu_ranges[category] += _duration(event) / 1000.0
    total_nccl = sum(item["duration_ms"] for item in result["collectives"])
    unattributed = sum(
        item["duration_ms"]
        for item in result["collectives"]
        if item["category"] == "unattributed"
    )
    result["cpu_ranges_ms"] = dict(sorted(cpu_ranges.items()))
    result["profile_window_ms"] = cpu_ranges.get("profile_window", 0.0)
    result["unattributed_nccl_fraction"] = (
        unattributed / total_nccl if total_nccl else 0.0
    )
    bucket_events = [
        event for event in events
        if event.get("ph") == "X"
        and str(event.get("name", "")).startswith((
            "arc_hook/bucket_ready",
            "greedylore_hook/bucket_ready",
        ))
        and _is_cpu_annotation(event)
    ]
    arc_bucket_events = [
        event for event in bucket_events
        if str(event.get("name", "")).startswith("arc_hook/bucket_ready")
    ]
    greedylore_bucket_events = [
        event for event in bucket_events
        if str(event.get("name", "")).startswith("greedylore_hook/bucket_ready")
    ]
    result["bucket_count"] = len(bucket_events)
    result["bucket_bytes"] = sum(_numeric_arg(event, "bucket_bytes") for event in bucket_events)
    for key in ("arc_bytes", "dense_bytes"):
        result[key] = sum(_numeric_arg(event, key) for event in arc_bucket_events)
    for key in (
        "matrix_bytes",
        "dense_aux_bytes",
        "score_bytes",
        "factor_bytes",
        "basis_bytes",
        "parameter_count",
    ):
        result[key] = sum(_numeric_arg(event, key) for event in greedylore_bucket_events)
    backward_ranges = [
        event for event in events
        if event.get("ph") == "X"
        and event.get("name") == "train/final_backward"
        and _is_cpu_annotation(event)
    ]
    bucket_ready_starts = [float(event.get("ts", 0.0)) for event in bucket_events]
    future_completion_ends = [
        float(event.get("ts", 0.0)) + _duration(event)
        for event in events
        if event.get("ph") == "X"
        and event.get("name") == "arc_hook/future_complete"
        and _is_cpu_annotation(event)
    ]
    backward_start = min(
        (float(event.get("ts", 0.0)) for event in backward_ranges),
        default=None,
    )
    backward_end = max(
        (float(event.get("ts", 0.0)) + _duration(event) for event in backward_ranges),
        default=None,
    )
    result["first_bucket_ready_from_backward_start_ms"] = (
        (min(bucket_ready_starts) - backward_start) / 1000.0
        if bucket_ready_starts and backward_start is not None
        else None
    )
    result["last_hook_future_completion_from_backward_end_ms"] = (
        (max(future_completion_ends) - backward_end) / 1000.0
        if future_completion_ends and backward_end is not None
        else None
    )
    return result


def parse_trace(path: str | Path) -> dict[str, Any]:
    return attribute_trace(path)


def summarize_trace(path: str | Path) -> dict[str, Any]:
    return attribute_trace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = attribute_trace(args.trace)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output: Path(args.output).write_text(payload)
    else: print(payload, end="")


if __name__ == "__main__": main()
