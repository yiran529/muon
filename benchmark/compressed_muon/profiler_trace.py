"""Offline Chrome trace attribution for benchmark communication ranges."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_CATEGORY_NAMES = {
    "benchmark/forward_backward": "compute",
    "benchmark/optimizer": "optimizer",
    "arc/projection": "arc_projection",
    "arc/topk": "arc_topk",
    "arc/selected_values": "arc_selected_values",
    "arc/ef21m": "arc_ef21m",
    "muon/newton_schulz": "muon_newton_schulz",
    "muon/result_collective": "muon_result",
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
    for key in ("correlation", "correlation_id", "External id", "external_id", "external id"):
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
    lower = name.lower().replace("-", "_")
    if "ddp" in lower and ("reduce" in lower or "bucket" in lower): return "ddp_gradient"
    if "arc" in lower and "sketch" in lower: return "arc_sketch"
    if "selected" in lower and ("value" in lower or "arc" in lower): return "arc_selected_values"
    if "result" in lower and "collective" in lower: return "muon_result"
    return None


def _is_nccl(event: dict[str, Any]) -> bool:
    text = " ".join(str(event.get(k, "")) for k in ("name", "cat", "s", "stream")).lower()
    return "nccl" in text


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
    correlation_categories = {str(r[3]): r[0] for r in ranges if r[3] is not None}
    for event in events:
        name = str(event.get("name", "")).lower()
        corr = _correlation(event)
        if corr is None or ("c10d" not in name and "allreduce" not in name
                            and "allgather" not in name and "broadcast" not in name):
            continue
        parent = _args(event).get("parent_correlation", _args(event).get("parent_external_id"))
        if parent is not None and str(parent) in correlation_categories:
            correlation_categories[str(corr)] = correlation_categories[str(parent)]
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"kernel_count": 0, "duration_us": 0.0, "message_bytes": 0})
    nccl_intervals = []; compute_intervals = []
    for event in events:
        if event.get("ph") not in {"X", "B"}: continue
        start = float(event.get("ts", 0)); end = start + _duration(event)
        if _is_nccl(event):
            corr = _correlation(event)
            candidates = [r for r in ranges if corr is not None and str(r[3]) == str(corr)]
            if not candidates:
                candidates = [r for r in ranges if r[1] <= start and end <= r[2]]
            category = (correlation_categories.get(str(corr)) if corr is not None else None) or (candidates[-1][0] if candidates else "unattributed")
            item = groups[category]
            item["kernel_count"] += 1; item["duration_us"] += _duration(event)
            nccl_intervals.append((start, end))
            if candidates:
                metadata = _args(candidates[-1][4])
                item["message_bytes"] += int(next((metadata.get(k) for k in ("bytes", "message_bytes", "size_bytes", "collective_bytes") if metadata.get(k) is not None), 0) or 0)
            else:
                metadata = _args(event)
                item["message_bytes"] += int(next((metadata.get(k) for k in ("bytes", "message_bytes", "size_bytes", "collective_bytes") if metadata.get(k) is not None), 0) or 0)
        elif ("kernel" in str(event.get("cat", "")).lower()
              or str(event.get("cat", "")).lower() in {"gpu", "cuda_kernel"}
              or ("stream" in event and "compute" in str(event.get("name", "")).lower())):
            compute_intervals.append((start, end))
    collectives = []
    for category in sorted(groups):
        item = groups[category]
        collectives.append({"category": category, "kernel_count": item["kernel_count"],
                            "duration_ms": item["duration_us"] / 1000.0,
                            "message_bytes": item["message_bytes"]})
    union_us = _interval_union(nccl_intervals)
    overlap_us = _overlap(nccl_intervals, compute_intervals)
    return {
        "nccl_kernel_time_ms": sum(item["duration_ms"] for item in collectives),
        "raw_nccl_total_ms": sum(item["duration_ms"] for item in collectives),
        "nccl_union_time_ms": union_us / 1000.0,
        "compute_overlap_ms": overlap_us / 1000.0,
        "nccl_compute_overlap_ms": overlap_us / 1000.0,
        "exposed_nccl_time_ms": max(0.0, union_us - overlap_us) / 1000.0,
        "exposed_time_ms": max(0.0, union_us - overlap_us) / 1000.0,
        "exposed_is_trace_estimate": True,
        "collectives": collectives,
    }


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
