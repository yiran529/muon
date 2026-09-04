"""Summarize repeated ARC benchmark JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


INVARIANTS = ("transport", "model", "workload")


def sample_stats(values: Iterable[float]) -> dict[str, float]:
    values = [float(v) for v in values]
    if not values: raise ValueError("at least one sample is required")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "mean_ms": mean, "std_ms": std,
            "cv": abs(std / mean) if mean else 0.0, "n": len(values)}


def _check_invariants(results: list[dict[str, Any]]) -> None:
    if not results: raise ValueError("no benchmark result files supplied")
    for field in (*INVARIANTS, "optimizer"):
        if field not in results[0] or any(field not in item for item in results):
            raise ValueError(f"missing {field} field")
        expected = results[0].get(field)
        if any(item.get(field) != expected for item in results[1:]):
            raise ValueError(f"mismatched {field} fields")
    for item in results:
        if item.get("schema_version") != 1:
            raise ValueError("unsupported or missing schema_version")
        for field in ("optimizer", "sync_mode"):
            if not item.get(field): raise ValueError(f"missing {field} field")
        profiler = item.get("profiler")
        if not isinstance(profiler, dict) or profiler.get("nccl_kernel_time_ms") is None or not profiler.get("collectives"):
            raise ValueError("missing profiler communication data")


def _variant_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    def per_result_samples(item, key):
        values = item.get("timing_ms", {}).get(key, [])
        if not values: raise ValueError(f"missing timing_ms.{key} samples")
        return statistics.mean(values)
    step = [per_result_samples(item, "step_samples") for item in results]
    optimizer = [per_result_samples(item, "optimizer_samples") for item in results]
    nccl = [item.get("profiler", {}).get("nccl_kernel_time_ms") for item in results]
    nccl = [v for v in nccl if v is not None]
    throughput = [item.get("throughput", {}).get("tokens_per_second", 0.0) for item in results]
    allocated = [item.get("memory", {}).get("peak_allocated_mib", 0.0) for item in results]
    reserved = [item.get("memory", {}).get("peak_reserved_mib", 0.0) for item in results]
    return {"step": sample_stats(step), "optimizer": sample_stats(optimizer),
            "nccl": sample_stats(nccl), "throughput": sample_stats(throughput),
            "memory": {"allocated_mib": sample_stats(allocated), "reserved_mib": sample_stats(reserved)}}


def _communication_bytes(item: dict[str, Any], arc: bool) -> float:
    c = item.get("communication", {})
    if arc:
        return sum(float(c.get(k, 0)) for k in ("arc_seed_bytes", "arc_sketch_bytes", "arc_selected_values_bytes", "uncompressed_bytes"))
    return float(c.get("dense_gradient_bytes", 0)) + float(c.get("uncompressed_bytes", 0))


def _gradient_comm_ms(item: dict[str, Any], arc: bool) -> float | None:
    collectives = item.get("profiler", {}).get("collectives", []) or []
    names = {"arc_sketch", "arc_selected_values", "arc_ef21m"} if arc else {"ddp_gradient"}
    values = [float(c.get("duration_ms", 0.0)) for c in collectives if c.get("category") in names]
    if values: return sum(values)
    return item.get("profiler", {}).get("nccl_kernel_time_ms")


def _ratio(dense: float, arc: float) -> float | None:
    return None if dense == 0 else 1.0 - arc / dense


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize three-or-more repetitions, optionally containing dense and ARC variants."""
    if results and not isinstance(results[0], dict):
        results = load_results(results)
    if len(results) < 3: raise ValueError("at least three result JSON files are required")
    _check_invariants(results)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results: grouped[item["sync_mode"]].append(item)
    if any(len(items) < 3 for items in grouped.values()):
        raise ValueError("at least three independent files are required per optimizer/sync cell")
    variants = {mode: _variant_summary(items) for mode, items in grouped.items()}
    primary_mode = "arc" if "arc" in variants else next(iter(variants))
    primary = variants[primary_mode]
    dense_items = grouped.get("dense", [])
    arc_items = grouped.get("arc", [])
    dense_bytes = statistics.mean([_communication_bytes(i, False) for i in dense_items]) if dense_items else None
    arc_bytes = statistics.mean([_communication_bytes(i, True) for i in arc_items]) if arc_items else None
    dense_grad = statistics.mean([_gradient_comm_ms(i, False) for i in dense_items if _gradient_comm_ms(i, False) is not None]) if dense_items else None
    arc_grad = statistics.mean([_gradient_comm_ms(i, True) for i in arc_items if _gradient_comm_ms(i, True) is not None]) if arc_items else None
    ratios = {
        "R_bytes": _ratio(dense_bytes, arc_bytes) if dense_bytes is not None and arc_bytes is not None else None,
        "R_grad_comm": _ratio(dense_grad, arc_grad) if dense_grad is not None and arc_grad is not None else None,
        "R_step": _ratio(variants["dense"]["step"]["mean_ms"], variants["arc"]["step"]["mean_ms"]) if "dense" in variants and "arc" in variants else None,
    }
    max_cv = max((v["step"]["cv"] for v in variants.values()), default=0.0)
    result = {**primary, "variants": variants, "ratios": ratios,
              "max_step_cv": max_cv, "unstable": max_cv > 0.05,
              "invariants": {field: results[0][field] for field in INVARIANTS}}
    return result


def load_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text()) for path in paths]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = summarize_results(load_results(args.results))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 2 if result["unstable"] else 0


if __name__ == "__main__": raise SystemExit(main())
