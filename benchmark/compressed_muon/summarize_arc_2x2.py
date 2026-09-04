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


def _check_invariants(results: list[dict[str, Any]], *, require_timing=False, require_profiler=False) -> None:
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
        if require_timing and not item.get("timing_ms", {}).get("step_samples"):
            raise ValueError("missing timing samples")
        communication = item.get("communication")
        required_comm = ("dense_gradient_bytes", "arc_seed_bytes", "arc_sketch_bytes",
                         "arc_selected_values_bytes", "uncompressed_bytes")
        if not isinstance(communication, dict) or any(key not in communication or communication[key] is None for key in required_comm):
            raise ValueError("missing communication byte field")
        throughput = item.get("throughput")
        memory = item.get("memory")
        if not isinstance(throughput, dict) or throughput.get("tokens_per_second") is None:
            raise ValueError("missing throughput field")
        if not isinstance(memory, dict) or any(memory.get(key) is None for key in ("peak_allocated_mib", "peak_reserved_mib")):
            raise ValueError("missing memory field")
        profiler = item.get("profiler")
        if require_profiler and (not isinstance(profiler, dict) or profiler.get("nccl_kernel_time_ms") is None or not profiler.get("collectives")):
            raise ValueError("missing profiler communication data")


def _variant_summary(results: list[dict[str, Any]], *, allow_missing_profiler=False) -> dict[str, Any]:
    def per_result_samples(item, key):
        values = item.get("timing_ms", {}).get(key, [])
        if not values: raise ValueError(f"missing timing_ms.{key} samples")
        return statistics.mean(values)
    step = [per_result_samples(item, "step_samples") for item in results]
    optimizer = [per_result_samples(item, "optimizer_samples") for item in results]
    nccl = [item.get("profiler", {}).get("nccl_kernel_time_ms") for item in results]
    nccl = [v for v in nccl if v is not None]
    gradient = []
    if not allow_missing_profiler or nccl:
        gradient = [_gradient_comm_ms(item, item.get("sync_mode") == "arc") for item in results]
    throughput = [item.get("throughput", {}).get("tokens_per_second", 0.0) for item in results]
    allocated = [item.get("memory", {}).get("peak_allocated_mib", 0.0) for item in results]
    reserved = [item.get("memory", {}).get("peak_reserved_mib", 0.0) for item in results]
    return {"step": sample_stats(step), "optimizer": sample_stats(optimizer),
            "nccl": sample_stats(nccl) if nccl else (None if allow_missing_profiler else sample_stats(nccl)),
            "gradient_comm": sample_stats(gradient) if gradient else (None if allow_missing_profiler else sample_stats(gradient)),
            "throughput": sample_stats(throughput),
            "memory": {"allocated_mib": sample_stats(allocated), "reserved_mib": sample_stats(reserved)}}


def _communication_bytes(item: dict[str, Any], arc: bool) -> float:
    c = item.get("communication", {})
    required = ("arc_seed_bytes", "arc_sketch_bytes", "arc_selected_values_bytes", "uncompressed_bytes") if arc else ("dense_gradient_bytes", "uncompressed_bytes")
    if any(key not in c for key in required):
        raise ValueError("missing communication byte field")
    if arc:
        return sum(float(c.get(k, 0)) for k in ("arc_seed_bytes", "arc_sketch_bytes", "arc_selected_values_bytes", "uncompressed_bytes"))
    return float(c.get("dense_gradient_bytes", 0)) + float(c.get("uncompressed_bytes", 0))


def _gradient_comm_ms(item: dict[str, Any], arc: bool) -> float | None:
    collectives = item.get("profiler", {}).get("collectives", []) or []
    names = {"arc_seed", "arc_sketch", "arc_selected_values",
             "arc_dense_uncompressed", "dense_uncompressed"} if arc else {"ddp_gradient"}
    values = [float(c.get("duration_ms", 0.0)) for c in collectives if c.get("category") in names]
    if not values:
        raise ValueError("missing recognized gradient communication category")
    return sum(values)


def _ratio(dense: float, arc: float) -> float | None:
    return None if dense == 0 else 1.0 - arc / dense


def summarize_results(results: list[dict[str, Any]], profiler_results=None) -> dict[str, Any]:
    """Summarize three-or-more repetitions, optionally containing dense and ARC variants."""
    if results and not isinstance(results[0], dict):
        results = load_results(results)
    if profiler_results is not None and profiler_results and not isinstance(profiler_results[0], dict):
        profiler_results = load_results(profiler_results)
    if len(results) < 3: raise ValueError("at least three result JSON files are required")
    separate_inputs = profiler_results is not None
    _check_invariants(results, require_timing=True, require_profiler=not separate_inputs)
    if separate_inputs:
        if len(profiler_results) < 3: raise ValueError("at least three profiler summaries are required")
        _check_invariants(profiler_results, require_profiler=True)
        _check_invariants(results + profiler_results)
        timing_comm = {item["sync_mode"]: item["communication"] for item in results}
        for item in results + profiler_results:
            if item["communication"] != timing_comm[item["sync_mode"]]:
                raise ValueError(
                    f"mismatched communication metadata for {item['sync_mode']} timing/profile inputs"
                )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results: grouped[item["sync_mode"]].append(item)
    if any(len(items) < 3 for items in grouped.values()):
        raise ValueError("at least three independent files are required per optimizer/sync cell")
    if set(grouped) != {"dense", "arc"}:
        raise ValueError("timing inputs must contain both dense and arc cells")
    variants = {mode: _variant_summary(items, allow_missing_profiler=separate_inputs) for mode, items in grouped.items()}
    if separate_inputs:
        profile_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in profiler_results: profile_grouped[item["sync_mode"]].append(item)
        if any(len(items) < 3 for items in profile_grouped.values()):
            raise ValueError("at least three independent profiler files are required per cell")
        if set(profile_grouped) != {"dense", "arc"}:
            raise ValueError("profiler inputs must contain both dense and arc cells")
        profile_stats = {}
        for mode, items in profile_grouped.items():
            nccl = [item["profiler"]["nccl_kernel_time_ms"] for item in items]
            profile_stats[mode] = sample_stats(nccl)
            profile_gradient = [_gradient_comm_ms(item, mode == "arc") for item in items]
            profile_stats[mode + "_gradient"] = sample_stats(profile_gradient)
        for mode in variants:
            if mode not in profile_stats: raise ValueError(f"missing profiler cell for {mode}")
            variants[mode]["nccl"] = profile_stats[mode]
            variants[mode]["gradient_comm"] = profile_stats[mode + "_gradient"]
    primary_mode = "arc" if "arc" in variants else next(iter(variants))
    primary = variants[primary_mode]
    dense_items = grouped.get("dense", [])
    arc_items = grouped.get("arc", [])
    dense_bytes = statistics.mean([_communication_bytes(i, False) for i in dense_items]) if dense_items else None
    arc_bytes = statistics.mean([_communication_bytes(i, True) for i in arc_items]) if arc_items else None
    profile_dense = profile_grouped.get("dense", []) if separate_inputs else dense_items
    profile_arc = profile_grouped.get("arc", []) if separate_inputs else arc_items
    dense_grad = statistics.mean([_gradient_comm_ms(i, False) for i in profile_dense if _gradient_comm_ms(i, False) is not None]) if profile_dense else None
    arc_grad = statistics.mean([_gradient_comm_ms(i, True) for i in profile_arc if _gradient_comm_ms(i, True) is not None]) if profile_arc else None
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
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = summarize_results(load_results(args.results), load_results(args.profiles))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 2 if result["unstable"] else 0


if __name__ == "__main__": raise SystemExit(main())
