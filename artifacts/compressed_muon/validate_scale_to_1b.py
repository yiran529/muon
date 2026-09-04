#!/usr/bin/env python3
"""Validate one ARC-TopK scale-out timing or profiler JSON artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


OBSERVED_NAMES = {
    "ddp_gradient": "ddp_gradient",
    "arc/seed": "arc_seed",
    "arc/sketch": "arc_sketch",
    "arc/selected_values": "arc_selected_values",
    "arc/dense_uncompressed": "arc_dense_uncompressed",
    "muon/result_collective": "muon_result",
}


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def finite(value, name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{name} is not finite")


def expected_categories(optimizer: str, sync: str) -> set[str]:
    if sync == "dense":
        return {"ddp_gradient"}
    categories = {"arc_seed", "arc_sketch", "arc_selected_values", "arc_dense_uncompressed"}
    if optimizer == "muon":
        categories.add("muon_result")
    return categories


def check_identity(item: dict, args: argparse.Namespace) -> None:
    for key, expected in {
        "experiment_id": args.experiment_id,
        "optimizer": args.optimizer,
        "sync_mode": args.sync,
        "transport": args.transport,
    }.items():
        if item.get(key) != expected:
            fail(f"{key}={item.get(key)!r}, expected {expected!r}")
    if item.get("model", {}).get("label") != args.model:
        fail("model label mismatch")
    workload = item.get("workload", {})
    for key, expected in {
        "world_size": args.expected_world_size,
        "local_batch": 1,
        "sequence_length": 256,
        "gradient_accumulation": 1,
        "dtype": "bfloat16",
    }.items():
        if workload.get(key) != expected:
            fail(f"workload.{key}={workload.get(key)!r}, expected {expected!r}")
    arc = item.get("arc", {})
    for key, expected in {"ratio": 0.2, "projection_rank": 4, "eta": 0.1, "seed": 42, "start_compress_step": 0}.items():
        if arc.get(key) != expected:
            fail(f"arc.{key}={arc.get(key)!r}, expected {expected!r}")


def check_correctness(item: dict) -> None:
    correctness = item.get("correctness")
    if not isinstance(correctness, dict):
        fail("missing correctness")
    for key in ("finite_loss", "finite_parameters", "parameter_checksum_agreement"):
        if correctness.get(key) is not True:
            fail(f"correctness.{key} is not true")
    signature = correctness.get("collective_signature", {})
    if signature.get("all_ranks_match") is not True:
        fail("collective signatures do not agree")
    for key in ("parameter_checksum", "parameter_checksum_squared"):
        finite(correctness.get(key), f"correctness.{key}")


def observed_bytes(profiler: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in profiler.get("observed_collectives", []) or []:
        category = OBSERVED_NAMES.get(entry.get("category"))
        if category is None:
            continue
        value = int(entry.get("bytes", 0))
        if value <= 0:
            fail(f"observer category {category} has no positive bytes")
        result[category] = result.get(category, 0) + value
    return result


def check_profiler(item: dict, root: Path, args: argparse.Namespace) -> None:
    profiler = item.get("profiler")
    if not isinstance(profiler, dict):
        fail("missing profiler")
    finite(profiler.get("nccl_kernel_time_ms"), "profiler.nccl_kernel_time_ms")
    if float(profiler["nccl_kernel_time_ms"]) <= 0:
        fail("profiler NCCL kernel time is not positive")
    trace_path = profiler.get("trace_path")
    if not trace_path:
        fail("missing profiler trace path")
    trace = Path(trace_path)
    if not trace.is_absolute():
        trace = root / trace
    if not trace.is_file():
        fail(f"missing trace {trace}")
    try:
        json.loads(trace.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid trace {trace}: {exc}")
    categories = {entry.get("category"): entry for entry in profiler.get("collectives", []) or []}
    required = expected_categories(args.optimizer, args.sync)
    if not required.issubset(categories):
        fail(f"missing profiler categories {sorted(required - set(categories))}")
    for category in required:
        entry = categories[category]
        if int(entry.get("kernel_count", 0)) <= 0:
            fail(f"category {category} has no kernels")
        if int(entry.get("message_bytes", 0)) <= 0:
            fail(f"category {category} has no message bytes")
        finite(entry.get("duration_ms"), f"collectives.{category}.duration_ms")
    unattributed = categories.get("unattributed")
    if unattributed and (int(unattributed.get("kernel_count", 0)) != 0 or float(unattributed.get("duration_ms", 0)) != 0):
        fail("unattributed active NCCL kernels present")
    observed = observed_bytes(profiler)
    for category in required:
        if observed.get(category) != int(categories[category]["message_bytes"]):
            fail(f"trace/observer byte disagreement for {category}")


def validate(args: argparse.Namespace) -> None:
    try:
        item = json.loads(Path(args.path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON: {exc}")
    if item.get("schema_version") != 1:
        fail("schema_version must be 1")
    check_identity(item, args)
    check_correctness(item)
    if args.kind == "timing":
        samples = item.get("timing_ms", {}).get("step_samples", [])
        if len(samples) != args.timing_samples:
            fail(f"expected {args.timing_samples} timing samples, got {len(samples)}")
        for index, sample in enumerate(samples):
            finite(sample, f"timing_ms.step_samples[{index}]")
            if float(sample) <= 0:
                fail("timing sample is not positive")
        if args.expected_world_size > 1:
            categories = expected_categories(args.optimizer, args.sync)
            observed = observed_bytes(item.get("profiler", {}))
            signature = item["correctness"]["collective_signature"].get("per_rank", [])
            signature_categories = {row[0] for rank in signature for row in rank}
            if not categories.intersection(signature_categories | set(observed)) == categories:
                fail("timing output missing expected nonzero collective categories")
    else:
        check_profiler(item, Path(args.root), args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--kind", choices=("timing", "profile"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), required=True)
    parser.add_argument("--sync", choices=("dense", "arc"), required=True)
    parser.add_argument("--model", choices=("gpt130m", "gpt350m", "gpt1b"), required=True)
    parser.add_argument("--transport", choices=("normal", "p2p_disabled"), required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--timing-samples", type=int, default=100)
    args = parser.parse_args()
    try:
        validate(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
