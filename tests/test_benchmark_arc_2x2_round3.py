import copy

import pytest
import torch

from benchmark.compressed_muon.benchmark_arc_2x2 import (
    build_result_skeleton,
    parse_args,
    reduce_correctness_flags,
)
from benchmark.compressed_muon.profiler_trace import attribute_trace
from benchmark.compressed_muon.summarize_arc_2x2 import summarize_results


def test_trace_maps_each_c10d_correlation_to_its_own_range_metadata():
    trace = {"traceEvents": [
        {"name": "arc/sketch", "ph": "X", "pid": 1, "tid": 1, "ts": 0, "dur": 40, "args": {"bytes": 10}},
        {"name": "arc/sketch", "ph": "X", "pid": 1, "tid": 1, "ts": 100, "dur": 40, "args": {"bytes": 20}},
        {"name": "c10d::allreduce", "ph": "X", "pid": 1, "tid": 1, "ts": 10, "dur": 10, "args": {"External id": 11}},
        {"name": "c10d::allreduce", "ph": "X", "pid": 1, "tid": 1, "ts": 110, "dur": 10, "args": {"External id": 22}},
        {"name": "ncclKernel_AllReduce", "ph": "X", "pid": 2, "tid": 1, "ts": 50, "dur": 5, "args": {"External id": 11}},
        {"name": "ncclKernel_AllReduce", "ph": "X", "pid": 2, "tid": 1, "ts": 150, "dur": 5, "args": {"External id": 22}},
    ]}
    sketch = next(x for x in attribute_trace(trace)["collectives"] if x["category"] == "arc_sketch")
    assert sketch["message_bytes"] == 30


def test_correctness_flag_reduction_is_logical_and_without_dist():
    assert reduce_correctness_flags(True, True) == (True, True)
    assert reduce_correctness_flags(False, True) == (False, True)


def test_cli_accepts_explicit_profile_inputs():
    config = parse_args(["--experiment-id", "CM002a-adamw-dense-gpt60m-ddp-ws4-s42", "--optimizer", "adamw", "--sync", "dense", "--model", "gpt60m", "--warmup-steps", "20", "--measure-steps", "100", "--seed", "42", "--output", "summary.json", "--profiles", "p1.json", "p2.json", "p3.json"])
    assert config.profiles == ["p1.json", "p2.json", "p3.json"]


def test_summary_reports_gradient_comm_statistics_from_profile_repetitions():
    from tests.test_benchmark_arc_2x2 import _config
    base = build_result_skeleton(_config())
    timing, profiles = [], []
    for mode in ("dense", "arc"):
        for _ in range(3):
            t = copy.deepcopy(base); t["sync_mode"] = mode; t["timing_ms"]["step_samples"] = [10.0 if mode == "dense" else 7.5]; t["timing_ms"]["optimizer_samples"] = [1.0]; t["communication"]["dense_gradient_bytes"] = 100; t["communication"]["arc_seed_bytes"] = 1; t["communication"]["arc_sketch_bytes"] = 9; t["communication"]["arc_selected_values_bytes"] = 15; timing.append(t)
            p = copy.deepcopy(t); p["timing_ms"] = {}; p["profiler"]["nccl_kernel_time_ms"] = 10.0 if mode == "dense" else 7.5; p["profiler"]["collectives"] = [{"category": "ddp_gradient" if mode == "dense" else "arc_sketch", "duration_ms": p["profiler"]["nccl_kernel_time_ms"]}]; profiles.append(p)
    result = summarize_results(timing, profiles)
    assert result["variants"]["arc"]["gradient_comm"]["mean"] == pytest.approx(7.5)


def test_summary_rejects_missing_evidence_fields():
    from tests.test_benchmark_arc_2x2 import _config
    base = build_result_skeleton(_config())
    values = []
    for mode in ("dense", "arc"):
        for _ in range(3):
            item = copy.deepcopy(base); item["sync_mode"] = mode; item["timing_ms"]["step_samples"] = [1.0]; item["timing_ms"]["optimizer_samples"] = [1.0]; item["profiler"]["nccl_kernel_time_ms"] = 1.0; item["profiler"]["collectives"] = [{"category": "ddp_gradient", "duration_ms": 1.0}]; values.append(item)
    del values[0]["throughput"]["tokens_per_second"]
    with pytest.raises(ValueError, match="throughput"):
        summarize_results(values)
