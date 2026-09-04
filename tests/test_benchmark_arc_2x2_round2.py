import copy

import pytest
import torch

from benchmark.compressed_muon.benchmark_arc_2x2 import (
    _correctness,
    _run_steps,
    build_result_skeleton,
    parse_args,
    validate_config,
)
from benchmark.compressed_muon.collective_observer import CollectiveObserver, signatures_agree
from benchmark.compressed_muon.summarize_arc_2x2 import summarize_results


def test_observer_signature_agreement_detects_missing_and_reordered_events():
    first = CollectiveObserver(); first.record("arc/sketch", "all_reduce", 8, torch.bfloat16, 16)
    second = CollectiveObserver(); second.record("arc/sketch", "all_reduce", 8, torch.bfloat16, 16)
    assert signatures_agree([first.signature(), second.signature()])
    second.record("arc/selected_values", "all_reduce", 4, torch.bfloat16, 8)
    assert not signatures_agree([first.signature(), second.signature()])
    reordered = CollectiveObserver(); reordered.record("arc/selected_values", "all_reduce", 4, torch.bfloat16, 8); reordered.record("arc/sketch", "all_reduce", 8, torch.bfloat16, 16)
    assert not signatures_agree([second.signature(), reordered.signature()])


def test_correctness_reports_nonfinite_actual_loss():
    config = parse_args(["--experiment-id", "CM002a-adamw-dense-gpt60m-ddp-ws4-s42", "--optimizer", "adamw", "--sync", "dense", "--model", "gpt60m", "--warmup-steps", "20", "--measure-steps", "100", "--seed", "42", "--output", "x.json"])
    model = torch.nn.Linear(2, 2)
    correctness = _correctness(model, config, torch.tensor(float("nan")), CollectiveObserver())
    assert correctness["finite_loss"] is False


def test_trace_hierarchy_maps_user_range_through_nested_c10d_launch():
    trace = {"traceEvents": [
        {"name": "arc/sketch", "ph": "X", "pid": 1, "tid": 2, "ts": 0, "dur": 35, "args": {"External id": 700, "bytes": 4096}},
        {"name": "c10d::allreduce", "ph": "X", "pid": 1, "tid": 2, "ts": 10, "dur": 20, "args": {"External id": 701}},
        {"name": "ncclDevKernel_AllReduce", "ph": "X", "pid": 2, "tid": 3, "ts": 40, "dur": 5, "args": {"External id": 701}},
    ]}
    from benchmark.compressed_muon.profiler_trace import attribute_trace
    result = attribute_trace(trace)
    sketch = next(item for item in result["collectives"] if item["category"] == "arc_sketch")
    assert sketch["kernel_count"] == 1 and sketch["message_bytes"] == 4096


def test_separate_timing_and_profiler_inputs_are_aggregated():
    from tests.test_benchmark_arc_2x2 import _config
    base = build_result_skeleton(_config())
    timing, profiles = [], []
    for value in (10.0, 11.0, 9.0):
        item = copy.deepcopy(base); item["sync_mode"] = "dense"; item["timing_ms"]["step_samples"] = [value]; item["timing_ms"]["optimizer_samples"] = [5.0]; item["communication"]["dense_gradient_bytes"] = 100; timing.append(item)
        profile = copy.deepcopy(base); profile["sync_mode"] = "dense"; profile["timing_ms"] = {}; profile["profiler"]["nccl_kernel_time_ms"] = 10.0; profile["profiler"]["collectives"] = [{"category": "ddp_gradient", "duration_ms": 10.0}]; profile["communication"]["dense_gradient_bytes"] = 100; profiles.append(profile)
    for _ in range(3):
        item = copy.deepcopy(base); item["sync_mode"] = "arc"; item["timing_ms"]["step_samples"] = [7.5]; item["timing_ms"]["optimizer_samples"] = [4.0]; item["communication"]["arc_seed_bytes"] = 1; item["communication"]["arc_sketch_bytes"] = 9; item["communication"]["arc_selected_values_bytes"] = 15; timing.append(item)
        profile = copy.deepcopy(base); profile["sync_mode"] = "arc"; profile["timing_ms"] = {}; profile["profiler"]["nccl_kernel_time_ms"] = 7.5; profile["profiler"]["collectives"] = [{"category": "arc_sketch", "duration_ms": 7.5}]; profile["communication"]["arc_seed_bytes"] = 1; profile["communication"]["arc_sketch_bytes"] = 9; profile["communication"]["arc_selected_values_bytes"] = 15; profiles.append(profile)
    summary = summarize_results(timing, profiles)
    assert summary["ratios"]["R_step"] == pytest.approx(0.25)
    assert summary["ratios"]["R_grad_comm"] == pytest.approx(0.25)
