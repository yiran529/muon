import json
import math

import pytest
import torch

from benchmark.compressed_muon.benchmark_arc_2x2 import (
    MODEL_PRESETS,
    BenchmarkConfig,
    build_result_skeleton,
    parse_args,
    validate_config,
    communication_result,
)
from benchmark.compressed_muon.profiler_trace import attribute_trace, _interval_union
from benchmark.compressed_muon.summarize_arc_2x2 import summarize_results


def test_model_presets_are_exact():
    assert MODEL_PRESETS == {
        "gpt60m": {"model_dim": 512, "n_layer": 4, "n_head": 8},
        "gpt130m": {"model_dim": 768, "n_layer": 8, "n_head": 12},
        "gpt350m": {"model_dim": 1024, "n_layer": 20, "n_head": 16},
        "gpt1b": {"model_dim": 1536, "n_layer": 30, "n_head": 24},
    }


def _config(**overrides):
    values = dict(
        experiment_id="CM002a-adamw-dense-gpt60m-ddp-ws4-s42",
        optimizer="adamw", sync="dense", model="gpt60m",
        warmup_steps=20, measure_steps=100, seed=42,
        output=None, profile_output=None, compile_model=False,
        world_size=4, formal=True,
    )
    values.update(overrides)
    return BenchmarkConfig(**values)


def test_config_validation_accepts_formal_cell():
    validate_config(_config())


@pytest.mark.parametrize("field,value", [("warmup_steps", 19), ("measure_steps", 99), ("world_size", 2)])
def test_config_validation_rejects_formal_minimums(field, value):
    with pytest.raises(ValueError):
        validate_config(_config(**{field: value}))


def test_config_validation_rejects_mismatched_experiment_id():
    with pytest.raises(ValueError, match="experiment_id"):
        validate_config(_config(experiment_id="CM002a-muon-arc-gpt60m-ddp-ws4-s42"))


def test_cli_parses_required_switches():
    config = parse_args([
        "--experiment-id", "CM002a-muon-arc-gpt60m-ddp-ws4-s42",
        "--optimizer", "muon", "--sync", "arc", "--model", "gpt60m",
        "--warmup-steps", "20", "--measure-steps", "100", "--seed", "42",
        "--output", "result.json",
        "--no-compile-model",
    ])
    assert config.optimizer == "muon" and config.sync == "arc"


def test_cli_exposes_independent_profile_mode():
    config = parse_args([
        "--experiment-id", "CM002d-m001-muon-arc-gpt60m-ddp-ws4-s42",
        "--optimizer", "muon", "--sync", "arc", "--model", "gpt60m",
        "--warmup-steps", "5", "--measure-steps", "5", "--seed", "42",
        "--output", "profile-summary.json", "--profile-output", "trace.json",
        "--profile", "--smoke",
    ])
    assert config.profile_only is True


def test_result_skeleton_has_schema_and_metadata():
    result = build_result_skeleton(_config())
    assert result["schema_version"] == 1
    assert result["model"]["label"] == "gpt60m"
    assert result["workload"]["dtype"] == "bfloat16"
    assert result["optimizer_config"] == {"lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01}
    assert set(result["communication"]) == {
        "dense_gradient_bytes", "arc_seed_bytes", "arc_sketch_bytes",
        "arc_selected_values_bytes", "uncompressed_bytes",
    }


def test_communication_result_uses_compressed_step_and_schema_keys():
    result = communication_result(_config(sync="arc"), [[torch.zeros(10, 8, dtype=torch.bfloat16)]], [torch.zeros(7, 8, dtype=torch.bfloat16)])
    assert result["arc_seed_bytes"] == 8
    assert result["arc_sketch_bytes"] > 0
    assert result["arc_selected_values_bytes"] > 0


def test_formal_plan_id_and_accumulation_validation():
    validate_config(_config(sync="arc", experiment_id="CM002b-m001-adamw-arc-gpt60m-ddp-ws4-s42"))
    with pytest.raises(ValueError, match="gradient_accumulation"):
        validate_config(_config(gradient_accumulation=2, formal=False))


def test_p2p_transport_requires_expected_environment(monkeypatch):
    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    monkeypatch.setenv("NCCL_SHM_DISABLE", "0")
    with pytest.raises(ValueError, match="NCCL_P2P_DISABLE"):
        validate_config(_config(transport="p2p_disabled"))


def _trace_fixture():
    return {
        "traceEvents": [
            {"name": "benchmark/forward_backward", "ph": "X", "ts": 0, "dur": 100, "pid": 1, "tid": 1, "args": {"correlation": 10}},
            {"name": "DDP bucket All-Reduce", "ph": "X", "ts": 10, "dur": 30, "pid": 1, "tid": 1, "args": {"correlation": 11, "bytes": 100}},
            {"name": "ncclKernel_AllReduce", "ph": "X", "ts": 20, "dur": 20, "pid": 2, "tid": 3, "args": {"correlation": 11}},
            {"name": "arc/sketch", "ph": "X", "ts": 100, "dur": 50, "pid": 1, "tid": 1, "args": {"correlation": 12, "bytes": 40}},
            {"name": "ncclKernel_AllReduce", "ph": "X", "ts": 110, "dur": 30, "pid": 2, "tid": 4, "args": {"correlation": 12}},
            {"name": "arc/selected_values", "ph": "X", "ts": 150, "dur": 40, "pid": 1, "tid": 1, "args": {"correlation": 13, "bytes": 20}},
            {"name": "ncclKernel_AllReduce", "ph": "X", "ts": 160, "dur": 10, "pid": 2, "tid": 4, "args": {"correlation": 13}},
            {"name": "muon/result_collective", "ph": "X", "ts": 200, "dur": 20, "pid": 1, "tid": 1, "args": {"correlation": 14, "bytes": 80}},
            {"name": "ncclKernel_AllGather", "ph": "X", "ts": 205, "dur": 10, "pid": 2, "tid": 4, "args": {"correlation": 14}},
            {"name": "aten::matmul", "ph": "X", "cat": "kernel", "ts": 15, "dur": 25, "pid": 2, "tid": 5},
            {"name": "ncclKernel_AllReduce", "ph": "X", "ts": 300, "dur": 7, "pid": 2, "tid": 4},
        ]
    }


def test_trace_attribution_preserves_unattributed_kernel_and_overlap_math():
    summary = attribute_trace(_trace_fixture())
    by_category = {item["category"]: item for item in summary["collectives"]}
    assert by_category["ddp_gradient"]["kernel_count"] == 1
    assert by_category["ddp_gradient"]["duration_ms"] == pytest.approx(0.02)
    assert by_category["ddp_gradient"]["message_bytes"] == 100
    assert by_category["arc_sketch"]["message_bytes"] == 40
    assert by_category["arc_selected_values"]["kernel_count"] == 1
    assert by_category["muon_result"]["kernel_count"] == 1
    assert by_category["unattributed"]["kernel_count"] == 1
    assert summary["nccl_union_time_ms"] == pytest.approx(0.077)
    assert summary["compute_overlap_ms"] == pytest.approx(0.02)
    assert summary["exposed_nccl_time_ms"] == pytest.approx(0.057)


def test_trace_union_flushes_tail_and_overlap_merges_concurrent_intervals():
    assert _interval_union([(0, 10)]) == pytest.approx(10)
    trace = {"traceEvents": [
        {"name": "arc/sketch", "ph": "X", "ts": 0, "dur": 100, "args": {"correlation": 1, "bytes": 1}},
        {"name": "ncclKernel_AllReduce", "ph": "X", "ts": 0, "dur": 50, "args": {"correlation": 1}},
        {"name": "ncclKernel_AllReduce", "ph": "X", "ts": 20, "dur": 50, "args": {"correlation": 1}},
        {"name": "compute_kernel", "cat": "kernel", "ph": "X", "ts": 40, "dur": 20},
    ]}
    summary = attribute_trace(trace)
    assert summary["nccl_union_time_ms"] == pytest.approx(0.07)
    assert summary["compute_overlap_ms"] == pytest.approx(0.02)
    assert summary["exposed_nccl_time_ms"] == pytest.approx(0.05)


def test_result_schema_exposes_smoke_correctness_and_profile_mode():
    result = build_result_skeleton(_config())
    assert result["correctness"] == {
        "finite_loss": None, "finite_parameters": None,
        "parameter_checksum": None, "parameter_checksum_agreement": None,
        "collective_signature": {
            "all_ranks_match": None, "per_rank": []
        }
    }


def test_summary_math_and_invariant_validation():
    base = build_result_skeleton(_config())
    cells = []
    for value in (10.0, 11.0, 9.0):
        item = json.loads(json.dumps(base))
        item["sync_mode"] = "dense"
        item["timing_ms"]["step_samples"] = [value]
        item["timing_ms"]["optimizer_samples"] = [value / 2]
        item["throughput"]["tokens_per_second"] = 100.0
        item["communication"]["dense_gradient_bytes"] = 100
        item["communication"]["arc_seed_bytes"] = 1
        item["communication"]["arc_sketch_bytes"] = 9
        item["communication"]["arc_selected_values_bytes"] = 15
        item["profiler"]["nccl_kernel_time_ms"] = value
        item["profiler"]["collectives"] = [{"category": "ddp_gradient", "duration_ms": value}]
        cells.append(item)
    for value in (7.5, 7.5, 7.5):
        item = json.loads(json.dumps(base))
        item["sync_mode"] = "arc"
        item["timing_ms"]["step_samples"] = [value]
        item["timing_ms"]["optimizer_samples"] = [value / 2]
        item["throughput"]["tokens_per_second"] = 100.0
        item["communication"]["uncompressed_bytes"] = 0
        item["communication"]["arc_seed_bytes"] = 1
        item["communication"]["arc_sketch_bytes"] = 9
        item["communication"]["arc_selected_values_bytes"] = 15
        item["profiler"]["nccl_kernel_time_ms"] = value
        item["profiler"]["collectives"] = [{"category": "arc_sketch", "duration_ms": value}]
        cells.append(item)
    summary = summarize_results(cells)
    assert summary["variants"]["dense"]["step"]["mean_ms"] == pytest.approx(10.0)
    assert summary["variants"]["dense"]["step"]["std_ms"] == pytest.approx(1.0)
    assert summary["variants"]["dense"]["step"]["cv"] == pytest.approx(0.1)
    assert summary["ratios"]["R_step"] == pytest.approx(0.25)
    assert summary["ratios"]["R_bytes"] == pytest.approx(0.75)
    assert summary["ratios"]["R_grad_comm"] == pytest.approx(0.25)

    bad = json.loads(json.dumps(cells[0]))
    bad["transport"] = "p2p_disabled"
    with pytest.raises(ValueError, match="transport"):
        summarize_results(cells + [bad])


def test_summary_rejects_missing_profiler_and_short_variant_cell():
    base = build_result_skeleton(_config())
    item = json.loads(json.dumps(base))
    item["timing_ms"]["step_samples"] = [10.0]
    item["timing_ms"]["optimizer_samples"] = [5.0]
    with pytest.raises(ValueError, match="profiler"):
        summarize_results([item, json.loads(json.dumps(item)), json.loads(json.dumps(item))])
