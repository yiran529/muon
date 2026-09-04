import copy

import pytest
import torch

import benchmark.compressed_muon.benchmark_arc_2x2 as bench
from benchmark.compressed_muon.profiler_trace import attribute_trace
from benchmark.compressed_muon.summarize_arc_2x2 import summarize_results
from dion.collective_observer import CollectiveObserver, aggregate_observed


def test_dense_muon_disables_internal_distributed_mesh(monkeypatch):
    captured = {}

    class FakeMuon:
        def __init__(self, groups, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bench, "Muon", FakeMuon)
    monkeypatch.setattr(bench, "_optimizer_parameters", lambda model: ([torch.nn.Parameter(torch.zeros(2, 2))], []))
    config = bench.BenchmarkConfig(
        experiment_id="CM002c-muon-dense-gpt60m-ddp-ws4-s42",
        optimizer="muon", sync="dense", model="gpt60m", warmup_steps=20,
        measure_steps=100, seed=42, output=None, profile_output=None,
        world_size=4, formal=True,
    )
    bench.build_optimizer(config, object(), process_group=object())
    assert captured["distributed_mesh"] is None


def test_actual_world_size_must_match_configuration(monkeypatch):
    monkeypatch.setattr(bench.dist, "is_available", lambda: True)
    monkeypatch.setattr(bench.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(bench.dist, "get_world_size", lambda group=None: 2)
    with pytest.raises(RuntimeError, match="actual distributed world size 2.*configured 4"):
        bench.validate_actual_world_size(4)


def test_arc_single_rank_context_does_not_require_ddp_wrapper():
    config = bench.BenchmarkConfig(
        experiment_id="smoke-adamw-arc-gpt60m-ddp-ws1-s42",
        optimizer="adamw", sync="arc", model="gpt60m", warmup_steps=1,
        measure_steps=1, seed=42, output=None, profile_output=None,
        world_size=1, formal=False,
    )
    with bench.sync_context(config, torch.nn.Linear(2, 2)):
        pass


def test_unlinked_nccl_inside_compute_range_stays_unattributed():
    trace = {"traceEvents": [
        {"name": "benchmark/forward_backward", "ph": "X", "pid": 1, "tid": 1,
         "ts": 0, "dur": 100, "args": {}},
        {"name": "ncclKernel_AllReduce", "ph": "X", "pid": 2, "tid": 1,
         "ts": 10, "dur": 20, "args": {}},
    ]}
    result = attribute_trace(trace)
    assert result["collectives"] == [{
        "category": "unattributed", "kernel_count": 1,
        "duration_ms": pytest.approx(0.02), "message_bytes": 0,
    }]


def test_parent_payload_is_counted_once_for_multiple_nccl_kernels():
    trace = {"traceEvents": [
        {"name": "arc/sketch", "ph": "X", "pid": 1, "tid": 1, "ts": 0,
         "dur": 40, "args": {"External id": 11, "bytes": 30}},
        {"name": "ncclKernel_1", "ph": "X", "pid": 2, "tid": 1, "ts": 50,
         "dur": 5, "args": {"External id": 11}},
        {"name": "ncclKernel_2", "ph": "X", "pid": 2, "tid": 1, "ts": 60,
         "dur": 5, "args": {"External id": 11}},
    ]}
    sketch = next(x for x in attribute_trace(trace)["collectives"] if x["category"] == "arc_sketch")
    assert sketch["kernel_count"] == 2
    assert sketch["message_bytes"] == 30


def test_trace_uses_record_param_comms_as_c10d_to_kernel_bridge():
    trace = {"traceEvents": [
        {"name": "arc/sketch", "ph": "X", "cat": "user_annotation", "pid": 1,
         "tid": 1, "ts": 0, "dur": 100, "args": {}},
        {"name": "c10d::allreduce_", "ph": "X", "cat": "cpu_op", "pid": 1,
         "tid": 1, "ts": 10, "dur": 70, "args": {"External id": 21}},
        {"name": "record_param_comms", "ph": "X", "cat": "cpu_op", "pid": 1,
         "tid": 1, "ts": 20, "dur": 50, "args": {"External id": 22}},
        {"name": "nccl:all_reduce", "ph": "X", "cat": "user_annotation", "pid": 1,
         "tid": 1, "ts": 30, "dur": 20, "args": {"External id": 23}},
        {"name": "ncclDevKernel_AllReduce", "ph": "X", "cat": "kernel", "pid": 0,
         "tid": 9, "ts": 200, "dur": 5, "args": {"External id": 22}},
        {"name": "nccl:all_reduce", "ph": "X", "cat": "gpu_user_annotation", "pid": 0,
         "tid": 9, "ts": 200, "dur": 5, "args": {"External id": 23}},
    ]}
    result = attribute_trace(trace)
    assert result["nccl_kernel_time_ms"] == pytest.approx(0.005)
    assert result["collectives"][0]["category"] == "arc_sketch"
    assert result["collectives"][0]["kernel_count"] == 1


def test_gradient_comm_excludes_local_ef21m_math():
    from tests.test_benchmark_arc_2x2 import _config

    base = bench.build_result_skeleton(_config())
    timing, profiles = [], []
    for mode in ("dense", "arc"):
        for _ in range(3):
            item = copy.deepcopy(base)
            item["sync_mode"] = mode
            item["timing_ms"]["step_samples"] = [10]
            item["timing_ms"]["optimizer_samples"] = [5]
            timing.append(item)
            profile = copy.deepcopy(item)
            profile["timing_ms"] = {}
            profile["profiler"]["nccl_kernel_time_ms"] = 100
            profile["profiler"]["collectives"] = ([{"category": "ddp_gradient", "duration_ms": 10}]
                if mode == "dense" else [
                    {"category": "arc_sketch", "duration_ms": 4},
                    {"category": "arc_ef21m", "duration_ms": 90},
                ])
            profiles.append(profile)
    summary = summarize_results(timing, profiles)
    assert summary["variants"]["arc"]["gradient_comm"]["mean"] == 4


def test_timing_and_profile_communication_metadata_must_match():
    from tests.test_benchmark_arc_2x2 import _config

    base = bench.build_result_skeleton(_config())
    timing, profiles = [], []
    for mode in ("dense", "arc"):
        for _ in range(3):
            item = copy.deepcopy(base)
            item["sync_mode"] = mode
            item["timing_ms"]["step_samples"] = [10]
            item["timing_ms"]["optimizer_samples"] = [5]
            timing.append(item)
            profile = copy.deepcopy(item)
            profile["timing_ms"] = {}
            profile["profiler"]["nccl_kernel_time_ms"] = 1
            profile["profiler"]["collectives"] = [{
                "category": "ddp_gradient" if mode == "dense" else "arc_sketch",
                "duration_ms": 1,
            }]
            profiles.append(profile)
    profiles[-1]["communication"]["arc_sketch_bytes"] = 999
    with pytest.raises(ValueError, match="communication metadata"):
        summarize_results(timing, profiles)


def test_checksum_pair_detects_equal_sum_but_different_parameters():
    assert not bench.checksums_agree([(3.0, 5.0), (3.0, 9.0)])


def test_observer_can_record_full_tensor_list_payload():
    observer = CollectiveObserver()
    tensors = [torch.zeros(2), torch.zeros(3)]
    with bench.observer_scope(observer):
        from dion.collective_observer import observe_collective
        observe_collective("muon/result_collective", "all_to_all", tensors[0],
                           numel=sum(t.numel() for t in tensors),
                           bytes=sum(t.numel() * t.element_size() for t in tensors))
    event = observer.events[0]
    assert event.numel == 5 and event.bytes == 20
    assert aggregate_observed(observer)[0]["numel"] == 5


def test_observed_payloads_enrich_trace_categories():
    trace_summary = {"collectives": [
        {"category": "arc_sketch", "message_bytes": 0},
        {"category": "muon_result", "message_bytes": 0},
    ]}
    observed = [
        {"category": "arc/sketch", "bytes": 123},
        {"category": "muon/result_collective", "bytes": 456},
    ]
    bench.merge_observed_message_bytes(trace_summary, observed)
    assert [x["message_bytes"] for x in trace_summary["collectives"]] == [123, 456]
