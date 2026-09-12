import pytest
import torch

import dion.greedy_lore_ddp_hook as hook_module
from benchmark.compressed_muon.profiler_trace import summarize_training_trace
from dion.collective_observer import CollectiveObserver, set_active_observer
from dion.greedy_lore_ddp_hook import GreedyLoreDDPParameterSpec, GreedyLoreDDPState
from dion.greedy_lore import GreedyLoreConfig


class FakeGradBucket:
    def __init__(self, parameters, gradients):
        self._parameters = tuple(parameters)
        self._gradients = tuple(gradients)
        self._buffer = torch.cat([gradient.reshape(-1) for gradient in gradients])

    def parameters(self):
        return list(self._parameters)

    def gradients(self):
        return list(self._gradients)

    def buffer(self):
        return self._buffer


class RecordingRange:
    calls = []

    def __init__(self, name, args=None):
        self.name = name
        self.args = args

    def __enter__(self):
        self.calls.append(("enter", self.name, self.args))
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.calls.append(("exit", self.name, self.args))
        return False


def _kernel(name, ts, dur, external_id):
    return {
        "ph": "X",
        "name": name,
        "cat": "kernel",
        "ts": ts,
        "dur": dur,
        "args": {"External id": external_id},
    }


def test_greedylore_collectives_payloads_operations_and_local_gpu_ranges():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1500},
        {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 1000},
        {"ph": "X", "name": "aten::mm_backward", "cat": "cpu_op", "ts": 110, "dur": 5,
         "args": {"External id": 90}},
        _kernel("backward_gemm", 120, 180, 90),
        {"ph": "X", "name": "greedylore_hook/dense/payload bytes=128", "cat": "cpu_op",
         "ts": 320, "dur": 20, "args": {"External id": 1}},
        _kernel("ncclDevKernel_AllReduce", 330, 50, 1),
        {"ph": "X", "name": "greedylore_hook/basis_broadcast/payload bytes=64", "cat": "cpu_op",
         "ts": 400, "dur": 20, "args": {"External id": 2}},
        _kernel("ncclDevKernel_Broadcast", 410, 60, 2),
        {"ph": "X", "name": "greedylore_hook/score_plus_aux_allreduce/payload bytes=32",
         "cat": "cpu_op", "ts": 500, "dur": 20, "args": {"External id": 3}},
        _kernel("ncclDevKernel_AllReduce", 510, 40, 3),
        {"ph": "X", "name": "greedylore_hook/factor_allreduce/payload bytes=48",
         "cat": "cpu_op", "ts": 600, "dur": 20, "args": {"External id": 4}},
        _kernel("ncclDevKernel_AllReduce", 610, 30, 4),
        {"ph": "X", "name": "greedylore_hook/local_svd", "cat": "cpu_op",
         "ts": 340, "dur": 40},
        {"ph": "X", "name": "aten::linalg_svd", "cat": "cpu_op", "ts": 345, "dur": 1,
         "args": {"External id": 10}},
        _kernel("gesvd_kernel", 350, 25, 10),
        {"ph": "X", "name": "greedylore_hook/score", "cat": "cpu_op", "ts": 460, "dur": 30},
        {"ph": "X", "name": "aten::matmul", "cat": "cpu_op", "ts": 465, "dur": 1,
         "args": {"External id": 11}},
        _kernel("score_kernel", 470, 20, 11),
        {"ph": "X", "name": "greedylore_hook/topr", "cat": "cpu_op", "ts": 555, "dur": 20},
        {"ph": "X", "name": "aten::topk", "cat": "cpu_op", "ts": 560, "dur": 1,
         "args": {"External id": 12}},
        _kernel("topk_kernel", 565, 15, 12),
        {"ph": "X", "name": "greedylore_hook/factor", "cat": "cpu_op", "ts": 580, "dur": 30},
        {"ph": "X", "name": "aten::matmul", "cat": "cpu_op", "ts": 585, "dur": 1,
         "args": {"External id": 13}},
        _kernel("factor_kernel", 590, 20, 13),
        {"ph": "X", "name": "greedylore_hook/error", "cat": "cpu_op", "ts": 625, "dur": 20},
        {"ph": "X", "name": "aten::sub", "cat": "cpu_op", "ts": 630, "dur": 1,
         "args": {"External id": 14}},
        _kernel("error_kernel", 635, 10, 14),
        {"ph": "X", "name": "greedylore_hook/reconstruction", "cat": "cpu_op",
         "ts": 760, "dur": 100},
        {"ph": "X", "name": "aten::matmul", "cat": "cpu_op", "ts": 765, "dur": 1,
         "args": {"External id": 15}},
        _kernel("reconstruct_kernel", 800, 80, 15),
        {"ph": "X", "name": "greedylore_hook/future_complete", "cat": "cpu_op",
         "ts": 890, "dur": 1},
    ]}

    result = summarize_training_trace(trace)

    assert [(item["category"], item["operation"], item["message_bytes"])
            for item in result["collective_launches"]] == [
        ("greedylore_hook_dense", "all_reduce", 128),
        ("greedylore_hook_basis_broadcast", "broadcast", 64),
        ("greedylore_hook_score_plus_aux_allreduce", "all_reduce", 32),
        ("greedylore_hook_factor_allreduce", "all_reduce", 48),
    ]
    assert result["gpu_ranges_ms"] == {
        "greedylore_hook_error": pytest.approx(0.010),
        "greedylore_hook_factor": pytest.approx(0.020),
        "greedylore_hook_local_svd": pytest.approx(0.025),
        "greedylore_hook_reconstruction": pytest.approx(0.080),
        "greedylore_hook_score": pytest.approx(0.020),
        "greedylore_hook_topr": pytest.approx(0.015),
    }
    assert result["nccl_compute_overlap_ms"] == 0.0
    assert result["exposed_gradient_sync_tail_ms"] == pytest.approx(0.18)
    assert result["compressor_critical_path_tail_ms"] == pytest.approx(0.591)


def test_backward_kernel_executing_during_hook_range_is_not_local_work():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op",
         "ts": 0, "dur": 1000},
        {"ph": "X", "name": "train/final_backward", "cat": "cpu_op",
         "ts": 100, "dur": 700, "pid": 1, "tid": 10},
        {"ph": "X", "name": "aten::mm_backward", "cat": "cpu_op",
         "ts": 110, "dur": 5, "pid": 1, "tid": 10,
         "args": {"External id": 90}},
        {"ph": "X", "name": "greedylore_hook/score", "cat": "cpu_op",
         "ts": 200, "dur": 200, "pid": 1, "tid": 20},
        {"ph": "X", "name": "aten::matmul", "cat": "cpu_op",
         "ts": 210, "dur": 5, "pid": 1, "tid": 20,
         "args": {"External id": 11}},
        _kernel("backward_gemm", 250, 30, 90),
        _kernel("score_kernel", 300, 20, 11),
        {"ph": "X", "name": "greedylore_hook/factor_allreduce/payload bytes=48",
         "cat": "cpu_op", "ts": 240, "dur": 10, "pid": 1, "tid": 20,
         "args": {"External id": 12}},
        _kernel("ncclDevKernel_AllReduce", 260, 50, 12),
    ]}

    result = summarize_training_trace(trace)

    assert result["gpu_ranges_ms"] == {
        "greedylore_hook_score": pytest.approx(0.020),
    }
    assert result["nccl_compute_overlap_ms"] == pytest.approx(0.020)


def test_gpu_user_annotation_payload_mirror_is_not_counted_as_a_second_launch():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
        {"ph": "X", "name": "greedylore_hook/dense/payload bytes=128", "cat": "cpu_op",
         "ts": 100, "dur": 20, "args": {"External id": 7}},
        {"ph": "X", "name": "greedylore_hook/dense/payload bytes=128",
         "cat": "gpu_user_annotation", "ts": 100, "dur": 20,
         "args": {"External id": 7}},
        _kernel("ncclDevKernel_AllReduce", 110, 40, 7),
    ]}

    result = summarize_training_trace(trace)

    assert result["collective_launches"] == [{
        "category": "greedylore_hook_dense",
        "operation": "all_reduce",
        "message_bytes": 128,
        "start_us": 100.0,
    }]
    assert next(item for item in result["collectives"]
                if item["category"] == "greedylore_hook_dense") == {
        "category": "greedylore_hook_dense",
        "kernel_count": 1,
        "operation": "all_reduce",
        "launch_count": 1,
        "duration_ms": pytest.approx(0.04),
        "message_bytes": 128,
    }


def test_compressor_tail_is_none_without_genuine_backward_gpu_kernel():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
        {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 700},
        {"ph": "X", "name": "greedylore_hook/score", "cat": "cpu_op", "ts": 200, "dur": 50},
        {"ph": "X", "name": "aten::matmul", "cat": "cpu_op", "ts": 205, "dur": 1,
         "args": {"External id": 7}},
        _kernel("score_kernel", 210, 40, 7),
        {"ph": "X", "name": "greedylore_hook/dense/payload bytes=24", "cat": "cpu_op",
         "ts": 300, "dur": 20, "args": {"External id": 8}},
        _kernel("ncclDevKernel_AllReduce", 310, 60, 8),
    ]}

    result = summarize_training_trace(trace)

    assert result["compressor_critical_path_tail_ms"] is None
    assert result["exposed_gradient_sync_tail_ms"] == pytest.approx(0.06)


def test_dense_greedylore_tail_uses_collective_completion_when_no_local_work():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
        {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 500},
        {"ph": "X", "name": "aten::mm_backward", "cat": "cpu_op", "ts": 110, "dur": 5,
         "args": {"External id": 1}},
        _kernel("backward_gemm", 120, 180, 1),
        {"ph": "X", "name": "greedylore_hook/dense/payload bytes=24", "cat": "cpu_op",
         "ts": 330, "dur": 20, "args": {"External id": 2}},
        _kernel("ncclDevKernel_AllReduce", 340, 90, 2),
    ]}

    result = summarize_training_trace(trace)

    assert result["exposed_gradient_sync_tail_ms"] == pytest.approx(0.09)
    assert result["compressor_critical_path_tail_ms"] == pytest.approx(0.13)


def test_hook_emits_bucket_collective_and_local_record_function_ranges(monkeypatch):
    matrix = torch.nn.Parameter(torch.zeros(2, 3))
    dense = torch.nn.Parameter(torch.zeros(2))
    bucket = FakeGradBucket(
        [matrix, dense],
        [torch.ones_like(matrix), torch.ones_like(dense)],
    )
    state = GreedyLoreDDPState(
        process_group=None,
        fingerprint="f" * 64,
        parameter_specs=[
            GreedyLoreDDPParameterSpec(matrix, "matrix", 0, "matrix"),
            GreedyLoreDDPParameterSpec(dense, "dense", 1, "dense_aux"),
        ],
        optimizer_parameters=[matrix, dense],
        config=GreedyLoreConfig(rank=1, start_compress_step=0, update_interval=100),
    )
    state.committed_step = 1
    state.world_size = 2
    observer = CollectiveObserver()
    RecordingRange.calls = []

    def fake_all_reduce(current_state, tensor, category):
        with hook_module._collective_profile_range(category, "all_reduce", tensor):
            pass
        future = torch.futures.Future()
        future.set_result(tensor)
        return future

    monkeypatch.setattr(hook_module, "record_function", RecordingRange)
    monkeypatch.setattr(hook_module, "_all_reduce_future", fake_all_reduce)
    set_active_observer(observer)
    try:
        state.begin_step()
        hook_module.greedy_lore_ddp_hook(state, bucket).wait()
        state.finish_step()
    finally:
        set_active_observer(None)

    entered = [name for kind, name, _args in RecordingRange.calls if kind == "enter"]
    bucket_ready = [name for name in entered
                    if name.startswith("greedylore_hook/bucket_ready")]
    assert len(bucket_ready) == 1
    assert "bucket_bytes=" in bucket_ready[0]
    assert "matrix_bytes=" in bucket_ready[0]
    assert "dense_aux_bytes=" in bucket_ready[0]
    assert "phase=" in bucket_ready[0]
    assert "greedylore_hook/score" in entered
    assert "greedylore_hook/topr" in entered
    assert "greedylore_hook/factor" in entered
    assert "greedylore_hook/error" in entered
    assert "greedylore_hook/reconstruction" in entered
    assert "greedylore_hook/score_plus_aux_allreduce/payload bytes=16" in entered
    assert "greedylore_hook/factor_allreduce/payload bytes=12" in entered
    assert observer.signature() == [
        ("greedylore_hook/score_plus_aux_allreduce", "all_reduce", 4, "float32", 16),
        ("greedylore_hook/factor_allreduce", "all_reduce", 3, "float32", 12),
    ]
