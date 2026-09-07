import pytest

from benchmark.compressed_muon.profiler_trace import summarize_training_trace


def test_training_trace_attributes_default_ddp_nccl_inside_backward():
    trace = {
        "traceEvents": [
            {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
            {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 600},
            {"ph": "X", "name": "backward_compute", "cat": "kernel", "ts": 120, "dur": 300},
            {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 300, "dur": 300},
        ]
    }

    result = summarize_training_trace(trace)

    ddp = next(item for item in result["collectives"] if item["category"] == "ddp_gradient")
    assert ddp["duration_ms"] == pytest.approx(0.3)
    assert result["nccl_union_time_ms"] == pytest.approx(0.3)
    assert result["nccl_compute_overlap_ms"] == pytest.approx(0.12)
    assert result["exposed_nccl_time_ms"] == pytest.approx(0.18)
    assert result["unattributed_nccl_fraction"] == pytest.approx(0.0)


def test_training_trace_reports_collective_wait_as_a_cpu_range():
    trace = {
        "traceEvents": [
            {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
            {"ph": "X", "name": "arc/sketch_wait", "cat": "cpu_op", "ts": 400, "dur": 250},
        ]
    }

    result = summarize_training_trace(trace)

    assert result["cpu_ranges_ms"]["arc_sketch_wait"] == pytest.approx(0.25)
    assert result["profile_window_ms"] == pytest.approx(1.0)


def test_default_ddp_record_param_correlation_survives_kernel_after_backward():
    trace = {
        "traceEvents": [
            {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 2000},
            {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 400,
             "pid": 10, "tid": 20},
            {"ph": "X", "name": "record_param_comms", "cat": "cpu_op", "ts": 300, "dur": 50,
             "pid": 10, "tid": 21, "args": {"External id": 7, "Collective name": "allreduce"}},
            {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 700, "dur": 500,
             "args": {"External id": 7}},
        ]
    }

    result = summarize_training_trace(trace)

    assert [(item["category"], item["duration_ms"]) for item in result["collectives"]] == [
        ("ddp_gradient", pytest.approx(0.5))
    ]
    assert result["unattributed_nccl_fraction"] == 0.0


def test_training_summary_ignores_profiler_alignment_outside_profile_window():
    trace = {
        "traceEvents": [
            {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 10, "dur": 50},
            {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 100, "dur": 500},
            {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 150, "dur": 200},
            {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 200, "dur": 100},
        ]
    }

    result = summarize_training_trace(trace)

    assert result["nccl_union_time_ms"] == pytest.approx(0.1)
    assert result["unattributed_nccl_fraction"] == 0.0


def test_training_summary_counts_only_cpu_user_annotations_as_cpu_ranges():
    trace = {
        "traceEvents": [
            {"ph": "X", "name": "train/profile_window", "cat": "user_annotation", "ts": 0, "dur": 1000},
            {"ph": "X", "name": "train/profile_window", "cat": "gpu_user_annotation", "ts": 0, "dur": 900},
            {"ph": "X", "name": "train/optimizer", "cat": "user_annotation", "ts": 600, "dur": 200},
            {"ph": "X", "name": "train/optimizer", "cat": "gpu_user_annotation", "ts": 600, "dur": 300},
        ]
    }

    result = summarize_training_trace(trace)

    assert result["profile_window_ms"] == pytest.approx(1.0)
    assert result["cpu_ranges_ms"]["optimizer"] == pytest.approx(0.2)
