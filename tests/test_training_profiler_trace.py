import pytest
import json

from benchmark.compressed_muon.profiler_trace import summarize_training_trace
from benchmark.compressed_muon.summarize_training_profiles import summarize_profile_root


def test_training_trace_attributes_default_ddp_nccl_inside_backward():
    trace = {
        "traceEvents": [
            {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
            {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 600},
            {"ph": "X", "name": "aten::mm_backward", "cat": "cpu_op", "ts": 110, "dur": 5,
             "args": {"External id": 10}},
            {"ph": "X", "name": "backward_compute", "cat": "kernel", "ts": 120, "dur": 300,
             "args": {"External id": 10}},
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


def test_hook_collectives_are_classified_explicitly_and_report_bucket_metrics():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 2000},
        {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 1000},
        {"ph": "X", "name": "arc_hook/bucket_ready", "cat": "cpu_op", "ts": 200, "dur": 1,
         "args": {"bucket_bytes": 100, "arc_bytes": 80, "dense_bytes": 20}},
        {"ph": "X", "name": "arc_hook/dense", "cat": "cpu_op", "ts": 250, "dur": 10,
         "args": {"External id": 1, "bytes": 20}},
        {"ph": "X", "name": "arc_hook/sketch", "cat": "cpu_op", "ts": 300, "dur": 10,
         "args": {"External id": 2, "bytes": 32}},
        {"ph": "X", "name": "arc_hook/selected_values", "cat": "cpu_op", "ts": 500, "dur": 10,
         "args": {"External id": 3, "bytes": 24}},
        {"ph": "X", "name": "aten::mm_backward", "cat": "cpu_op", "ts": 140, "dur": 5,
         "args": {"External id": 99}},
        {"ph": "X", "name": "backward_gemm", "cat": "kernel", "ts": 150, "dur": 500,
         "args": {"External id": 99}},
        {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 270, "dur": 100,
         "args": {"External id": 1}},
        {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 380, "dur": 100,
         "args": {"External id": 2}},
        {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 550, "dur": 200,
         "args": {"External id": 3}},
        {"ph": "X", "name": "arc_hook/future_complete", "cat": "cpu_op", "ts": 760, "dur": 1},
    ]}

    result = summarize_training_trace(trace)

    assert [item["category"] for item in result["collectives"]] == [
        "arc_hook_dense", "arc_hook_selected_values", "arc_hook_sketch"
    ]
    assert result["bucket_count"] == 1
    assert result["bucket_bytes"] == 100
    assert result["arc_bytes"] == 80
    assert result["dense_bytes"] == 20
    assert result["first_bucket_ready_from_backward_start_ms"] == pytest.approx(0.1)
    assert result["last_hook_future_completion_from_backward_end_ms"] == pytest.approx(-0.339)
    assert result["first_arc_collective_from_backward_start_ms"] == pytest.approx(0.17)
    assert result["last_arc_completion_from_backward_end_ms"] == pytest.approx(-0.35)
    assert result["arc_collective_backward_compute_overlap_ms"] == pytest.approx(0.30)
    assert result["exposed_gradient_sync_tail_ms"] == pytest.approx(0.10)


def test_training_summary_reports_greedylore_bucket_payload_totals():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 2000},
        {"ph": "X", "name": "greedylore_hook/bucket_ready "
         "basis_bytes=128 bucket_bytes=176 dense_aux_bytes=80 "
         "factor_bytes=24 matrix_bytes=96 parameter_count=3 "
         "phase=compressed score_bytes=12", "cat": "user_annotation",
         "ts": 200, "dur": 2},
    ]}

    result = summarize_training_trace(trace)

    assert result["bucket_count"] == 1
    assert result["bucket_bytes"] == 176
    assert result["arc_bytes"] == 0
    assert result["dense_bytes"] == 0
    assert result["matrix_bytes"] == 96
    assert result["dense_aux_bytes"] == 80
    assert result["score_bytes"] == 12
    assert result["factor_bytes"] == 24
    assert result["basis_bytes"] == 128
    assert result["parameter_count"] == 3
    assert result["cpu_ranges_ms"]["greedylore_hook_bucket_ready"] == pytest.approx(0.002)


def test_host_backward_containment_does_not_fake_gpu_compute_overlap():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 2000},
        {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 1200},
        {"ph": "X", "name": "backward_gemm", "cat": "kernel", "ts": 150, "dur": 200},
        {"ph": "X", "name": "arc_hook/sketch", "cat": "cpu_op", "ts": 500, "dur": 10,
         "args": {"External id": 4}},
        {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 600, "dur": 300,
         "args": {"External id": 4}},
    ]}

    result = summarize_training_trace(trace)

    assert result["arc_collective_backward_compute_overlap_ms"] == 0.0
    assert result["exposed_gradient_sync_tail_ms"] == pytest.approx(0.3)


def test_uncorrelated_gpu_kernel_inside_backward_is_not_genuine_backward_compute():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 2000},
        {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 1200},
        {"ph": "X", "name": "unrelated_compute", "cat": "kernel", "ts": 550, "dur": 300},
        {"ph": "X", "name": "arc_hook/sketch", "cat": "cpu_op", "ts": 500, "dur": 10,
         "args": {"External id": 4}},
        {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 600, "dur": 200,
         "args": {"External id": 4}},
    ]}

    result = summarize_training_trace(trace)

    assert result["arc_collective_backward_compute_overlap_ms"] == 0.0
    assert result["exposed_gradient_sync_tail_ms"] == pytest.approx(0.2)


def _write_profile_cell(root, cell, rank, trace):
    path = root / cell / "profiler" / f"rank-{rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace))
    (root / cell / "exit_code.txt").write_text("0\n")
    (root / cell / "stdout.log").write_text(
        "step:13/13 val_loss:1.0 train_time:10ms step_avg:10ms\n"
    )


@pytest.mark.parametrize("failure", ["signature", "seed"])
def test_profile_summary_fails_closed_on_rank_divergence_or_seed(tmp_path, failure):
    cell = "arc_ddp_hook-r1"
    (tmp_path / "plan.json").write_text(json.dumps({
        "world_size": 2,
        "cells": [cell],
        "require_final_timing": True,
    }))
    base = [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
        {"ph": "X", "name": "arc_hook/sketch", "cat": "cpu_op", "ts": 100, "dur": 10,
         "args": {"External id": 1, "bytes": 32}},
        {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 110, "dur": 5,
         "args": {"External id": 1}},
    ]
    _write_profile_cell(tmp_path, cell, 0, {"traceEvents": base})
    second = list(base)
    if failure == "signature":
        second = second + [
            {"ph": "X", "name": "arc_hook/sketch/payload bytes=32", "cat": "cpu_op",
             "ts": 400, "dur": 10, "args": {"External id": 2}},
            {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 400, "dur": 100,
             "args": {"External id": 2}},
        ]
    else:
        second = second + [
            {"ph": "X", "name": "arc/seed", "cat": "cpu_op", "ts": 400, "dur": 10,
             "args": {"External id": 2}},
            {"ph": "X", "name": "ncclDevKernel_Broadcast", "cat": "kernel", "ts": 450, "dur": 50,
             "args": {"External id": 2}},
        ]
    _write_profile_cell(tmp_path, cell, 1, {"traceEvents": second})

    with pytest.raises(SystemExit, match="signature|seed"):
        summarize_profile_root(tmp_path, require_plan=True)


def test_payload_ranges_define_signature_even_if_kernel_correlation_differs():
    trace = {"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
        {"ph": "X", "name": "arc_hook/sketch", "cat": "cpu_op", "ts": 100, "dur": 30},
        {"ph": "X", "name": "arc_hook/sketch/payload bytes=64", "cat": "cpu_op", "ts": 105, "dur": 20,
         "args": {"External id": 9}},
        {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 110, "dur": 5,
         "args": {"External id": 999}},
    ]}

    result = summarize_training_trace(trace)
    sketch = next(item for item in result["collectives"] if item["category"] == "arc_hook_sketch")

    assert sketch["launch_count"] == 1
    assert sketch["message_bytes"] == 64


def test_profile_summary_rejects_rank_divergent_collective_launch_order(tmp_path):
    cell = "arc_ddp_hook-r1"
    (tmp_path / "plan.json").write_text(json.dumps({
        "world_size": 2,
        "cells": [cell],
        "require_final_timing": True,
    }))

    def trace(order):
        events = [
            {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
        ]
        for index, category in enumerate(order, start=1):
            start = index * 100
            events.extend([
                {
                    "ph": "X",
                    "name": f"arc_hook/{category}/payload bytes=64",
                    "cat": "cpu_op",
                    "ts": start,
                    "dur": 20,
                    "args": {"External id": index},
                },
                {
                    "ph": "X",
                    "name": "ncclDevKernel_AllReduce",
                    "cat": "kernel",
                    "ts": start + 5,
                    "dur": 5,
                    "args": {"External id": index},
                },
            ])
        return {"traceEvents": events}

    _write_profile_cell(tmp_path, cell, 0, trace(["sketch", "selected_values"]))
    _write_profile_cell(tmp_path, cell, 1, trace(["selected_values", "sketch"]))

    with pytest.raises(SystemExit, match="signature"):
        summarize_profile_root(tmp_path, require_plan=True)


def test_profile_summary_aggregates_greedylore_gpu_ranges_and_compressor_tail(tmp_path):
    cell = "greedylore_local_svd-refresh-r1"
    (tmp_path / "plan.json").write_text(json.dumps({
        "world_size": 2,
        "cells": [cell],
        "require_final_timing": True,
    }))

    def trace(local_kernel_us):
        return {"traceEvents": [
            {"ph": "X", "name": "train/profile_window", "cat": "cpu_op", "ts": 0, "dur": 1000},
            {"ph": "X", "name": "train/final_backward", "cat": "cpu_op", "ts": 100, "dur": 500},
            {"ph": "X", "name": "aten::mm_backward", "cat": "cpu_op", "ts": 110, "dur": 1,
             "args": {"External id": 1}},
            {"ph": "X", "name": "backward_gemm", "cat": "kernel", "ts": 120, "dur": 180,
             "args": {"External id": 1}},
            {"ph": "X", "name": "greedylore_hook/dense/payload bytes=24", "cat": "cpu_op",
             "ts": 320, "dur": 10, "args": {"External id": 2}},
            {"ph": "X", "name": "ncclDevKernel_AllReduce", "cat": "kernel", "ts": 330, "dur": 40,
             "args": {"External id": 2}},
            {"ph": "X", "name": "greedylore_hook/local_svd", "cat": "cpu_op", "ts": 380, "dur": 80},
            {"ph": "X", "name": "aten::linalg_svd", "cat": "cpu_op", "ts": 390, "dur": 1,
             "args": {"External id": 3}},
            {"ph": "X", "name": "gesvd_kernel", "cat": "kernel", "ts": 400, "dur": local_kernel_us,
             "args": {"External id": 3}},
        ]}

    _write_profile_cell(tmp_path, cell, 0, trace(50))
    _write_profile_cell(tmp_path, cell, 1, trace(70))

    result = summarize_profile_root(tmp_path, require_plan=True)
    cell_summary = result["cells"][0]

    assert cell_summary["rank_max_gpu_ranges_ms"]["greedylore_hook_local_svd"] == pytest.approx(0.07)
    assert cell_summary["rank_max_compressor_critical_path_tail_ms"] == pytest.approx(0.17)
    assert result["by_mode"]["greedylore_local_svd-refresh"]["mean_rank_max_gpu_ranges_ms"] == {
        "greedylore_hook_local_svd": pytest.approx(0.07)
    }
