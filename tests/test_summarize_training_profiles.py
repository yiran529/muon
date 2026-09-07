import json

import pytest

from benchmark.compressed_muon.summarize_training_profiles import summarize_profile_root


def _write_trace(path, *, window_us, optimizer_us, nccl_us):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "user_annotation", "ts": 0, "dur": window_us},
        {"ph": "X", "name": "train/optimizer", "cat": "user_annotation", "ts": 100, "dur": optimizer_us},
        {"ph": "X", "name": "arc/selected_values", "cat": "user_annotation", "ts": 200, "dur": nccl_us},
        {"ph": "X", "name": "ncclDevKernel_AllGather", "cat": "kernel", "ts": 200, "dur": nccl_us},
    ]}))


def test_cross_repeat_summary_aggregates_hotspots_and_collectives(tmp_path):
    _write_trace(tmp_path / "arc-r1/profiler/rank-0.json", window_us=1000, optimizer_us=400, nccl_us=200)
    _write_trace(tmp_path / "arc-r2/profiler/rank-0.json", window_us=1400, optimizer_us=600, nccl_us=400)

    summary = summarize_profile_root(tmp_path)
    arc = summary["by_mode"]["arc"]

    assert arc["n"] == 2
    assert arc["mean_rank_max_profile_window_ms"] == pytest.approx(1.2)
    assert arc["mean_rank_max_nccl_union_time_ms"] == pytest.approx(0.3)
    assert arc["mean_rank_max_exposed_nccl_time_ms"] == pytest.approx(0.3)
    assert arc["mean_rank_max_cpu_ranges_ms"]["optimizer"] == pytest.approx(0.5)
    assert arc["mean_rank_max_collective_time_ms"]["arc_selected_values"] == pytest.approx(0.3)
