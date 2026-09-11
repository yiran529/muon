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


def _write_profile_cell(root, cell, rank=0):
    _write_trace(
        root / cell / "profiler" / f"rank-{rank}.json",
        window_us=1000,
        optimizer_us=100,
        nccl_us=20,
    )
    cell_dir = root / cell
    (cell_dir / "exit_code.txt").write_text("0\n")
    (cell_dir / "stdout.log").write_text(
        "step:4/4 val_loss:1.0 step_avg:10ms\nPeak memory consumption: 8 MiB\n"
    )


def _write_timing_cell(
    root,
    cell,
    *,
    command="python train.py",
    exit_code="0",
    finished_at="2026-09-11T00:00:00+00:00",
    stdout="step:4/4 val_loss:1.0 step_avg:10ms\nPeak memory consumption: 8 MiB\n",
):
    cell_dir = root / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "command.txt").write_text(command + "\n")
    (cell_dir / "exit_code.txt").write_text(exit_code + "\n")
    (cell_dir / "finished_at.txt").write_text(finished_at + "\n")
    (cell_dir / "stdout.log").write_text(stdout)
    (cell_dir / "stderr.log").write_text("")


def _write_plan(root, *, cells=("dense-refresh-r1",), timing_cells=("dense-timing-r1",)):
    (root / "plan.json").write_text(json.dumps({
        "world_size": 1,
        "cells": list(cells),
        "timing_cells": list(timing_cells),
        "require_final_timing": True,
        "global_batch_size": 1,
        "sequence_length": 1,
    }))


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


def test_plan_completeness_rejects_missing_timing_cell(tmp_path):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")

    with pytest.raises(SystemExit, match="timing cell"):
        summarize_profile_root(tmp_path, require_plan=True)


def test_plan_completeness_rejects_nonzero_timing_cell(tmp_path):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")
    _write_timing_cell(tmp_path, "dense-timing-r1", exit_code="7")

    with pytest.raises(SystemExit, match="nonzero.*timing cell"):
        summarize_profile_root(tmp_path, require_plan=True)


def test_plan_completeness_rejects_timing_cell_without_command(tmp_path):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")
    _write_timing_cell(tmp_path, "dense-timing-r1")
    (tmp_path / "dense-timing-r1/command.txt").unlink()

    with pytest.raises(SystemExit, match="required command"):
        summarize_profile_root(tmp_path, require_plan=True)


def test_plan_completeness_rejects_timing_cell_without_completion_timestamp(tmp_path):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")
    _write_timing_cell(tmp_path, "dense-timing-r1")
    (tmp_path / "dense-timing-r1/finished_at.txt").unlink()

    with pytest.raises(SystemExit, match="completion timestamp"):
        summarize_profile_root(tmp_path, require_plan=True)


def test_plan_completeness_rejects_timing_cell_without_final_timing(tmp_path):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")
    _write_timing_cell(tmp_path, "dense-timing-r1", stdout="training finished\n")

    with pytest.raises(SystemExit, match="final timing"):
        summarize_profile_root(tmp_path, require_plan=True)


@pytest.mark.parametrize("terminal_value", ["invalid", "-1", "nan", "inf", "1e999"])
def test_plan_completeness_rejects_invalid_terminal_timing_marker(
    tmp_path,
    terminal_value,
):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")
    _write_timing_cell(
        tmp_path,
        "dense-timing-r1",
        stdout=(
            "step:3/4 val_loss:1.0 step_avg:10ms\n"
            f"step:4/4 val_loss:1.0 step_avg:{terminal_value}ms\n"
        ),
    )

    with pytest.raises(SystemExit, match="final timing"):
        summarize_profile_root(tmp_path, require_plan=True)


@pytest.mark.parametrize(
    "terminal_marker",
    ["step_avg:", "step_avg:10", "step_avg:10seconds"],
)
def test_plan_completeness_rejects_malformed_terminal_timing_marker(
    tmp_path,
    terminal_marker,
):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")
    _write_timing_cell(
        tmp_path,
        "dense-timing-r1",
        stdout=(
            "step:3/4 val_loss:1.0 step_avg:10ms\n"
            f"step:4/4 val_loss:1.0 {terminal_marker}\n"
        ),
    )

    with pytest.raises(SystemExit, match="final timing"):
        summarize_profile_root(tmp_path, require_plan=True)


def test_plan_completeness_produces_valid_timing_summary(tmp_path):
    _write_plan(tmp_path)
    _write_profile_cell(tmp_path, "dense-refresh-r1")
    _write_timing_cell(tmp_path, "dense-timing-r1")

    summary = summarize_profile_root(tmp_path, require_plan=True)

    assert summary["cells"][0]["cell"] == "dense-refresh-r1"
    timing_summary = json.loads((tmp_path / "timing-summary.json").read_text())
    assert timing_summary["modes"]["dense"]["cells"][0]["exit_code"] == 0
    assert timing_summary["modes"]["dense"]["cells"][0]["step_avg_ms"] == 10.0


def test_plan_completeness_supports_timing_only_artifacts(tmp_path):
    _write_plan(tmp_path, cells=(), timing_cells=("dense-timing-r1",))
    _write_timing_cell(tmp_path, "dense-timing-r1")

    summary = summarize_profile_root(tmp_path, require_plan=True)

    assert summary == {"schema_version": 1, "cells": []}
    timing_summary = json.loads((tmp_path / "timing-summary.json").read_text())
    assert timing_summary["modes"]["dense"]["mean_step_ms"] == 10.0
