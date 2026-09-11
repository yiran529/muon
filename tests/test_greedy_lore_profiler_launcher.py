import json
import subprocess

from pathlib import Path

from dion.greedy_lore import GreedyLoreConfig, compressed_phase, is_refresh_step


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "benchmark/compressed_muon/run_greedy_lore_profiler.sh"


def _plan(*args, check=True):
    return subprocess.run(
        ["bash", str(LAUNCHER), *args, "--print-plan"],
        cwd=REPO,
        check=check,
        capture_output=True,
        text=True,
    )


def test_print_plan_rotates_greedylore_modes_and_parameterizes_resources(tmp_path):
    plan = json.loads(_plan(
        "--world-size", "2",
        "--global-batch-size", "16",
        "--device-batch-size", "2",
        "--model-dim", "128",
        "--layers", "3",
        "--heads", "4",
        "--sequence-length", "64",
        "--bucket-cap-mb", "7",
        "--timing-warmup-steps", "5",
        "--measured-full-periods", "3",
        "--greedy-lore-rank", "2",
        "--greedy-lore-update-interval", "4",
        "--gpu-list", "1,3",
        "--exclude-gpus", "0",
        "--artifact-root", str(tmp_path),
        "--repeats", "2",
    ).stdout)

    assert plan["world_size"] == 2
    assert plan["global_batch_size"] == 16
    assert plan["device_batch_size"] == 2
    assert plan["gradient_accumulation_steps"] == 4
    assert plan["model"] == {"dim": 128, "layers": 3, "heads": 4}
    assert plan["sequence_length"] == 64
    assert plan["bucket_cap_mb"] == 7
    assert plan["timing_warmup_steps"] == 5
    assert plan["measured_full_periods"] == 3
    assert plan["timing_num_iterations"] == 17
    assert plan["greedy_lore"] == {
        "rank": 2,
        "update_interval": 4,
        "start_compress_step": 5,
        "refresh_profile_step": 5,
        "compressed_profile_step": 6,
    }
    assert plan["gpu_list"] == "1,3"
    assert plan["exclude_gpus"] == "0"
    assert plan["artifact_root"] == str(tmp_path)
    assert plan["cells"] == [
        "dense-refresh-r1",
        "dense-compressed-r1",
        "greedylore_local_svd-refresh-r1",
        "greedylore_local_svd-compressed-r1",
        "greedylore_broadcast-refresh-r1",
        "greedylore_broadcast-compressed-r1",
        "greedylore_local_svd-refresh-r2",
        "greedylore_local_svd-compressed-r2",
        "greedylore_broadcast-refresh-r2",
        "greedylore_broadcast-compressed-r2",
        "dense-refresh-r2",
        "dense-compressed-r2",
    ]
    assert plan["timing_cells"] == [
        "dense-timing-r1",
        "greedylore_local_svd-timing-r1",
        "greedylore_broadcast-timing-r1",
        "greedylore_local_svd-timing-r2",
        "greedylore_broadcast-timing-r2",
        "dense-timing-r2",
    ]
    assert "CUDA_VISIBLE_DEVICES=0" not in json.dumps(plan)


def test_print_plan_profiles_exact_refresh_then_nonrefresh_lifecycle_step():
    plan = json.loads(_plan(
        "--world-size", "1",
        "--global-batch-size", "1",
        "--device-batch-size", "1",
        "--timing-warmup-steps", "5",
        "--measured-full-periods", "2",
        "--greedy-lore-update-interval", "2",
    ).stdout)

    greedy_lore = plan["greedy_lore"]
    config = GreedyLoreConfig(
        start_compress_step=greedy_lore["start_compress_step"],
        update_interval=greedy_lore["update_interval"],
    )
    # train.py's begin_step assigns active step s + 1 before loop step s.
    refresh_loop_step = greedy_lore["refresh_profile_step"]
    compressed_loop_step = greedy_lore["compressed_profile_step"]
    assert refresh_loop_step == 5
    assert compressed_loop_step == 6
    assert compressed_phase(refresh_loop_step + 1, config.start_compress_step) == 0
    assert is_refresh_step(refresh_loop_step + 1, config)
    assert compressed_phase(compressed_loop_step + 1, config.start_compress_step) == 1
    assert not is_refresh_step(compressed_loop_step + 1, config)


def test_print_plan_filters_profile_and_timing_modes_independently():
    plan = json.loads(_plan(
        "--repeats", "3",
        "--profile-modes", "dense,greedylore_local_svd,greedylore_broadcast",
        "--timing-modes", "dense,greedylore_local_svd",
    ).stdout)

    assert plan["profile_modes"] == [
        "dense",
        "greedylore_local_svd",
        "greedylore_broadcast",
    ]
    assert plan["timing_modes"] == ["dense", "greedylore_local_svd"]
    assert len(plan["cells"]) == 18
    assert len(plan["timing_cells"]) == 6
    assert not any("broadcast-timing" in cell for cell in plan["timing_cells"])
    assert plan["timing_cells"] == [
        "dense-timing-r1",
        "greedylore_local_svd-timing-r1",
        "greedylore_local_svd-timing-r2",
        "dense-timing-r2",
        "dense-timing-r3",
        "greedylore_local_svd-timing-r3",
    ]


def test_print_plan_supports_profile_only_preflight():
    plan = json.loads(_plan(
        "--profile-modes", "dense,greedylore_local_svd,greedylore_broadcast",
        "--timing-modes", "none",
    ).stdout)

    assert len(plan["cells"]) == 6
    assert plan["timing_cells"] == []


def test_print_plan_rejects_unknown_profile_mode():
    completed = _plan("--profile-modes", "unknown", check=False)

    assert completed.returncode == 64
    assert "invalid profile mode" in completed.stderr


def _run_normal(*args):
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )


def _minimal_run_args(artifact_root):
    return (
        "--world-size", "1",
        "--global-batch-size", "1",
        "--device-batch-size", "1",
        "--artifact-root", str(artifact_root),
    )


def _write_summarize_only_fixture(root, *, exit_code="0", stdout=None):
    (root / "plan.json").write_text(json.dumps({
        "world_size": 1,
        "cells": ["dense-refresh-r1"],
        "timing_cells": ["dense-timing-r1"],
        "require_final_timing": True,
        "global_batch_size": 1,
        "sequence_length": 1,
    }))
    profile_trace = root / "dense-refresh-r1/profiler/rank-0.json"
    profile_trace.parent.mkdir(parents=True)
    profile_trace.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "user_annotation", "ts": 0, "dur": 1000},
    ]}))
    profile_dir = profile_trace.parent.parent
    (profile_dir / "exit_code.txt").write_text("0\n")
    (profile_dir / "stdout.log").write_text(
        "step:1/1 val_loss:1.0 step_avg:10ms\n"
    )
    timing_dir = root / "dense-timing-r1"
    timing_dir.mkdir()
    (timing_dir / "command.txt").write_text("python train.py\n")
    (timing_dir / "exit_code.txt").write_text(f"{exit_code}\n")
    (timing_dir / "finished_at.txt").write_text("2026-09-11T00:00:00+00:00\n")
    (timing_dir / "stdout.log").write_text(
        "training finished\n" if stdout is None else stdout
    )
    (timing_dir / "stderr.log").write_text("")


def test_normal_run_refuses_existing_nonempty_artifact_root(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    sentinel = artifact_root / "sentinel.txt"
    sentinel.write_text("keep me\n")

    completed = _run_normal(*_minimal_run_args(artifact_root))

    assert completed.returncode == 73
    assert "refusing" in completed.stderr
    assert sentinel.read_text() == "keep me\n"
    assert not (artifact_root / "started_at.txt").exists()
    assert not (artifact_root / "plan.json").exists()


def test_summarize_only_produces_timing_summary_before_completion(tmp_path):
    _write_summarize_only_fixture(
        tmp_path,
        stdout="step:1/1 val_loss:1.0 step_avg:10ms\nPeak memory consumption: 8 MiB\n",
    )

    completed = _run_normal(
        "--artifact-root", str(tmp_path),
        "--summarize-only",
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "finished_at.txt").read_text().strip()
    assert "EXPERIMENT_DONE" in (tmp_path / "status.log").read_text()
    assert json.loads((tmp_path / "timing-summary.json").read_text())["modes"]["dense"]


def test_summarize_only_rejects_nonzero_timing_cell(tmp_path):
    _write_summarize_only_fixture(
        tmp_path,
        exit_code="9",
        stdout="step:1/1 val_loss:1.0 step_avg:10ms\n",
    )

    completed = _run_normal(
        "--artifact-root", str(tmp_path),
        "--summarize-only",
    )

    assert completed.returncode != 0
    assert not (tmp_path / "finished_at.txt").exists()
    assert "SUMMARY_FAILED" in (tmp_path / "status.log").read_text()


def test_summarize_only_rejects_timing_cell_without_final_timing(tmp_path):
    _write_summarize_only_fixture(tmp_path)

    completed = _run_normal(
        "--artifact-root", str(tmp_path),
        "--summarize-only",
    )

    assert completed.returncode != 0
    assert not (tmp_path / "finished_at.txt").exists()
    assert "SUMMARY_FAILED" in (tmp_path / "status.log").read_text()


def test_normal_run_refuses_existing_started_artifact_root(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    sentinel = artifact_root / "started_at.txt"
    sentinel.write_text("original\n")

    completed = _run_normal(*_minimal_run_args(artifact_root))

    assert completed.returncode == 73
    assert sentinel.read_text() == "original\n"
    assert not (artifact_root / "plan.json").exists()


def test_normal_run_refuses_artifact_root_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep me\n")
    artifact_root = tmp_path / "artifacts-link"
    artifact_root.symlink_to(target, target_is_directory=True)

    completed = _run_normal(*_minimal_run_args(artifact_root))

    assert completed.returncode == 73
    assert "refusing" in completed.stderr
    assert artifact_root.is_symlink()
    assert sentinel.read_text() == "keep me\n"
    assert not (target / "started_at.txt").exists()
    assert not (target / "plan.json").exists()


def test_print_plan_rejects_nondivisible_greedylore_global_batch():
    completed = _plan(
        "--world-size", "3",
        "--global-batch-size", "64",
        "--device-batch-size", "2",
        check=False,
    )

    assert completed.returncode != 0
    assert "divisible" in completed.stderr
