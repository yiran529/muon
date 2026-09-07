import json
import subprocess

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "benchmark/compressed_muon/run_training_critical_path_profiler.sh"


def _plan(*args, check=True):
    return subprocess.run(
        ["bash", str(LAUNCHER), *args, "--print-plan"],
        cwd=REPO,
        check=check,
        capture_output=True,
        text=True,
    )


def test_default_plan_is_three_card_gpt350m_and_rotates_three_modes():
    plan = json.loads(_plan().stdout)

    assert plan["world_size"] == 3
    assert plan["global_batch_size"] == 768
    assert plan["device_batch_size"] == 1
    assert plan["gradient_accumulation_steps"] == 256
    assert plan["model"] == {"dim": 1024, "layers": 20, "heads": 16}
    assert plan["sequence_length"] == 1024
    assert plan["cells"] == [
        "dense-r1", "arc_optimizer-r1", "arc_ddp_hook-r1"
    ]
    assert plan["bucket_cap_mb"] == 25
    assert plan["timing_warmup_steps"] == 12
    assert plan["measured_steps"] == 1
    assert plan["num_iterations"] == 13
    assert plan["profile_scope"] == "final_microstep_and_optimizer"
    assert plan["uses_time_optimizer"] is False


def test_four_card_plan_needs_only_cli_overrides():
    plan = json.loads(
        _plan("--world-size", "4", "--global-batch-size", "1024").stdout
    )

    assert plan["world_size"] == 4
    assert plan["global_batch_size"] == 1024
    assert plan["gradient_accumulation_steps"] == 256


def test_six_repeats_rotate_all_three_mode_orders_once():
    plan = json.loads(_plan("--repeats", "6").stdout)
    orders = [
        tuple(cell.rsplit("-r", 1)[0] for cell in plan["cells"][i:i + 3])
        for i in range(0, 18, 3)
    ]

    assert set(orders) == {
        ("dense", "arc_optimizer", "arc_ddp_hook"),
        ("dense", "arc_ddp_hook", "arc_optimizer"),
        ("arc_optimizer", "dense", "arc_ddp_hook"),
        ("arc_optimizer", "arc_ddp_hook", "dense"),
        ("arc_ddp_hook", "dense", "arc_optimizer"),
        ("arc_ddp_hook", "arc_optimizer", "dense"),
    }


def test_plan_accepts_bucket_timing_and_artifact_overrides(tmp_path):
    plan = json.loads(_plan(
        "--bucket-cap-mb", "5",
        "--timing-warmup-steps", "7",
        "--measured-steps", "20",
        "--artifact-root", str(tmp_path),
        "--gpu-list", "2,3,4",
        "--exclude-gpus", "0,1",
    ).stdout)

    assert plan["bucket_cap_mb"] == 5
    assert plan["timing_warmup_steps"] == 7
    assert plan["measured_steps"] == 20
    assert plan["num_iterations"] == 27
    assert plan["gpu_list"] == "2,3,4"
    assert plan["exclude_gpus"] == "0,1"


def test_plan_rejects_nondivisible_global_batch():
    completed = _plan(
        "--world-size", "3", "--global-batch-size", "1024", check=False
    )

    assert completed.returncode != 0
    assert "divisible" in completed.stderr


def test_gpu_selector_uses_requested_count_and_numeric_memory():
    completed = subprocess.run(
        [
            "bash", str(LAUNCHER), "--world-size", "3",
            "--exclude-gpus", "0,1", "--select-gpus-from-stdin",
        ],
        cwd=REPO,
        input="0, 3\n1, 3\n2, 100\n3, 1024\n4, 999\n5, 2\n",
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == "2,4,5\n"


def test_summarize_only_fails_closed_without_rank_traces(tmp_path: Path):
    completed = subprocess.run(
        [
            "bash", str(LAUNCHER), "--artifact-root", str(tmp_path),
            "--summarize-only",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "finished_at.txt").exists()
    assert "SUMMARY_FAILED" in (tmp_path / "status.log").read_text()


def test_summarize_only_rejects_partial_cell_and_rank_sets(tmp_path: Path):
    (tmp_path / "plan.json").write_text(json.dumps({
        "world_size": 3,
        "cells": ["dense-r1", "arc-r1"],
    }))
    trace = tmp_path / "dense-r1/profiler/rank-0.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "name": "train/profile_window", "cat": "user_annotation", "ts": 0, "dur": 1000},
    ]}))

    completed = subprocess.run(
        [
            "bash", str(LAUNCHER), "--artifact-root", str(tmp_path),
            "--summarize-only",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "finished_at.txt").exists()
    assert "SUMMARY_FAILED" in (tmp_path / "status.log").read_text()
