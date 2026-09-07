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


def test_default_plan_is_three_card_gpt350m_and_serial():
    plan = json.loads(_plan().stdout)

    assert plan["world_size"] == 3
    assert plan["global_batch_size"] == 768
    assert plan["device_batch_size"] == 1
    assert plan["gradient_accumulation_steps"] == 256
    assert plan["model"] == {"dim": 1024, "layers": 20, "heads": 16}
    assert plan["sequence_length"] == 1024
    assert plan["cells"] == ["dense-r1", "arc-r1"]
    assert plan["profile_scope"] == "final_microstep_and_optimizer"
    assert plan["uses_time_optimizer"] is False


def test_four_card_plan_needs_only_cli_overrides():
    plan = json.loads(
        _plan("--world-size", "4", "--global-batch-size", "1024").stdout
    )

    assert plan["world_size"] == 4
    assert plan["global_batch_size"] == 1024
    assert plan["gradient_accumulation_steps"] == 256


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
