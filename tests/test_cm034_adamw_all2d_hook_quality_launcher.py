import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
DENSE_CONFIG = REPO / "configs/compressed_muon/cm034a_dense_adamw_gpt60m_paperlike.yaml"
ARC_CONFIG = REPO / "configs/compressed_muon/cm034b_all2d_hook_adamw_gpt60m_paperlike.yaml"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm034_adamw_all2d_hook_quality.sh"
ARTIFACT_ROOT = Path(
    "/home/wyr/dion/artifacts/compressed_muon/"
    "CM034-m001-adamw-all2d-hook-gpt60m-paperlike-train-ddp-ws4-s42"
)


def test_cm034_configs_are_matched_gpt60m_adamw_quality_cells():
    dense = yaml.safe_load(DENSE_CONFIG.read_text())
    arc = yaml.safe_load(ARC_CONFIG.read_text())

    for config in (dense, arc):
        assert (config["model_dim"], config["n_layer"], config["n_head"]) == (512, 4, 8)
        assert config["sequence_length"] == 256
        assert config["batch_size"] == 512
        assert config["device_batch_size"] == 128
        assert config["num_iterations"] == 8393
        assert config["num_iterations"] * config["batch_size"] * config["sequence_length"] == 1_100_087_296
        assert config["val_tokens"] == 10_485_760
        assert config["lr"] == 0.001
        assert "mu" not in config
        assert "adjust_lr" not in config
        assert "use_polar_express" not in config

    for field in (
        "batch_size",
        "device_batch_size",
        "num_iterations",
        "warmup_ratio",
        "warmdown_ratio",
        "weight_decay",
        "lr",
    ):
        assert dense[field] == arc[field]

    assert dense["optimizer"] == "adamw"
    assert set(dense).isdisjoint(
        {
            "arc_sync_mode",
            "arc_topk_ratio",
            "arc_projection_rank",
            "arc_eta",
            "arc_seed",
            "arc_start_compress_step",
            "bucket_cap_mb",
        }
    )
    assert arc["optimizer"] == "arc_topk_adamw"
    assert {field: arc[field] for field in (
        "arc_sync_mode",
        "arc_topk_ratio",
        "arc_projection_rank",
        "arc_eta",
        "arc_seed",
        "arc_start_compress_step",
        "bucket_cap_mb",
    )} == {
        "arc_sync_mode": "ddp_hook",
        "arc_topk_ratio": 0.2,
        "arc_projection_rank": 4,
        "arc_eta": 1.0,
        "arc_seed": 42,
        "arc_start_compress_step": 1000,
        "bucket_cap_mb": 160,
    }


def test_cm034_print_plan_orders_dense_then_arc_without_side_effects():
    artifact_root_existed_before = os.path.lexists(ARTIFACT_ROOT)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["cells"] == [
        "dense_adamw_gpt60m-s42",
        "arc_ddp_hook_adamw_gpt60m-s42",
    ]
    assert plan["world_size"] == 4
    assert plan["gpu_list"] == [4, 5, 6, 7]
    assert plan["sequence_length"] == 256
    assert plan["global_batch"] == 512
    assert plan["preferred_device_batch"] == 128
    assert plan["preferred_gradient_accumulation"] == 1
    assert plan["oom_fallback_device_batches"] == [128, 64, 32, 16]
    assert plan["num_iterations"] == 8393
    assert plan["training_tokens"] == 1_100_087_296
    assert plan["total_tokens"] == 1_100_087_296
    assert plan["validation_tokens"] == 10_485_760
    assert plan["training_seed"] == 42
    assert plan["cells_detail"][1]["hook_arc_scope"] == "all_ndim_2_parameters"
    assert os.path.lexists(ARTIFACT_ROOT) is artifact_root_existed_before


@pytest.mark.parametrize("oom_mode", ["dense", "arc"])
@pytest.mark.parametrize("passing_batch", [128, 64, 32, 16, 0])
def test_cm034_selects_common_batch_before_either_formal_cell(
    tmp_path, oom_mode, passing_batch
):
    completed = _run_controller_selection(tmp_path, oom_mode, passing_batch)
    candidates = [128, 64, 32, 16]
    attempted = (
        candidates[:candidates.index(passing_batch) + 1]
        if passing_batch else candidates
    )
    expected = [
        f"probe {mode} {batch}" for batch in attempted for mode in ("dense", "arc")
    ]
    if passing_batch:
        ga = {128: 1, 64: 2, 32: 4, 16: 8}[passing_batch]
        expected += [
            f"formal dense {passing_batch} {ga}", f"formal arc {passing_batch} {ga}"
        ]
    assert completed.returncode == (0 if passing_batch else 42), completed.stderr
    assert completed.stdout.splitlines() == expected
    if passing_batch:
        assert (tmp_path / "selected_device_batch.txt").read_text().strip() == str(passing_batch)
        assert (tmp_path / "selected_ga.txt").read_text().strip() == str(ga)


@pytest.mark.parametrize("failure_mode", ["dense", "arc"])
def test_cm034_non_oom_probe_failure_prevents_all_formal_runs(tmp_path, failure_mode):
    completed = _run_controller_selection(tmp_path, "arc", 64, failure_mode)
    assert completed.returncode == 1, completed.stderr
    expected = ["probe dense 128"]
    if failure_mode == "arc":
        expected.append("probe arc 128")
    assert completed.stdout.splitlines() == expected
    assert not list(tmp_path.glob("*selected*"))


def _run_controller_selection(tmp_path, oom_mode, passing_batch, failure_mode=""):
    # Execute the real selection functions and controller calls, replacing only
    # GPU/training boundaries. Never execute controller artifact/bootstrap code.
    definitions, boundary, controller = LAUNCHER.read_text().partition(
        '\ncd "$repo_dir" || exit 2\n'
    )
    assert boundary, "launcher bootstrap boundary must be isolated from this no-GPU test"
    calls = re.findall(r"^(?:select_\w+|run_cell) .+$", controller, re.MULTILINE)
    assert calls, "test must exercise the actual controller selection/formal ordering"
    stubs = r'''
artifact_root="$CM034_TEST_ROOT"
preflight() { return 0; }
log() { :; }
run_probe() {
    printf 'probe %s %s\n' "$2" "$5"
    [[ "$2" != "$CM034_TEST_FAIL_MODE" ]] || return 1
    if [[ "$2" == "$CM034_TEST_OOM_MODE" && "$5" -gt "$CM034_TEST_PASS_BATCH" ]]; then
        return 42
    fi
    return 0
}
run_cell() {
    printf 'formal %s %s %s\n' "$2" "$SELECTED_DEVICE_BATCH" "$SELECTED_GA"
}
'''
    return subprocess.run(
        ["bash"],
        input=definitions + stubs + "\n".join(calls) + "\n",
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CM034_TEST_ROOT": str(tmp_path),
            "CM034_TEST_OOM_MODE": oom_mode,
            "CM034_TEST_PASS_BATCH": str(passing_batch),
            "CM034_TEST_FAIL_MODE": failure_mode,
        },
    )


def test_cm034_summary_reports_final_step_time_and_throughput_read_only(tmp_path):
    _write_summary_results(tmp_path, "250")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--summarize", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["cells"] == {
        "dense": {
            "final_validation_loss": 0.0,
            "perplexity": 1.0,
            "peak_memory_mib": 1234,
            "step_avg_ms": 250.0,
            "tokens_per_second": 524288.0,
        },
        "arc": {
            "final_validation_loss": 0.0,
            "perplexity": 1.0,
            "peak_memory_mib": 1234,
            "step_avg_ms": 500.0,
            "tokens_per_second": 262144.0,
        },
    }
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("step_avg", [None, "0", "-1", "nan", "inf", "1e999", "invalid"])
def test_cm034_summary_rejects_missing_or_invalid_final_timing(tmp_path, step_avg):
    _write_summary_results(tmp_path, step_avg)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--summarize", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert completed.returncode == 1, completed.stderr
    assert "step_avg" in completed.stderr
    assert completed.stdout == ""
    assert not (tmp_path / "summary.json").exists()


def _write_summary_results(root, dense_step_avg):
    (root / "plan.json").write_text(json.dumps({
        "experiment_id": "synthetic-cm034",
        "cells": ["dense", "arc"],
        "num_iterations": 8393,
        "global_batch": 512,
        "sequence_length": 256,
    }))
    for cell, step_avg in (("dense", dense_step_avg), ("arc", "500")):
        cell_dir = root / cell
        cell_dir.mkdir()
        timing = "" if step_avg is None else f" step_avg:{step_avg}ms"
        (cell_dir / "result.txt").write_text(
            "step:500/8393 val_loss:2.0000 train_time:50000ms step_avg:100.00ms\n"
            f"step:8393/8393 val_loss:0.0000 train_time:2000000ms{timing}\n"
            "Peak memory consumption: 1234 MiB\n"
        )
