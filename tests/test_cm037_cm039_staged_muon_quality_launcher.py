"""Contracts for the ordered CM037--CM039 Muon quality experiment."""

import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/compressed_muon"
CONFIGS = {
    "cm037_dense": CONFIG_DIR / "cm037a_dense_muon_scalar_adamw.yaml",
    "cm037_arc": CONFIG_DIR / "cm037b_ef21m_muon_scalar_adamw.yaml",
    "cm038_dense": CONFIG_DIR / "cm038a_dense_muon_scalar_adamw_repeat.yaml",
    "cm038_arc": CONFIG_DIR / "cm038b_ef14_muon_scalar_adamw.yaml",
    "cm039_dense": CONFIG_DIR / "cm039a_dense_muon_scalar_adamw_warmup_cosine_clip.yaml",
    "cm039_arc": CONFIG_DIR / "cm039b_ef14_muon_scalar_adamw_warmup_cosine_clip.yaml",
}
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm037_cm039_staged_muon_quality.sh"
ARTIFACT_ROOT = REPO / "artifacts/compressed_muon/CM037-CM039-m001-staged-scalar-adamw-ef14-muon-gpt60m-ws4-s42"


def _load_configs():
    return {name: yaml.safe_load(path.read_text()) for name, path in CONFIGS.items()}


def _without(config, *keys):
    return {key: value for key, value in config.items() if key not in keys}


def test_six_configs_encode_only_the_declared_stage_differences():
    configs = _load_configs()
    common = {
        "model_dim": 512,
        "n_layer": 4,
        "n_head": 8,
        "sequence_length": 256,
        "batch_size": 512,
        "device_batch_size": 128,
        "num_iterations": 8393,
        "val_tokens": 10_485_760,
        "scalar_opt": "adamw",
        "adjust_lr": "spectral_norm",
        "lr": 0.02,
        "mu": 0.95,
        "weight_decay": 0.01,
        "scalar_lr": 0.001,
        "scalar_adam_beta1": 0.9,
        "scalar_adam_beta2": 0.999,
        "scalar_adam_eps": 1e-8,
        "scalar_weight_decay": 0.0,
    }
    for config in configs.values():
        for key, value in common.items():
            assert config[key] == value
        assert config["num_iterations"] * config["batch_size"] * config["sequence_length"] == 1_100_087_296

    for name in ("cm037_dense", "cm038_dense"):
        assert configs[name]["optimizer"] == "muon"
        assert configs[name]["warmup_ratio"] == 0.0
        assert configs[name]["warmdown_ratio"] == 0.2
        assert configs[name].get("grad_clip_norm") is None

    arc_common = {
        "optimizer": "arc_topk_muon",
        "arc_sync_mode": "ddp_hook",
        "arc_topk_ratio": 0.2,
        "arc_projection_rank": 4,
        "arc_eta": 1.0,
        "arc_seed": 42,
        "arc_start_compress_step": 1000,
        "bucket_cap_mb": 160,
    }
    for name in ("cm037_arc", "cm038_arc", "cm039_arc"):
        for key, value in arc_common.items():
            assert configs[name][key] == value
    assert configs["cm037_arc"]["arc_error_feedback"] == "ef21m"
    assert configs["cm038_arc"]["arc_error_feedback"] == "ef14"
    assert configs["cm039_arc"]["arc_error_feedback"] == "ef14"

    identity = {"wandb_job_name"}
    assert _without(configs["cm038_dense"], *identity) == _without(configs["cm037_dense"], *identity)
    assert _without(configs["cm038_arc"], *identity, "arc_error_feedback") == _without(
        configs["cm037_arc"], *identity, "arc_error_feedback"
    )

    for name in ("cm039_dense", "cm039_arc"):
        config = configs[name]
        assert config["warmup_ratio"] == 0.0
        assert config["warmdown_ratio"] == 0.0
        assert config["warmup_steps"] == 1000
        assert config["lr_schedule"] == "cosine"
        assert config["grad_clip_norm"] == 1.0
    schedule_keys = {"warmup_steps", "lr_schedule", "grad_clip_norm"}
    assert _without(configs["cm039_dense"], *identity, *schedule_keys) == {
        **_without(configs["cm038_dense"], *identity),
        "warmdown_ratio": 0.0,
    }
    assert _without(configs["cm039_arc"], *identity, *schedule_keys) == {
        **_without(configs["cm038_arc"], *identity),
        "warmdown_ratio": 0.0,
    }


def test_launcher_print_plan_has_three_ordered_two_cell_stages_and_no_side_effects():
    existed = os.path.lexists(ARTIFACT_ROOT)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["experiment_id"] == ARTIFACT_ROOT.name
    assert [stage["id"] for stage in plan["stages"]] == ["CM037", "CM038", "CM039"]
    assert all(len(stage["cells"]) == 2 for stage in plan["stages"])
    assert [stage["cells"][1]["error_feedback"] for stage in plan["stages"]] == [
        "ef21m", "ef14", "ef14"
    ]
    assert plan["world_size"] == 4
    assert plan["gpu_list"] == [4, 5, 6, 7]
    assert plan["global_batch"] == 512
    assert plan["num_iterations"] == 8393
    assert plan["training_tokens"] == 1_100_087_296
    assert plan["training_seed"] == 42
    assert plan["scalar_adamw"] == {
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    assert plan["stages"][0]["schedule"] == {
        "warmup_steps": 0,
        "decay": "final_20_percent_linear",
        "grad_clip_norm": None,
    }
    assert plan["stages"][2]["schedule"] == {
        "warmup_steps": 1000,
        "decay": "cosine_to_zero",
        "grad_clip_norm": 1.0,
    }
    assert os.path.lexists(ARTIFACT_ROOT) is existed


def _write_synthetic_results(root: Path, invalid_timing: bool = False):
    plan = json.loads(
        subprocess.run(
            ["bash", str(LAUNCHER), "--print-plan"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))
    for index, cell in enumerate(plan["cells"]):
        cell_dir = root / cell
        cell_dir.mkdir()
        timing = "nan" if invalid_timing and index == 0 else str(100 + index)
        loss = 4.0 + index / 10
        (cell_dir / "result.txt").write_text(
            f"step:8393/8393 val_loss:{loss} step_avg:{timing}ms\n"
            f"Peak memory consumption: {1000 + index} MiB\n"
        )


def test_launcher_summarizes_all_six_cells(tmp_path):
    root = tmp_path / "results"
    _write_synthetic_results(root)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--summarize", str(root)],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert len(summary["cells"]) == 6
    first = summary["cells"][summary["cell_order"][0]]
    assert first["final_validation_loss"] == 4.0
    assert first["perplexity"] == pytest.approx(math.exp(4.0))
    assert first["step_avg_ms"] == 100.0
    assert first["tokens_per_second"] == pytest.approx(512 * 256 * 10)


def test_launcher_summary_rejects_nonfinite_timing(tmp_path):
    root = tmp_path / "invalid-results"
    _write_synthetic_results(root, invalid_timing=True)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--summarize", str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "invalid final step_avg" in completed.stderr
