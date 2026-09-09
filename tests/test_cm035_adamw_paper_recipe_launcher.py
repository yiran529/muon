"""Contract tests for the paired CM035 paper-recipe AdamW experiment."""

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
DENSE_CONFIG = REPO / "configs/compressed_muon/cm035a_dense_adamw_gpt60m_paper_recipe.yaml"
ARC_CONFIG = REPO / "configs/compressed_muon/cm035b_all2d_hook_adamw_gpt60m_paper_recipe.yaml"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm035_adamw_paper_recipe_quality.sh"
ARTIFACT_ROOT = Path(
    "/home/wyr/dion/artifacts/compressed_muon/"
    "CM035-m001-adamw-paper-recipe-all2d-hook-gpt60m-train-ddp-ws4-s42"
)


def test_cm035_configs_match_paper_adamw_recipe_and_cm034_arc_settings():
    assert DENSE_CONFIG.is_file()
    assert ARC_CONFIG.is_file()
    dense = yaml.safe_load(DENSE_CONFIG.read_text())
    arc = yaml.safe_load(ARC_CONFIG.read_text())

    for config in (dense, arc):
        assert (config["model_dim"], config["n_layer"], config["n_head"]) == (512, 4, 8)
        assert config["sequence_length"] == 256
        assert config["batch_size"] == 512
        assert config["device_batch_size"] == 128
        assert config["num_iterations"] == 8393
        assert config["num_iterations"] * config["batch_size"] * config["sequence_length"] == 1_100_087_296
        assert config["lr"] == 0.001
        assert config["adam_beta1"] == 0.9
        assert config["adam_beta2"] == 0.999
        assert config["adam_eps"] == 1e-8
        assert config["grad_clip_norm"] == 1.0
        assert config["warmup_steps"] == 1000
        assert config["lr_schedule"] == "cosine"
        assert config["weight_decay"] == 0.0

    paired_fields = {
        "batch_size", "device_batch_size", "num_iterations", "val_loss_every",
        "val_tokens", "lr", "adam_beta1", "adam_beta2", "adam_eps",
        "grad_clip_norm", "warmup_steps", "lr_schedule", "weight_decay",
    }
    assert {field: dense[field] for field in paired_fields} == {
        field: arc[field] for field in paired_fields
    }
    assert dense["optimizer"] == "adamw"
    assert arc["optimizer"] == "arc_topk_adamw"
    assert {field: arc[field] for field in (
        "arc_sync_mode", "arc_topk_ratio", "arc_projection_rank", "arc_eta",
        "arc_seed", "arc_start_compress_step", "bucket_cap_mb",
    )} == {
        "arc_sync_mode": "ddp_hook",
        "arc_topk_ratio": 0.2,
        "arc_projection_rank": 4,
        "arc_eta": 1.0,
        "arc_seed": 42,
        "arc_start_compress_step": 1000,
        "bucket_cap_mb": 160,
    }


def test_cm035_print_plan_is_paired_and_side_effect_free():
    assert LAUNCHER.is_file()
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
    assert plan["cells"] == [
        "dense_adamw_paper_recipe_gpt60m-s42",
        "arc_ef21m_all2d_adamw_paper_recipe_gpt60m-s42",
    ]
    assert plan["world_size"] == 4
    assert plan["gpu_list"] == [4, 5, 6, 7]
    assert plan["global_batch"] == 512
    assert plan["preferred_device_batch"] == 128
    assert plan["oom_fallback_device_batches"] == [128, 64, 32, 16]
    assert plan["num_iterations"] == 8393
    assert plan["training_tokens"] == 1_100_087_296
    assert plan["adamw_recipe"] == {
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "grad_clip_norm": 1.0,
        "warmup_steps": 1000,
        "lr_schedule": "cosine",
        "weight_decay": 0.0,
    }
    assert plan["cells_detail"][1]["error_feedback"] == "ef21m"
    assert plan["cells_detail"][1]["hook_arc_scope"] == "all_ndim_2_parameters"
    assert os.path.lexists(ARTIFACT_ROOT) is existed


def test_cm035_probe_config_disables_long_training_warmup(tmp_path):
    source = tmp_path / "source.yaml"
    output = tmp_path / "probe.yaml"
    source.write_text(DENSE_CONFIG.read_text())
    definitions = LAUNCHER.read_text().partition('\ncd "$repo_dir" || exit 2\n')[0]
    completed = subprocess.run(
        ["bash"],
        input=(
            definitions
            + f'\nmake_probe_config "{source}" "{output}" 131072 dense\n'
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    probe = yaml.safe_load(output.read_text())
    assert probe["num_iterations"] == 3
    assert probe["warmup_steps"] == 0
    assert probe["no_wandb"] is True
