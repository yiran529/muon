"""Contracts for the paper-aligned M002 Muon quality queue."""

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/compressed_muon"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm052_cm053_m002_quality.sh"
CONTROLLER_ROOT = REPO / "artifacts/compressed_muon/CM052-CM053-m002-paper-aligned-quality-controller"
CONFIGS = {
    "60_dense": CONFIG_DIR / "cm052a_dense_muon_gpt60m_paper_aligned_s1234.yaml",
    "60_m002": CONFIG_DIR / "cm052b_m002_greedy_lore_muon_gpt60m_paper_aligned_s1234.yaml",
    "130_dense": CONFIG_DIR / "cm053a_dense_muon_gpt130m_paper_aligned_s1234.yaml",
    "130_m002": CONFIG_DIR / "cm053b_m002_greedy_lore_muon_gpt130m_paper_aligned_s1234.yaml",
}


def test_configs_match_paper_scale_and_keep_muon_pairing_fair():
    configs = {name: yaml.safe_load(path.read_text()) for name, path in CONFIGS.items()}
    model_settings = {
        "60": {"model_dim": 512, "n_layer": 4, "n_head": 8, "num_iterations": 10_000, "warmup_steps": 1_000},
        "130": {"model_dim": 768, "n_layer": 8, "n_head": 12, "num_iterations": 20_000, "warmup_steps": 2_000},
    }
    common = {
        "sequence_length": 256,
        "batch_size": 512,
        "device_batch_size": 128,
        "lr_schedule": "cosine",
        "min_lr_ratio": 0.1,
        "grad_clip_norm": 1.0,
        "val_loss_every": 500,
        "val_tokens": 10_485_760,
        "bucket_cap_mb": 160,
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
    for model in ("60", "130"):
        dense = configs[f"{model}_dense"]
        compressed = configs[f"{model}_m002"]
        for config in (dense, compressed):
            for key, value in {**common, **model_settings[model]}.items():
                assert config[key] == value
            assert config["warmup_ratio"] == 0.0
            assert config["warmdown_ratio"] == 0.0
            assert config["no_wandb"] is False
        assert dense["optimizer"] == "muon"
        assert compressed["optimizer"] == "greedy_lore_muon"
        assert compressed["greedy_lore_rank"] == 32
        assert compressed["greedy_lore_update_interval"] == 200
        assert compressed["greedy_lore_start_compress_step"] == 1000
        assert compressed["greedy_lore_seed"] == 1234
        assert compressed["greedy_lore_basis_sync"] == "local_svd"


def test_launcher_print_plan_is_serial_and_has_no_artifact_side_effect():
    existed = os.path.lexists(CONTROLLER_ROOT)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["controller_id"] == CONTROLLER_ROOT.name
    assert plan["cell_order"] == [
        "CM052a-dense-muon-gpt60m-paper-aligned-ddp-ws4-s1234",
        "CM052b-m002-greedylore-muon-gpt60m-paper-aligned-ddp-ws4-s1234",
        "CM053a-dense-muon-gpt130m-paper-aligned-ddp-ws4-s1234",
        "CM053b-m002-greedylore-muon-gpt130m-paper-aligned-ddp-ws4-s1234",
    ]
    assert plan["world_size"] == 4
    assert plan["gpu_policy"] == {
        "selection": "any_four",
        "minimum_free_mib": 18_000,
        "poll_seconds": 60,
    }
    assert plan["training_seed"] == 1234
    assert plan["dataset"] == "fineweb10B"
    assert plan["quality_gate"]["max_perplexity_ratio"] == 1.10
    assert plan["profiler_trace"] is False
    assert os.path.lexists(CONTROLLER_ROOT) is existed


def test_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=REPO, check=True)
