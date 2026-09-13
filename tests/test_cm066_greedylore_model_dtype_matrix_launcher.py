"""Contracts for the strictly paired GreedyLore model-dtype matrix."""

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/compressed_muon"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm066_greedylore_model_dtype_matrix.sh"
CONTROLLER_ROOT = (
    REPO
    / "artifacts/compressed_muon/CM066-m002-greedylore-model-dtype-matrix-controller"
)
CONFIGS = {
    ("float32", "dense"): CONFIG_DIR
    / "cm066a_dense_muon_gpt130m_fp32_dtype_matrix_s1234.yaml",
    ("float32", "greedylore"): CONFIG_DIR
    / "cm066b_m002_greedy_lore_muon_gpt130m_fp32_dtype_matrix_s1234.yaml",
    ("bfloat16", "dense"): CONFIG_DIR
    / "cm066c_dense_muon_gpt130m_bf16_dtype_matrix_s1234.yaml",
    ("bfloat16", "greedylore"): CONFIG_DIR
    / "cm066d_m002_greedy_lore_muon_gpt130m_bf16_dtype_matrix_s1234.yaml",
}


def test_configs_form_strict_model_dtype_and_sync_method_cartesian_product():
    configs = {key: yaml.safe_load(path.read_text()) for key, path in CONFIGS.items()}
    common = {
        "model_dim": 768,
        "n_layer": 8,
        "n_head": 12,
        "sequence_length": 256,
        "batch_size": 512,
        "device_batch_size": 128,
        "num_iterations": 20_000,
        "warmup_steps": 2_000,
        "lr_schedule": "cosine",
        "min_lr_ratio": 0.1,
        "grad_clip_norm": 1.0,
        "val_loss_every": 500,
        "val_tokens": 10_485_760,
        "bucket_cap_mb": 80,
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
    for (dtype, method), config in configs.items():
        for key, value in common.items():
            assert config[key] == value
        assert config["model_dtype"] == dtype
        assert config["optimizer"] == (
            "muon" if method == "dense" else "greedy_lore_muon"
        )
        assert config["no_wandb"] is False
        if method == "greedylore":
            assert config["greedy_lore_rank"] == 32
            assert config["greedy_lore_update_interval"] == 200
            assert config["greedy_lore_start_compress_step"] == 1000
            assert config["greedy_lore_seed"] == 1234
            assert config["greedy_lore_basis_sync"] == "local_svd"
            assert config["greedy_lore_dense_aux_communication_dtype"] == "bucket"


def test_launcher_print_plan_is_serial_new_artifacts_only_and_side_effect_free():
    existed = os.path.lexists(CONTROLLER_ROOT)
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan == {
        "controller_id": CONTROLLER_ROOT.name,
        "execution": "serial_fail_fast",
        "world_size": 4,
        "gpu_policy": {
            "selection": "any_four",
            "minimum_free_mib": 18000,
            "poll_seconds": 60,
        },
        "dataset": "fineweb10B",
        "training_seed": 1234,
        "cells": [
            {
                "id": "CM066a-dense-muon-gpt130m-fp32-ddp-ws4-s1234",
                "entry": "train.py",
                "model_dtype": "float32",
                "method": "dense",
            },
            {
                "id": "CM066b-m002-greedylore-muon-gpt130m-fp32-ddp-ws4-s1234",
                "entry": "train_greedylore.py",
                "model_dtype": "float32",
                "method": "greedylore",
            },
            {
                "id": "CM066c-dense-muon-gpt130m-bf16-ddp-ws4-s1234",
                "entry": "train.py",
                "model_dtype": "bfloat16",
                "method": "dense",
            },
            {
                "id": "CM066d-m002-greedylore-muon-gpt130m-bf16-ddp-ws4-s1234",
                "entry": "train_greedylore.py",
                "model_dtype": "bfloat16",
                "method": "greedylore",
            },
        ],
        "pairing_rule": "compare dense and GreedyLore only within the same model_dtype",
        "formal_wandb": True,
        "smoke_mode": "--smoke-only disables W&B and does not launch formal cells",
    }
    assert os.path.lexists(CONTROLLER_ROOT) is existed


def test_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=REPO, check=True)
