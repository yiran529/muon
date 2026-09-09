import json
import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
DENSE_CONFIG = REPO / "configs/compressed_muon/cm034a_dense_adamw_gpt60m_paperlike.yaml"
ARC_CONFIG = REPO / "configs/compressed_muon/cm034b_all2d_hook_adamw_gpt60m_paperlike.yaml"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm034_adamw_all2d_hook_quality.sh"


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
    assert plan["validation_tokens"] == 10_485_760
    assert plan["training_seed"] == 42
    assert plan["cells_detail"][1]["hook_arc_scope"] == "all_ndim_2_parameters"
