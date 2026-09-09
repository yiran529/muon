import json
import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/compressed_muon/cm033_all2d_hook_gpt60m_paperlike.yaml"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm033_all2d_hook_quality.sh"


def test_cm033_config_matches_cm027_scale_with_all2d_ddp_hook():
    config = yaml.safe_load(CONFIG.read_text())

    assert (config["model_dim"], config["n_layer"], config["n_head"]) == (512, 4, 8)
    assert config["sequence_length"] == 256
    assert config["batch_size"] == 512
    assert config["device_batch_size"] == 128
    assert config["num_iterations"] == 8393
    assert config["num_iterations"] * config["batch_size"] * config["sequence_length"] == 1_100_087_296
    assert config["optimizer"] == "arc_topk_muon"
    assert config["arc_sync_mode"] == "ddp_hook"
    assert config["arc_topk_ratio"] == 0.2
    assert config["arc_projection_rank"] == 4
    assert config["arc_eta"] == 1.0
    assert config["arc_seed"] == 42
    assert config["arc_start_compress_step"] == 1000
    assert config["bucket_cap_mb"] == 160
    assert config["no_wandb"] is False


def test_cm033_plan_runs_only_the_all2d_hook_quality_cell():
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["experiment_id"] == (
        "CM033-m001-arc-ddp-hook-all2d-gpt60m-paperlike-train-ddp-ws4-s42"
    )
    assert plan["cells"] == ["all2d_hook_gpt60m-s42"]
    assert plan["world_size"] == 4
    assert plan["gpu_list"] == [4, 5, 6, 7]
    assert plan["shared_gpu"] is True
    assert plan["sequence_length"] == 256
    assert plan["global_batch"] == 512
    assert plan["preferred_device_batch"] == 128
    assert plan["preferred_gradient_accumulation"] == 1
    assert plan["num_iterations"] == 8393
    assert plan["training_tokens"] == 1_100_087_296
    assert plan["hook_arc_scope"] == "all_ndim_2_parameters"
    assert plan["arc_start_compress_step"] == 1000
    assert plan["wandb"] is True
