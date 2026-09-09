"""Contract tests for the queued CM036 EF14 paper-recipe run."""

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/compressed_muon/cm036_ef14_all2d_hook_adamw_gpt60m_paper_recipe.yaml"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm036_ef14_adamw_paper_recipe_quality.sh"
EXPERIMENT_ID = "CM036-m001-ef14-all2d-hook-adamw-paper-recipe-gpt60m-ddp-ws4-s42"
ARTIFACT_ROOT = Path("/home/wyr/dion/artifacts/compressed_muon") / EXPERIMENT_ID


def test_cm036_config_changes_only_error_feedback_from_cm035_arc():
    assert CONFIG.is_file()
    cm035 = yaml.safe_load((
        REPO / "configs/compressed_muon/cm035b_all2d_hook_adamw_gpt60m_paper_recipe.yaml"
    ).read_text())
    cm036 = yaml.safe_load(CONFIG.read_text())
    assert cm036["arc_error_feedback"] == "ef14"
    ignored = {"wandb_job_name", "arc_error_feedback"}
    assert {k: v for k, v in cm036.items() if k not in ignored} == {
        k: v for k, v in cm035.items() if k not in ignored
    }
    assert cm036["adam_beta2"] == 0.999
    assert cm036["grad_clip_norm"] == 1.0
    assert cm036["warmup_steps"] == 1000
    assert cm036["lr_schedule"] == "cosine"
    assert cm036["weight_decay"] == 0.0


def test_cm036_plan_reuses_cm035_selected_batch_and_declares_ef14():
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
    assert plan["experiment_id"] == EXPERIMENT_ID
    assert plan["cells"] == ["arc_ef14_all2d_adamw_paper_recipe_gpt60m-s42"]
    assert plan["baseline_experiment_id"].startswith("CM035-")
    assert plan["batch_selection"] == "reuse_cm035_common_selection"
    assert plan["error_feedback"] == "ef14"
    assert plan["hook_arc_scope"] == "all_ndim_2_parameters"
    assert plan["arc_ratio"] == 0.2
    assert plan["adamw_recipe"]["betas"] == [0.9, 0.999]
    assert os.path.lexists(ARTIFACT_ROOT) is existed


def test_cm036_controller_installs_exit_trap_only_after_new_artifact_root():
    script = LAUNCHER.read_text()
    guard = '[[ ! -e "$artifact_root" && ! -L "$artifact_root" ]]'
    mkdir = 'mkdir -p "$artifact_root"'
    trap = 'trap finish_controller EXIT'
    assert guard in script
    assert script.index(guard) < script.index(mkdir) < script.index(trap)


def test_cm036_adamw_launcher_does_not_pass_muon_only_options():
    script = LAUNCHER.read_text()
    assert "--use_polar_express" not in script
