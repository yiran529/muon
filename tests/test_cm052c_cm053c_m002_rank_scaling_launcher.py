"""Contracts for the high-rank M002 paper-aligned quality queue."""

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/compressed_muon"
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm052c_cm053c_m002_rank_scaling_quality.sh"
CONTROLLER_ROOT = REPO / "artifacts/compressed_muon/CM052c-CM053c-m002-rank-scaling-quality-controller"
CONFIG_PAIRS = {
    "60": (
        CONFIG_DIR / "cm052b_m002_greedy_lore_muon_gpt60m_paper_aligned_s1234.yaml",
        CONFIG_DIR / "cm052c_m002_greedy_lore_muon_gpt60m_paper_aligned_r128_s1234.yaml",
        128,
    ),
    "130": (
        CONFIG_DIR / "cm053b_m002_greedy_lore_muon_gpt130m_paper_aligned_s1234.yaml",
        CONFIG_DIR / "cm053c_m002_greedy_lore_muon_gpt130m_paper_aligned_r256_s1234.yaml",
        256,
    ),
}


def test_configs_only_change_rank_name_and_comment_from_b_runs():
    for source_path, target_path, expected_rank in CONFIG_PAIRS.values():
        source = yaml.safe_load(source_path.read_text())
        target = yaml.safe_load(target_path.read_text())

        differing_keys = {
            key for key in source | target if source.get(key) != target.get(key)
        }
        assert differing_keys == {"wandb_job_name", "greedy_lore_rank"}
        assert target["greedy_lore_rank"] == expected_rank
        assert target["wandb_job_name"].startswith(
            "CM052c-" if expected_rank == 128 else "CM053c-"
        )


def test_launcher_print_plan_is_compressed_only_serial_and_side_effect_free():
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
        "CM052c-m002-greedylore-muon-gpt60m-paper-aligned-r128-ddp-ws4-s1234",
        "CM053c-m002-greedylore-muon-gpt130m-paper-aligned-r256-ddp-ws4-s1234",
    ]
    assert plan["execution"] == "serial"
    assert plan["world_size"] == 4
    assert plan["greedy_lore_ranks"] == {"gpt60m": 128, "gpt130m": 256}
    assert plan["gpu_policy"] == {
        "selection": "any_four",
        "minimum_free_mib": 18_000,
        "poll_seconds": 60,
    }
    assert plan["training_seed"] == 1234
    assert plan["dataset"] == "fineweb10B"
    assert plan["profiler_trace"] is False
    assert "quality_gate" not in plan
    assert os.path.lexists(CONTROLLER_ROOT) is existed


def test_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=REPO, check=True)
