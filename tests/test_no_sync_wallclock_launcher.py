"""Contract tests for the corrected no_sync serial GPU launcher."""

import json
import subprocess

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "benchmark/compressed_muon/run_no_sync_wallclock_diagnostic.sh"
DENSE_CONFIG = REPO_ROOT / "configs/compressed_muon/cm020a_dense_muon_gpt350m.yaml"
ARC_CONFIG = REPO_ROOT / "configs/compressed_muon/cm020b_arc_muon_gpt350m.yaml"


def test_launcher_print_plan_is_serial_matched_and_unperturbed():
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["cells"] == [
        "CM020a-muon-dense-gpt350m-corrected-nosync",
        "CM020b-m001-arc-muon-gpt350m-corrected-nosync",
    ]
    assert plan["cuda_visible_devices"] == "2,3,4,5"
    assert plan["model"] == {"dim": 1024, "layers": 20, "heads": 16}
    assert plan["sequence_length"] == 1024
    assert plan["batch_size"] == 1024
    assert plan["device_batch_size"] == 1
    assert plan["num_iterations"] == 15
    assert plan["primary_uses_time_optimizer"] is False
    assert plan["compile"] is True
    assert plan["serial"] is True
    assert plan["arc"] == {
        "ratio": 0.2,
        "projection_rank": 4,
        "eta": 0.1,
        "start_compress_step": 0,
    }


def test_launcher_configs_match_the_printed_plan():
    dense = yaml.safe_load(DENSE_CONFIG.read_text())
    arc = yaml.safe_load(ARC_CONFIG.read_text())

    shared = {
        "model_dim": 1024,
        "n_layer": 20,
        "n_head": 16,
        "sequence_length": 1024,
        "batch_size": 1024,
        "device_batch_size": 1,
        "num_iterations": 15,
        "no_compile": False,
    }
    for key, expected in shared.items():
        assert dense[key] == expected
        assert arc[key] == expected

    assert dense["optimizer"] == "muon"
    assert dense["replicate_mesh_grad_sync"] is False
    assert arc["optimizer"] == "arc_topk_muon"
    assert arc["replicate_mesh_grad_sync"] is True
    assert arc["arc_topk_ratio"] == 0.2
    assert arc["arc_projection_rank"] == 4
    assert arc["arc_eta"] == 0.1
    assert arc["arc_start_compress_step"] == 0
