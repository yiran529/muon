import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm031_dense_supplement.sh"


def test_plan_adds_only_matching_dense_profiles_to_cm031():
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["experiment_id"] == "CM031-shared-gpu-hook-optimizer-profile"
    assert plan["supplement_cells"] == ["dense_gpt60m-r1", "dense_gpt130m-r1"]
    assert plan["existing_cells"] == [
        "arc_optimizer_gpt60m-r1",
        "arc_ddp_hook_gpt60m-r1",
        "arc_ddp_hook_gpt130m-r1",
        "arc_optimizer_gpt130m-r1",
    ]
    assert plan["world_size"] == 4
    assert plan["gpu_list"] == [4, 5, 6, 7]
    assert plan["device_batch"] == 128
    assert plan["gradient_accumulation"] == 1
    assert plan["global_batch"] == 512
    assert plan["sequence_length"] == 256
    assert plan["profile_step"] == 20
    assert plan["shared_gpu"] is True
    assert plan["require_final_timing"] is True
