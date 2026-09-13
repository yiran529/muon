import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm064_greedylore_bucket_sweep.sh"


def test_print_plan_has_one_differential_pair_per_bucket_and_alternates_order():
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(completed.stdout)

    assert plan["bucket_caps_mib"] == [80, 160, 256, 384]
    assert plan["world_size"] == 4
    assert plan["model"] == {"dim": 768, "layers": 8, "heads": 12}
    assert plan["global_batch_size"] == 512
    assert plan["device_batch_size"] == 128
    assert plan["sequence_length"] == 256
    assert plan["cells"] == [
        "cap80-i100",
        "cap80-i200",
        "cap160-i200",
        "cap160-i100",
        "cap256-i100",
        "cap256-i200",
        "cap384-i200",
        "cap384-i100",
    ]
    assert plan["measured_updates_per_cell"] == 200
    assert plan["profile_modes"] == []
    assert plan["timing_modes"] == ["greedylore_local_svd"]
