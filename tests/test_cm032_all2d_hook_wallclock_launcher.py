import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm032_all2d_hook_wallclock.sh"


def test_plan_compares_dense_optimizer_and_all2d_hook_once_per_model():
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["experiment_id"] == "CM032-all2d-hook-wallclock-ws4-single"
    assert plan["cells"] == [
        "dense_gpt60m-r1",
        "arc_optimizer_gpt60m-r1",
        "arc_ddp_hook_all2d_gpt60m-r1",
        "arc_ddp_hook_all2d_gpt130m-r1",
        "arc_optimizer_gpt130m-r1",
        "dense_gpt130m-r1",
    ]
    assert plan["world_size"] == 4
    assert plan["gpu_list"] == [4, 5, 6, 7]
    assert plan["shared_gpu"] is True
    assert plan["repeats_per_cell"] == 1
    assert plan["sequence_length"] == 256
    assert plan["global_batch"] == 512
    assert plan["preferred_device_batch"] == 128
    assert plan["preferred_gradient_accumulation"] == 1
    assert plan["warmup_steps"] == 20
    assert plan["measured_steps"] == 200
    assert plan["bucket_cap_mb"] == 160
    assert plan["hook_arc_scope"] == "all_ndim_2_parameters"
    assert plan["optimizer_arc_scope"] == "existing_transformer_block_parameters"
    assert plan["profiler"] is False
    assert plan["wandb"] is False
