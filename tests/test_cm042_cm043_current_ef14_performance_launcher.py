import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "benchmark/compressed_muon/run_cm042_cm043_current_ef14_performance.sh"


def test_plan_runs_balanced_wallclock_before_isolated_profiles():
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--print-plan"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)

    assert plan["execution"] == "serial_wallclock_then_profiler"
    assert plan["gpu_list"] == [4, 5, 6, 7]
    assert plan["world_size"] == 4
    assert plan["common"] == {
        "sequence_length": 256,
        "global_batch": 512,
        "device_batch": 128,
        "gradient_accumulation": 1,
        "training_seed": 42,
        "bucket_cap_mb": 160,
        "wandb": False,
        "recipe": "independent_scalar_adamw_warmup_cosine_clip",
        "training_warmup_steps": 20,
        "lr_schedule": "cosine",
        "grad_clip_norm": 1.0,
    }

    wallclock = plan["wallclock"]
    assert wallclock["experiment_id"].startswith("CM042-")
    assert wallclock["warmup_steps"] == 20
    assert wallclock["measured_steps"] == 200
    assert wallclock["repeats_per_model"] == 4
    assert wallclock["pair_orders"] == [
        ["dense", "arc_ef14_all2d"],
        ["arc_ef14_all2d", "dense"],
        ["arc_ef14_all2d", "dense"],
        ["dense", "arc_ef14_all2d"],
    ]
    assert wallclock["profiler"] is False

    profiler = plan["profiler"]
    assert profiler["experiment_id"].startswith("CM043-")
    assert profiler["starts_after"] == wallclock["experiment_id"]
    assert profiler["profile_step"] == 20
    assert profiler["num_iterations"] == 22
    assert profiler["profiles_per_cell"] == 1
    assert profiler["cells"] == [
        "dense_gpt60m-r1",
        "arc_ef14_all2d_gpt60m-r1",
        "dense_gpt130m-r1",
        "arc_ef14_all2d_gpt130m-r1",
    ]

    arc = plan["arc"]
    assert arc == {
        "sync_mode": "ddp_hook",
        "error_feedback": "ef14",
        "scope": "all_ndim_2_parameters",
        "ratio": 0.2,
        "projection_rank": 4,
        "eta": 1.0,
        "start_compress_step": 0,
    }
