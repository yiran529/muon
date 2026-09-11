import json
import subprocess

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "benchmark/compressed_muon/run_greedy_lore_profiler.sh"


def _plan(*args, check=True):
    return subprocess.run(
        ["bash", str(LAUNCHER), *args, "--print-plan"],
        cwd=REPO,
        check=check,
        capture_output=True,
        text=True,
    )


def test_print_plan_rotates_greedylore_modes_and_parameterizes_resources(tmp_path):
    plan = json.loads(_plan(
        "--world-size", "2",
        "--global-batch-size", "16",
        "--device-batch-size", "2",
        "--model-dim", "128",
        "--layers", "3",
        "--heads", "4",
        "--sequence-length", "64",
        "--bucket-cap-mb", "7",
        "--timing-warmup-steps", "5",
        "--measured-full-periods", "3",
        "--greedy-lore-rank", "2",
        "--greedy-lore-update-interval", "4",
        "--gpu-list", "1,3",
        "--exclude-gpus", "0",
        "--artifact-root", str(tmp_path),
        "--repeats", "2",
    ).stdout)

    assert plan["world_size"] == 2
    assert plan["global_batch_size"] == 16
    assert plan["device_batch_size"] == 2
    assert plan["gradient_accumulation_steps"] == 4
    assert plan["model"] == {"dim": 128, "layers": 3, "heads": 4}
    assert plan["sequence_length"] == 64
    assert plan["bucket_cap_mb"] == 7
    assert plan["timing_warmup_steps"] == 5
    assert plan["measured_full_periods"] == 3
    assert plan["timing_num_iterations"] == 17
    assert plan["greedy_lore"] == {
        "rank": 2,
        "update_interval": 4,
        "start_compress_step": 5,
        "refresh_profile_step": 6,
        "compressed_profile_step": 7,
    }
    assert plan["gpu_list"] == "1,3"
    assert plan["exclude_gpus"] == "0"
    assert plan["artifact_root"] == str(tmp_path)
    assert plan["cells"] == [
        "dense-refresh-r1",
        "dense-compressed-r1",
        "greedylore_local_svd-refresh-r1",
        "greedylore_local_svd-compressed-r1",
        "greedylore_broadcast-refresh-r1",
        "greedylore_broadcast-compressed-r1",
        "greedylore_local_svd-refresh-r2",
        "greedylore_local_svd-compressed-r2",
        "greedylore_broadcast-refresh-r2",
        "greedylore_broadcast-compressed-r2",
        "dense-refresh-r2",
        "dense-compressed-r2",
    ]
    assert plan["timing_cells"] == [
        "dense-timing-r1",
        "greedylore_local_svd-timing-r1",
        "greedylore_broadcast-timing-r1",
        "greedylore_local_svd-timing-r2",
        "greedylore_broadcast-timing-r2",
        "dense-timing-r2",
    ]
    assert "CUDA_VISIBLE_DEVICES=0" not in json.dumps(plan)


def test_print_plan_rejects_nondivisible_greedylore_global_batch():
    completed = _plan(
        "--world-size", "3",
        "--global-batch-size", "64",
        "--device-batch-size", "2",
        check=False,
    )

    assert completed.returncode != 0
    assert "divisible" in completed.stderr
