#!/usr/bin/env bash
# Serial CM042 wall-clock benchmark followed by isolated CM043 Kineto profiles.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
dense_config="$repo_dir/configs/compressed_muon/cm039a_dense_muon_scalar_adamw_warmup_cosine_clip.yaml"
arc_config="$repo_dir/configs/compressed_muon/cm039b_ef14_muon_scalar_adamw_warmup_cosine_clip.yaml"
wall_id=CM042-m001-current-ef14-all2d-wallclock-ws4-s42
profile_id=CM043-m001-current-ef14-all2d-profile-ws4-s42
wall_root="$repo_dir/artifacts/compressed_muon/$wall_id"
profile_root="$repo_dir/artifacts/compressed_muon/$profile_id"
gpu_list="${GPUS:-4,5,6,7}"
world_size=4
global_batch=512
device_batch=128
sequence_length=256
gradient_accumulation=1
training_seed=42
bucket_cap_mb=160
timing_warmup_steps=20
measured_steps=200
wall_iterations=$((timing_warmup_steps + measured_steps))
profile_step=20
profile_iterations=22
minimum_free_mib=20000
poll_seconds=60
mode=run

if [[ "${1:-}" == "--print-plan" && $# -eq 1 ]]; then
    mode=print
elif (($#)); then
    printf 'usage: %s [--print-plan]\n' "$0" >&2
    exit 64
fi

print_plan() {
    "$python_bin" - <<PY
import json

wall_id = "$wall_id"
profile_id = "$profile_id"
print(json.dumps({
    "execution": "serial_wallclock_then_profiler",
    "gpu_list": [4, 5, 6, 7],
    "world_size": 4,
    "common": {
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
    },
    "wallclock": {
        "experiment_id": wall_id,
        "models": ["gpt60m", "gpt130m"],
        "warmup_steps": 20,
        "measured_steps": 200,
        "repeats_per_model": 4,
        "pair_orders": [
            ["dense", "arc_ef14_all2d"],
            ["arc_ef14_all2d", "dense"],
            ["arc_ef14_all2d", "dense"],
            ["dense", "arc_ef14_all2d"],
        ],
        "profiler": False,
    },
    "profiler": {
        "experiment_id": profile_id,
        "starts_after": wall_id,
        "profile_step": 20,
        "num_iterations": 22,
        "profiles_per_cell": 1,
        "cells": [
            "dense_gpt60m-r1",
            "arc_ef14_all2d_gpt60m-r1",
            "dense_gpt130m-r1",
            "arc_ef14_all2d_gpt130m-r1",
        ],
        "interpretation": "critical-path attribution only; separate from wall-clock evidence",
    },
    "arc": {
        "sync_mode": "ddp_hook",
        "error_feedback": "ef14",
        "scope": "all_ndim_2_parameters",
        "ratio": 0.2,
        "projection_rank": 4,
        "eta": 1.0,
        "start_compress_step": 0,
    },
}, indent=2))
PY
}

if [[ "$mode" == print ]]; then
    print_plan
    exit 0
fi

timestamp() { date --iso-8601=seconds; }
queue_log="$wall_root/queue_status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$queue_log"; }

model_shape() {
    case "$1" in
        gpt60m) MODEL_DIM=512; MODEL_LAYERS=4; MODEL_HEADS=8 ;;
        gpt130m) MODEL_DIM=768; MODEL_LAYERS=8; MODEL_HEADS=12 ;;
        *) return 64 ;;
    esac
}

capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
            --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

selected_gpus_idle() {
    local gpu free
    [[ "$gpu_list" == "4,5,6,7" ]] || return 1
    for gpu in 4 5 6 7; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || return 1
    done
}

wait_for_gpus() {
    while ! selected_gpus_idle; do
        log "GPU_WAIT list=$gpu_list required_free_mib=$minimum_free_mib"
        sleep "$poll_seconds"
    done
}

write_wall_plan() {
    ROOT="$wall_root" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
orders = [("dense", "arc_ef14_all2d"), ("arc_ef14_all2d", "dense"),
          ("arc_ef14_all2d", "dense"), ("dense", "arc_ef14_all2d")]
cells = []
for model in ("gpt60m", "gpt130m"):
    for repeat, order in enumerate(orders, 1):
        cells.extend(f"{mode}_{model}-r{repeat}" for mode in order)
(root / "plan.json").write_text(json.dumps({
    "experiment_id": root.name,
    "world_size": 4,
    "cells": cells,
    "models": ["gpt60m", "gpt130m"],
    "pair_orders": [list(order) for order in orders],
    "timing_warmup_steps": 20,
    "measured_steps": 200,
    "profiler": False,
}, indent=2) + "\n")
PY
}

write_profile_plan() {
    ROOT="$profile_root" WALL_ID="$wall_id" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
(root / "plan.json").write_text(json.dumps({
    "experiment_id": root.name,
    "starts_after": os.environ["WALL_ID"],
    "world_size": 4,
    "cells": [
        "dense_gpt60m-r1", "arc_ef14_all2d_gpt60m-r1",
        "dense_gpt130m-r1", "arc_ef14_all2d_gpt130m-r1",
    ],
    "profile_step": 20,
    "num_iterations": 22,
    "profiles_per_cell": 1,
    "require_final_timing": True,
    "interpretation": "critical-path attribution only; separate from CM042 wall-clock",
}, indent=2) + "\n")
PY
}

run_cell() {
    local phase="$1" cell="$2" model="$3" cell_mode="$4"
    local root entry config iterations cell_dir rc trace_count
    model_shape "$model" || return $?
    if [[ "$phase" == wallclock ]]; then
        root="$wall_root"; iterations="$wall_iterations"
    else
        root="$profile_root"; iterations="$profile_iterations"
    fi
    if [[ "$cell_mode" == dense ]]; then
        entry="$repo_dir/train.py"; config="$dense_config"
    else
        entry="$repo_dir/train_arctopk.py"; config="$arc_config"
    fi
    cell_dir="$root/$cell"
    [[ ! -e "$cell_dir" ]] || { log "BLOCKED refusing_to_overwrite=$cell_dir"; return 73; }
    wait_for_gpus
    mkdir -p "$cell_dir"
    [[ "$phase" == profiler ]] && mkdir -p "$cell_dir/profiler"
    cp "$config" "$cell_dir/config.yaml"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    {
        printf 'phase=%s\ncell=%s\nmodel=%s\nmode=%s\n' "$phase" "$cell" "$model" "$cell_mode"
        printf 'gpu_list=%s\nworld_size=%s\nglobal_batch=%s\ndevice_batch=%s\nga=%s\n' \
            "$gpu_list" "$world_size" "$global_batch" "$device_batch" "$gradient_accumulation"
        printf 'sequence_length=%s\nbucket_cap_mb=%s\ntraining_seed=%s\n' \
            "$sequence_length" "$bucket_cap_mb" "$training_seed"
        printf 'recipe=independent_scalar_adamw_warmup_cosine_clip\n'
        printf 'arc_scope=all_ndim_2_parameters\narc_error_feedback=ef14\narc_start_compress_step=0\n'
        printf 'git_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$entry" --config "$config" --data_dir "$data_dir"
        --model_dim "$MODEL_DIM" --n_layer "$MODEL_LAYERS" --n_head "$MODEL_HEADS"
        --sequence_length "$sequence_length" --batch_size "$global_batch"
        --device_batch_size "$device_batch" --val_tokens 131072
        --num_iterations "$iterations" --timing-warmup-steps "$timing_warmup_steps"
        --warmup_steps "$timing_warmup_steps" --lr_schedule cosine --grad_clip_norm 1.0
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb"
        --no_wandb --use_polar_express)
    if [[ "$cell_mode" != dense ]]; then
        command+=(--arc_start_compress_step 0)
    fi
    if [[ "$phase" == profiler ]]; then
        command+=(--profile-output-dir "$cell_dir/profiler" --profile-step "$profile_step")
    fi
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"; printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL_START phase=$phase cell=$cell"
    timeout --signal=TERM --kill-after=60s 2h "${command[@]}" \
        > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    ((rc == 0)) || { log "CELL_FAILED phase=$phase cell=$cell exit=$rc"; return "$rc"; }
    rg -o 'step_avg:[0-9.]+ms|Peak memory consumption: [0-9]+ MiB' "$cell_dir/stdout.log" \
        | tail -n 2 > "$cell_dir/result.txt"
    [[ "$(wc -l < "$cell_dir/result.txt")" == 2 ]] || {
        log "CELL_INVALID phase=$phase cell=$cell missing_metrics"; return 65;
    }
    if [[ "$phase" == profiler ]]; then
        trace_count="$(find "$cell_dir/profiler" -maxdepth 1 -name 'rank-*.json' | wc -l)"
        ((trace_count == world_size)) || {
            log "CELL_INVALID phase=$phase cell=$cell traces=$trace_count"; return 65;
        }
    fi
    log "CELL_DONE phase=$phase cell=$cell $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

summarize_wallclock() {
    ROOT="$wall_root" "$python_bin" - <<'PY'
import json
import os
import re
import statistics
from pathlib import Path

root = Path(os.environ["ROOT"])
plan = json.loads((root / "plan.json").read_text())
cells = {}
for name in plan["cells"]:
    text = (root / name / "result.txt").read_text()
    timing = re.search(r"step_avg:([0-9.]+)ms", text)
    memory = re.search(r"Peak memory consumption: ([0-9]+) MiB", text)
    if not timing or not memory:
        raise SystemExit(f"missing result fields for {name}")
    cells[name] = {"step_avg_ms": float(timing.group(1)), "peak_memory_mib": int(memory.group(1))}

by_model = {}
for model in plan["models"]:
    dense = [cells[f"dense_{model}-r{repeat}"]["step_avg_ms"] for repeat in range(1, 5)]
    arc = [cells[f"arc_ef14_all2d_{model}-r{repeat}"]["step_avg_ms"] for repeat in range(1, 5)]
    paired = [(d - a) / d * 100.0 for d, a in zip(dense, arc)]
    by_model[model] = {
        "dense": {"mean_ms": statistics.mean(dense), "median_ms": statistics.median(dense),
                  "stdev_ms": statistics.stdev(dense), "cv_percent": statistics.stdev(dense) / statistics.mean(dense) * 100.0},
        "arc_ef14_all2d": {"mean_ms": statistics.mean(arc), "median_ms": statistics.median(arc),
                           "stdev_ms": statistics.stdev(arc), "cv_percent": statistics.stdev(arc) / statistics.mean(arc) * 100.0},
        "paired_arc_speedup_percent": paired,
        "mean_paired_arc_speedup_percent": statistics.mean(paired),
    }
(root / "summary.json").write_text(json.dumps({
    "experiment_id": plan["experiment_id"], "cells": cells, "by_model": by_model,
}, indent=2) + "\n")
PY
}

finish_queue() {
    local rc=$?
    printf '%s\n' "$rc" > "$wall_root/queue_exit_code.txt"
    timestamp > "$wall_root/queue_finished_at.txt"
}

cd "$repo_dir" || exit 2
for required in "$python_bin" "$torchrun_bin" "$data_dir" "$dense_config" "$arc_config"; do
    [[ -e "$required" ]] || { printf 'missing required path: %s\n' "$required" >&2; exit 66; }
done
[[ ! -e "$wall_root" && ! -e "$profile_root" ]] || {
    printf 'refusing to overwrite CM042 or CM043 artifacts\n' >&2; exit 73;
}
mkdir -p "$wall_root" "$profile_root"
: > "$queue_log"
trap finish_queue EXIT
print_plan > "$wall_root/queue_plan.json"
write_wall_plan
write_profile_plan
cp "$0" "$wall_root/controller.sh"
git rev-parse HEAD > "$wall_root/git_commit.txt"
git status --short > "$wall_root/git_status.txt"
capture_gpu_state "$wall_root/nvidia_smi_start.txt"

log "BEGIN experiment=$wall_id"
for model in gpt60m gpt130m; do
    for repeat in 1 2 3 4; do
        case "$repeat" in
            1|4) order=(dense arc_ef14_all2d) ;;
            2|3) order=(arc_ef14_all2d dense) ;;
        esac
        for cell_mode in "${order[@]}"; do
            run_cell wallclock "${cell_mode}_${model}-r${repeat}" "$model" "$cell_mode" || exit $?
        done
    done
done
summarize_wallclock || exit $?
timestamp > "$wall_root/finished_at.txt"
log "COMPLETE experiment=$wall_id summary=$wall_root/summary.json"

log "BEGIN experiment=$profile_id"
for model in gpt60m gpt130m; do
    run_cell profiler "dense_${model}-r1" "$model" dense || exit $?
    run_cell profiler "arc_ef14_all2d_${model}-r1" "$model" arc_ef14_all2d || exit $?
done
PYTHONPATH="$repo_dir" "$python_bin" \
    "$repo_dir/benchmark/compressed_muon/summarize_training_profiles.py" \
    "$profile_root" --output "$profile_root/summary.json" --require-plan || exit $?
timestamp > "$profile_root/finished_at.txt"
capture_gpu_state "$profile_root/nvidia_smi_end.txt"
log "COMPLETE experiment=$profile_id summary=$profile_root/summary.json"
