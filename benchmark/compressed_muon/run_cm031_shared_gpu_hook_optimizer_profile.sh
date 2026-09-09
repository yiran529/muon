#!/usr/bin/env bash
# Targeted one-step profiles for CM029/CM030 on explicitly shared GPUs.
set -uo pipefail

repo_dir=/home/wyr/dion
artifact_root="$repo_dir/artifacts/compressed_muon/CM031-shared-gpu-hook-optimizer-profile"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
optimizer_config="$repo_dir/configs/compressed_muon/cm029_arc_optimizer_paperlike_wallclock.yaml"
hook_config="$repo_dir/configs/compressed_muon/cm029_arc_ddp_hook_paperlike_wallclock.yaml"
gpu_list=4,5,6,7
world_size=4
device_batch=128
global_batch=512
sequence_length=256
profile_step=20
num_iterations=22
bucket_cap_mb=160
training_seed=42
minimum_free_mib=16384
status_log="$artifact_root/status.log"

mkdir -p "$artifact_root"
cd "$repo_dir" || exit 2
: > "$status_log"

timestamp() { date -Is; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$artifact_root/controller_exit_code.txt"
    timestamp > "$artifact_root/controller_finished_at.txt"
}
trap finish_controller EXIT

capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
            --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

shared_gpu_preflight() {
    local gpu free
    for gpu in 4 5 6 7; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || {
            log "BLOCKED shared_gpu=true gpu=$gpu free_mib=$free required_mib=$minimum_free_mib"
            return 1
        }
    done
    [[ -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing data or repository virtualenv executables"
        return 1
    }
}

model_shape() {
    case "$1" in
        gpt60m) MODEL_DIM=512; MODEL_LAYERS=4; MODEL_HEADS=8 ;;
        gpt130m) MODEL_DIM=768; MODEL_LAYERS=8; MODEL_HEADS=12 ;;
        *) return 2 ;;
    esac
}

run_cell() {
    local cell="$1" model="$2" mode="$3"
    local config cell_dir rc trace_count
    model_shape "$model" || return $?
    if [[ "$mode" == "arc_ddp_hook" ]]; then
        config="$hook_config"
    else
        config="$optimizer_config"
    fi
    shared_gpu_preflight || return 78
    cell_dir="$artifact_root/$cell"
    if [[ -e "$cell_dir/exit_code.txt" || -d "$cell_dir/profiler" ]]; then
        log "BLOCKED refusing to overwrite cell=$cell"
        return 73
    fi
    mkdir -p "$cell_dir/profiler"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    {
        printf 'shared_gpu=true\nexternal_gpu_processes_expected=true\n'
        printf 'cell=%s\nmodel=%s\nmode=%s\n' "$cell" "$model" "$mode"
        printf 'gpu_list=%s\nworld_size=%s\ndevice_batch=%s\nga=1\n' \
            "$gpu_list" "$world_size" "$device_batch"
        printf 'global_batch=%s\nsequence_length=%s\nprofile_step=%s\n' \
            "$global_batch" "$sequence_length" "$profile_step"
        printf 'bucket_cap_mb=%s\ntraining_seed=%s\ngit_head=%s\n' \
            "$bucket_cap_mb" "$training_seed" "$(git rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/train_arctopk.py" --config "$config" --data_dir "$data_dir"
        --model_dim "$MODEL_DIM" --n_layer "$MODEL_LAYERS" --n_head "$MODEL_HEADS"
        --sequence_length "$sequence_length" --batch_size "$global_batch"
        --device_batch_size "$device_batch" --val_tokens 131072
        --num_iterations "$num_iterations" --timing-warmup-steps 20
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb"
        --arc_start_compress_step 0 --no_wandb --use_polar_express
        --profile-output-dir "$cell_dir/profiler" --profile-step "$profile_step")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL START shared_gpu=true cell=$cell model=$model mode=$mode"
    timeout --signal=TERM --kill-after=60s 1h "${command[@]}" \
        > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    ((rc == 0)) || {
        log "CELL FAIL shared_gpu=true cell=$cell exit=$rc"
        return "$rc"
    }
    trace_count="$(find "$cell_dir/profiler" -maxdepth 1 -name 'rank-*.json' | wc -l)"
    ((trace_count == world_size)) || {
        log "CELL FAIL shared_gpu=true cell=$cell traces=$trace_count"
        return 65
    }
    log "CELL PASS shared_gpu=true cell=$cell traces=$trace_count"
}

if [[ -e "$artifact_root/controller_started_at.txt" ]]; then
    log "BLOCKED refusing to overwrite prior controller"
    exit 73
fi
timestamp > "$artifact_root/controller_started_at.txt"
cp "$0" "$artifact_root/controller.sh"
git rev-parse HEAD > "$artifact_root/git_commit.txt"
git status --short > "$artifact_root/git_status.txt"
capture_gpu_state "$artifact_root/nvidia_smi_start.txt"

WS="$world_size" ROOT="$artifact_root" "$python_bin" - <<'PY' > "$artifact_root/plan.json"
import json
import os

print(json.dumps({
    "world_size": int(os.environ["WS"]),
    "cells": [
        "arc_optimizer_gpt60m-r1",
        "arc_ddp_hook_gpt60m-r1",
        "arc_ddp_hook_gpt130m-r1",
        "arc_optimizer_gpt130m-r1",
    ],
    "shared_gpu": True,
    "external_gpu_processes_expected": True,
    "gpu_list": [4, 5, 6, 7],
    "device_batch": 128,
    "gradient_accumulation": 1,
    "global_batch": 512,
    "sequence_length": 256,
    "arc": {"ratio": 0.2, "projection_rank": 4, "eta": 1.0, "start_compress_step": 0},
    "bucket_cap_mb": 160,
    "profile_step": 20,
    "profiles_per_cell": 1,
    "require_final_timing": True,
    "interpretation": "critical-path attribution only; not stable wall-clock evidence",
    "artifact_root": os.environ["ROOT"],
}, indent=2))
PY

log "BEGIN CM031 targeted profile queue shared_gpu=true gpu_list=$gpu_list"
run_cell arc_optimizer_gpt60m-r1 gpt60m arc_optimizer || exit $?
run_cell arc_ddp_hook_gpt60m-r1 gpt60m arc_ddp_hook || exit $?
run_cell arc_ddp_hook_gpt130m-r1 gpt130m arc_ddp_hook || exit $?
run_cell arc_optimizer_gpt130m-r1 gpt130m arc_optimizer || exit $?

PYTHONPATH="$repo_dir" "$python_bin" \
    "$repo_dir/benchmark/compressed_muon/summarize_training_profiles.py" \
    "$artifact_root" --output "$artifact_root/summary.json" --require-plan || exit $?
capture_gpu_state "$artifact_root/nvidia_smi_end.txt"
log "COMPLETE CM031 targeted profile queue shared_gpu=true summary=$artifact_root/summary.json"
