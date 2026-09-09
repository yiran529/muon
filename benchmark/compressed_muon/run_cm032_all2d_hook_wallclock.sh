#!/usr/bin/env bash
# Shared-GPU, single-run wall-clock comparison after extending the ARC hook to all 2-D parameters.
set -uo pipefail

repo_dir=/home/wyr/dion
experiment_id=CM032-all2d-hook-wallclock-ws4-single
artifact_root="$repo_dir/artifacts/compressed_muon/$experiment_id"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
dense_60m_config="$repo_dir/configs/compressed_muon/cm027a_dense_muon_gpt60m_paperlike.yaml"
dense_130m_config="$repo_dir/configs/compressed_muon/cm028a_dense_muon_gpt130m_paperlike.yaml"
optimizer_config="$repo_dir/configs/compressed_muon/cm029_arc_optimizer_paperlike_wallclock.yaml"
hook_config="$repo_dir/configs/compressed_muon/cm029_arc_ddp_hook_paperlike_wallclock.yaml"
gpu_list="${GPUS:-4,5,6,7}"
world_size=4
global_batch=512
sequence_length=256
warmup_steps=20
measured_steps=200
num_iterations=$((warmup_steps + measured_steps))
bucket_cap_mb=160
training_seed=42
minimum_free_mib=16384
status_log="$artifact_root/status.log"
mode=run

if [[ "${1:-}" == "--print-plan" && $# -eq 1 ]]; then
    mode=print
elif (($#)); then
    printf 'usage: %s [--print-plan]\n' "$0" >&2
    exit 64
fi

print_plan() {
    "$python_bin" - <<'PY'
import json

print(json.dumps({
    "experiment_id": "CM032-all2d-hook-wallclock-ws4-single",
    "cells": [
        "dense_gpt60m-r1",
        "arc_optimizer_gpt60m-r1",
        "arc_ddp_hook_all2d_gpt60m-r1",
        "arc_ddp_hook_all2d_gpt130m-r1",
        "arc_optimizer_gpt130m-r1",
        "dense_gpt130m-r1",
    ],
    "world_size": 4,
    "shared_gpu": True,
    "external_gpu_processes_expected": True,
    "gpu_list": [4, 5, 6, 7],
    "repeats_per_cell": 1,
    "sequence_length": 256,
    "global_batch": 512,
    "preferred_device_batch": 128,
    "preferred_gradient_accumulation": 1,
    "oom_fallback_device_batches": [128, 64, 32, 16],
    "warmup_steps": 20,
    "measured_steps": 200,
    "bucket_cap_mb": 160,
    "arc_ratio": 0.2,
    "arc_projection_rank": 4,
    "arc_eta": 1.0,
    "arc_start_compress_step": 0,
    "hook_arc_scope": "all_ndim_2_parameters",
    "optimizer_arc_scope": "existing_transformer_block_parameters",
    "profiler": False,
    "wandb": False,
    "interpretation": "shared-GPU single-run exploratory wall-clock evidence",
}, indent=2))
PY
}

if [[ "$mode" == print ]]; then
    print_plan
    exit 0
fi

timestamp() { date -Is; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$artifact_root/controller_exit_code.txt"
    timestamp > "$artifact_root/controller_finished_at.txt"
}
trap finish_controller EXIT

is_oom() {
    rg -qi 'CUDA out of memory|out of memory|cuda error: out of memory' "$1"
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

shared_gpu_preflight() {
    local gpu free
    [[ "$gpu_list" == "4,5,6,7" ]] || {
        log "BLOCKED registered shared-GPU run requires GPU 4,5,6,7; got $gpu_list"
        return 1
    }
    for gpu in 4 5 6 7; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || {
            log "BLOCKED shared_gpu=true gpu=$gpu free_mib=$free required_mib=$minimum_free_mib"
            return 1
        }
    done
    [[ -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing data or Python executables"
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

dense_config_for_model() {
    case "$1" in
        gpt60m) DENSE_CONFIG="$dense_60m_config" ;;
        gpt130m) DENSE_CONFIG="$dense_130m_config" ;;
        *) return 2 ;;
    esac
}

run_hook_probe() {
    local model="$1" device_batch="$2" probe_dir output rc ga
    model_shape "$model" || return $?
    ga=$((global_batch / (world_size * device_batch)))
    probe_dir="$artifact_root/probes/${model}-hook-all2d-db${device_batch}"
    output="$probe_dir/stdout.log"
    mkdir -p "$probe_dir"
    capture_gpu_state "$probe_dir/nvidia_smi_before.txt"
    log "PROBE START model=$model mode=hook-all2d device_batch=$device_batch ga=$ga"
    timeout --signal=TERM --kill-after=60s 30m \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" \
        "$repo_dir/train_arctopk.py" --config "$hook_config" --data_dir "$data_dir" \
        --model_dim "$MODEL_DIM" --n_layer "$MODEL_LAYERS" --n_head "$MODEL_HEADS" \
        --sequence_length "$sequence_length" --batch_size "$global_batch" \
        --device_batch_size "$device_batch" --val_tokens 131072 \
        --num_iterations 3 --timing-warmup-steps 0 --training-seed "$training_seed" \
        --bucket-cap-mb "$bucket_cap_mb" --arc_start_compress_step 0 \
        --no_wandb --use_polar_express \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_after.txt"
    if [[ "$rc" == 0 ]] && rg -q 'Peak memory consumption:' "$output"; then
        log "PROBE PASS model=$model mode=hook-all2d device_batch=$device_batch ga=$ga"
        return 0
    fi
    if is_oom "$output"; then
        log "PROBE OOM model=$model mode=hook-all2d device_batch=$device_batch ga=$ga"
        return 42
    fi
    log "PROBE FAIL model=$model mode=hook-all2d device_batch=$device_batch exit=$rc"
    return 1
}

select_batch() {
    local model="$1" candidate rc
    for candidate in 128 64 32 16; do
        shared_gpu_preflight || return 78
        run_hook_probe "$model" "$candidate"
        rc=$?
        if [[ "$rc" == 0 ]]; then
            SELECTED_DEVICE_BATCH="$candidate"
            SELECTED_GA=$((global_batch / (world_size * candidate)))
            printf '%s\n' "$candidate" > "$artifact_root/${model}_selected_device_batch.txt"
            printf '%s\n' "$SELECTED_GA" > "$artifact_root/${model}_selected_ga.txt"
            log "SELECTED model=$model device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
            return 0
        fi
        [[ "$rc" == 42 ]] || return "$rc"
    done
    log "BLOCKED model=$model OOM through device_batch=16/GA8"
    return 42
}

run_cell() {
    local cell="$1" model="$2" cell_mode="$3"
    local cell_dir="$artifact_root/$cell" entry config output rc
    model_shape "$model" || return $?
    dense_config_for_model "$model" || return $?
    case "$cell_mode" in
        dense) entry="$repo_dir/train.py"; config="$DENSE_CONFIG" ;;
        optimizer) entry="$repo_dir/train_arctopk.py"; config="$optimizer_config" ;;
        hook-all2d) entry="$repo_dir/train_arctopk.py"; config="$hook_config" ;;
        *) return 2 ;;
    esac
    shared_gpu_preflight || return 78
    [[ ! -e "$cell_dir" ]] || {
        log "BLOCKED refusing to overwrite cell=$cell"
        return 73
    }
    mkdir -p "$cell_dir"
    output="$cell_dir/stdout.log"
    cp "$config" "$cell_dir/config.yaml"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    {
        printf 'experiment_id=%s\ncell=%s\nmodel=%s\nmode=%s\n' \
            "$experiment_id" "$cell" "$model" "$cell_mode"
        printf 'shared_gpu=true\nexternal_gpu_processes_expected=true\n'
        printf 'gpu_list=%s\nworld_size=%s\nsequence_length=%s\n' \
            "$gpu_list" "$world_size" "$sequence_length"
        printf 'global_batch=%s\ndevice_batch=%s\nga=%s\n' \
            "$global_batch" "$SELECTED_DEVICE_BATCH" "$SELECTED_GA"
        printf 'warmup_steps=%s\nmeasured_steps=%s\nbucket_cap_mb=%s\n' \
            "$warmup_steps" "$measured_steps" "$bucket_cap_mb"
        printf 'hook_arc_scope=all_ndim_2_parameters\n'
        printf 'optimizer_arc_scope=existing_transformer_block_parameters\n'
        printf 'training_seed=%s\ngit_head=%s\n' "$training_seed" "$(git rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$entry" --config "$config" --data_dir "$data_dir"
        --model_dim "$MODEL_DIM" --n_layer "$MODEL_LAYERS" --n_head "$MODEL_HEADS"
        --sequence_length "$sequence_length" --batch_size "$global_batch"
        --device_batch_size "$SELECTED_DEVICE_BATCH" --val_tokens 131072
        --num_iterations "$num_iterations" --timing-warmup-steps "$warmup_steps"
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb"
        --no_wandb --use_polar_express)
    if [[ "$cell_mode" != dense ]]; then
        command+=(--arc_start_compress_step 0)
    fi
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL START cell=$cell model=$model mode=$cell_mode device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
    timeout --signal=TERM --kill-after=60s 2h "${command[@]}" > "$output" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    if [[ "$rc" != 0 ]]; then
        log "CELL FAIL cell=$cell mode=$cell_mode exit=$rc"
        return "$rc"
    fi
    tr '\r' '\n' < "$output" |
        rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$cell_dir/result.txt"
    if [[ "$(wc -l < "$cell_dir/result.txt")" != 2 ]]; then
        log "CELL INVALID cell=$cell missing final timing or memory metric"
        return 3
    fi
    log "CELL PASS cell=$cell $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

summarize() {
    ROOT="$artifact_root" "$python_bin" - <<'PY'
import json
import os
import re
from pathlib import Path

root = Path(os.environ["ROOT"])
plan = json.loads((root / "plan.json").read_text())
summary = {"experiment_id": plan["experiment_id"], "shared_gpu": True, "cells": {}}
for cell in plan["cells"]:
    text = (root / cell / "result.txt").read_text()
    step = re.search(r"step_avg:([0-9.]+)ms", text)
    loss = re.search(r"val_loss:([0-9.]+)", text)
    memory = re.search(r"Peak memory consumption: ([0-9]+) MiB", text)
    if not (step and loss and memory):
        raise SystemExit(f"failed to parse result for {cell}")
    summary["cells"][cell] = {
        "step_avg_ms": float(step.group(1)),
        "final_validation_loss": float(loss.group(1)),
        "peak_memory_mib": int(memory.group(1)),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
}

cd "$repo_dir" || exit 2
[[ ! -e "$artifact_root/controller_started_at.txt" ]] || {
    printf 'refusing to overwrite prior %s run\n' "$experiment_id" >&2
    exit 73
}
mkdir -p "$artifact_root"
: > "$status_log"
timestamp > "$artifact_root/controller_started_at.txt"
print_plan > "$artifact_root/plan.json"
cp "$0" "$artifact_root/controller.sh"
git rev-parse HEAD > "$artifact_root/git_commit.txt"
git status --short > "$artifact_root/git_status.txt"
capture_gpu_state "$artifact_root/nvidia_smi_start.txt"
log "BEGIN $experiment_id shared_gpu=true gpu_list=$gpu_list"

select_batch gpt60m || exit $?
run_cell dense_gpt60m-r1 gpt60m dense || exit $?
run_cell arc_optimizer_gpt60m-r1 gpt60m optimizer || exit $?
run_cell arc_ddp_hook_all2d_gpt60m-r1 gpt60m hook-all2d || exit $?

select_batch gpt130m || exit $?
run_cell arc_ddp_hook_all2d_gpt130m-r1 gpt130m hook-all2d || exit $?
run_cell arc_optimizer_gpt130m-r1 gpt130m optimizer || exit $?
run_cell dense_gpt130m-r1 gpt130m dense || exit $?

summarize || exit $?
capture_gpu_state "$artifact_root/nvidia_smi_end.txt"
log "COMPLETE $experiment_id summary=$artifact_root/summary.json"
