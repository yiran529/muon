#!/usr/bin/env bash
# CM027-scale GPT-60M quality run using the all-2D ARC DDP hook.
set -uo pipefail

repo_dir=/home/wyr/dion
experiment_id=CM033-m001-arc-ddp-hook-all2d-gpt60m-paperlike-train-ddp-ws4-s42
artifact_root="$repo_dir/artifacts/compressed_muon/$experiment_id"
config="$repo_dir/configs/compressed_muon/cm033_all2d_hook_gpt60m_paperlike.yaml"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
gpu_list="${GPUS:-4,5,6,7}"
world_size=4
global_batch=512
sequence_length=256
num_iterations=8393
training_tokens=1100087296
bucket_cap_mb=160
training_seed=42
minimum_free_mib=16384
probe_timeout="${PROBE_TIMEOUT:-30m}"
formal_timeout="${FORMAL_TIMEOUT:-12h}"
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
    "experiment_id": "CM033-m001-arc-ddp-hook-all2d-gpt60m-paperlike-train-ddp-ws4-s42",
    "cells": ["all2d_hook_gpt60m-s42"],
    "model": "gpt60m",
    "world_size": 4,
    "shared_gpu": True,
    "external_gpu_processes_expected": True,
    "gpu_list": [4, 5, 6, 7],
    "sequence_length": 256,
    "global_batch": 512,
    "effective_local_batch": 128,
    "preferred_device_batch": 128,
    "preferred_gradient_accumulation": 1,
    "oom_fallback_device_batches": [128, 64, 32, 16],
    "num_iterations": 8393,
    "training_tokens": 1_100_087_296,
    "validation_interval": 500,
    "validation_tokens": 10_485_760,
    "hook_arc_scope": "all_ndim_2_parameters",
    "arc_ratio": 0.2,
    "arc_projection_rank": 4,
    "arc_eta": 1.0,
    "arc_start_compress_step": 1000,
    "bucket_cap_mb": 160,
    "training_seed": 42,
    "wandb": True,
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

preflight() {
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
    [[ -f "$config" && -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing config, dataset, or repository virtualenv executables"
        return 1
    }
    "$python_bin" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)' || {
        log "BLOCKED W&B authentication unavailable"
        return 1
    }
}

make_probe_config() {
    local output_config="$1" val_tokens="$2"
    cp "$config" "$output_config"
    sed -E -i \
        -e 's/^num_iterations:.*/num_iterations: 3/' \
        -e 's/^val_loss_every:.*/val_loss_every: 0/' \
        -e "s/^val_tokens:.*/val_tokens: $val_tokens/" \
        -e 's/^no_wandb:.*/no_wandb: true/' \
        -e 's/^arc_start_compress_step:.*/arc_start_compress_step: 0/' \
        "$output_config"
}

run_probe() {
    local device_batch="$1" ga probe_dir probe_config output val_tokens rc
    ga=$((global_batch / (world_size * device_batch)))
    probe_dir="$artifact_root/probes/hook-all2d-db${device_batch}"
    probe_config="$probe_dir/config.yaml"
    output="$probe_dir/stdout.log"
    val_tokens=$((device_batch * sequence_length * world_size))
    mkdir -p "$probe_dir"
    make_probe_config "$probe_config" "$val_tokens"
    capture_gpu_state "$probe_dir/nvidia_smi_before.txt"
    log "PROBE START mode=hook-all2d device_batch=$device_batch ga=$ga"
    timeout --signal=TERM --kill-after=60s "$probe_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" \
        "$repo_dir/train_arctopk.py" --config "$probe_config" --data_dir "$data_dir" \
        --device_batch_size "$device_batch" --batch_size "$global_batch" \
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb" \
        --no_wandb --use_polar_express > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_after.txt"
    if [[ "$rc" == 0 ]] && rg -q 'Peak memory consumption:' "$output"; then
        tr '\r' '\n' < "$output" |
            rg 'step:3/3 val_loss:|Peak memory consumption:' |
            tail -n 2 > "$probe_dir/result.txt"
        log "PROBE PASS mode=hook-all2d device_batch=$device_batch ga=$ga $(tr '\n' ' ' < "$probe_dir/result.txt")"
        return 0
    fi
    if is_oom "$output"; then
        log "PROBE OOM mode=hook-all2d device_batch=$device_batch ga=$ga"
        return 42
    fi
    log "PROBE FAIL mode=hook-all2d device_batch=$device_batch exit=$rc"
    return 1
}

select_batch() {
    local candidate rc
    for candidate in 128 64 32 16; do
        preflight || return 78
        run_probe "$candidate"
        rc=$?
        if [[ "$rc" == 0 ]]; then
            SELECTED_DEVICE_BATCH="$candidate"
            SELECTED_GA=$((global_batch / (world_size * candidate)))
            printf '%s\n' "$candidate" > "$artifact_root/selected_device_batch.txt"
            printf '%s\n' "$SELECTED_GA" > "$artifact_root/selected_ga.txt"
            log "SELECTED device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA effective_local_batch=128"
            return 0
        fi
        [[ "$rc" == 42 ]] || return "$rc"
    done
    log "BLOCKED OOM through device_batch=16/GA8"
    return 42
}

run_formal() {
    local output="$artifact_root/stdout.log" rc
    preflight || return 78
    cp "$config" "$artifact_root/config.yaml"
    capture_gpu_state "$artifact_root/nvidia_smi_before.txt"
    {
        printf 'experiment_id=%s\nmodel=gpt60m\nmode=arc_ddp_hook_all2d\n' "$experiment_id"
        printf 'dataset=fineweb10B\nshared_gpu=true\nexternal_gpu_processes_expected=true\n'
        printf 'gpu_list=%s\nworld_size=%s\nsequence_length=%s\n' \
            "$gpu_list" "$world_size" "$sequence_length"
        printf 'global_batch=%s\neffective_local_batch=128\ndevice_batch=%s\nga=%s\n' \
            "$global_batch" "$SELECTED_DEVICE_BATCH" "$SELECTED_GA"
        printf 'num_iterations=%s\ntraining_tokens=%s\nbucket_cap_mb=%s\n' \
            "$num_iterations" "$training_tokens" "$bucket_cap_mb"
        printf 'hook_arc_scope=all_ndim_2_parameters\ntraining_seed=%s\ngit_head=%s\n' \
            "$training_seed" "$(git rev-parse HEAD)"
    } > "$artifact_root/environment.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/train_arctopk.py" --config "$config" --data_dir "$data_dir"
        --device_batch_size "$SELECTED_DEVICE_BATCH" --batch_size "$global_batch"
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb"
        --wandb_job_name "$experiment_id" --use_polar_express)
    printf '%q ' "${command[@]}" > "$artifact_root/command.txt"
    printf '\n' >> "$artifact_root/command.txt"
    timestamp > "$artifact_root/formal_started_at.txt"
    log "FORMAL START id=$experiment_id device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" "${command[@]}" \
        > "$output" 2> "$artifact_root/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$artifact_root/exit_code.txt"
    timestamp > "$artifact_root/formal_finished_at.txt"
    capture_gpu_state "$artifact_root/nvidia_smi_after.txt"
    if [[ "$rc" != 0 ]]; then
        log "FORMAL FAIL id=$experiment_id exit=$rc"
        return "$rc"
    fi
    tr '\r' '\n' < "$output" |
        rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$artifact_root/result.txt"
    if [[ "$(wc -l < "$artifact_root/result.txt")" != 2 ]]; then
        log "FORMAL INVALID id=$experiment_id missing final validation or memory metric"
        return 3
    fi
    log "FORMAL PASS id=$experiment_id $(tr '\n' ' ' < "$artifact_root/result.txt")"
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

select_batch || exit $?
run_formal || exit $?

log "COMPLETE $experiment_id"
