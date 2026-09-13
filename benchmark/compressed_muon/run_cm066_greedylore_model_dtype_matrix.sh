#!/usr/bin/env bash
# Serial, fail-fast FP32/BF16 x dense/GreedyLore GPT-130M quality matrix.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_id=CM066-m002-greedylore-model-dtype-matrix-controller
controller_dir="$artifact_base/$controller_id"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
requested_gpu_list=""
world_size=4
training_seed=1234
minimum_free_mib=18000
gpu_wait_seconds="${GPU_WAIT_SECONDS:-60}"
probe_timeout="${PROBE_TIMEOUT:-30m}"
formal_timeout="${FORMAL_TIMEOUT:-12h}"
mode=run

ids=(
    CM066a-dense-muon-gpt130m-fp32-ddp-ws4-s1234
    CM066b-m002-greedylore-muon-gpt130m-fp32-ddp-ws4-s1234
    CM066c-dense-muon-gpt130m-bf16-ddp-ws4-s1234
    CM066d-m002-greedylore-muon-gpt130m-bf16-ddp-ws4-s1234
)
entries=(train.py train_greedylore.py train.py train_greedylore.py)
model_dtypes=(float32 float32 bfloat16 bfloat16)
methods=(dense greedylore dense greedylore)
configs=(
    "$repo_dir/configs/compressed_muon/cm066a_dense_muon_gpt130m_fp32_dtype_matrix_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm066b_m002_greedy_lore_muon_gpt130m_fp32_dtype_matrix_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm066c_dense_muon_gpt130m_bf16_dtype_matrix_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm066d_m002_greedy_lore_muon_gpt130m_bf16_dtype_matrix_s1234.yaml"
)

while (($#)); do
    case "$1" in
        --print-plan) mode=print; shift ;;
        --smoke-only) mode=smoke; shift ;;
        --gpu-list) requested_gpu_list="$2"; shift 2 ;;
        *) printf 'usage: %s [--print-plan|--smoke-only] [--gpu-list 0,1,2,3]\n' "$0" >&2; exit 64 ;;
    esac
done

print_plan() {
    "$python_bin" - <<'PY'
import json

print(json.dumps({
    "controller_id": "CM066-m002-greedylore-model-dtype-matrix-controller",
    "execution": "serial_fail_fast",
    "world_size": 4,
    "gpu_policy": {
        "selection": "any_four",
        "minimum_free_mib": 18000,
        "poll_seconds": 60,
    },
    "dataset": "fineweb10B",
    "training_seed": 1234,
    "cells": [
        {"id": "CM066a-dense-muon-gpt130m-fp32-ddp-ws4-s1234", "entry": "train.py", "model_dtype": "float32", "method": "dense"},
        {"id": "CM066b-m002-greedylore-muon-gpt130m-fp32-ddp-ws4-s1234", "entry": "train_greedylore.py", "model_dtype": "float32", "method": "greedylore"},
        {"id": "CM066c-dense-muon-gpt130m-bf16-ddp-ws4-s1234", "entry": "train.py", "model_dtype": "bfloat16", "method": "dense"},
        {"id": "CM066d-m002-greedylore-muon-gpt130m-bf16-ddp-ws4-s1234", "entry": "train_greedylore.py", "model_dtype": "bfloat16", "method": "greedylore"},
    ],
    "pairing_rule": "compare dense and GreedyLore only within the same model_dtype",
    "formal_wandb": True,
    "smoke_mode": "--smoke-only disables W&B and does not launch formal cells",
}, indent=2))
PY
}

if [[ "$mode" == print ]]; then
    print_plan
    exit 0
fi

if [[ "$mode" == smoke ]]; then
    controller_dir="${controller_dir}-smoke"
fi
status_log="$controller_dir/status.log"
gpu_list=""

timestamp() { date --iso-8601=seconds; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_dir/controller_exit_code.txt"
    timestamp > "$controller_dir/controller_finished_at.txt"
}

capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

preflight() {
    local config
    for config in "${configs[@]}"; do
        [[ -f "$config" ]] || { log "BLOCKED missing config=$config"; return 1; }
    done
    [[ -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing dataset or virtualenv executables"
        return 1
    }
    if [[ "$mode" == run ]]; then
        "$python_bin" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)' || {
            log "BLOCKED W&B authentication unavailable"
            return 1
        }
    fi
}

select_free_gpus() {
    local -a candidates requested
    local gpu free
    candidates=()
    if [[ -n "$requested_gpu_list" ]]; then
        IFS=',' read -r -a requested <<< "$requested_gpu_list"
        [[ "${#requested[@]}" == "$world_size" ]] || return 1
        for gpu in "${requested[@]}"; do
            [[ "$gpu" =~ ^[0-9]+$ ]] || return 1
            free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
            [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || return 1
        done
        gpu_list="$requested_gpu_list"
        return 0
    fi
    while read -r gpu free; do
        [[ "$gpu" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        ((free >= minimum_free_mib)) && candidates+=("$gpu")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    ((${#candidates[@]} >= world_size)) || return 1
    gpu_list="$(IFS=,; printf '%s' "${candidates[*]:0:$world_size}")"
}

wait_for_gpus() {
    local attempts=0
    until select_free_gpus; do
        if ((attempts == 0 || attempts % 10 == 0)); then
            log "WAITING for four GPUs with at least ${minimum_free_mib} MiB free each"
        fi
        attempts=$((attempts + 1))
        sleep "$gpu_wait_seconds"
    done
    log "GPU READY cuda_visible_devices=$gpu_list"
}

make_probe_config() {
    local source_config="$1" output_config="$2" method="$3"
    cp "$source_config" "$output_config"
    sed -E -i \
        -e 's/^num_iterations:.*/num_iterations: 3/' \
        -e 's/^val_loss_every:.*/val_loss_every: 0/' \
        -e 's/^val_tokens:.*/val_tokens: 131072/' \
        -e 's/^checkpoint_freq:.*/checkpoint_freq: 0/' \
        -e 's/^warmup_steps:.*/warmup_steps: 0/' \
        -e 's/^no_wandb:.*/no_wandb: true/' \
        "$output_config"
    if [[ "$method" == greedylore ]]; then
        sed -E -i \
            -e 's/^greedy_lore_start_compress_step:.*/greedy_lore_start_compress_step: 0/' \
            -e 's/^greedy_lore_update_interval:.*/greedy_lore_update_interval: 2/' \
            "$output_config"
    fi
}

run_probe() {
    local index="$1" id entry method config probe_dir probe_config output rc
    id="${ids[$index]}"
    entry="$repo_dir/${entries[$index]}"
    method="${methods[$index]}"
    config="${configs[$index]}"
    probe_dir="$controller_dir/probes/$id"
    probe_config="$probe_dir/config.yaml"
    output="$probe_dir/stdout.log"
    mkdir -p "$probe_dir"
    make_probe_config "$config" "$probe_config" "$method"
    printf '%s\n' "CUDA_VISIBLE_DEVICES=$gpu_list $torchrun_bin --standalone --nproc_per_node=$world_size $entry --config $probe_config --data_dir $data_dir --training-seed $training_seed --no_wandb" > "$probe_dir/command.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_before.txt"
    log "PROBE START id=$id dtype=${model_dtypes[$index]} method=$method"
    timeout --signal=TERM --kill-after=60s "$probe_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$probe_config" --data_dir "$data_dir" \
        --training-seed "$training_seed" --no_wandb > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_after.txt"
    ((rc == 0)) || { log "PROBE FAIL id=$id exit=$rc"; return "$rc"; }
    log "PROBE PASS id=$id"
}

run_formal() {
    local index="$1" id entry method config cell_dir checkpoint_dir output rc
    id="${ids[$index]}"
    entry="$repo_dir/${entries[$index]}"
    method="${methods[$index]}"
    config="${configs[$index]}"
    cell_dir="$artifact_base/$id"
    checkpoint_dir="$cell_dir/checkpoints"
    output="$cell_dir/stdout.log"
    if [[ -e "$cell_dir" ]]; then
        log "BLOCKED refusing to overwrite formal artifact=$cell_dir"
        return 73
    fi
    mkdir -p "$cell_dir" "$checkpoint_dir"
    cp "$config" "$cell_dir/config.yaml"
    printf '%s\n' "CUDA_VISIBLE_DEVICES=$gpu_list $torchrun_bin --standalone --nproc_per_node=$world_size $entry --config $config --data_dir $data_dir --training-seed $training_seed --checkpoint_dir $checkpoint_dir --wandb_job_name $id" > "$cell_dir/command.txt"
    {
        printf 'experiment_id=%s\nmethod=%s\nmodel_dtype=%s\n' "$id" "$method" "${model_dtypes[$index]}"
        printf 'dataset=fineweb10B\nworld_size=%s\ncuda_visible_devices=%s\n' "$world_size" "$gpu_list"
        printf 'training_seed=%s\ngit_head=%s\n' "$training_seed" "$(git -C "$repo_dir" rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "FORMAL START id=$id dtype=${model_dtypes[$index]} method=$method"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$config" --data_dir "$data_dir" --training-seed "$training_seed" \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$id" > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    ((rc == 0)) || { log "FORMAL FAIL id=$id exit=$rc"; return "$rc"; }
    tr '\r' '\n' < "$output" | rg 'step:20000/20000 val_loss:|Peak memory consumption:' | tail -n 2 > "$cell_dir/result.txt"
    [[ "$(wc -l < "$cell_dir/result.txt")" == 2 ]] || {
        log "FORMAL INVALID id=$id missing final validation or memory metric"
        return 3
    }
    log "FORMAL PASS id=$id $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

if [[ -e "$controller_dir" ]]; then
    printf 'refusing to overwrite controller artifact: %s\n' "$controller_dir" >&2
    exit 73
fi
mkdir -p "$controller_dir"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$controller_dir/controller_started_at.txt"
cp "$0" "$controller_dir/controller.sh"
print_plan > "$controller_dir/plan.json"
capture_gpu_state "$controller_dir/nvidia_smi_at_start.txt"
preflight || exit 78

for index in 0 1 2 3; do
    wait_for_gpus
    run_probe "$index" || exit $?
done

if [[ "$mode" == smoke ]]; then
    log "SMOKE COMPLETE; formal cells not launched"
    exit 0
fi

for index in 0 1 2 3; do
    wait_for_gpus
    run_formal "$index" || exit $?
done

log "COMPLETE CM066 strict model-dtype matrix"
