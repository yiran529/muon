#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_root="$repo_dir/artifacts/compressed_muon"
controller_dir="$artifact_root/CM020-no-sync-wallclock-diagnostic"
status_log="$controller_dir/status.log"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
gpu_list="2,3,4,5"
world_size=4
run_limit=1h

dense_id="CM020a-muon-dense-gpt350m-corrected-nosync"
arc_id="CM020b-m001-arc-muon-gpt350m-corrected-nosync"
dense_config="$repo_dir/configs/compressed_muon/cm020a_dense_muon_gpt350m.yaml"
arc_config="$repo_dir/configs/compressed_muon/cm020b_arc_muon_gpt350m.yaml"

print_plan() {
    printf '%s\n' \
        '{' \
        '  "cells": [' \
        '    "CM020a-muon-dense-gpt350m-corrected-nosync",' \
        '    "CM020b-m001-arc-muon-gpt350m-corrected-nosync"' \
        '  ],' \
        '  "cuda_visible_devices": "2,3,4,5",' \
        '  "model": {"dim": 1024, "layers": 20, "heads": 16},' \
        '  "sequence_length": 1024,' \
        '  "batch_size": 1024,' \
        '  "device_batch_size": 1,' \
        '  "num_iterations": 15,' \
        '  "primary_uses_time_optimizer": false,' \
        '  "compile": true,' \
        '  "serial": true,' \
        '  "arc": {"ratio": 0.2, "projection_rank": 4, "eta": 0.1, "start_compress_step": 0}' \
        '}'
}

if [[ "${1:-}" == "--print-plan" ]]; then
    print_plan
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf 'usage: %s [--print-plan]\n' "$0" >&2
    exit 64
fi

mkdir -p "$controller_dir"
: > "$status_log"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$status_log"
}

run_cpu_gate() {
    log "CPU_GATE START"
    cd "$repo_dir" || return 1
    PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --frozen --extra dev python -m pytest \
        tests/test_no_sync_diagnostic.py \
        tests/test_train_ddp_sync.py \
        tests/test_train_arctopk.py \
        tests/test_muon_arctopk.py \
        tests/test_muon_arctopk_distributed.py \
        tests/test_no_sync_wallclock_launcher.py -q \
        > "$controller_dir/cpu-tests.log" 2>&1 || return 1
    PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
        "$python_bin" "$repo_dir/benchmark/compressed_muon/no_sync_diagnostic.py" \
        --output "$controller_dir/no-sync-diagnostic.json" \
        > "$controller_dir/no-sync-diagnostic.log" 2>&1 || return 1
    log "CPU_GATE PASS"
}

preflight_static() {
    [[ -d "$data_dir" ]] || { log "BLOCKED missing dataset: $data_dir"; return 1; }
    command -v timeout >/dev/null || { log "BLOCKED GNU timeout unavailable"; return 1; }
    [[ -x "$torchrun_bin" ]] || { log "BLOCKED missing torchrun: $torchrun_bin"; return 1; }
}

preflight_gpus() {
    local rows count bad
    rows="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)" || {
        log "BLOCKED nvidia-smi unavailable"
        return 1
    }
    count="$(awk -F', ' '$1 >= 2 && $1 <= 5 {n++} END {print n+0}' <<< "$rows")"
    bad="$(awk -F', ' '$1 >= 2 && $1 <= 5 && $2 >= 1024 {n++} END {print n+0}' <<< "$rows")"
    [[ "$count" == 4 && "$bad" == 0 ]] || {
        log "BLOCKED GPU 2-5 unavailable or using >=1024 MiB"
        return 1
    }
}

run_cell() {
    local mode="$1" id entry config cell_dir rc
    if [[ "$mode" == dense ]]; then
        id="$dense_id"
        entry="$repo_dir/train.py"
        config="$dense_config"
    else
        id="$arc_id"
        entry="$repo_dir/train_arctopk.py"
        config="$arc_config"
    fi
    cell_dir="$artifact_root/$id"
    if [[ -e "$cell_dir/started_at.txt" ]]; then
        log "BLOCKED refusing to overwrite existing cell: $id"
        return 1
    fi
    mkdir -p "$cell_dir"
    cp "$config" "$cell_dir/config.yaml"
    {
        printf 'experiment_id=%s\nmode=%s\n' "$id" "$mode"
        printf 'git_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
        printf 'cuda_visible_devices=%s\nworld_size=%s\n' "$gpu_list" "$world_size"
        printf 'model_dim=1024\nn_layer=20\nn_head=16\nsequence_length=1024\n'
        printf 'batch_size=1024\ndevice_batch_size=1\ngrad_accumulation=256\n'
        printf 'num_iterations=15\ncompile=true\nprimary_uses_time_optimizer=false\n'
    } > "$cell_dir/environment.txt"
    printf 'CUDA_VISIBLE_DEVICES=%q %q --standalone --nproc_per_node=%q %q --config %q --data_dir %q --no_wandb\n' \
        "$gpu_list" "$torchrun_bin" "$world_size" "$entry" "$config" "$data_dir" \
        > "$cell_dir/command.txt"
    date -Is > "$cell_dir/started_at.txt"
    log "GPU CELL START mode=$mode id=$id"
    timeout --signal=TERM --kill-after=60s "$run_limit" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone --nproc_per_node="$world_size" \
        "$entry" --config "$config" --data_dir "$data_dir" --no_wandb \
        > "$cell_dir/stdout.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    date -Is > "$cell_dir/finished_at.txt"
    if [[ "$rc" -ne 0 ]]; then
        log "GPU CELL FAIL mode=$mode id=$id exit=$rc"
        return "$rc"
    fi
    tr '\r' '\n' < "$cell_dir/stdout.log" \
        | grep -E 'step:15/15 val_loss:.*step_avg:' \
        | tail -n 1 > "$cell_dir/result.txt"
    if [[ ! -s "$cell_dir/result.txt" ]]; then
        log "GPU CELL FAIL mode=$mode id=$id missing_final_timing"
        return 65
    fi
    log "GPU CELL PASS mode=$mode id=$id"
}

log "BEGIN corrected no_sync GPT-350M serial diagnostic"
run_cpu_gate || { log "BLOCKED CPU gate failed"; exit 78; }
preflight_static || exit 78
preflight_gpus || exit 78
run_cell dense || exit "$?"
preflight_gpus || exit 78
run_cell arc || exit "$?"
log "COMPLETE corrected no_sync GPT-350M serial diagnostic"
