#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
profiler_launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
artifact_root="$repo_dir/artifacts/compressed_muon/CM064-m002-gpt130m-bucket-cap-sweep-ws4-s42"
gpu_list=""
mode="run"

while (($#)); do
    case "$1" in
        --artifact-root) artifact_root="$2"; shift 2 ;;
        --gpu-list) gpu_list="$2"; shift 2 ;;
        --print-plan) mode="print"; shift ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

print_plan() {
    "$repo_dir/.venv/bin/python" - <<'PY'
import json

print(json.dumps({
    "bucket_caps_mib": [80, 160, 256, 384],
    "world_size": 4,
    "model": {"dim": 768, "layers": 8, "heads": 12},
    "global_batch_size": 512,
    "device_batch_size": 128,
    "sequence_length": 256,
    "timing_warmup_steps": 20,
    "measured_updates_per_cell": 200,
    "profile_modes": [],
    "timing_modes": ["greedylore_local_svd"],
    "cells": [
        "cap80-i100", "cap80-i200",
        "cap160-i200", "cap160-i100",
        "cap256-i100", "cap256-i200",
        "cap384-i200", "cap384-i100",
    ],
}, indent=2))
PY
}

if [[ "$mode" == "print" ]]; then
    print_plan
    exit 0
fi

if [[ -e "$artifact_root" && -n "$(find "$artifact_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    printf 'refusing to overwrite existing artifact root: %s\n' "$artifact_root" >&2
    exit 73
fi
mkdir -p "$artifact_root"
print_plan > "$artifact_root/plan.json"
status_log="$artifact_root/status.log"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$status_log"; }

if [[ -z "$gpu_list" ]]; then
    while [[ -z "$gpu_list" ]]; do
        gpu_list="$({ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
            | "$profiler_launcher" --world-size 4 --exclude-gpus 0,1,7 --select-gpus-from-stdin; } || true)"
        [[ -n "$gpu_list" ]] || { log "GPU_WAIT required=4 eligible=2-6"; sleep 60; }
    done
fi
[[ "$(awk -F, '{print NF}' <<<"$gpu_list")" -eq 4 ]] || {
    printf 'gpu-list must contain exactly four GPUs\n' >&2
    exit 64
}
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

run_cell() {
    local cap="$1" interval="$2" periods cell child
    periods=1
    [[ "$interval" -eq 100 ]] && periods=2
    cell="cap${cap}-i${interval}"
    child="$artifact_root/$cell"
    log "CELL_START cell=$cell"
    "$profiler_launcher" \
        --world-size 4 \
        --global-batch-size 512 \
        --device-batch-size 128 \
        --model-dim 768 \
        --layers 8 \
        --heads 12 \
        --sequence-length 256 \
        --bucket-cap-mb "$cap" \
        --timing-warmup-steps 20 \
        --measured-full-periods "$periods" \
        --repeats 1 \
        --profile-modes none \
        --timing-modes greedylore_local_svd \
        --training-seed 42 \
        --greedy-lore-rank 32 \
        --greedy-lore-update-interval "$interval" \
        --gpu-list "$gpu_list" \
        --artifact-root "$child"
    log "CELL_DONE cell=$cell"
}

run_cell 80 100
run_cell 80 200
run_cell 160 200
run_cell 160 100
run_cell 256 100
run_cell 256 200
run_cell 384 200
run_cell 384 100

date --iso-8601=seconds > "$artifact_root/finished_at.txt"
log "EXPERIMENT_DONE root=$artifact_root"
