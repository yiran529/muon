#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
dense_config="$repo_dir/configs/compressed_muon/cm020a_dense_muon_gpt350m.yaml"
arc_config="$repo_dir/configs/compressed_muon/m001_arc_topk_muon_ddp.yaml"
world_size=3
global_batch_size=768
device_batch_size=1
model_dim=1024
layers=20
heads=16
sequence_length=1024
timing_warmup_steps=12
measured_steps=1
requested_num_iterations=""
bucket_cap_mb=25
repeats=1
training_seed=42
gpu_list=""
exclude_gpus=""
artifact_root=""
memory_limit_mib=1024
poll_seconds=60
mode="run"

while (($#)); do
    case "$1" in
        --world-size) world_size="$2"; shift 2 ;;
        --global-batch-size) global_batch_size="$2"; shift 2 ;;
        --device-batch-size) device_batch_size="$2"; shift 2 ;;
        --model-dim) model_dim="$2"; shift 2 ;;
        --layers) layers="$2"; shift 2 ;;
        --heads) heads="$2"; shift 2 ;;
        --sequence-length) sequence_length="$2"; shift 2 ;;
        --profile-step) timing_warmup_steps="$2"; shift 2 ;;
        --num-iterations) requested_num_iterations="$2"; shift 2 ;;
        --timing-warmup-steps) timing_warmup_steps="$2"; shift 2 ;;
        --measured-steps) measured_steps="$2"; shift 2 ;;
        --bucket-cap-mb) bucket_cap_mb="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --training-seed) training_seed="$2"; shift 2 ;;
        --gpu-list) gpu_list="$2"; shift 2 ;;
        --exclude-gpus) exclude_gpus="$2"; shift 2 ;;
        --artifact-root) artifact_root="$2"; shift 2 ;;
        --print-plan) mode="print"; shift ;;
        --select-gpus-from-stdin) mode="select"; shift ;;
        --summarize-only) mode="summarize"; shift ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

denominator=$((world_size * device_batch_size))
if ((world_size < 1 || device_batch_size < 1 || global_batch_size % denominator != 0)); then
    printf 'global batch must be divisible by world_size * device_batch_size\n' >&2
    exit 64
fi
if [[ -n "$requested_num_iterations" ]]; then
    measured_steps=$((requested_num_iterations - timing_warmup_steps))
fi
num_iterations=$((timing_warmup_steps + measured_steps))
profile_step="$timing_warmup_steps"
if ((timing_warmup_steps < 1 || measured_steps < 1)); then
    printf 'timing warmup and measured steps must both be positive\n' >&2
    exit 64
fi
grad_accum_steps=$((global_batch_size / denominator))
val_tokens=$((world_size * device_batch_size * sequence_length))
artifact_root="${artifact_root:-$repo_dir/artifacts/compressed_muon/CM024-gpt350m-three-mode-profiler-ws${world_size}}"

select_gpus() {
    awk -F, -v limit="$memory_limit_mib" -v excluded="$exclude_gpus" '
        BEGIN {n=split(excluded, values, ","); for (i=1; i<=n; i++) skip[values[i]]=1}
        {gsub(/ /, "", $1); gsub(/ /, "", $2)}
        ($2 + 0) < (limit + 0) && !($1 in skip) {print $1}
    ' | sort -n | head -n "$world_size" | paste -sd, -
}

if [[ "$mode" == "select" ]]; then
    selected="$(select_gpus)"
    [[ -n "$selected" && "$(awk -F, '{print NF}' <<<"$selected")" -eq "$world_size" ]] || exit 78
    printf '%s\n' "$selected"
    exit 0
fi

print_plan() {
    WS="$world_size" GBS="$global_batch_size" DBS="$device_batch_size" GA="$grad_accum_steps" \
    MD="$model_dim" NL="$layers" NH="$heads" SEQ="$sequence_length" PS="$profile_step" \
    NI="$num_iterations" WARMUP="$timing_warmup_steps" MEASURED="$measured_steps" BUCKET="$bucket_cap_mb" \
    REPS="$repeats" GPU_LIST_VALUE="${gpu_list:-dynamic}" EXCLUDED="$exclude_gpus" ROOT="$artifact_root" \
    "$python_bin" - <<'PY'
import json, os
import itertools
repeats=int(os.environ["REPS"])
cells=[]
orders=list(itertools.permutations(("dense", "arc_optimizer", "arc_ddp_hook")))
for repeat in range(1, repeats+1):
    cells.extend(f"{mode}-r{repeat}" for mode in orders[(repeat-1) % len(orders)])
print(json.dumps({
  "world_size": int(os.environ["WS"]), "global_batch_size": int(os.environ["GBS"]),
  "device_batch_size": int(os.environ["DBS"]), "gradient_accumulation_steps": int(os.environ["GA"]),
  "model": {"dim": int(os.environ["MD"]), "layers": int(os.environ["NL"]), "heads": int(os.environ["NH"])},
  "sequence_length": int(os.environ["SEQ"]), "profile_step": int(os.environ["PS"]),
  "num_iterations": int(os.environ["NI"]), "timing_warmup_steps": int(os.environ["WARMUP"]),
  "measured_steps": int(os.environ["MEASURED"]), "bucket_cap_mb": float(os.environ["BUCKET"]), "cells": cells,
  "gpu_list": os.environ["GPU_LIST_VALUE"], "exclude_gpus": os.environ["EXCLUDED"], "artifact_root": os.environ["ROOT"],
  "profile_scope": "final_microstep_and_optimizer", "uses_time_optimizer": False,
  "require_final_timing": True,
}, indent=2))
PY
}

if [[ "$mode" == "print" ]]; then
    print_plan
    exit 0
fi

timestamp() { date --iso-8601=seconds; }
status_log="$artifact_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }
summarize() {
    if ! PYTHONPATH="$repo_dir" "$python_bin" "$repo_dir/benchmark/compressed_muon/summarize_training_profiles.py" \
        "$artifact_root" --output "$artifact_root/summary.json" --require-plan; then
        log "SUMMARY_FAILED root=$artifact_root"
        return 65
    fi
}
if [[ "$mode" == "summarize" ]]; then
    mkdir -p "$artifact_root"
    summarize || exit $?
    timestamp > "$artifact_root/finished_at.txt"
    log "EXPERIMENT_DONE summary=$artifact_root/summary.json"
    exit 0
fi

mkdir -p "$artifact_root"
if [[ -e "$artifact_root/started_at.txt" ]]; then
    printf 'refusing to overwrite existing run: %s\n' "$artifact_root" >&2
    exit 73
fi
timestamp > "$artifact_root/started_at.txt"
print_plan > "$artifact_root/plan.json"

for required in "$python_bin" "$torchrun_bin" "$data_dir" "$dense_config" "$arc_config"; do
    [[ -e "$required" ]] || { log "BLOCKED missing=$required"; exit 66; }
done

if [[ -z "$gpu_list" ]]; then
    while true; do
        gpu_list="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | select_gpus)"
        count=0; [[ -n "$gpu_list" ]] && count="$(awk -F, '{print NF}' <<<"$gpu_list")"
        ((count == world_size)) && break
        gpu_list=""
        log "GPU_WAIT idle_count=$count required=$world_size"
        sleep "$poll_seconds"
    done
fi
[[ "$(awk -F, '{print NF}' <<<"$gpu_list")" -eq "$world_size" ]] || { log "BLOCKED gpu_list_count"; exit 64; }
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

selected_idle() {
    local gpu used
    IFS=',' read -ra ids <<<"$gpu_list"
    for gpu in "${ids[@]}"; do
        used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
        [[ "$used" =~ ^[0-9]+$ ]] && ((used < memory_limit_mib)) || return 1
    done
}

run_cell() {
    local cell="$1" mode_name="$2" entry config cell_dir rc
    cell_dir="$artifact_root/$cell"
    entry="$repo_dir/train.py"; config="$dense_config"
    [[ "$mode_name" != "dense" ]] && { entry="$repo_dir/train_arctopk.py"; config="$arc_config"; }
    while ! selected_idle; do log "GPU_WAIT_SELECTED list=$gpu_list"; sleep "$poll_seconds"; done
    mkdir -p "$cell_dir/profiler"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry"
        --config "$config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim "$model_dim" --n_layer "$layers" --n_head "$heads"
        --sequence_length "$sequence_length" --batch_size "$global_batch_size"
        --device_batch_size "$device_batch_size" --val_tokens "$val_tokens"
        --num_iterations "$num_iterations" --training-seed "$training_seed"
        --timing-warmup-steps "$timing_warmup_steps" --bucket-cap-mb "$bucket_cap_mb"
        --profile-output-dir "$cell_dir/profiler" --profile-step "$profile_step")
    if [[ "$mode_name" == "arc_optimizer" ]]; then
        command+=(--arc_sync_mode optimizer --arc_start_compress_step 0)
    elif [[ "$mode_name" == "arc_ddp_hook" ]]; then
        command+=(--arc_sync_mode ddp_hook --arc_start_compress_step 0)
    fi
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"; printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"; log "CELL_START cell=$cell mode=$mode_name"
    timeout --signal=TERM --kill-after=60 3600 "${command[@]}" > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?; printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"; timestamp > "$cell_dir/finished_at.txt"
    ((rc == 0)) || { log "CELL_FAILED cell=$cell exit=$rc"; return "$rc"; }
    trace_count="$(find "$cell_dir/profiler" -maxdepth 1 -name 'rank-*.json' | wc -l)"
    ((trace_count == world_size)) || { log "CELL_FAILED cell=$cell traces=$trace_count"; return 65; }
    log "CELL_DONE cell=$cell traces=$trace_count"
}

for ((repeat=1; repeat<=repeats; repeat++)); do
    case $(((repeat - 1) % 6)) in
        0) order=(dense arc_optimizer arc_ddp_hook) ;;
        1) order=(dense arc_ddp_hook arc_optimizer) ;;
        2) order=(arc_optimizer dense arc_ddp_hook) ;;
        3) order=(arc_optimizer arc_ddp_hook dense) ;;
        4) order=(arc_ddp_hook dense arc_optimizer) ;;
        5) order=(arc_ddp_hook arc_optimizer dense) ;;
    esac
    for mode_name in "${order[@]}"; do run_cell "${mode_name}-r${repeat}" "$mode_name" || exit $?; done
done

summarize || exit $?
timestamp > "$artifact_root/finished_at.txt"
log "EXPERIMENT_DONE summary=$artifact_root/summary.json"
