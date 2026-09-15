#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
dense_config="$repo_dir/configs/compressed_muon/cm037a_dense_muon_scalar_adamw.yaml"
greedy_lore_config="$repo_dir/configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml"
world_size=4
global_batch_size=1024
device_batch_size=1
model_dim=1024
layers=20
heads=16
sequence_length=1024
model_dtype="float32"
timing_warmup_steps=20
measured_full_periods=1
bucket_cap_mb=160
repeats=1
profile_modes_csv="dense,greedylore_local_svd,greedylore_broadcast"
timing_modes_csv="dense,greedylore_local_svd,greedylore_broadcast"
training_seed=42
greedy_lore_rank=32
greedy_lore_update_interval=200
greedy_lore_dense_aux_communication_dtype="bucket"
greedy_lore_compress_embedding_lm_head=0
greedy_lore_isolate_dense_aux_buckets=0
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
        --model-dtype) model_dtype="$2"; shift 2 ;;
        --bucket-cap-mb) bucket_cap_mb="$2"; shift 2 ;;
        --timing-warmup-steps) timing_warmup_steps="$2"; shift 2 ;;
        --measured-full-periods) measured_full_periods="$2"; shift 2 ;;
        --repeats) repeats="$2"; shift 2 ;;
        --profile-modes) profile_modes_csv="$2"; shift 2 ;;
        --timing-modes) timing_modes_csv="$2"; shift 2 ;;
        --training-seed) training_seed="$2"; shift 2 ;;
        --greedy-lore-rank) greedy_lore_rank="$2"; shift 2 ;;
        --greedy-lore-update-interval) greedy_lore_update_interval="$2"; shift 2 ;;
        --greedy-lore-dense-aux-communication-dtype) greedy_lore_dense_aux_communication_dtype="$2"; shift 2 ;;
        --greedy-lore-compress-embedding-lm-head) greedy_lore_compress_embedding_lm_head=1; shift ;;
        --greedy-lore-isolate-dense-aux-buckets) greedy_lore_isolate_dense_aux_buckets=1; shift ;;
        --gpu-list) gpu_list="$2"; shift 2 ;;
        --exclude-gpus) exclude_gpus="$2"; shift 2 ;;
        --artifact-root) artifact_root="$2"; shift 2 ;;
        --print-plan) mode="print"; shift ;;
        --select-gpus-from-stdin) mode="select"; shift ;;
        --summarize-only) mode="summarize"; shift ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

validate_modes() {
    local raw="$1" label="$2" allow_none="$3" value seen=","
    local -a values
    if [[ "$raw" == "none" ]]; then
        ((allow_none == 1)) && return 0
        printf '%s modes cannot be none\n' "$label" >&2
        return 64
    fi
    IFS=',' read -ra values <<<"$raw"
    for value in "${values[@]}"; do
        case "$value" in
            dense|greedylore_local_svd|greedylore_broadcast) ;;
            *) printf 'invalid %s mode: %s\n' "$label" "$value" >&2; return 64 ;;
        esac
        if [[ "$seen" == *",$value,"* ]]; then
            printf 'duplicate %s mode: %s\n' "$label" "$value" >&2
            return 64
        fi
        seen+="$value,"
    done
}

validate_modes "$profile_modes_csv" profile 1 || exit $?
validate_modes "$timing_modes_csv" timing 1 || exit $?
case "$greedy_lore_dense_aux_communication_dtype" in
    bucket|float32|bfloat16) ;;
    *) printf 'invalid GreedyLore dense auxiliary communication dtype: %s\n' "$greedy_lore_dense_aux_communication_dtype" >&2; exit 64 ;;
esac
case "$model_dtype" in
    float32|bfloat16) ;;
    *) printf 'invalid model dtype: %s\n' "$model_dtype" >&2; exit 64 ;;
esac
if ((greedy_lore_compress_embedding_lm_head && greedy_lore_isolate_dense_aux_buckets)); then
    printf 'embedding/lm-head compression and dense auxiliary bucket isolation are mutually exclusive\n' >&2
    exit 64
fi
if [[ "$profile_modes_csv" == "none" ]]; then
    profile_modes=()
else
    IFS=',' read -ra profile_modes <<<"$profile_modes_csv"
fi
if [[ "$timing_modes_csv" == "none" ]]; then
    timing_modes=()
else
    IFS=',' read -ra timing_modes <<<"$timing_modes_csv"
fi

denominator=$((world_size * device_batch_size))
if ((world_size < 1 || device_batch_size < 1 || global_batch_size % denominator != 0)); then
    printf 'global batch must be divisible by world_size * device_batch_size\n' >&2
    exit 64
fi
if ((timing_warmup_steps < 1 || measured_full_periods < 1 || greedy_lore_update_interval < 2)); then
    printf 'warmup, measured periods, and GreedyLore interval must be positive; interval must be at least 2\n' >&2
    exit 64
fi
grad_accum_steps=$((global_batch_size / denominator))
refresh_profile_step="$timing_warmup_steps"
compressed_profile_step=$((timing_warmup_steps + 1))
timing_num_iterations=$((timing_warmup_steps + measured_full_periods * greedy_lore_update_interval))
val_tokens=$((world_size * device_batch_size * sequence_length))
artifact_root="${artifact_root:-$repo_dir/artifacts/compressed_muon/CM-greedylore-profiler-ws${world_size}}"

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
    MD="$model_dim" NL="$layers" NH="$heads" SEQ="$sequence_length" MODEL_DTYPE="$model_dtype" REFRESH="$refresh_profile_step" \
    COMPRESSED="$compressed_profile_step" WARMUP="$timing_warmup_steps" PERIODS="$measured_full_periods" \
    TIMING_NI="$timing_num_iterations" BUCKET="$bucket_cap_mb" REPS="$repeats" GL_RANK="$greedy_lore_rank" \
    GL_INTERVAL="$greedy_lore_update_interval" GL_DENSE_AUX_DTYPE="$greedy_lore_dense_aux_communication_dtype" \
    GL_COMPRESS_EMBEDDING_LM_HEAD="$greedy_lore_compress_embedding_lm_head" \
    GL_ISOLATE_DENSE_AUX_BUCKETS="$greedy_lore_isolate_dense_aux_buckets" \
    GPU_LIST_VALUE="${gpu_list:-dynamic}" EXCLUDED="$exclude_gpus" \
    PROFILE_MODES="$profile_modes_csv" TIMING_MODES="$timing_modes_csv" \
    ROOT="$artifact_root" "$python_bin" - <<'PY'
import json
import os

repeats = int(os.environ["REPS"])
profile_modes = (
    ()
    if os.environ["PROFILE_MODES"] == "none"
    else tuple(os.environ["PROFILE_MODES"].split(","))
)
timing_modes = (
    ()
    if os.environ["TIMING_MODES"] == "none"
    else tuple(os.environ["TIMING_MODES"].split(","))
)
cells = []
timing_cells = []
for repeat in range(1, repeats + 1):
    profile_offset = (repeat - 1) % len(profile_modes) if profile_modes else 0
    profile_order = profile_modes[profile_offset:] + profile_modes[:profile_offset]
    for mode in profile_order:
        cells.append(f"{mode}-refresh-r{repeat}")
        cells.append(f"{mode}-compressed-r{repeat}")
    timing_offset = (repeat - 1) % len(timing_modes) if timing_modes else 0
    timing_order = timing_modes[timing_offset:] + timing_modes[:timing_offset]
    for mode in timing_order:
        timing_cells.append(f"{mode}-timing-r{repeat}")
print(json.dumps({
    "world_size": int(os.environ["WS"]),
    "global_batch_size": int(os.environ["GBS"]),
    "device_batch_size": int(os.environ["DBS"]),
    "gradient_accumulation_steps": int(os.environ["GA"]),
    "model": {
        "dim": int(os.environ["MD"]),
        "layers": int(os.environ["NL"]),
        "heads": int(os.environ["NH"]),
    },
    "sequence_length": int(os.environ["SEQ"]),
    "model_dtype": os.environ["MODEL_DTYPE"],
    "bucket_cap_mb": float(os.environ["BUCKET"]),
    "timing_warmup_steps": int(os.environ["WARMUP"]),
    "measured_full_periods": int(os.environ["PERIODS"]),
    "timing_num_iterations": int(os.environ["TIMING_NI"]),
    "profile_modes": list(profile_modes),
    "timing_modes": list(timing_modes),
    "greedy_lore": {
        "rank": int(os.environ["GL_RANK"]),
        "update_interval": int(os.environ["GL_INTERVAL"]),
        "dense_aux_communication_dtype": os.environ["GL_DENSE_AUX_DTYPE"],
        "compress_embedding_lm_head": bool(int(os.environ["GL_COMPRESS_EMBEDDING_LM_HEAD"])),
        "isolate_dense_aux_buckets": bool(int(os.environ["GL_ISOLATE_DENSE_AUX_BUCKETS"])),
        "start_compress_step": int(os.environ["WARMUP"]),
        "refresh_profile_step": int(os.environ["REFRESH"]),
        "compressed_profile_step": int(os.environ["COMPRESSED"]),
    },
    "cells": cells,
    "timing_cells": timing_cells,
    "gpu_list": os.environ["GPU_LIST_VALUE"],
    "exclude_gpus": os.environ["EXCLUDED"],
    "artifact_root": os.environ["ROOT"],
    "profile_scope": "final_microstep_and_optimizer",
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

if [[ -e "$artifact_root" || -L "$artifact_root" ]]; then
    if [[ -L "$artifact_root" || ! -d "$artifact_root" || -n "$(find "$artifact_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        printf 'refusing to overwrite existing artifact root: %s\n' "$artifact_root" >&2
        exit 73
    fi
fi
mkdir -p "$artifact_root"
timestamp > "$artifact_root/started_at.txt"
print_plan > "$artifact_root/plan.json"

for required in "$python_bin" "$torchrun_bin" "$data_dir" "$dense_config" "$greedy_lore_config"; do
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
    local cell="$1" mode_name="$2" phase="$3" run_kind="$4" entry config cell_dir rc profile_step run_iterations
    cell_dir="$artifact_root/$cell"
    entry="$repo_dir/train.py"
    config="$dense_config"
    if [[ "$mode_name" == greedylore_* ]]; then
        entry="$repo_dir/train_greedylore.py"
        config="$greedy_lore_config"
    fi
    if [[ "$phase" == "refresh" ]]; then
        profile_step="$refresh_profile_step"
    else
        profile_step="$compressed_profile_step"
    fi
    run_iterations="$timing_num_iterations"
    if [[ "$run_kind" == "profile" ]]; then
        run_iterations=$((profile_step + 1))
    fi
    while ! selected_idle; do log "GPU_WAIT_SELECTED list=$gpu_list"; sleep "$poll_seconds"; done
    mkdir -p "$cell_dir"
    if [[ "$run_kind" == "profile" ]]; then
        mkdir -p "$cell_dir/profiler"
    fi
    env | sort > "$cell_dir/environment.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry"
        --config "$config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim "$model_dim" --n_layer "$layers" --n_head "$heads"
        --model_dtype "$model_dtype"
        --sequence_length "$sequence_length" --batch_size "$global_batch_size"
        --device_batch_size "$device_batch_size" --val_tokens "$val_tokens"
        --num_iterations "$run_iterations" --training-seed "$training_seed"
        --timing-warmup-steps "$timing_warmup_steps" --bucket-cap-mb "$bucket_cap_mb")
    if [[ "$mode_name" == greedylore_* ]]; then
        command+=(--greedy_lore_rank "$greedy_lore_rank"
            --greedy_lore_update_interval "$greedy_lore_update_interval"
            --greedy_lore_start_compress_step "$timing_warmup_steps"
            --greedy_lore_dense_aux_communication_dtype "$greedy_lore_dense_aux_communication_dtype")
        if ((greedy_lore_compress_embedding_lm_head)); then
            command+=(--greedy_lore_compress_embedding_lm_head)
        fi
        if ((greedy_lore_isolate_dense_aux_buckets)); then
            command+=(--greedy_lore_isolate_dense_aux_buckets)
        fi
        if [[ "$mode_name" == "greedylore_broadcast" ]]; then
            command+=(--greedy_lore_basis_sync broadcast)
        else
            command+=(--greedy_lore_basis_sync local_svd)
        fi
    fi
    if [[ "$run_kind" == "profile" ]]; then
        command+=(--profile-output-dir "$cell_dir/profiler" --profile-step "$profile_step")
    fi
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"; printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"; log "CELL_START cell=$cell mode=$mode_name kind=$run_kind"
    timeout --signal=TERM --kill-after=60 3600 "${command[@]}" > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?; printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"; timestamp > "$cell_dir/finished_at.txt"
    ((rc == 0)) || { log "CELL_FAILED cell=$cell exit=$rc"; return "$rc"; }
    if [[ "$run_kind" == "profile" ]]; then
        trace_count="$(find "$cell_dir/profiler" -maxdepth 1 -name 'rank-*.json' | wc -l)"
        ((trace_count == world_size)) || { log "CELL_FAILED cell=$cell traces=$trace_count"; return 65; }
        log "CELL_DONE cell=$cell traces=$trace_count"
    else
        log "CELL_DONE cell=$cell"
    fi
}

run_repeat() {
    local repeat="$1"
    local index mode_name
    for ((index=0; index<${#profile_modes[@]}; index++)); do
        mode_name="${profile_modes[$(((repeat - 1 + index) % ${#profile_modes[@]}))]}"
        run_cell "${mode_name}-refresh-r${repeat}" "$mode_name" refresh profile || exit $?
        run_cell "${mode_name}-compressed-r${repeat}" "$mode_name" compressed profile || exit $?
    done
    for ((index=0; index<${#timing_modes[@]}; index++)); do
        mode_name="${timing_modes[$(((repeat - 1 + index) % ${#timing_modes[@]}))]}"
        run_cell "${mode_name}-timing-r${repeat}" "$mode_name" compressed timing || exit $?
    done
}

for ((repeat=1; repeat<=repeats; repeat++)); do
    run_repeat "$repeat"
done

summarize || exit $?
timestamp > "$artifact_root/finished_at.txt"
log "EXPERIMENT_DONE summary=$artifact_root/summary.json"
