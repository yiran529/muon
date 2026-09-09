#!/usr/bin/env bash
# Paired GPT-60M paper-recipe AdamW quality run: dense, then EF21M all-2D hook.
set -uo pipefail

repo_dir=/home/wyr/dion
experiment_id=CM035-m001-adamw-paper-recipe-all2d-hook-gpt60m-train-ddp-ws4-s42
artifact_root="$repo_dir/artifacts/compressed_muon/$experiment_id"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
dense_config="$repo_dir/configs/compressed_muon/cm035a_dense_adamw_gpt60m_paper_recipe.yaml"
arc_config="$repo_dir/configs/compressed_muon/cm035b_all2d_hook_adamw_gpt60m_paper_recipe.yaml"
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
elif [[ "${1:-}" == "--summarize" && $# -eq 2 ]]; then
    mode=summarize
    artifact_root="$2"
elif (($#)); then
    printf 'usage: %s [--print-plan | --summarize ARTIFACT_ROOT]\n' "$0" >&2
    exit 64
fi

print_plan() {
    "$python_bin" - <<'PY'
import json

print(json.dumps({
    "experiment_id": "CM035-m001-adamw-paper-recipe-all2d-hook-gpt60m-train-ddp-ws4-s42",
    "cells": [
        "dense_adamw_paper_recipe_gpt60m-s42",
        "arc_ef21m_all2d_adamw_paper_recipe_gpt60m-s42",
    ],
    "cells_detail": [
        {
            "cell": "dense_adamw_paper_recipe_gpt60m-s42",
            "entrypoint": "train.py",
            "optimizer": "adamw",
        },
        {
            "cell": "arc_ef21m_all2d_adamw_paper_recipe_gpt60m-s42",
            "entrypoint": "train_arctopk.py",
            "optimizer": "arc_topk_adamw",
            "error_feedback": "ef21m",
            "hook_arc_scope": "all_ndim_2_parameters",
        },
    ],
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
    "total_tokens": 1_100_087_296,
    "validation_interval": 500,
    "validation_tokens": 10_485_760,
    "adamw_recipe": {
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "grad_clip_norm": 1.0,
        "warmup_steps": 1000,
        "lr_schedule": "cosine",
        "weight_decay": 0.0
    },
    "training_seed": 42,
    "hook_arc_scope": "all_ndim_2_parameters",
    "arc_ratio": 0.2,
    "arc_projection_rank": 4,
    "arc_eta": 1.0,
    "arc_start_compress_step": 1000,
    "bucket_cap_mb": 160,
    "wandb": True,
}, indent=2))
PY
}

# This branch intentionally precedes every filesystem, GPU, W&B, and data action.
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
    local gpu free config
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
    for config in "$@"; do
        [[ -f "$config" ]] || {
            log "BLOCKED missing config=$config"
            return 1
        }
    done
    [[ -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing dataset or repository virtualenv executables"
        return 1
    }
    "$python_bin" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)' || {
        log "BLOCKED W&B authentication unavailable"
        return 1
    }
}

make_probe_config() {
    local source_config="$1" output_config="$2" val_tokens="$3" cell_mode="$4"
    cp "$source_config" "$output_config"
    sed -E -i \
        -e 's/^num_iterations:.*/num_iterations: 3/' \
        -e 's/^val_loss_every:.*/val_loss_every: 0/' \
        -e "s/^val_tokens:.*/val_tokens: $val_tokens/" \
        -e 's/^warmup_steps:.*/warmup_steps: 0/' \
        -e 's/^no_wandb:.*/no_wandb: true/' \
        "$output_config"
    if [[ "$cell_mode" == arc ]]; then
        sed -E -i 's/^arc_start_compress_step:.*/arc_start_compress_step: 0/' "$output_config"
    fi
}

run_probe() {
    local cell="$1" cell_mode="$2" entry="$3" config="$4" device_batch="$5"
    local ga probe_dir probe_config output val_tokens rc
    ga=$((global_batch / (world_size * device_batch)))
    probe_dir="$artifact_root/probes/${cell}-db${device_batch}"
    probe_config="$probe_dir/config.yaml"
    output="$probe_dir/stdout.log"
    val_tokens=$((device_batch * sequence_length * world_size))
    mkdir -p "$probe_dir"
    make_probe_config "$config" "$probe_config" "$val_tokens" "$cell_mode"
    capture_gpu_state "$probe_dir/nvidia_smi_before.txt"
    log "PROBE START cell=$cell mode=$cell_mode device_batch=$device_batch ga=$ga"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$entry" --config "$probe_config" --data_dir "$data_dir"
        --device_batch_size "$device_batch" --batch_size "$global_batch"
        --training-seed "$training_seed" --no_wandb)
    if [[ "$cell_mode" == arc ]]; then
        command+=(--bucket-cap-mb "$bucket_cap_mb")
    fi
    timeout --signal=TERM --kill-after=60s "$probe_timeout" "${command[@]}" > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_after.txt"
    if [[ "$rc" == 0 ]] && rg -q 'Peak memory consumption:' "$output"; then
        tr '\r' '\n' < "$output" |
            rg 'step:3/3 val_loss:|Peak memory consumption:' |
            tail -n 2 > "$probe_dir/result.txt"
        log "PROBE PASS cell=$cell mode=$cell_mode device_batch=$device_batch ga=$ga $(tr '\n' ' ' < "$probe_dir/result.txt")"
        return 0
    fi
    if is_oom "$output"; then
        log "PROBE OOM cell=$cell mode=$cell_mode device_batch=$device_batch ga=$ga"
        return 42
    fi
    log "PROBE FAIL cell=$cell mode=$cell_mode device_batch=$device_batch exit=$rc"
    return 1
}

select_common_batch() {
    local candidate dense_rc arc_rc
    for candidate in 128 64 32 16; do
        preflight "$dense_config" "$arc_config" || return 78
        run_probe dense_adamw_paper_recipe_gpt60m-s42 dense "$repo_dir/train.py" "$dense_config" "$candidate"
        dense_rc=$?
        [[ "$dense_rc" == 0 || "$dense_rc" == 42 ]] || return "$dense_rc"
        run_probe arc_ef21m_all2d_adamw_paper_recipe_gpt60m-s42 arc "$repo_dir/train_arctopk.py" "$arc_config" "$candidate"
        arc_rc=$?
        [[ "$arc_rc" == 0 || "$arc_rc" == 42 ]] || return "$arc_rc"
        if [[ "$dense_rc" == 0 && "$arc_rc" == 0 ]]; then
            SELECTED_DEVICE_BATCH="$candidate"
            SELECTED_GA=$((global_batch / (world_size * candidate)))
            printf '%s\n' "$candidate" > "$artifact_root/selected_device_batch.txt"
            printf '%s\n' "$SELECTED_GA" > "$artifact_root/selected_ga.txt"
            log "SELECTED common_device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA effective_local_batch=128"
            return 0
        fi
    done
    log "BLOCKED no common device batch; OOM through device_batch=16/GA8"
    return 42
}

run_cell() {
    local cell="$1" cell_mode="$2" entry="$3" config="$4"
    local cell_dir="$artifact_root/$cell" output rc
    preflight "$config" || return 78
    [[ ! -e "$cell_dir" ]] || {
        log "BLOCKED refusing to overwrite cell=$cell"
        return 73
    }
    mkdir -p "$cell_dir"
    output="$cell_dir/stdout.log"
    cp "$config" "$cell_dir/config.yaml"
    git rev-parse HEAD > "$cell_dir/git_commit.txt"
    git status --short > "$cell_dir/git_status.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    {
        printf 'experiment_id=%s\ncell=%s\nmode=%s\n' "$experiment_id" "$cell" "$cell_mode"
        printf 'model=gpt60m\nshared_gpu=true\nexternal_gpu_processes_expected=true\n'
        printf 'gpu_list=%s\nworld_size=%s\nsequence_length=%s\n' \
            "$gpu_list" "$world_size" "$sequence_length"
        printf 'global_batch=%s\neffective_local_batch=128\ndevice_batch=%s\nga=%s\n' \
            "$global_batch" "$SELECTED_DEVICE_BATCH" "$SELECTED_GA"
        printf 'num_iterations=%s\ntraining_tokens=%s\ntraining_seed=%s\n' \
            "$num_iterations" "$training_tokens" "$training_seed"
        printf 'git_head=%s\n' "$(git rev-parse HEAD)"
        if [[ "$cell_mode" == arc ]]; then
            printf 'hook_arc_scope=all_ndim_2_parameters\nbucket_cap_mb=%s\n' "$bucket_cap_mb"
        fi
    } > "$cell_dir/environment.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$entry" --config "$config" --data_dir "$data_dir"
        --device_batch_size "$SELECTED_DEVICE_BATCH" --batch_size "$global_batch"
        --training-seed "$training_seed" --wandb_job_name "$experiment_id-$cell")
    if [[ "$cell_mode" == arc ]]; then
        command+=(--bucket-cap-mb "$bucket_cap_mb")
    fi
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL START cell=$cell mode=$cell_mode device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" "${command[@]}" \
        > "$output" 2> "$cell_dir/stderr.log"
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
        log "CELL INVALID cell=$cell missing final validation or memory metric"
        return 3
    fi
    log "CELL PASS cell=$cell $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

summarize() {
    "$python_bin" - "$artifact_root" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
plan = json.loads((root / "plan.json").read_text())
summary = {"experiment_id": plan["experiment_id"], "cells": {}}
for cell in plan["cells"]:
    result = (root / cell / "result.txt").read_text()
    final_step = plan["num_iterations"]
    final_lines = re.findall(
        rf"step:{final_step}/{final_step} val_loss:[^\r\n]*", result
    )
    final_line = final_lines[-1] if final_lines else ""
    loss = re.search(r"val_loss:([0-9.eE+-]+)", final_line)
    memory = re.search(r"Peak memory consumption: ([0-9]+) MiB", result)
    if not (loss and memory):
        raise SystemExit(f"failed to parse final validation or memory metric for {cell}")
    timing = re.search(r"step_avg:([^\s]+)ms\b", final_line)
    try:
        step_avg_ms = float(timing.group(1)) if timing else float("nan")
    except ValueError:
        step_avg_ms = float("nan")
    if not math.isfinite(step_avg_ms) or step_avg_ms <= 0:
        raise SystemExit(f"invalid final step_avg for {cell}: expected finite positive milliseconds")
    final_loss = float(loss.group(1))
    summary["cells"][cell] = {
        "final_validation_loss": final_loss,
        "perplexity": math.exp(final_loss),
        "peak_memory_mib": int(memory.group(1)),
        "step_avg_ms": step_avg_ms,
        "tokens_per_second": plan["global_batch"] * plan["sequence_length"] * 1000 / step_avg_ms,
    }
print(json.dumps(summary, indent=2, allow_nan=False))
PY
}

# Read-only result inspection bypasses controller artifacts, GPUs, data and W&B.
if [[ "$mode" == summarize ]]; then
    summarize
    exit $?
fi

cd "$repo_dir" || exit 2
[[ -e "$artifact_root" || -L "$artifact_root" ]] && {
    printf 'refusing to overwrite existing controller artifact root: %s\n' "$artifact_root" >&2
    exit 73
}
mkdir -p "$artifact_root"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$artifact_root/controller_started_at.txt"
print_plan > "$artifact_root/plan.json"
cp "$0" "$artifact_root/controller.sh"
git rev-parse HEAD > "$artifact_root/git_commit.txt"
git status --short > "$artifact_root/git_status.txt"
log "BEGIN $experiment_id shared_gpu=true gpu_list=$gpu_list"

select_common_batch || exit $?
run_cell dense_adamw_paper_recipe_gpt60m-s42 dense "$repo_dir/train.py" "$dense_config" || exit $?
run_cell arc_ef21m_all2d_adamw_paper_recipe_gpt60m-s42 arc "$repo_dir/train_arctopk.py" "$arc_config" || exit $?

summarize > "$artifact_root/summary.json" || exit $?
capture_gpu_state "$artifact_root/nvidia_smi_end.txt"
log "COMPLETE $experiment_id summary=$artifact_root/summary.json"
