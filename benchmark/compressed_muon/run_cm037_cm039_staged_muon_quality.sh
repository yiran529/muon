#!/usr/bin/env bash
# Three ordered GPT-60M stages; each stage runs a dense/ARC pair.
set -uo pipefail

repo_dir=/home/wyr/dion
experiment_id=CM037-CM039-m001-staged-scalar-adamw-ef14-muon-gpt60m-ws4-s42
artifact_root="$repo_dir/artifacts/compressed_muon/$experiment_id"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
gpu_list="${GPUS:-4,5,6,7}"
world_size=4
global_batch=512
sequence_length=256
num_iterations=8393
training_tokens=1100087296
training_seed=42
bucket_cap_mb=160
minimum_free_mib=16384
probe_timeout="${PROBE_TIMEOUT:-30m}"
formal_timeout="${FORMAL_TIMEOUT:-12h}"
status_log="$artifact_root/status.log"
mode=run

stage_ids=(CM037 CM038 CM039)
dense_cells=(
    cm037a_dense_muon_scalar_adamw_gpt60m-s42
    cm038a_dense_muon_scalar_adamw_repeat_gpt60m-s42
    cm039a_dense_muon_scalar_adamw_warmup_cosine_clip_gpt60m-s42
)
arc_cells=(
    cm037b_ef21m_muon_scalar_adamw_gpt60m-s42
    cm038b_ef14_muon_scalar_adamw_gpt60m-s42
    cm039b_ef14_muon_scalar_adamw_warmup_cosine_clip_gpt60m-s42
)
dense_configs=(
    "$repo_dir/configs/compressed_muon/cm037a_dense_muon_scalar_adamw.yaml"
    "$repo_dir/configs/compressed_muon/cm038a_dense_muon_scalar_adamw_repeat.yaml"
    "$repo_dir/configs/compressed_muon/cm039a_dense_muon_scalar_adamw_warmup_cosine_clip.yaml"
)
arc_configs=(
    "$repo_dir/configs/compressed_muon/cm037b_ef21m_muon_scalar_adamw.yaml"
    "$repo_dir/configs/compressed_muon/cm038b_ef14_muon_scalar_adamw.yaml"
    "$repo_dir/configs/compressed_muon/cm039b_ef14_muon_scalar_adamw_warmup_cosine_clip.yaml"
)

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

stages = [
    {
        "id": "CM037",
        "cells": [
            {"name": "cm037a_dense_muon_scalar_adamw_gpt60m-s42", "mode": "dense"},
            {"name": "cm037b_ef21m_muon_scalar_adamw_gpt60m-s42", "mode": "arc", "error_feedback": "ef21m"},
        ],
        "schedule": {"warmup_steps": 0, "decay": "final_20_percent_linear", "grad_clip_norm": None},
    },
    {
        "id": "CM038",
        "cells": [
            {"name": "cm038a_dense_muon_scalar_adamw_repeat_gpt60m-s42", "mode": "dense"},
            {"name": "cm038b_ef14_muon_scalar_adamw_gpt60m-s42", "mode": "arc", "error_feedback": "ef14"},
        ],
        "schedule": {"warmup_steps": 0, "decay": "final_20_percent_linear", "grad_clip_norm": None},
    },
    {
        "id": "CM039",
        "cells": [
            {"name": "cm039a_dense_muon_scalar_adamw_warmup_cosine_clip_gpt60m-s42", "mode": "dense"},
            {"name": "cm039b_ef14_muon_scalar_adamw_warmup_cosine_clip_gpt60m-s42", "mode": "arc", "error_feedback": "ef14"},
        ],
        "schedule": {"warmup_steps": 1000, "decay": "cosine_to_zero", "grad_clip_norm": 1.0},
    },
]
print(json.dumps({
    "experiment_id": "CM037-CM039-m001-staged-scalar-adamw-ef14-muon-gpt60m-ws4-s42",
    "stages": stages,
    "cells": [cell["name"] for stage in stages for cell in stage["cells"]],
    "model": "gpt60m",
    "world_size": 4,
    "gpu_list": [4, 5, 6, 7],
    "shared_gpu": True,
    "global_batch": 512,
    "effective_local_batch": 128,
    "preferred_device_batch": 128,
    "oom_fallback_device_batches": [128, 64, 32, 16],
    "sequence_length": 256,
    "num_iterations": 8393,
    "training_tokens": 1_100_087_296,
    "validation_tokens": 10_485_760,
    "training_seed": 42,
    "muon": {"lr": 0.02, "momentum": 0.95, "weight_decay": 0.01, "adjust_lr": "spectral_norm"},
    "scalar_adamw": {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    "arc": {"scope": "all_ndim_2_parameters", "ratio": 0.2, "projection_rank": 4, "eta": 1.0, "start_step": 1000, "bucket_cap_mb": 160},
}, indent=2))
PY
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
summary = {
    "experiment_id": plan["experiment_id"],
    "cell_order": plan["cells"],
    "cells": {},
}
for cell in plan["cells"]:
    result = (root / cell / "result.txt").read_text()
    final_step = plan["num_iterations"]
    lines = re.findall(rf"step:{final_step}/{final_step} val_loss:[^\r\n]*", result)
    final_line = lines[-1] if lines else ""
    loss_match = re.search(r"val_loss:([0-9.eE+-]+)", final_line)
    timing_match = re.search(r"step_avg:([^\s]+)ms\b", final_line)
    memory_match = re.search(r"Peak memory consumption: ([0-9]+) MiB", result)
    if not (loss_match and timing_match and memory_match):
        raise SystemExit(f"failed to parse final validation, timing, or memory metric for {cell}")
    loss = float(loss_match.group(1))
    step_avg_ms = float(timing_match.group(1))
    if not math.isfinite(loss):
        raise SystemExit(f"invalid final validation loss for {cell}")
    if not math.isfinite(step_avg_ms) or step_avg_ms <= 0:
        raise SystemExit(f"invalid final step_avg for {cell}: expected finite positive milliseconds")
    summary["cells"][cell] = {
        "final_validation_loss": loss,
        "perplexity": math.exp(loss),
        "step_avg_ms": step_avg_ms,
        "tokens_per_second": plan["global_batch"] * plan["sequence_length"] * 1000 / step_avg_ms,
        "peak_memory_mib": int(memory_match.group(1)),
    }
print(json.dumps(summary, indent=2, allow_nan=False))
PY
}

# Read-only operations precede all artifact, GPU, data, and W&B actions.
if [[ "$mode" == print ]]; then
    print_plan
    exit 0
elif [[ "$mode" == summarize ]]; then
    summarize
    exit $?
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
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

preflight() {
    local gpu free config
    [[ "$gpu_list" == "4,5,6,7" ]] || {
        log "BLOCKED registered run requires GPU 4,5,6,7; got $gpu_list"
        return 1
    }
    for gpu in 4 5 6 7; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || {
            log "BLOCKED gpu=$gpu free_mib=$free required_mib=$minimum_free_mib"
            return 1
        }
    done
    for config in "$@"; do
        [[ -f "$config" ]] || { log "BLOCKED missing config=$config"; return 1; }
    done
    [[ -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing dataset or virtualenv executables"
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
    local stage="$1" cell="$2" cell_mode="$3" entry="$4" config="$5" device_batch="$6"
    local ga probe_dir probe_config output val_tokens rc
    ga=$((global_batch / (world_size * device_batch)))
    probe_dir="$artifact_root/probes/${stage}-${cell}-db${device_batch}"
    probe_config="$probe_dir/config.yaml"
    output="$probe_dir/stdout.log"
    val_tokens=$((device_batch * sequence_length * world_size))
    mkdir -p "$probe_dir"
    make_probe_config "$config" "$probe_config" "$val_tokens" "$cell_mode"
    capture_gpu_state "$probe_dir/nvidia_smi_before.txt"
    log "PROBE START stage=$stage cell=$cell device_batch=$device_batch ga=$ga"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$entry" --config "$probe_config" --data_dir "$data_dir"
        --device_batch_size "$device_batch" --batch_size "$global_batch"
        --training-seed "$training_seed" --no_wandb)
    [[ "$cell_mode" == arc ]] && command+=(--bucket-cap-mb "$bucket_cap_mb")
    timeout --signal=TERM --kill-after=60s "$probe_timeout" "${command[@]}" > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_after.txt"
    if [[ "$rc" == 0 ]] && rg -q 'Peak memory consumption:' "$output"; then
        log "PROBE PASS stage=$stage cell=$cell device_batch=$device_batch ga=$ga"
        return 0
    elif is_oom "$output"; then
        log "PROBE OOM stage=$stage cell=$cell device_batch=$device_batch ga=$ga"
        return 42
    fi
    log "PROBE FAIL stage=$stage cell=$cell exit=$rc"
    return 1
}

select_common_batch() {
    local index="$1" candidate dense_rc arc_rc stage
    stage="${stage_ids[$index]}"
    for candidate in 128 64 32 16; do
        preflight "${dense_configs[$index]}" "${arc_configs[$index]}" || return 78
        run_probe "$stage" "${dense_cells[$index]}" dense "$repo_dir/train.py" "${dense_configs[$index]}" "$candidate"
        dense_rc=$?
        [[ "$dense_rc" == 0 || "$dense_rc" == 42 ]] || return "$dense_rc"
        run_probe "$stage" "${arc_cells[$index]}" arc "$repo_dir/train_arctopk.py" "${arc_configs[$index]}" "$candidate"
        arc_rc=$?
        [[ "$arc_rc" == 0 || "$arc_rc" == 42 ]] || return "$arc_rc"
        if [[ "$dense_rc" == 0 && "$arc_rc" == 0 ]]; then
            SELECTED_DEVICE_BATCH="$candidate"
            SELECTED_GA=$((global_batch / (world_size * candidate)))
            printf '%s\n' "$candidate" > "$artifact_root/${stage}_selected_device_batch.txt"
            printf '%s\n' "$SELECTED_GA" > "$artifact_root/${stage}_selected_ga.txt"
            log "SELECTED stage=$stage common_device_batch=$candidate ga=$SELECTED_GA"
            return 0
        fi
    done
    log "BLOCKED stage=$stage no common device batch through 16"
    return 42
}

run_cell() {
    local stage="$1" cell="$2" cell_mode="$3" entry="$4" config="$5"
    local cell_dir="$artifact_root/$cell" output rc
    preflight "$config" || return 78
    [[ ! -e "$cell_dir" ]] || { log "BLOCKED refusing to overwrite cell=$cell"; return 73; }
    mkdir -p "$cell_dir"
    output="$cell_dir/stdout.log"
    cp "$config" "$cell_dir/config.yaml"
    git rev-parse HEAD > "$cell_dir/git_commit.txt"
    git status --short > "$cell_dir/git_status.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$entry" --config "$config" --data_dir "$data_dir"
        --device_batch_size "$SELECTED_DEVICE_BATCH" --batch_size "$global_batch"
        --training-seed "$training_seed" --wandb_job_name "$experiment_id-$cell")
    [[ "$cell_mode" == arc ]] && command+=(--bucket-cap-mb "$bucket_cap_mb")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    {
        printf 'experiment_id=%s\nstage=%s\ncell=%s\nmode=%s\n' "$experiment_id" "$stage" "$cell" "$cell_mode"
        printf 'gpu_list=%s\nworld_size=%s\ndevice_batch=%s\nga=%s\n' "$gpu_list" "$world_size" "$SELECTED_DEVICE_BATCH" "$SELECTED_GA"
        printf 'global_batch=%s\nsequence_length=%s\nnum_iterations=%s\ntraining_tokens=%s\ntraining_seed=%s\n' "$global_batch" "$sequence_length" "$num_iterations" "$training_tokens" "$training_seed"
    } > "$cell_dir/environment.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL START stage=$stage cell=$cell device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" "${command[@]}" > "$output" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    if [[ "$rc" != 0 ]]; then
        log "CELL FAIL stage=$stage cell=$cell exit=$rc"
        return "$rc"
    fi
    tr '\r' '\n' < "$output" | rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" | tail -n 2 > "$cell_dir/result.txt"
    if [[ "$(wc -l < "$cell_dir/result.txt")" != 2 ]]; then
        log "CELL INVALID stage=$stage cell=$cell missing final metrics"
        return 3
    fi
    log "CELL PASS stage=$stage cell=$cell $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

cd "$repo_dir" || exit 2
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
    printf 'refusing formal launch with tracked worktree changes\n' >&2
    exit 74
}
[[ -e "$artifact_root" || -L "$artifact_root" ]] && {
    printf 'refusing to overwrite existing artifact root: %s\n' "$artifact_root" >&2
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
log "BEGIN $experiment_id gpu_list=$gpu_list"

for index in 0 1 2; do
    stage="${stage_ids[$index]}"
    log "STAGE START stage=$stage"
    select_common_batch "$index" || exit $?
    run_cell "$stage" "${dense_cells[$index]}" dense "$repo_dir/train.py" "${dense_configs[$index]}" || exit $?
    run_cell "$stage" "${arc_cells[$index]}" arc "$repo_dir/train_arctopk.py" "${arc_configs[$index]}" || exit $?
    log "STAGE COMPLETE stage=$stage"
done

summarize > "$artifact_root/summary.json" || exit $?
capture_gpu_state "$artifact_root/nvidia_smi_end.txt"
log "COMPLETE $experiment_id summary=$artifact_root/summary.json"
