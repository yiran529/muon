#!/usr/bin/env bash
# Immutable retry for the two 130M Sparse-K quality cells interrupted by ENOSPC.
set -uo pipefail

repo_dir=/home/wyr/dion
experiment_id=CM040cd-r2-m003-m004-sparse-k-ef14-muon-ws4-s42
artifact_root="$repo_dir/artifacts/compressed_muon/$experiment_id"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
gpu_list="${GPUS:-4,5,6,7}"
world_size=4
global_batch=512
sequence_length=256
training_seed=42
bucket_cap_mb=160
minimum_free_gpu_mib=16384
minimum_free_disk_mib=16384
probe_timeout="${PROBE_TIMEOUT:-30m}"
formal_timeout="${FORMAL_TIMEOUT:-12h}"
status_log="$artifact_root/status.log"
mode=run

cells=(
    CM040d-r2-m004-topk-ef14-muon-gpt130m-ddp-ws4-s42
    CM040c-r2-m003-randk-ef14-muon-gpt130m-ddp-ws4-s42
)
configs=(
    "$repo_dir/configs/compressed_muon/cm040d_topk_ef14_muon_scalar_adamw_gpt130m.yaml"
    "$repo_dir/configs/compressed_muon/cm040c_randk_ef14_muon_scalar_adamw_gpt130m.yaml"
)
methods=(topk randk)
num_iterations=16785
training_tokens=2200043520

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
    GPU_LIST="$gpu_list" "$python_bin" - <<'PY'
import json
import os

print(json.dumps({
    "experiment_id": "CM040cd-r2-m003-m004-sparse-k-ef14-muon-ws4-s42",
    "cells": [
        "CM040d-r2-m004-topk-ef14-muon-gpt130m-ddp-ws4-s42",
        "CM040c-r2-m003-randk-ef14-muon-gpt130m-ddp-ws4-s42",
    ],
    "execution": "serial",
    "world_size": 4,
    "gpu_list": [int(value) for value in os.environ["GPU_LIST"].split(",")],
    "minimum_free_gpu_mib": 16384,
    "minimum_free_disk_mib": 16384,
    "global_batch": 512,
    "device_batch": 128,
    "sequence_length": 256,
    "num_iterations": 16785,
    "training_tokens": 2200043520,
    "training_seed": 42,
    "checkpointing": False,
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
summary = {"experiment_id": plan["experiment_id"], "cell_order": plan["cells"], "cells": {}}
for cell in plan["cells"]:
    result = (root / cell / "result.txt").read_text()
    final_lines = re.findall(r"step:16785/16785 val_loss:[^\r\n]*", result)
    final_line = final_lines[-1] if final_lines else ""
    loss_match = re.search(r"val_loss:([0-9.eE+-]+)", final_line)
    timing_match = re.search(r"step_avg:([^\s]+)ms\b", final_line)
    memory_match = re.search(r"Peak memory consumption: ([0-9]+) MiB", result)
    if not (loss_match and timing_match and memory_match):
        raise SystemExit(f"failed to parse final metrics for {cell}")
    loss = float(loss_match.group(1))
    step_avg_ms = float(timing_match.group(1))
    if not math.isfinite(loss) or not math.isfinite(step_avg_ms) or step_avg_ms <= 0:
        raise SystemExit(f"invalid final metrics for {cell}")
    summary["cells"][cell] = {
        "method": "topk" if "m004-topk" in cell else "randk",
        "final_validation_loss": loss,
        "perplexity": math.exp(loss),
        "step_avg_ms": step_avg_ms,
        "tokens_per_second": 512 * 256 * 1000 / step_avg_ms,
        "peak_memory_mib": int(memory_match.group(1)),
    }
print(json.dumps(summary, indent=2, allow_nan=False))
PY
}

if [[ "$mode" == print ]]; then print_plan; exit 0; fi
if [[ "$mode" == summarize ]]; then summarize; exit $?; fi

timestamp() { date -Is; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }
finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$artifact_root/controller_exit_code.txt"
    timestamp > "$artifact_root/controller_finished_at.txt"
}
capture_gpu_state() {
    { nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true; } > "$1"
}
check_disk() {
    local available
    available="$(df --output=avail -BM "$repo_dir" | tail -n 1 | tr -dc '0-9')"
    [[ "$available" =~ ^[0-9]+$ ]] && ((available >= minimum_free_disk_mib)) || {
        log "BLOCKED disk_free_mib=${available:-unknown} require=$minimum_free_disk_mib"
        return 1
    }
}

preflight() {
    local gpu free config
    [[ "$(awk -F, '{print NF}' <<< "$gpu_list")" == "$world_size" ]] || {
        printf 'expected %s GPU ids, got %s\n' "$world_size" "$gpu_list" >&2
        return 1
    }
    for gpu in ${gpu_list//,/ }; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_gpu_mib)) || {
            printf 'GPU %s has %s MiB free; require %s MiB\n' "$gpu" "$free" "$minimum_free_gpu_mib" >&2
            return 1
        }
    done
    for config in "${configs[@]}"; do
        [[ -f "$config" ]] || { printf 'missing config: %s\n' "$config" >&2; return 1; }
    done
    [[ -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        printf 'missing dataset or virtualenv executables\n' >&2
        return 1
    }
    "$python_bin" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)' || {
        printf 'W&B authentication unavailable\n' >&2
        return 1
    }
    [[ -z "$(git -C "$repo_dir" status --porcelain -- train_sparsek.py dion/sparse_k.py dion/sparse_k_layout.py dion/sparse_k_ddp_hook.py "${configs[@]}")" ]] || {
        printf 'refusing launch with uncommitted Sparse-K experiment inputs\n' >&2
        return 1
    }
    check_disk
}

make_probe_config() {
    cp "$1" "$2"
    sed -E -i \
        -e 's/^num_iterations:.*/num_iterations: 3/' \
        -e 's/^val_loss_every:.*/val_loss_every: 0/' \
        -e 's/^val_tokens:.*/val_tokens: 131072/' \
        -e 's/^no_wandb:.*/no_wandb: true/' \
        -e 's/^sparse_k_start_compress_step:.*/sparse_k_start_compress_step: 0/' \
        "$2"
}

run_probe() {
    local index="$1" cell="${cells[$1]}" probe_dir probe_config output rc
    probe_dir="$artifact_root/probes/$cell"
    probe_config="$probe_dir/config.yaml"
    output="$probe_dir/stdout.log"
    mkdir -p "$probe_dir"
    make_probe_config "${configs[$index]}" "$probe_config"
    log "PROBE START cell=$cell"
    timeout --signal=TERM --kill-after=60s "$probe_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone --nproc_per_node="$world_size" \
        "$repo_dir/train_sparsek.py" --config "$probe_config" --data_dir "$data_dir" \
        --device_batch_size 128 --batch_size "$global_batch" --training-seed "$training_seed" \
        --bucket-cap-mb "$bucket_cap_mb" --no_wandb > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    if [[ "$rc" == 0 ]] && rg -q 'Peak memory consumption:' "$output"; then
        log "PROBE PASS cell=$cell"
        return 0
    fi
    log "PROBE FAIL cell=$cell exit=$rc"
    return "$rc"
}

run_cell() {
    local index="$1" cell="${cells[$1]}" cell_dir output rc
    check_disk || return 75
    cell_dir="$artifact_root/$cell"
    output="$cell_dir/stdout.log"
    [[ ! -e "$cell_dir" ]] || { log "BLOCKED refusing to overwrite cell=$cell"; return 73; }
    mkdir -p "$cell_dir"
    cp "${configs[$index]}" "$cell_dir/config.yaml"
    git -C "$repo_dir" rev-parse HEAD > "$cell_dir/git_commit.txt"
    git -C "$repo_dir" status --short > "$cell_dir/git_status.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$repo_dir/train_sparsek.py"
        --config "${configs[$index]}" --data_dir "$data_dir" --device_batch_size 128
        --batch_size "$global_batch" --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb"
        --wandb_job_name "$cell")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    printf 'experiment_id=%s\ncell=%s\nmodel=gpt130m\nmethod=%s\ngpu_list=%s\nworld_size=%s\ndevice_batch=128\nga=1\nglobal_batch=%s\nsequence_length=%s\nnum_iterations=%s\ntraining_tokens=%s\ntraining_seed=%s\n' \
        "$experiment_id" "$cell" "${methods[$index]}" "$gpu_list" "$world_size" \
        "$global_batch" "$sequence_length" "$num_iterations" "$training_tokens" "$training_seed" \
        > "$cell_dir/environment.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL START cell=$cell"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" "${command[@]}" \
        > "$output" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    [[ "$rc" == 0 ]] || { log "CELL FAIL cell=$cell exit=$rc"; return "$rc"; }
    tr '\r' '\n' < "$output" | \
        rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" | \
        tail -n 2 > "$cell_dir/result.txt"
    [[ "$(wc -l < "$cell_dir/result.txt")" == 2 ]] || {
        log "CELL INVALID cell=$cell missing final metrics"
        return 3
    }
    log "CELL PASS cell=$cell $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

cd "$repo_dir" || exit 2
preflight || exit 78
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
capture_gpu_state "$artifact_root/nvidia_smi_start.txt"
log "BEGIN $experiment_id gpu_list=$gpu_list"
for index in 0 1; do run_probe "$index" || exit $?; done
for index in 0 1; do run_cell "$index" || exit $?; done
summarize > "$artifact_root/summary.json" || exit $?
capture_gpu_state "$artifact_root/nvidia_smi_end.txt"
log "COMPLETE $experiment_id summary=$artifact_root/summary.json"
