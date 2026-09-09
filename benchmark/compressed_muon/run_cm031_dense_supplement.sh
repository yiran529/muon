#!/usr/bin/env bash
# Add matching dense targeted profiles to the existing CM031 artifact.
set -uo pipefail

repo_dir=/home/wyr/dion
artifact_root="$repo_dir/artifacts/compressed_muon/CM031-shared-gpu-hook-optimizer-profile"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
dense_60m_config="$repo_dir/configs/compressed_muon/cm027a_dense_muon_gpt60m_paperlike.yaml"
dense_130m_config="$repo_dir/configs/compressed_muon/cm028a_dense_muon_gpt130m_paperlike.yaml"
gpu_list=4,5,6,7
world_size=4
device_batch=128
global_batch=512
sequence_length=256
profile_step=20
num_iterations=22
bucket_cap_mb=160
training_seed=42
minimum_free_mib=16384
status_log="$artifact_root/dense_supplement_status.log"
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
    "experiment_id": "CM031-shared-gpu-hook-optimizer-profile",
    "existing_cells": [
        "arc_optimizer_gpt60m-r1",
        "arc_ddp_hook_gpt60m-r1",
        "arc_ddp_hook_gpt130m-r1",
        "arc_optimizer_gpt130m-r1",
    ],
    "supplement_cells": ["dense_gpt60m-r1", "dense_gpt130m-r1"],
    "world_size": 4,
    "shared_gpu": True,
    "external_gpu_processes_expected": True,
    "gpu_list": [4, 5, 6, 7],
    "device_batch": 128,
    "gradient_accumulation": 1,
    "global_batch": 512,
    "sequence_length": 256,
    "bucket_cap_mb": 160,
    "profile_step": 20,
    "profiles_per_cell": 1,
    "require_final_timing": True,
    "interpretation": "critical-path attribution only; not stable wall-clock evidence",
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
    printf '%s\n' "$rc" > "$artifact_root/dense_supplement_exit_code.txt"
    timestamp > "$artifact_root/dense_supplement_finished_at.txt"
}
trap finish_controller EXIT

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
    for gpu in 4 5 6 7; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || {
            log "BLOCKED shared_gpu=true gpu=$gpu free_mib=$free required_mib=$minimum_free_mib"
            return 1
        }
    done
}

verify_existing_cm031() {
    ROOT="$artifact_root" WS="$world_size" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
world_size = int(os.environ["WS"])
expected = [
    "arc_optimizer_gpt60m-r1",
    "arc_ddp_hook_gpt60m-r1",
    "arc_ddp_hook_gpt130m-r1",
    "arc_optimizer_gpt130m-r1",
]
plan = json.loads((root / "plan.json").read_text())
if plan.get("cells") != expected:
    raise SystemExit(f"unexpected existing CM031 cells: {plan.get('cells')}")
for cell in expected:
    cell_dir = root / cell
    if (cell_dir / "exit_code.txt").read_text().strip() != "0":
        raise SystemExit(f"existing cell is incomplete: {cell}")
    traces = list((cell_dir / "profiler").glob("rank-*.json"))
    if len(traces) != world_size:
        raise SystemExit(f"existing cell has {len(traces)} traces: {cell}")
PY
}

run_cell() {
    local cell="$1" model="$2" config="$3" model_dim="$4" layers="$5" heads="$6"
    local cell_dir="$artifact_root/$cell" rc trace_count
    shared_gpu_preflight || return 78
    if [[ -e "$cell_dir" ]]; then
        log "BLOCKED refusing to overwrite cell=$cell"
        return 73
    fi
    mkdir -p "$cell_dir/profiler"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    {
        printf 'shared_gpu=true\nexternal_gpu_processes_expected=true\n'
        printf 'cell=%s\nmodel=%s\nmode=dense\n' "$cell" "$model"
        printf 'gpu_list=%s\nworld_size=%s\ndevice_batch=%s\nga=1\n' \
            "$gpu_list" "$world_size" "$device_batch"
        printf 'global_batch=%s\nsequence_length=%s\nprofile_step=%s\n' \
            "$global_batch" "$sequence_length" "$profile_step"
        printf 'bucket_cap_mb=%s\ntraining_seed=%s\ngit_head=%s\n' \
            "$bucket_cap_mb" "$training_seed" "$(git rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/train.py" --config "$config" --data_dir "$data_dir"
        --model_dim "$model_dim" --n_layer "$layers" --n_head "$heads"
        --sequence_length "$sequence_length" --batch_size "$global_batch"
        --device_batch_size "$device_batch" --val_tokens 131072
        --num_iterations "$num_iterations" --timing-warmup-steps 20
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb"
        --no_wandb --use_polar_express
        --profile-output-dir "$cell_dir/profiler" --profile-step "$profile_step")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL START shared_gpu=true cell=$cell model=$model mode=dense"
    timeout --signal=TERM --kill-after=60s 1h "${command[@]}" \
        > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    ((rc == 0)) || {
        log "CELL FAIL shared_gpu=true cell=$cell exit=$rc"
        return "$rc"
    }
    trace_count="$(find "$cell_dir/profiler" -maxdepth 1 -name 'rank-*.json' | wc -l)"
    ((trace_count == world_size)) || {
        log "CELL FAIL shared_gpu=true cell=$cell traces=$trace_count"
        return 65
    }
    log "CELL PASS shared_gpu=true cell=$cell traces=$trace_count"
}

cd "$repo_dir" || exit 2
[[ -x "$python_bin" && -x "$torchrun_bin" && -d "$data_dir" ]] || exit 66
[[ -f "$dense_60m_config" && -f "$dense_130m_config" ]] || exit 66
[[ ! -e "$artifact_root/dense_supplement_started_at.txt" ]] || {
    printf 'refusing to overwrite prior CM031 dense supplement\n' >&2
    exit 73
}
verify_existing_cm031 || exit $?
[[ ! -e "$artifact_root/dense_gpt60m-r1" && ! -e "$artifact_root/dense_gpt130m-r1" ]] || exit 73

: > "$status_log"
timestamp > "$artifact_root/dense_supplement_started_at.txt"
print_plan > "$artifact_root/dense_supplement_plan.json"
cp "$artifact_root/plan.json" "$artifact_root/plan.before_dense_supplement.json"
cp "$artifact_root/summary.json" "$artifact_root/summary.before_dense_supplement.json"
capture_gpu_state "$artifact_root/nvidia_smi_dense_supplement_start.txt"

log "BEGIN CM031 dense supplement shared_gpu=true gpu_list=$gpu_list"
run_cell dense_gpt60m-r1 gpt60m "$dense_60m_config" 512 4 8 || exit $?
run_cell dense_gpt130m-r1 gpt130m "$dense_130m_config" 768 8 12 || exit $?

ROOT="$artifact_root" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
plan = json.loads((root / "plan.before_dense_supplement.json").read_text())
plan["cells"].extend(["dense_gpt60m-r1", "dense_gpt130m-r1"])
plan["dense_supplement"] = {
    "shared_gpu": True,
    "cells": ["dense_gpt60m-r1", "dense_gpt130m-r1"],
    "interpretation": "critical-path attribution only; not stable wall-clock evidence",
}
temporary = root / "plan.json.tmp"
temporary.write_text(json.dumps(plan, indent=2) + "\n")
temporary.replace(root / "plan.json")
PY

PYTHONPATH="$repo_dir" "$python_bin" \
    "$repo_dir/benchmark/compressed_muon/summarize_training_profiles.py" \
    "$artifact_root" --output "$artifact_root/summary.json" --require-plan || exit $?
capture_gpu_state "$artifact_root/nvidia_smi_dense_supplement_end.txt"
log "COMPLETE CM031 dense supplement shared_gpu=true summary=$artifact_root/summary.json"
