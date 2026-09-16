#!/usr/bin/env bash
# Time optimized rank-32 PowerSGD on the CM089 GPT-130M training geometry.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_root="$repo_dir/artifacts/compressed_muon/CM095b-m005-powersgd-optimized-cm089-geometry-ws4-s42"
cell_name=powersgd-timing-r1
cell_dir="$artifact_root/$cell_name"
config="$repo_dir/configs/compressed_muon/cm095a_m005_power_sgd_optimized_timing.yaml"
entry="$repo_dir/train_powersgd.py"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
python_bin="$repo_dir/.venv/bin/python"
data_dir="$repo_dir/data/fineweb10B"
world_size=4
minimum_free_mib=18000
maximum_idle_used_mib=1024
gpu_wait_seconds=300

timestamp() { date --iso-8601=seconds; }
status_log="$artifact_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

if [[ -e "$artifact_root" || -L "$artifact_root" ]]; then
    printf 'refusing to overwrite existing artifact root: %s\n' "$artifact_root" >&2
    exit 73
fi
mkdir -p "$cell_dir"
timestamp > "$artifact_root/started_at.txt"
cp "$0" "$artifact_root/controller.sh"
git -C "$repo_dir" rev-parse HEAD > "$artifact_root/git_head.txt"

finish() {
    local rc=$?
    printf '%s\n' "$rc" > "$artifact_root/controller_exit_code.txt"
    timestamp > "$artifact_root/finished_at.txt"
}
trap finish EXIT

select_gpus() {
    local busy_uuids
    busy_uuids="$(
        nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
        awk '/^GPU-/ {gsub(/ /, "", $0); print}' | paste -sd, -
    )"
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.free \
        --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v minimum="$minimum_free_mib" -v maximum="$maximum_idle_used_mib" \
        -v busy="$busy_uuids" '
        {for (i = 1; i <= 4; i++) gsub(/ /, "", $i)}
        ($3 + 0) < maximum && ($4 + 0) >= minimum &&
            index("," busy ",", "," $2 ",") == 0 {print $1}
    ' | sort -n | head -n "$world_size" | paste -sd, -
}
while true; do
    gpu_list="$(select_gpus)"
    if [[ -n "$gpu_list" && "$(awk -F, '{print NF}' <<<"$gpu_list")" -eq "$world_size" ]]; then
        break
    fi
    log "GPU_WAIT required=$world_size maximum_idle_used_mib=$maximum_idle_used_mib no_compute_process=true retry_seconds=$gpu_wait_seconds"
    sleep "$gpu_wait_seconds"
done
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

ROOT="$artifact_root" CELL="$cell_name" "$python_bin" - <<'PY' > "$artifact_root/plan.json"
import json
import os

print(json.dumps({
    "experiment_id": "CM095b-m005-powersgd-optimized-cm089-geometry-ws4-s42",
    "world_size": 4,
    "global_batch_size": 512,
    "device_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "model": {"dim": 768, "layers": 8, "heads": 12},
    "sequence_length": 256,
    "model_dtype": "bfloat16",
    "bucket_cap_mb": 80,
    "timing_warmup_steps": 20,
    "timing_num_iterations": 820,
    "timing_cells": [os.environ["CELL"]],
    "power_sgd": {
        "rank": 32,
        "start_compress_step": 0,
        "error_feedback": "ef14",
        "warm_start": True,
    },
    "artifact_root": os.environ["ROOT"],
    "profile_scope": "profiler-off timing only",
    "reference_geometry": "CM089/CM078 without GreedyLore-specific role isolation",
}, indent=2))
PY

env | sort > "$cell_dir/environment.txt"
command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
    "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
    "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry"
    --config "$config" --data_dir "$data_dir" --no_wandb --use_polar_express
    --model_dim 768 --n_layer 8 --n_head 12 --model_dtype bfloat16
    --sequence_length 256 --batch_size 512 --device_batch_size 128
    --num_iterations 820 --timing-warmup-steps 20 --bucket-cap-mb 80
    --training-seed 42 --warmup_steps 0 --warmdown_ratio 0
    --lr_schedule linear --checkpoint_freq 0
    --wandb_job_name CM095b-m005-powersgd-optimized-cm089-geometry-ws4-s42
    --power_sgd_rank 32 --power_sgd_start_compress_step 0
    --power_sgd_error_feedback ef14 --power_sgd_warm_start
    --power_sgd_seed 42)
printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
printf '\n' >> "$cell_dir/command.txt"

timestamp > "$cell_dir/started_at.txt"
log "CELL_START cell=$cell_name geometry=CM089 rank=32 compressed_warmup=20 measured_updates=800"
timeout --signal=TERM --kill-after=60s 2h "${command[@]}" \
    > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
timestamp > "$cell_dir/finished_at.txt"
if ((rc != 0)); then
    log "CELL_FAILED cell=$cell_name exit=$rc"
    exit "$rc"
fi

PYTHONPATH="$repo_dir" "$python_bin" \
    "$repo_dir/benchmark/compressed_muon/summarize_training_profiles.py" \
    "$artifact_root" --output "$artifact_root/summary.json" --require-plan
log "EXPERIMENT_DONE summary=$artifact_root/summary.json"
