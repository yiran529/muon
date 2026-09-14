#!/usr/bin/env bash
# Wait for four GPUs, run all-2D GreedyLore timing, then a full GPT-60M train.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_id="CM074-CM075-m002-greedylore-both-controller"
controller_dir="$artifact_base/$controller_id"
timing_id="CM074-m002-greedylore-both-gpt130m-bf16-timing-ws4-s42"
timing_dir="$artifact_base/$timing_id"
training_id="CM075-m002-greedylore-both-gpt60m-bf16-ddp-ws4-s1234"
training_dir="$artifact_base/$training_id"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
training_config="$repo_dir/configs/compressed_muon/cm075_m002_greedylore_both_gpt60m_bf16_s1234.yaml"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
world_size=4
minimum_free_mib=18000
gpu_wait_seconds="${GPU_WAIT_SECONDS:-300}"
formal_timeout="${FORMAL_TIMEOUT:-12h}"
requested_gpu_list="${GPUS:-}"
gpu_list=""
status_log="$controller_dir/status.log"

timestamp() { date --iso-8601=seconds; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

print_plan() {
    "$python_bin" - <<'PY'
import json

print(json.dumps({
    "controller_id": "CM074-CM075-m002-greedylore-both-controller",
    "execution": "wait for four GPUs; CM074 timing; CM075 full training; fail-fast",
    "gpu_policy": {"selection": "any_four", "minimum_free_mib": 18000, "poll_seconds": 300},
    "CM074": {
        "model": "GPT-130M BF16", "world_size": 4, "sequence_length": 256,
        "global_batch_size": 512, "device_batch_size": 128, "bucket_cap_mb": 80,
        "rank": 32, "update_interval": 200, "timing_warmup_steps": 20,
        "measured_updates": 800, "repeats": 3,
        "modes": ["dense", "greedylore_both"], "training_seed": 42,
    },
    "CM075": {
        "model": "GPT-60M BF16", "world_size": 4, "sequence_length": 256,
        "global_batch_size": 512, "device_batch_size": 128, "bucket_cap_mb": 160,
        "updates": 10000, "training_tokens": 1310720000, "rank": 32,
        "update_interval": 200, "start_compress_step": 1000,
        "compress_embedding_lm_head": True, "training_seed": 1234,
        "wandb": True,
    },
}, indent=2))
PY
}

if [[ "${1:-}" == "--print-plan" && $# -eq 1 ]]; then
    print_plan
    exit 0
elif (($#)); then
    printf 'usage: %s [--print-plan]\n' "$0" >&2
    exit 64
fi

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_dir/controller_exit_code.txt"
    timestamp > "$controller_dir/finished_at.txt"
}

capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

preflight() {
    local required
    for required in "$launcher" "$training_config" "$python_bin" "$torchrun_bin" "$data_dir"; do
        [[ -e "$required" ]] || { log "BLOCKED missing=$required"; return 78; }
    done
    "$python_bin" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)' || {
        log "BLOCKED W&B authentication unavailable"
        return 78
    }
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
        if ((attempts == 0 || attempts % 12 == 0)); then
            log "GPU_WAIT required=4 minimum_free_mib=$minimum_free_mib"
        fi
        attempts=$((attempts + 1))
        sleep "$gpu_wait_seconds"
    done
    log "GPU_SELECTED list=$gpu_list"
}

run_timing() {
    log "TIMING_START id=$timing_id gpu_list=$gpu_list"
    "$launcher" \
        --world-size 4 \
        --global-batch-size 512 \
        --device-batch-size 128 \
        --model-dim 768 \
        --layers 8 \
        --heads 12 \
        --model-dtype bfloat16 \
        --sequence-length 256 \
        --bucket-cap-mb 80 \
        --timing-warmup-steps 20 \
        --measured-full-periods 4 \
        --repeats 3 \
        --profile-modes none \
        --timing-modes dense,greedylore_local_svd \
        --training-seed 42 \
        --greedy-lore-rank 32 \
        --greedy-lore-update-interval 200 \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --greedy-lore-compress-embedding-lm-head \
        --gpu-list "$gpu_list" \
        --artifact-root "$timing_dir"
    local rc=$?
    printf '%s\n' "$rc" > "$controller_dir/cm074_exit_code.txt"
    ((rc == 0)) || { log "TIMING_FAIL id=$timing_id exit=$rc"; return "$rc"; }
    log "TIMING_DONE id=$timing_id"
}

run_training() {
    local output="$training_dir/stdout.log" checkpoint_dir="$training_dir/checkpoints" rc
    if [[ -e "$training_dir" ]]; then
        log "BLOCKED refusing_to_overwrite=$training_dir"
        return 73
    fi
    mkdir -p "$training_dir" "$checkpoint_dir"
    cp "$training_config" "$training_dir/config.yaml"
    printf '%q ' env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" \
        "$repo_dir/train_greedylore.py" --config "$training_config" \
        --data_dir "$data_dir" --training-seed 1234 --bucket-cap-mb 160 \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$training_id" \
        > "$training_dir/command.txt"
    printf '\n' >> "$training_dir/command.txt"
    {
        printf 'experiment_id=%s\nmodel=gpt60m\nmode=greedylore_both\n' "$training_id"
        printf 'model_dtype=bfloat16\ndataset=fineweb10B\nworld_size=4\ncuda_visible_devices=%s\n' "$gpu_list"
        printf 'training_seed=1234\ngit_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
    } > "$training_dir/environment.txt"
    capture_gpu_state "$training_dir/nvidia_smi_before.txt"
    timestamp > "$training_dir/started_at.txt"
    log "TRAINING_START id=$training_id gpu_list=$gpu_list"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" \
        "$repo_dir/train_greedylore.py" --config "$training_config" \
        --data_dir "$data_dir" --training-seed 1234 --bucket-cap-mb 160 \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$training_id" \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$training_dir/exit_code.txt"
    timestamp > "$training_dir/finished_at.txt"
    capture_gpu_state "$training_dir/nvidia_smi_after.txt"
    printf '%s\n' "$rc" > "$controller_dir/cm075_exit_code.txt"
    ((rc == 0)) || { log "TRAINING_FAIL id=$training_id exit=$rc"; return "$rc"; }
    tr '\r' '\n' < "$output" | \
        rg 'step:10000/10000 val_loss:|Peak memory consumption:' | \
        tail -n 2 > "$training_dir/result.txt"
    [[ "$(wc -l < "$training_dir/result.txt")" == 2 ]] || {
        log "TRAINING_INVALID id=$training_id missing_final_metrics"
        return 65
    }
    log "TRAINING_DONE id=$training_id $(tr '\n' ' ' < "$training_dir/result.txt")"
}

for path in "$controller_dir" "$timing_dir" "$training_dir"; do
    [[ ! -e "$path" ]] || { printf 'refusing to overwrite existing artifact: %s\n' "$path" >&2; exit 73; }
done
mkdir -p "$controller_dir"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$controller_dir/started_at.txt"
cp "$0" "$controller_dir/controller.sh"
print_plan > "$controller_dir/plan.json"
git -C "$repo_dir" rev-parse HEAD > "$controller_dir/git_head.txt"
capture_gpu_state "$controller_dir/nvidia_smi_at_start.txt"

log "CONTROLLER_START timing_then_full_training"
preflight || exit $?
wait_for_gpus
run_timing || exit $?
wait_for_gpus
run_training || exit $?
log "CONTROLLER_DONE"
