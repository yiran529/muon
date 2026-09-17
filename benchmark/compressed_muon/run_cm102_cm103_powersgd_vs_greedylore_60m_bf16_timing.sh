#!/usr/bin/env bash
# Paired optimized-PowerSGD vs sharded-SVD GreedyLore timing at two GPT-60M batches.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_id=CM102-CM103-powersgd-vs-greedylore-60m-bf16-controller
controller_root="$artifact_base/$controller_id"
cm102_root="$artifact_base/CM102-m005-vs-m002-gpt60m-bf16-batch128-timing-ws4-s42"
cm103_root="$artifact_base/CM103-m005-vs-m002-gpt60m-bf16-batch8-timing-ws4-s42"
power_config="$repo_dir/configs/compressed_muon/m005_power_sgd_timing.yaml"
power_entry="$repo_dir/train_powersgd.py"
greedy_launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
python_bin="$repo_dir/.venv/bin/python"
data_dir="$repo_dir/data/fineweb10B"
world_size=4
timing_warmup_steps=20
measured_updates=800
training_seed=42
maximum_idle_used_mib=1024
poll_seconds=300
cell_timeout="${CELL_TIMEOUT:-2h}"
gpu_list=""
status_log="$controller_root/status.log"

timestamp() { date --iso-8601=seconds; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_root/controller_exit_code.txt"
    timestamp > "$controller_root/finished_at.txt"
}

capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

eligible_gpus() {
    local busy_uuids
    busy_uuids="$(
        nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
        awk '/^GPU-/ {gsub(/ /, "", $0); print}' | paste -sd, -
    )"
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v maximum="$maximum_idle_used_mib" -v busy="$busy_uuids" '
        {for (i = 1; i <= 3; i++) gsub(/ /, "", $i)}
        ($3 + 0) < maximum && index("," busy ",", "," $2 ",") == 0 {print $1}
    ' | sort -n
}

select_idle_gpu_list() {
    eligible_gpus | head -n "$world_size" | paste -sd, -
}

selected_gpus_idle() {
    local eligible requested gpu
    eligible=",$(eligible_gpus | paste -sd, -),"
    IFS=',' read -r -a requested <<< "$gpu_list"
    ((${#requested[@]} == world_size)) || return 1
    for gpu in "${requested[@]}"; do
        [[ "$eligible" == *",$gpu,"* ]] || return 1
    done
}

wait_for_initial_gpus() {
    while true; do
        gpu_list="$(select_idle_gpu_list)"
        if [[ -n "$gpu_list" && "$(awk -F, '{print NF}' <<< "$gpu_list")" -eq "$world_size" ]]; then
            printf '%s\n' "$gpu_list" > "$controller_root/gpu_list.txt"
            log "GPU_SELECTED list=$gpu_list"
            return
        fi
        gpu_list=""
        log "GPU_WAIT required=$world_size maximum_idle_used_mib=$maximum_idle_used_mib no_compute_process=true retry_seconds=$poll_seconds"
        sleep "$poll_seconds"
    done
}

wait_for_selected_gpus() {
    while ! selected_gpus_idle; do
        log "GPU_WAIT_SELECTED list=$gpu_list maximum_idle_used_mib=$maximum_idle_used_mib no_compute_process=true retry_seconds=$poll_seconds"
        sleep "$poll_seconds"
    done
}

write_plan() {
    "$python_bin" - <<'PY'
import json

print(json.dumps({
    "experiments": {
        "CM102": {"device_batch_size": 128, "global_batch_size": 512},
        "CM103": {"device_batch_size": 8, "global_batch_size": 32},
    },
    "model": {"name": "GPT-60M", "dim": 512, "layers": 4, "heads": 8},
    "world_size": 4,
    "model_dtype": "bfloat16",
    "sequence_length": 256,
    "gradient_accumulation_steps": 1,
    "bucket_cap_mb": 80,
    "rank": 32,
    "error_feedback": "ef14",
    "training_seed": 42,
    "timing_warmup_steps": 20,
    "measured_updates": 800,
    "repeats": 3,
    "methods": {
        "powersgd": {"warm_start": True, "start_compress_step": 0},
        "greedylore": {
            "basis_sync": "sharded_svd",
            "score_randomization": "independent",
            "update_interval": 200,
            "dense_aux_communication_dtype": "bucket",
        },
    },
    "pairing_order": [
        ["powersgd", "greedylore"],
        ["greedylore", "powersgd"],
        ["powersgd", "greedylore"],
    ],
    "profiler": False,
    "wandb": False,
    "checkpoint": False,
}, indent=2))
PY
}

run_powersgd() {
    local root="$1" repeat="$2" global_batch="$3" device_batch="$4"
    local cell_dir="$root/powersgd-r$repeat" num_iterations val_tokens rc
    num_iterations=$((timing_warmup_steps + measured_updates))
    val_tokens=$((world_size * device_batch * 256))
    wait_for_selected_gpus
    mkdir -p "$cell_dir"
    env | sort > "$cell_dir/environment.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$power_entry"
        --config "$power_config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim 512 --n_layer 4 --n_head 8 --model_dtype bfloat16
        --sequence_length 256 --batch_size "$global_batch" --device_batch_size "$device_batch"
        --val_tokens "$val_tokens" --num_iterations "$num_iterations"
        --timing-warmup-steps "$timing_warmup_steps" --bucket-cap-mb 80
        --training-seed "$training_seed" --warmup_steps 0 --warmdown_ratio 0
        --lr_schedule linear --checkpoint_freq 0 --power_sgd_rank 32
        --power_sgd_start_compress_step 0 --power_sgd_error_feedback ef14
        --power_sgd_warm_start --power_sgd_seed "$training_seed")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL_START experiment=$(basename "$root") method=powersgd repeat=$repeat device_batch=$device_batch gpu_list=$gpu_list"
    timeout --signal=TERM --kill-after=60s "$cell_timeout" "${command[@]}" \
        > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    ((rc == 0)) || { log "CELL_FAILED method=powersgd repeat=$repeat exit=$rc"; return "$rc"; }
    tr '\r' '\n' < "$cell_dir/stdout.log" |
        rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$cell_dir/result.txt"
    log "CELL_DONE method=powersgd repeat=$repeat $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

run_greedylore() {
    local root="$1" repeat="$2" global_batch="$3" device_batch="$4"
    local cell_dir="$root/greedylore-r$repeat" rc
    wait_for_selected_gpus
    log "CELL_START experiment=$(basename "$root") method=greedylore_sharded_svd repeat=$repeat device_batch=$device_batch gpu_list=$gpu_list"
    timeout --signal=TERM --kill-after=60s "$cell_timeout" bash "$greedy_launcher" \
        --world-size "$world_size" --global-batch-size "$global_batch" --device-batch-size "$device_batch" \
        --model-dim 512 --layers 4 --heads 8 --model-dtype bfloat16 \
        --sequence-length 256 --bucket-cap-mb 80 --timing-warmup-steps "$timing_warmup_steps" \
        --measured-full-periods 4 --repeats 1 --profile-modes none \
        --timing-modes greedylore_sharded_svd --training-seed "$training_seed" \
        --greedy-lore-rank 32 --greedy-lore-update-interval 200 \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" --artifact-root "$cell_dir"
    rc=$?
    printf '%s\n' "$rc" > "$root/greedylore-r${repeat}-exit_code.txt"
    ((rc == 0)) || { log "CELL_FAILED method=greedylore_sharded_svd repeat=$repeat exit=$rc"; return "$rc"; }
    log "CELL_DONE method=greedylore_sharded_svd repeat=$repeat"
}

run_geometry() {
    local root="$1" global_batch="$2" device_batch="$3" repeat
    for repeat in 1 2 3; do
        if ((repeat % 2 == 1)); then
            run_powersgd "$root" "$repeat" "$global_batch" "$device_batch" || return $?
            run_greedylore "$root" "$repeat" "$global_batch" "$device_batch" || return $?
        else
            run_greedylore "$root" "$repeat" "$global_batch" "$device_batch" || return $?
            run_powersgd "$root" "$repeat" "$global_batch" "$device_batch" || return $?
        fi
    done
}

for required in "$power_config" "$power_entry" "$greedy_launcher" "$torchrun_bin" "$python_bin" "$data_dir"; do
    [[ -e "$required" ]] || { printf 'missing required path: %s\n' "$required" >&2; exit 66; }
done
for root in "$controller_root" "$cm102_root" "$cm103_root"; do
    [[ ! -e "$root" && ! -L "$root" ]] || { printf 'refusing to overwrite existing path: %s\n' "$root" >&2; exit 73; }
done

mkdir -p "$controller_root" "$cm102_root" "$cm103_root"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$controller_root/started_at.txt"
cp "$0" "$controller_root/controller.sh"
git -C "$repo_dir" rev-parse HEAD > "$controller_root/git_head.txt"
write_plan > "$controller_root/plan.json"
capture_gpu_state "$controller_root/nvidia_smi_at_start.txt"

log "CONTROLLER_START experiments=2 cells=12 execution=serial idle_gate=true"
wait_for_initial_gpus
run_geometry "$cm102_root" 512 128 || exit $?
run_geometry "$cm103_root" 32 8 || exit $?
log "CONTROLLER_DONE"
