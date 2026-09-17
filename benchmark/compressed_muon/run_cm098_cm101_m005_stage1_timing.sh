#!/usr/bin/env bash
# Stage-one PowerSGD timing on representative historical GreedyLore geometries.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_id=CM098-CM101-m005-stage1-timing-controller
controller_root="$artifact_base/$controller_id"
dependency_root="$artifact_base/CM096-CM097-m005-optimized-rank32-main-controller"
config="$repo_dir/configs/compressed_muon/m005_power_sgd_timing.yaml"
entry="$repo_dir/train_powersgd.py"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
world_size=4
timing_warmup_steps=20
training_seed=42
maximum_idle_used_mib=1024
poll_seconds=300
cell_timeout="${CELL_TIMEOUT:-3h}"
gpu_list=""
status_log="$controller_root/status.log"

cm098_root="$artifact_base/CM098-m005-powersgd-bf16-scale-timing-ws4-s42"
cm099_root="$artifact_base/CM099-m005-powersgd-gpt350m-fp32-batch64-timing-ws4-s42"
cm100_root="$artifact_base/CM100-m005-powersgd-gpt350m-fp32-bucket-bridge-ws4-s42"
cm101_root="$artifact_base/CM101-m005-powersgd-gpt720m-fp32-batch-timing-ws4-s42"

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

wait_for_dependency() {
    while [[ ! -f "$dependency_root/controller_exit_code.txt" ]]; do
        log "DEPENDENCY_WAIT controller=$(basename "$dependency_root") retry_seconds=$poll_seconds"
        sleep "$poll_seconds"
    done
    log "DEPENDENCY_DONE controller=$(basename "$dependency_root") exit=$(tr -d '[:space:]' < "$dependency_root/controller_exit_code.txt")"
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

run_cell() {
    local root="$1" label="$2" dim="$3" layers="$4" heads="$5"
    local dtype="$6" global_batch="$7" device_batch="$8" bucket_mib="$9"
    local measured_updates="${10}"
    local num_iterations=$((timing_warmup_steps + measured_updates))
    local val_tokens=$((world_size * device_batch * 256))
    local cell_dir="$root/$label" rc

    wait_for_selected_gpus
    mkdir -p "$cell_dir"
    env | sort > "$cell_dir/environment.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry"
        --config "$config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim "$dim" --n_layer "$layers" --n_head "$heads"
        --model_dtype "$dtype" --sequence_length 256
        --batch_size "$global_batch" --device_batch_size "$device_batch"
        --val_tokens "$val_tokens" --num_iterations "$num_iterations"
        --timing-warmup-steps "$timing_warmup_steps" --bucket-cap-mb "$bucket_mib"
        --training-seed "$training_seed" --warmup_steps 0 --warmdown_ratio 0
        --lr_schedule linear --checkpoint_freq 0
        --power_sgd_rank 32 --power_sgd_start_compress_step 0
        --power_sgd_error_feedback ef14 --power_sgd_warm_start
        --power_sgd_seed "$training_seed")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL_START root=$(basename "$root") label=$label dim=$dim layers=$layers dtype=$dtype global_batch=$global_batch device_batch=$device_batch bucket_mib=$bucket_mib measured_updates=$measured_updates gpu_list=$gpu_list"
    timeout --signal=TERM --kill-after=60s "$cell_timeout" "${command[@]}" \
        > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    if ((rc != 0)); then
        log "CELL_FAILED root=$(basename "$root") label=$label exit=$rc continuing=true"
        return "$rc"
    fi
    tr '\r' '\n' < "$cell_dir/stdout.log" |
        rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$cell_dir/result.txt"
    log "CELL_DONE root=$(basename "$root") label=$label $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

for root in "$controller_root" "$cm098_root" "$cm099_root" "$cm100_root" "$cm101_root"; do
    if [[ -e "$root" || -L "$root" ]]; then
        printf 'refusing to overwrite existing artifact root: %s\n' "$root" >&2
        exit 73
    fi
done
mkdir -p "$controller_root" "$cm098_root" "$cm099_root" "$cm100_root" "$cm101_root"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$controller_root/started_at.txt"
cp "$0" "$controller_root/controller.sh"
git -C "$repo_dir" rev-parse HEAD > "$controller_root/git_head.txt"
capture_gpu_state "$controller_root/nvidia_smi_at_start.txt"

log "CONTROLLER_START cells=7 execution=serial dependency=CM096-CM097 idle_gate=true"
wait_for_dependency
wait_for_initial_gpus

# CM079/CM080 BF16 maximum-feasible-batch geometries.
run_cell "$cm098_root" gpt350m-bf16-batch72 1024 20 16 bfloat16 288 72 80 800 || true
run_cell "$cm098_root" gpt1b-bf16-batch24 1536 30 24 bfloat16 96 24 80 800 || true

# CM085/CM090 FP32 GPT-350M maximum-feasible-batch geometry.
run_cell "$cm099_root" gpt350m-fp32-batch64 1024 20 16 float32 256 64 80 800 || true

# CM087/CM091 FP32 GPT-350M low-batch bucket bridge.
run_cell "$cm100_root" gpt350m-fp32-batch8-bucket160 1024 20 16 float32 32 8 160 200 || true
run_cell "$cm100_root" gpt350m-fp32-batch8-bucket80 1024 20 16 float32 32 8 80 200 || true

# CM086/CM088/CM092 FP32 GPT-720M batch geometries.
run_cell "$cm101_root" gpt720m-fp32-batch12 1280 30 20 float32 48 12 80 800 || true
run_cell "$cm101_root" gpt720m-fp32-batch8 1280 30 20 float32 32 8 80 800 || true

log "CONTROLLER_DONE"
