#!/usr/bin/env bash
# Fill the nine independent-score sharded-SVD geometries missing from CM104.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_base="$repo_dir/artifacts/compressed_muon"
experiment_id=CM105-m002-sharded-svd-missing-cm104-timing-ws4-s42
controller_root="$artifact_base/$experiment_id"
config="$repo_dir/configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml"
entry="$repo_dir/train_greedylore.py"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
data_dir="$repo_dir/data/fineweb10B"
world_size=4
sequence_length=256
timing_warmup_steps=20
training_seed=42
maximum_idle_used_mib=1024
poll_seconds=300
cell_timeout="${CELL_TIMEOUT:-3h}"
gpu_list=""
status_log="$controller_root/status.log"
failed_cells=0

labels=(
    gpt60m-bf16-b128-bucket160
    gpt130m-bf16-b128-bucket160
    gpt350m-bf16-b72-bucket80
    gpt1b-bf16-b24-bucket80
    gpt60m-fp32-b128-bucket80
    gpt60m-fp32-b8-bucket80
    gpt130m-fp32-b128-bucket80
    gpt130m-fp32-b8-bucket80
    gpt720m-fp32-b12-bucket80
)
dims=(512 768 1024 1536 512 512 768 768 1280)
layers=(4 8 20 30 4 4 8 8 30)
heads=(8 12 16 24 8 8 12 12 20)
dtypes=(bfloat16 bfloat16 bfloat16 bfloat16 float32 float32 float32 float32 float32)
global_batches=(512 512 288 96 512 32 512 32 48)
device_batches=(128 128 72 24 128 8 128 8 12)
bucket_mib=(160 160 80 80 80 80 80 80 80)
measured_updates=(800 800 800 800 800 800 800 800 800)

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
    local eligible gpu
    local -a requested
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

run_cell() {
    local index="$1"
    local label="${labels[$index]}"
    local dim="${dims[$index]}" layer_count="${layers[$index]}" head_count="${heads[$index]}"
    local dtype="${dtypes[$index]}" global_batch="${global_batches[$index]}"
    local device_batch="${device_batches[$index]}" bucket="${bucket_mib[$index]}"
    local measured="${measured_updates[$index]}"
    local num_iterations=$((timing_warmup_steps + measured))
    local val_tokens=$((world_size * device_batch * sequence_length))
    local cell_dir="$controller_root/cells/$label" rc

    wait_for_selected_gpus
    mkdir -p "$cell_dir"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry"
        --config "$config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim "$dim" --n_layer "$layer_count" --n_head "$head_count"
        --model_dtype "$dtype" --sequence_length "$sequence_length"
        --batch_size "$global_batch" --device_batch_size "$device_batch"
        --val_tokens "$val_tokens" --num_iterations "$num_iterations"
        --timing-warmup-steps "$timing_warmup_steps" --bucket-cap-mb "$bucket"
        --training-seed "$training_seed"
        --greedy_lore_rank 32 --greedy_lore_update_interval 200
        --greedy_lore_start_compress_step "$timing_warmup_steps"
        --greedy_lore_dense_aux_communication_dtype bucket
        --greedy_lore_score_randomization independent
        --greedy_lore_basis_sync sharded_svd)
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL_START index=$((index + 1))/${#labels[@]} label=$label dtype=$dtype global_batch=$global_batch device_batch=$device_batch bucket_mib=$bucket measured_updates=$measured gpu_list=$gpu_list"
    timeout --signal=TERM --kill-after=60s "$cell_timeout" "${command[@]}" \
        > "$cell_dir/stdout.log" 2> "$cell_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    if ((rc != 0)); then
        log "CELL_FAILED index=$((index + 1))/${#labels[@]} label=$label exit=$rc continuing=true"
        return "$rc"
    fi
    tr '\r' '\n' < "$cell_dir/stdout.log" |
        rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$cell_dir/result.txt"
    log "CELL_DONE index=$((index + 1))/${#labels[@]} label=$label $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

for required in "$config" "$entry" "$torchrun_bin" "$data_dir"; do
    [[ -e "$required" ]] || { printf 'missing required path: %s\n' "$required" >&2; exit 66; }
done
[[ ! -e "$controller_root" && ! -L "$controller_root" ]] || {
    printf 'refusing to overwrite existing artifact root: %s\n' "$controller_root" >&2
    exit 73
}

mkdir -p "$controller_root/cells"
trap finish_controller EXIT
: > "$status_log"
printf '%s\n' "$$" > "$controller_root/controller_pid.txt"
timestamp > "$controller_root/started_at.txt"
cp "$0" "$controller_root/controller.sh"
git -C "$repo_dir" rev-parse HEAD > "$controller_root/git_head.txt"
git -C "$repo_dir" status --short > "$controller_root/git_status.txt"
git -C "$repo_dir" diff --binary > "$controller_root/code.patch"
capture_gpu_state "$controller_root/nvidia_smi_at_start.txt"

log "CONTROLLER_START cells=${#labels[@]} execution=serial idle_gate=true repeats=1"
wait_for_initial_gpus
for index in "${!labels[@]}"; do
    if run_cell "$index"; then
        printf '0\n' > "$controller_root/cells/${labels[$index]}/queue_exit_code.txt"
    else
        rc=$?
        printf '%s\n' "$rc" > "$controller_root/cells/${labels[$index]}/queue_exit_code.txt"
        failed_cells=$((failed_cells + 1))
    fi
done
log "CONTROLLER_DONE failed_cells=$failed_cells"
((failed_cells == 0))
