#!/usr/bin/env bash
# Rerun CM093/CM094 with optimized PowerSGD, serially on four genuinely idle GPUs.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_id=CM096-CM097-m005-optimized-rank32-main-controller
controller_dir="$artifact_base/$controller_id"
data_dir="$repo_dir/data/fineweb10B"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
entry="$repo_dir/train_powersgd.py"
world_size=4
training_seed=1234
maximum_idle_used_mib=1024
gpu_wait_seconds=300
formal_timeout="${FORMAL_TIMEOUT:-12h}"
status_log="$controller_dir/status.log"
gpu_list=""

ids=(
    CM096-m005-powersgd-optimized-gpt60m-bf16-rank32-ddp-ws4-s1234
    CM097-m005-powersgd-optimized-gpt130m-bf16-rank32-ddp-ws4-s1234
)
models=(gpt60m gpt130m)
final_steps=(10000 20000)
configs=(
    "$repo_dir/configs/compressed_muon/cm096_m005_power_sgd_muon_gpt60m_bf16_rank32_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm097_m005_power_sgd_muon_gpt130m_bf16_rank32_s1234.yaml"
)

timestamp() { date -Is; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

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
            printf '%s\n' "$gpu_list" > "$controller_dir/gpu_list.txt"
            log "GPU_SELECTED list=$gpu_list"
            return
        fi
        gpu_list=""
        log "GPU_WAIT required=$world_size maximum_idle_used_mib=$maximum_idle_used_mib no_compute_process=true retry_seconds=$gpu_wait_seconds"
        sleep "$gpu_wait_seconds"
    done
}

wait_for_selected_gpus() {
    while ! selected_gpus_idle; do
        log "GPU_WAIT_SELECTED list=$gpu_list maximum_idle_used_mib=$maximum_idle_used_mib no_compute_process=true retry_seconds=$gpu_wait_seconds"
        sleep "$gpu_wait_seconds"
    done
}

run_formal() {
    local index="$1"
    local id="${ids[$index]}" model="${models[$index]}"
    local final_step="${final_steps[$index]}" config="${configs[$index]}"
    local cell_dir="$artifact_base/$id" output checkpoint_dir rc
    output="$cell_dir/stdout.log"
    checkpoint_dir="$cell_dir/checkpoints"
    if [[ -e "$cell_dir" ]]; then
        log "BLOCKED refusing_to_overwrite=$cell_dir"
        return 73
    fi
    wait_for_selected_gpus
    mkdir -p "$cell_dir" "$checkpoint_dir"
    cp "$config" "$cell_dir/config.yaml"
    printf '%q ' env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$config" --data_dir "$data_dir" \
        --training-seed "$training_seed" --bucket-cap-mb 160 \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$id" \
        > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    {
        printf 'experiment_id=%s\nmodel=%s\nmethod=M005\n' "$id" "$model"
        printf 'dataset=fineweb10B\nworld_size=%s\ncuda_visible_devices=%s\n' "$world_size" "$gpu_list"
        printf 'model_dtype=bfloat16\nrank=32\ntraining_seed=%s\n' "$training_seed"
        printf 'git_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "FORMAL_START id=$id gpu_list=$gpu_list"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$config" --data_dir "$data_dir" \
        --training-seed "$training_seed" --bucket-cap-mb 160 \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$id" \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    ((rc == 0)) || { log "FORMAL_FAIL id=$id exit=$rc"; return "$rc"; }
    tr '\r' '\n' < "$output" |
        rg "step:${final_step}/${final_step} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$cell_dir/result.txt"
    log "FORMAL_DONE id=$id $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

if [[ -e "$controller_dir" ]]; then
    printf 'refusing to overwrite existing controller artifact: %s\n' "$controller_dir" >&2
    exit 73
fi
mkdir -p "$controller_dir"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$controller_dir/started_at.txt"
cp "$0" "$controller_dir/controller.sh"
git -C "$repo_dir" rev-parse HEAD > "$controller_dir/git_head.txt"
capture_gpu_state "$controller_dir/nvidia_smi_at_start.txt"

log "CONTROLLER_START cells=2 execution=serial idle_gate=true"
wait_for_initial_gpus
for index in "${!ids[@]}"; do
    if run_formal "$index"; then
        printf '0\n' > "$controller_dir/${ids[$index]}_queue_exit_code.txt"
    else
        rc=$?
        printf '%s\n' "$rc" > "$controller_dir/${ids[$index]}_queue_exit_code.txt"
        log "FORMAL_ALLOWED_FAILURE id=${ids[$index]} exit=$rc continuing_queue=true"
    fi
done
log "CONTROLLER_DONE"
