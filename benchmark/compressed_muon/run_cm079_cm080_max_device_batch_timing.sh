#!/usr/bin/env bash
# Find the largest common BF16 device batch for dense/M002, then run paired timing.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
dense_config="$repo_dir/configs/compressed_muon/cm037a_dense_muon_scalar_adamw.yaml"
greedy_config="$repo_dir/configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml"
data_dir="$repo_dir/data/fineweb10B"
controller_root="$repo_dir/artifacts/compressed_muon/CM079-CM080-max-device-batch-timing-controller"
cm079_root="$repo_dir/artifacts/compressed_muon/CM079-m002-gpt350m-bf16-max-device-batch-timing-ws4-s42"
cm080_root="$repo_dir/artifacts/compressed_muon/CM080-m002-gpt1b-bf16-max-device-batch-timing-ws4-s42"
world_size=4
minimum_free_mib=23500
poll_seconds=60
gpu_list=""

timestamp() { date --iso-8601=seconds; }
status_log="$controller_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

for root in "$controller_root" "$cm079_root" "$cm080_root"; do
    if [[ -e "$root" || -L "$root" ]]; then
        printf 'refusing to overwrite existing artifact root: %s\n' "$root" >&2
        exit 73
    fi
done
mkdir -p "$controller_root"
timestamp > "$controller_root/started_at.txt"

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_root/controller_exit_code.txt"
    timestamp > "$controller_root/finished_at.txt"
}
trap finish_controller EXIT

select_free_gpus() {
    gpu_list="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
        awk -F, -v minimum="$minimum_free_mib" '
            {gsub(/ /, "", $1); gsub(/ /, "", $2)}
            ($2 + 0) >= minimum {print $1}
        ' | sort -n | head -n "$world_size" | paste -sd, -
    )"
    [[ -n "$gpu_list" && "$(awk -F, '{print NF}' <<<"$gpu_list")" -eq "$world_size" ]]
}

until select_free_gpus; do
    gpu_list=""
    log "GPU_WAIT required=$world_size minimum_free_mib=$minimum_free_mib poll_seconds=$poll_seconds"
    sleep "$poll_seconds"
done
printf '%s\n' "$gpu_list" > "$controller_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

selected_idle() {
    local gpu free
    IFS=',' read -ra ids <<<"$gpu_list"
    for gpu in "${ids[@]}"; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || return 1
    done
}

wait_selected_idle() {
    until selected_idle; do
        log "GPU_WAIT_SELECTED list=$gpu_list minimum_free_mib=$minimum_free_mib"
        sleep "$poll_seconds"
    done
}

# Return 0 for feasible, 1 for an expected CUDA OOM, and >1 for an unexpected failure.
probe_m002() {
    local experiment_root="$1" label="$2" dim="$3" layers="$4" heads="$5" device_batch="$6"
    local probe_dir="$experiment_root/probes/m002-db${device_batch}" global_batch rc
    global_batch=$((world_size * device_batch))
    wait_selected_idle
    mkdir -p "$probe_dir"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/train_greedylore.py"
        --config "$greedy_config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim "$dim" --n_layer "$layers" --n_head "$heads" --model_dtype bfloat16
        --sequence_length 256 --batch_size "$global_batch" --device_batch_size "$device_batch"
        --val_tokens "$((world_size * device_batch * 256))" --num_iterations 3 --training-seed 42
        --timing-warmup-steps 1 --bucket-cap-mb 80
        --greedy_lore_rank 32 --greedy_lore_update_interval 2
        --greedy_lore_start_compress_step 1 --greedy_lore_basis_sync local_svd
        --greedy_lore_dense_aux_communication_dtype bucket)
    printf '%q ' "${command[@]}" > "$probe_dir/command.txt"; printf '\n' >> "$probe_dir/command.txt"
    timestamp > "$probe_dir/started_at.txt"
    log "PROBE_START model=$label mode=m002 device_batch=$device_batch global_batch=$global_batch"
    timeout --signal=TERM --kill-after=60 1800 "${command[@]}" \
        > "$probe_dir/stdout.log" 2> "$probe_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    timestamp > "$probe_dir/finished_at.txt"
    if ((rc == 0)); then
        log "PROBE_PASS model=$label mode=m002 device_batch=$device_batch"
        return 0
    fi
    if grep -Eiq 'CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
        "$probe_dir/stdout.log" "$probe_dir/stderr.log"; then
        log "PROBE_OOM model=$label mode=m002 device_batch=$device_batch exit=$rc"
        return 1
    fi
    log "PROBE_FAILED model=$label mode=m002 device_batch=$device_batch exit=$rc"
    return "$rc"
}

probe_dense_selected() {
    local experiment_root="$1" label="$2" dim="$3" layers="$4" heads="$5" device_batch="$6"
    local probe_dir="$experiment_root/probes/dense-db${device_batch}" global_batch rc
    global_batch=$((world_size * device_batch))
    wait_selected_idle
    mkdir -p "$probe_dir"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/train.py"
        --config "$dense_config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim "$dim" --n_layer "$layers" --n_head "$heads" --model_dtype bfloat16
        --sequence_length 256 --batch_size "$global_batch" --device_batch_size "$device_batch"
        --val_tokens "$((world_size * device_batch * 256))" --num_iterations 3 --training-seed 42
        --timing-warmup-steps 1 --bucket-cap-mb 80)
    printf '%q ' "${command[@]}" > "$probe_dir/command.txt"; printf '\n' >> "$probe_dir/command.txt"
    timestamp > "$probe_dir/started_at.txt"
    log "PROBE_START model=$label mode=dense device_batch=$device_batch global_batch=$global_batch"
    timeout --signal=TERM --kill-after=60 1800 "${command[@]}" \
        > "$probe_dir/stdout.log" 2> "$probe_dir/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    timestamp > "$probe_dir/finished_at.txt"
    ((rc == 0)) || { log "PROBE_FAILED model=$label mode=dense device_batch=$device_batch exit=$rc"; return "$rc"; }
    log "PROBE_PASS model=$label mode=dense device_batch=$device_batch"
}

find_max_batch() {
    local experiment_root="$1" label="$2" dim="$3" layers="$4" heads="$5" candidates_csv="$6"
    local candidate selected=0 rc
    local -a candidates
    mkdir -p "$experiment_root"
    printf '%s\n' "$gpu_list" > "$experiment_root/gpu_list.txt"
    timestamp > "$experiment_root/started_at.txt"
    IFS=',' read -ra candidates <<<"$candidates_csv"
    for candidate in "${candidates[@]}"; do
        probe_m002 "$experiment_root" "$label" "$dim" "$layers" "$heads" "$candidate"
        rc=$?
        if ((rc == 0)); then
            selected=$candidate
            break
        elif ((rc == 1)); then
            continue
        else
            return "$rc"
        fi
    done
    if ((selected == 0)); then
        printf '0\n' > "$experiment_root/selected_device_batch.txt"
        log "MODEL_BLOCKED model=$label reason=m002_oom_at_device_batch_1"
        return 0
    fi
    probe_dense_selected "$experiment_root" "$label" "$dim" "$layers" "$heads" "$selected" || return $?
    printf '%s\n' "$selected" > "$experiment_root/selected_device_batch.txt"
    printf '%s\n' "$candidates_csv" > "$experiment_root/device_batch_candidates.txt"
    log "BATCH_SELECTED model=$label device_batch=$selected candidates=$candidates_csv global_batch=$((world_size * selected)) ga=1"
}

run_timing() {
    local experiment_root="$1" label="$2" dim="$3" layers="$4" heads="$5" device_batch global_batch rc
    device_batch="$(<"$experiment_root/selected_device_batch.txt")"
    if ((device_batch == 0)); then
        printf 'blocked: M002 OOM at device batch 1\n' > "$experiment_root/timing_status.txt"
        timestamp > "$experiment_root/finished_at.txt"
        return 0
    fi
    global_batch=$((world_size * device_batch))
    log "TIMING_START model=$label device_batch=$device_batch global_batch=$global_batch ga=1"
    "$launcher" \
        --world-size 4 --global-batch-size "$global_batch" --device-batch-size "$device_batch" \
        --model-dim "$dim" --layers "$layers" --heads "$heads" \
        --model-dtype bfloat16 --sequence-length 256 --bucket-cap-mb 80 \
        --timing-warmup-steps 20 --measured-full-periods 4 --repeats 3 \
        --profile-modes none --timing-modes dense,greedylore_local_svd \
        --training-seed 42 --greedy-lore-rank 32 --greedy-lore-update-interval 200 \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" --artifact-root "$experiment_root/timing"
    rc=$?
    printf '%s\n' "$rc" > "$experiment_root/timing_exit_code.txt"
    timestamp > "$experiment_root/finished_at.txt"
    ((rc == 0)) || { log "TIMING_FAILED model=$label exit=$rc"; return "$rc"; }
    printf 'completed\n' > "$experiment_root/timing_status.txt"
    log "TIMING_DONE model=$label"
}

log "EXPERIMENT_START gpu_list=$gpu_list dtype=bfloat16 seq=256 bucket_mib=80"
find_max_batch "$cm079_root" gpt350m 1024 20 16 128,96,72,64,48,32,24,16,8,4,2,1 || exit $?
run_timing "$cm079_root" gpt350m 1024 20 16 || exit $?
find_max_batch "$cm080_root" gpt1b 1536 30 24 32,24,16,12,8,4,2,1 || exit $?
run_timing "$cm080_root" gpt1b 1536 30 24 || exit $?
log "EXPERIMENT_DONE"
