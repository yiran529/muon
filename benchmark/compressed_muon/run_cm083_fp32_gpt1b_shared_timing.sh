#!/usr/bin/env bash
# Find a common feasible FP32 sequence/batch pair, then run four timings serially.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
dense_config="$repo_dir/configs/compressed_muon/cm037a_dense_muon_scalar_adamw.yaml"
greedy_config="$repo_dir/configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml"
data_dir="$repo_dir/data/fineweb10B"
experiment_id="${CM_EXPERIMENT_ID:-CM083-m002-shared-score-gpt1b-fp32-max-batch-timing-ws4-s42}"
model_dim="${CM_MODEL_DIM:-1536}"
layers="${CM_LAYERS:-30}"
heads="${CM_HEADS:-24}"
sequence_lengths="${CM_SEQUENCE_LENGTHS:-256}"
batch_candidates="${CM_BATCH_CANDIDATES:-24 16 12 8 4 2 1}"
artifact_root="$repo_dir/artifacts/compressed_muon/$experiment_id"
world_size=4
minimum_free_mib=23500
poll_seconds=60
gpu_list=""

timestamp() { date --iso-8601=seconds; }
status_log="$artifact_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

if [[ -e "$artifact_root" || -L "$artifact_root" ]]; then
    printf 'refusing to overwrite existing artifact root: %s\n' "$artifact_root" >&2
    exit 73
fi
mkdir -p "$artifact_root/probes"
timestamp > "$artifact_root/started_at.txt"

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$artifact_root/controller_exit_code.txt"
    timestamp > "$artifact_root/finished_at.txt"
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

until select_free_gpus; do
    gpu_list=""
    log "GPU_WAIT required=$world_size minimum_free_mib=$minimum_free_mib"
    sleep "$poll_seconds"
done
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

probe_shared() {
    local sequence_length="$1"
    local device_batch="$2"
    local global_batch=$((world_size * device_batch))
    local probe_root="$artifact_root/probes/shared-seq${sequence_length}-db${device_batch}"
    local rc
    wait_selected_idle
    mkdir -p "$probe_root"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/train_greedylore.py"
        --config "$greedy_config" --data_dir "$data_dir" --no_wandb --use_polar_express
        --model_dim "$model_dim" --n_layer "$layers" --n_head "$heads" --model_dtype float32
        --sequence_length "$sequence_length" --batch_size "$global_batch" --device_batch_size "$device_batch"
        --val_tokens "$((global_batch * sequence_length))" --num_iterations 4 --training-seed 42
        --timing-warmup-steps 1 --bucket-cap-mb 80
        --greedy_lore_rank 32 --greedy_lore_update_interval 2
        --greedy_lore_start_compress_step 1 --greedy_lore_basis_sync local_svd
        --greedy_lore_dense_aux_communication_dtype bucket
        --greedy_lore_score_randomization shared)
    printf '%q ' "${command[@]}" > "$probe_root/command.txt"
    printf '\n' >> "$probe_root/command.txt"
    timestamp > "$probe_root/started_at.txt"
    log "PROBE_START mode=shared dtype=float32 sequence_length=$sequence_length device_batch=$device_batch"
    timeout --signal=TERM --kill-after=60 1800 "${command[@]}" \
        > "$probe_root/stdout.log" 2> "$probe_root/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$probe_root/exit_code.txt"
    timestamp > "$probe_root/finished_at.txt"
    if ((rc == 0)); then
        grep -q 'Model parameter dtype: float32' "$probe_root/stdout.log" || return 65
        grep -q 'GreedyLore dense auxiliary communication dtype: bucket' "$probe_root/stdout.log" || return 65
        grep -q 'GreedyLore score randomization: shared' "$probe_root/stdout.log" || return 65
        log "PROBE_PASS mode=shared dtype=float32 sequence_length=$sequence_length device_batch=$device_batch"
        return 0
    fi
    if grep -Eiq 'CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
        "$probe_root/stdout.log" "$probe_root/stderr.log"; then
        log "PROBE_OOM mode=shared dtype=float32 sequence_length=$sequence_length device_batch=$device_batch exit=$rc"
        return 1
    fi
    log "PROBE_FAILED mode=shared dtype=float32 sequence_length=$sequence_length device_batch=$device_batch exit=$rc"
    return "$rc"
}

selected_batch=0
selected_sequence_length=0
for sequence_length in $sequence_lengths; do
    for candidate in $batch_candidates; do
        probe_shared "$sequence_length" "$candidate"
        rc=$?
        if ((rc == 0)); then
            selected_sequence_length="$sequence_length"
            selected_batch="$candidate"
            break 2
        elif ((rc != 1)); then
            exit "$rc"
        fi
    done
done
printf '%s\n' "$selected_batch" > "$artifact_root/selected_device_batch.txt"
printf '%s\n' "$selected_sequence_length" > "$artifact_root/selected_sequence_length.txt"
if ((selected_batch == 0)); then
    log "EXPERIMENT_BLOCKED shared_oom sequence_lengths=$sequence_lengths batch_candidates=$batch_candidates"
    exit 0
fi

global_batch=$((world_size * selected_batch))
run_timing_cell() {
    local label="$1" mode="$2" interval="$3" periods="$4"
    local cell_root="$artifact_root/$label"
    local expected_score_randomization rc
    log "CELL_START label=$label mode=$mode dtype=float32 sequence_length=$selected_sequence_length device_batch=$selected_batch"
    "$launcher" \
        --world-size 4 --global-batch-size "$global_batch" --device-batch-size "$selected_batch" \
        --model-dim "$model_dim" --layers "$layers" --heads "$heads" \
        --model-dtype float32 --sequence-length "$selected_sequence_length" --bucket-cap-mb 80 \
        --timing-warmup-steps 20 --measured-full-periods "$periods" --repeats 1 \
        --profile-modes none --timing-modes "$mode" \
        --training-seed 42 --greedy-lore-rank 32 \
        --greedy-lore-update-interval "$interval" \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" --artifact-root "$cell_root"
    rc=$?
    printf '%s\n' "$rc" > "$artifact_root/${label}_exit_code.txt"
    ((rc == 0)) || { log "CELL_FAILED label=$label exit=$rc"; return "$rc"; }
    stdout_path="$(find "$cell_root" -mindepth 2 -maxdepth 2 -name stdout.log -print -quit)"
    grep -q 'Model parameter dtype: float32' "$stdout_path" || return 65
    if [[ "$mode" == greedylore_* ]]; then
        grep -q 'GreedyLore dense auxiliary communication dtype: bucket' "$stdout_path" || return 65
        expected_score_randomization=independent
        [[ "$mode" == "greedylore_local_svd_shared" ]] && expected_score_randomization=shared
        grep -q "GreedyLore score randomization: $expected_score_randomization" "$stdout_path" || return 65
    fi
    log "CELL_DONE label=$label dtype_audit=passed"
}

run_timing_cell dense dense 200 4 || exit $?
run_timing_cell independent-interval200 greedylore_local_svd 200 4 || exit $?
run_timing_cell shared-interval200 greedylore_local_svd_shared 200 4 || exit $?
run_timing_cell shared-interval800 greedylore_local_svd_shared 800 1 || exit $?

"$python_bin" "$repo_dir/benchmark/compressed_muon/summarize_cm083_fp32.py" \
    "$artifact_root" --model-dim "$model_dim" --layers "$layers" --heads "$heads" \
    --output "$artifact_root/summary.json" || exit $?
log "EXPERIMENT_DONE summary=$artifact_root/summary.json"
