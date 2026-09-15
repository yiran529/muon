#!/usr/bin/env bash
# Run four single-cell GPT-1B timing diagnostics serially on one idle GPU set.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
dense_config="$repo_dir/configs/compressed_muon/cm037a_dense_muon_scalar_adamw.yaml"
artifact_root="$repo_dir/artifacts/compressed_muon/CM082-m002-shared-phase-gradient-sync-oracle-gpt1b-ws4-s42"
dense_summary="$repo_dir/artifacts/compressed_muon/CM080-m002-gpt1b-bf16-max-device-batch-timing-ws4-s42/timing/timing-summary.json"
shared_interval200_summary="$repo_dir/artifacts/compressed_muon/CM081-m002-shared-score-gpt1b-nccl-timing-ws4-s42/shared-only-timing/timing-summary.json"
world_size=4
minimum_free_mib=23500
poll_seconds=60
gpu_list=""
resume_oracle_only=0

timestamp() { date --iso-8601=seconds; }
status_log="$artifact_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

if [[ -e "$artifact_root" || -L "$artifact_root" ]]; then
    if [[ -d "$artifact_root" && -f "$artifact_root/shared-interval800/timing-summary.json" && ! -e "$artifact_root/no-grad-sync" ]]; then
        resume_oracle_only=1
    else
        printf 'refusing to overwrite or ambiguously resume artifact root: %s\n' "$artifact_root" >&2
        exit 73
    fi
else
    mkdir -p "$artifact_root"
    timestamp > "$artifact_root/started_at.txt"
fi

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

until select_free_gpus; do
    gpu_list=""
    log "GPU_WAIT required=$world_size minimum_free_mib=$minimum_free_mib"
    sleep "$poll_seconds"
done
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

run_launcher_cell() {
    local label="$1" mode="$2" interval="$3" periods="$4" rc
    local cell_root="$artifact_root/$label"
    log "CELL_START label=$label mode=$mode interval=$interval periods=$periods"
    "$launcher" \
        --world-size 4 --global-batch-size 96 --device-batch-size 24 \
        --model-dim 1536 --layers 30 --heads 24 \
        --model-dtype bfloat16 --sequence-length 256 --bucket-cap-mb 80 \
        --timing-warmup-steps 20 --measured-full-periods "$periods" --repeats 1 \
        --profile-modes none --timing-modes "$mode" \
        --training-seed 42 --greedy-lore-rank 32 \
        --greedy-lore-update-interval "$interval" \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" --artifact-root "$cell_root"
    rc=$?
    printf '%s\n' "$rc" > "$artifact_root/${label}_exit_code.txt"
    ((rc == 0)) || { log "CELL_FAILED label=$label exit=$rc"; return "$rc"; }
    log "CELL_DONE label=$label"
}

run_no_grad_sync_oracle() {
    local cell_root="$artifact_root/no-grad-sync" rc
    while ! selected_idle; do
        log "GPU_WAIT_SELECTED list=$gpu_list minimum_free_mib=$minimum_free_mib"
        sleep "$poll_seconds"
    done
    mkdir -p "$cell_root"
    local -a command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/benchmark/compressed_muon/train_no_grad_sync_oracle.py"
        --config "$dense_config" --data_dir "$repo_dir/data/fineweb10B"
        --no_wandb --use_polar_express
        --model_dim 1536 --n_layer 30 --n_head 24 --model_dtype bfloat16
        --sequence_length 256 --batch_size 96 --device_batch_size 24
        --val_tokens 24576 --num_iterations 820 --training-seed 42
        --timing-warmup-steps 20 --bucket-cap-mb 80)
    printf '%q ' "${command[@]}" > "$cell_root/command.txt"
    printf '\n' >> "$cell_root/command.txt"
    env | sort > "$cell_root/environment.txt"
    timestamp > "$cell_root/started_at.txt"
    log "CELL_START label=no-grad-sync mode=timing-oracle"
    timeout --signal=TERM --kill-after=60 3600 "${command[@]}" \
        > "$cell_root/stdout.log" 2> "$cell_root/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$cell_root/exit_code.txt"
    timestamp > "$cell_root/finished_at.txt"
    ((rc == 0)) || { log "CELL_FAILED label=no-grad-sync exit=$rc"; return "$rc"; }
    log "CELL_DONE label=no-grad-sync"
}

if ((!resume_oracle_only)); then
    run_launcher_cell shared-interval800 greedylore_local_svd_shared 800 1 || exit $?
else
    log "RESUME_ORACLE_ONLY shared_interval800=preserved"
fi
run_no_grad_sync_oracle || exit $?

"$python_bin" "$repo_dir/benchmark/compressed_muon/summarize_cm082_phase_oracle.py" \
    "$artifact_root" \
    --dense-summary "$dense_summary" \
    --shared-interval200-summary "$shared_interval200_summary" \
    --output "$artifact_root/summary.json" || exit $?
log "EXPERIMENT_DONE summary=$artifact_root/summary.json"
