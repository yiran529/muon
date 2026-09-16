#!/usr/bin/env bash
# Current-code 350M bridge controls followed by a lower-batch 720M timing.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
controller_root="$repo_dir/artifacts/compressed_muon/CM087-CM088-batch-bucket-diagnostics-controller"
cm087_root="$repo_dir/artifacts/compressed_muon/CM087-m002-gpt350m-fp32-current-code-batch8-bucket-bridge-ws4-s42"
cm088_root="$repo_dir/artifacts/compressed_muon/CM088-m002-gpt720m-fp32-device-batch8-timing-ws4-s42"
world_size=4
minimum_free_mib=23500
poll_seconds=60
gpu_list=""

timestamp() { date --iso-8601=seconds; }
status_log="$controller_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

for root in "$controller_root" "$cm087_root" "$cm088_root"; do
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
printf '%s\n' "$gpu_list" > "$controller_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

run_cell() {
    local experiment_root="$1" label="$2" dim="$3" layers="$4" heads="$5"
    local device_batch="$6" bucket_mib="$7" mode="$8" interval="$9" periods="${10}"
    local global_batch=$((world_size * device_batch))
    local cell_root="$experiment_root/$label"
    local expected_score_randomization rc stdout_path
    wait_selected_idle
    log "CELL_START label=$label dim=$dim layers=$layers device_batch=$device_batch bucket_mib=$bucket_mib mode=$mode interval=$interval"
    "$launcher" \
        --world-size "$world_size" --global-batch-size "$global_batch" --device-batch-size "$device_batch" \
        --model-dim "$dim" --layers "$layers" --heads "$heads" \
        --model-dtype float32 --sequence-length 256 --bucket-cap-mb "$bucket_mib" \
        --timing-warmup-steps 20 --measured-full-periods "$periods" --repeats 1 \
        --profile-modes none --timing-modes "$mode" \
        --training-seed 42 --greedy-lore-rank 32 \
        --greedy-lore-update-interval "$interval" \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" --artifact-root "$cell_root"
    rc=$?
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

# CM087: exact historical geometry first, then change only bucket cap.
run_cell "$cm087_root" bucket160-dense 1024 20 16 8 160 dense 200 1 || exit $?
run_cell "$cm087_root" bucket160-independent200 1024 20 16 8 160 greedylore_local_svd 200 1 || exit $?
run_cell "$cm087_root" bucket80-independent200 1024 20 16 8 80 greedylore_local_svd 200 1 || exit $?
run_cell "$cm087_root" bucket80-dense 1024 20 16 8 80 dense 200 1 || exit $?

# CM088: change only device batch relative to CM086, retaining 800 measured updates.
run_cell "$cm088_root" dense 1280 30 20 8 80 dense 200 4 || exit $?
run_cell "$cm088_root" independent-interval200 1280 30 20 8 80 greedylore_local_svd 200 4 || exit $?
run_cell "$cm088_root" shared-interval200 1280 30 20 8 80 greedylore_local_svd_shared 200 4 || exit $?
run_cell "$cm088_root" shared-interval800 1280 30 20 8 80 greedylore_local_svd_shared 800 1 || exit $?

"$python_bin" "$repo_dir/benchmark/compressed_muon/summarize_cm087_cm088.py" \
    --cm087-root "$cm087_root" --cm088-root "$cm088_root" || exit $?
log "EXPERIMENT_DONE cm087_summary=$cm087_root/summary.json cm088_summary=$cm088_root/summary.json"
