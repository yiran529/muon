#!/usr/bin/env bash
# Run one sharded-SVD timing cell for each requested historical geometry.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
controller_root="$repo_dir/artifacts/compressed_muon/CM089-CM092-sharded-svd-controller"
cm089_root="$repo_dir/artifacts/compressed_muon/CM089-m002-sharded-svd-cm078-geometry-ws4-s42"
cm090_root="$repo_dir/artifacts/compressed_muon/CM090-m002-sharded-svd-cm085-geometry-ws4-s42"
cm091_root="$repo_dir/artifacts/compressed_muon/CM091-m002-sharded-svd-cm087-geometry-ws4-s42"
cm092_root="$repo_dir/artifacts/compressed_muon/CM092-m002-sharded-svd-cm088-geometry-ws4-s42"
world_size=4
minimum_free_mib=23500

timestamp() { date --iso-8601=seconds; }
status_log="$controller_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

for root in "$controller_root" "$cm089_root" "$cm090_root" "$cm091_root" "$cm092_root"; do
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

gpu_list="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v minimum="$minimum_free_mib" '
        {gsub(/ /, "", $1); gsub(/ /, "", $2)}
        ($2 + 0) >= minimum {print $1}
    ' | sort -n | head -n "$world_size" | paste -sd, -
)"
if [[ -z "$gpu_list" || "$(awk -F, '{print NF}' <<<"$gpu_list")" -ne "$world_size" ]]; then
    log "BLOCKED insufficient_free_gpus required=$world_size minimum_free_mib=$minimum_free_mib"
    exit 78
fi
printf '%s\n' "$gpu_list" > "$controller_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

run_cell() {
    local root="$1" label="$2" dim="$3" layers="$4" heads="$5"
    local dtype="$6" global_batch="$7" device_batch="$8" bucket_mib="$9"
    local mode="${10}" interval="${11}" periods="${12}"
    shift 12
    local cell_root="$root/$label" rc
    log "CELL_START label=$label dim=$dim layers=$layers dtype=$dtype global_batch=$global_batch device_batch=$device_batch bucket_mib=$bucket_mib mode=$mode interval=$interval"
    bash "$launcher" \
        --world-size "$world_size" --global-batch-size "$global_batch" --device-batch-size "$device_batch" \
        --model-dim "$dim" --layers "$layers" --heads "$heads" \
        --model-dtype "$dtype" --sequence-length 256 --bucket-cap-mb "$bucket_mib" \
        --timing-warmup-steps 20 --measured-full-periods "$periods" --repeats 1 \
        --profile-modes none --timing-modes "$mode" \
        --training-seed 42 --greedy-lore-rank 32 \
        --greedy-lore-update-interval "$interval" \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" --artifact-root "$cell_root" "$@"
    rc=$?
    printf '%s\n' "$rc" > "$root/${label}_exit_code.txt"
    ((rc == 0)) || { log "CELL_FAILED label=$label exit=$rc"; return "$rc"; }
    log "CELL_DONE label=$label"
}

# CM078: calibrated full role isolation, BF16 GPT-130M, 20 warmup + 800 measured.
run_cell "$cm089_root" cm078-full 768 8 12 bfloat16 512 128 80 \
    greedylore_sharded_svd_full 200 4 \
    --greedy-lore-calibrated-bucket-cap-mb-list 73.6875,78.75,29.25,73.6875 || exit $?

# CM085: FP32 GPT-350M at the maximum feasible device batch.
run_cell "$cm090_root" cm085-independent200 1024 20 16 float32 256 64 80 \
    greedylore_sharded_svd 200 4 || exit $?

# CM087: the same GPT-350M batch with both historical bucket caps.
run_cell "$cm091_root" cm087-bucket160-independent200 1024 20 16 float32 32 8 160 \
    greedylore_sharded_svd 200 1 || exit $?
run_cell "$cm091_root" cm087-bucket80-independent200 1024 20 16 float32 32 8 80 \
    greedylore_sharded_svd 200 1 || exit $?

# CM088: FP32 GPT-720M independent/shared score and interval settings.
run_cell "$cm092_root" cm088-independent200 1280 30 20 float32 32 8 80 \
    greedylore_sharded_svd 200 4 || exit $?
run_cell "$cm092_root" cm088-shared200 1280 30 20 float32 32 8 80 \
    greedylore_sharded_svd_shared 200 4 || exit $?
run_cell "$cm092_root" cm088-shared800 1280 30 20 float32 32 8 80 \
    greedylore_sharded_svd_shared 800 1 || exit $?

log "EXPERIMENT_DONE"
