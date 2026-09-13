#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
artifact_root="$repo_dir/artifacts/compressed_muon/CM065-m002-gpt130m-bf16-dense-aux-bucket80-ddp-ws4-s42"
gpu_list=""

while (($#)); do
    case "$1" in
        --artifact-root) artifact_root="$2"; shift 2 ;;
        --gpu-list) gpu_list="$2"; shift 2 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

if [[ -e "$artifact_root" && -n "$(find "$artifact_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    printf 'refusing to overwrite existing artifact root: %s\n' "$artifact_root" >&2
    exit 73
fi
mkdir -p "$artifact_root"
status_log="$artifact_root/status.log"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$status_log"; }

if [[ -z "$gpu_list" ]]; then
    gpu_list="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | "$launcher" --world-size 4 --exclude-gpus 0,1,6,7 --select-gpus-from-stdin)"
fi
[[ "$(awk -F, '{print NF}' <<<"$gpu_list")" -eq 4 ]] || {
    printf 'gpu-list must contain exactly four GPUs\n' >&2
    exit 64
}
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"

run_cell() {
    local repeat="$1" dtype="$2" child
    child="$artifact_root/${dtype}-r${repeat}"
    log "CELL_START dtype=$dtype repeat=$repeat"
    "$launcher" \
        --world-size 4 \
        --global-batch-size 512 \
        --device-batch-size 128 \
        --model-dim 768 \
        --layers 8 \
        --heads 12 \
        --sequence-length 256 \
        --bucket-cap-mb 80 \
        --timing-warmup-steps 20 \
        --measured-full-periods 1 \
        --repeats 1 \
        --profile-modes none \
        --timing-modes greedylore_local_svd \
        --training-seed 42 \
        --greedy-lore-rank 32 \
        --greedy-lore-update-interval 200 \
        --greedy-lore-dense-aux-communication-dtype "$dtype" \
        --gpu-list "$gpu_list" \
        --artifact-root "$child"
    log "CELL_DONE dtype=$dtype repeat=$repeat"
}

run_cell 1 float32
run_cell 1 bfloat16
run_cell 2 bfloat16
run_cell 2 float32
run_cell 3 float32
run_cell 3 bfloat16

date --iso-8601=seconds > "$artifact_root/finished_at.txt"
log "EXPERIMENT_DONE root=$artifact_root"
