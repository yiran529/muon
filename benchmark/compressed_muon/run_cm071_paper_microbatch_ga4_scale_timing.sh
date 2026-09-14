#!/usr/bin/env bash
# Paper-batch-semantics BF16 dense/GreedyLore timing at 60M, 130M, then best-effort 350M.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
artifact_root="$repo_dir/artifacts/compressed_muon/CM071-m002-paper-microbatch-ga4-bf16-scale-timing-ws4-s42"
gpu_list=""

while (($#)); do
    case "$1" in
        --artifact-root) artifact_root="$2"; shift 2 ;;
        --gpu-list) gpu_list="$2"; shift 2 ;;
        --print-plan) mode=print; shift ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done
mode="${mode:-run}"

print_plan() {
    "$repo_dir/.venv/bin/python" - <<'PY'
import json

print(json.dumps({
    "experiment_id": "CM071-m002-paper-microbatch-ga4-bf16-scale-timing-ws4-s42",
    "execution": "60M then 130M fail-fast; 350M best-effort",
    "model_dtype": "bfloat16",
    "world_size": 4,
    "global_batch_size": 512,
    "device_micro_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "effective_batch_size_per_device": 128,
    "sequence_length": 256,
    "bucket_cap_mb": 80,
    "timing_warmup_optimizer_steps": 20,
    "measured_optimizer_steps": 800,
    "measured_full_periods": 4,
    "repeats": 3,
    "training_seed": 42,
    "greedy_lore": {
        "rank": 32,
        "update_interval": 200,
        "basis_sync": "local_svd",
        "dense_aux_communication_dtype": "bucket",
    },
    "models": {
        "60m": {"dim": 512, "layers": 4, "heads": 8},
        "130m": {"dim": 768, "layers": 8, "heads": 12},
        "350m": {"dim": 1024, "layers": 20, "heads": 16},
    },
    "timing_modes": ["dense", "greedylore_local_svd"],
    "pairing_order": ["dense,greedylore", "greedylore,dense", "dense,greedylore"],
    "oom_policy": "350M failure is recorded and does not fail the controller",
}, indent=2))
PY
}

if [[ "$mode" == print ]]; then
    print_plan
    exit 0
fi

if [[ -e "$artifact_root" && -n "$(find "$artifact_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    printf 'refusing to overwrite existing artifact root: %s\n' "$artifact_root" >&2
    exit 73
fi
mkdir -p "$artifact_root"
status_log="$artifact_root/status.log"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$status_log"; }
finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$artifact_root/controller_exit_code.txt"
    date --iso-8601=seconds > "$artifact_root/finished_at.txt"
}
trap finish_controller EXIT
print_plan > "$artifact_root/plan.json"
git -C "$repo_dir" rev-parse HEAD > "$artifact_root/git_head.txt"

if [[ -z "$gpu_list" ]]; then
    gpu_list="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | "$launcher" --world-size 4 --exclude-gpus 0,1 --select-gpus-from-stdin)"
fi
[[ "$(awk -F, '{print NF}' <<<"$gpu_list")" -eq 4 ]] || {
    printf 'gpu-list must contain exactly four GPUs\n' >&2
    exit 64
}
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"

run_scale() {
    local label="$1" dim="$2" layers="$3" heads="$4" tolerate_failure="$5"
    local child="$artifact_root/$label" rc
    log "SCALE_START model=$label dim=$dim layers=$layers heads=$heads"
    "$launcher" \
        --world-size 4 \
        --global-batch-size 512 \
        --device-batch-size 32 \
        --model-dim "$dim" \
        --layers "$layers" \
        --heads "$heads" \
        --model-dtype bfloat16 \
        --sequence-length 256 \
        --bucket-cap-mb 80 \
        --timing-warmup-steps 20 \
        --measured-full-periods 4 \
        --repeats 3 \
        --profile-modes none \
        --timing-modes dense,greedylore_local_svd \
        --training-seed 42 \
        --greedy-lore-rank 32 \
        --greedy-lore-update-interval 200 \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" \
        --artifact-root "$child"
    rc=$?
    printf '%s\n' "$rc" > "$artifact_root/${label}_exit_code.txt"
    if ((rc != 0)); then
        if [[ "$tolerate_failure" == 1 ]]; then
            log "SCALE_ALLOWED_FAILURE model=$label exit=$rc"
            if [[ ! -e "$child/greedylore_local_svd-timing-r1/exit_code.txt" ]]; then
                log "SCALE_350M_GREEDYLORE_FALLBACK_START"
                "$launcher" \
                    --world-size 4 --global-batch-size 512 --device-batch-size 32 \
                    --model-dim "$dim" --layers "$layers" --heads "$heads" \
                    --model-dtype bfloat16 --sequence-length 256 --bucket-cap-mb 80 \
                    --timing-warmup-steps 20 --measured-full-periods 4 --repeats 1 \
                    --profile-modes none --timing-modes greedylore_local_svd \
                    --training-seed 42 --greedy-lore-rank 32 \
                    --greedy-lore-update-interval 200 \
                    --greedy-lore-dense-aux-communication-dtype bucket \
                    --gpu-list "$gpu_list" \
                    --artifact-root "$artifact_root/${label}-greedylore-fallback"
                printf '%s\n' "$?" > "$artifact_root/${label}_greedylore_fallback_exit_code.txt"
            fi
            return 0
        fi
        log "SCALE_FAILED model=$label exit=$rc"
        return "$rc"
    fi
    log "SCALE_DONE model=$label"
}

log "EXPERIMENT_START gpu_list=$gpu_list micro_batch_per_device=32 grad_accum=4 global_batch=512"
run_scale 60m 512 4 8 0 || exit $?
run_scale 130m 768 8 12 0 || exit $?
run_scale 350m 1024 20 16 1 || exit $?
log "EXPERIMENT_DONE root=$artifact_root"
