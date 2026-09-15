#!/usr/bin/env bash
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
generic_runner="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
derive_caps="$repo_dir/benchmark/compressed_muon/derive_role_aligned_bucket_caps.py"
config="$repo_dir/configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml"
data_dir="$repo_dir/data/fineweb10B"
artifact_root="$repo_dir/artifacts/compressed_muon/CM078-m002-gpt130m-bf16-dense-aux-full-isolation-timing-profile-ws4-s42"
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
mkdir -p "$artifact_root"
timestamp > "$artifact_root/started_at.txt"

for required in "$python_bin" "$torchrun_bin" "$generic_runner" "$derive_caps" "$config" "$data_dir"; do
    [[ -e "$required" ]] || { log "BLOCKED missing=$required"; exit 66; }
done

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
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

calibration_root="$artifact_root/calibration"
layout_dir="$calibration_root/layout"
mkdir -p "$layout_dir"
calibration_command=(env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE
    "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list"
    "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
    "$repo_dir/train_greedylore.py"
    --config "$config" --data_dir "$data_dir" --no_wandb --use_polar_express
    --model_dim 768 --n_layer 8 --n_head 12 --model_dtype bfloat16
    --sequence_length 256 --batch_size 512 --device_batch_size 128
    --val_tokens 131072 --num_iterations 3 --training-seed 42
    --timing-warmup-steps 2 --bucket-cap-mb 80
    --greedy_lore_rank 32 --greedy_lore_update_interval 200
    --greedy_lore_start_compress_step 20 --greedy_lore_basis_sync local_svd
    --greedy_lore_dense_aux_communication_dtype bucket
    --greedy_lore_isolate_dense_aux_buckets
    --greedy-lore-bucket-layout-output-dir "$layout_dir"
    --greedy-lore-bucket-layout-capture-step 2)
printf '%q ' "${calibration_command[@]}" > "$calibration_root/command.txt"
printf '\n' >> "$calibration_root/command.txt"
env | sort > "$calibration_root/environment.txt"
timestamp > "$calibration_root/started_at.txt"
log "CALIBRATION_START"
timeout --signal=TERM --kill-after=60 1800 "${calibration_command[@]}" \
    > "$calibration_root/stdout.log" 2> "$calibration_root/stderr.log"
calibration_rc=$?
printf '%s\n' "$calibration_rc" > "$calibration_root/exit_code.txt"
timestamp > "$calibration_root/finished_at.txt"
((calibration_rc == 0)) || { log "CALIBRATION_FAILED exit=$calibration_rc"; exit "$calibration_rc"; }

caps_json="$calibration_root/calibrated-caps.json"
caps_csv="$($python_bin "$derive_caps" "$layout_dir" --target-cap-mb 80 --output "$caps_json")" || {
    rc=$?; log "CAP_DERIVATION_FAILED exit=$rc"; exit "$rc"
}
CAPS_JSON="$caps_json" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["CAPS_JSON"]).read_text())
if payload.get("rank_count") != 4:
    raise SystemExit("calibration did not capture all four ranks")
if len(payload.get("expected_buckets", [])) != 4:
    raise SystemExit("calibration did not derive the expected four buckets")
roles = [bucket.get("role") for bucket in payload["expected_buckets"]]
if roles != ["dense_aux", "matrix", "matrix", "dense_aux"]:
    raise SystemExit(f"unexpected calibrated bucket roles: {roles}")
PY
rc=$?
((rc == 0)) || { log "CAP_VALIDATION_FAILED exit=$rc"; exit "$rc"; }
log "CALIBRATION_DONE caps_mb=$caps_csv"

common_args=(
    --world-size 4 --global-batch-size 512 --device-batch-size 128
    --model-dim 768 --layers 8 --heads 12 --sequence-length 256
    --model-dtype bfloat16 --bucket-cap-mb 80 --timing-warmup-steps 20
    --greedy-lore-rank 32 --greedy-lore-update-interval 200
    --greedy-lore-dense-aux-communication-dtype bucket
    --greedy-lore-calibrated-bucket-cap-mb-list "$caps_csv"
    --training-seed 42 --gpu-list "$gpu_list")

log "TIMING_START"
bash "$generic_runner" "${common_args[@]}" \
    --profile-modes none \
    --timing-modes dense,greedylore_local_svd_partial,greedylore_local_svd_full \
    --measured-full-periods 4 --repeats 3 \
    --artifact-root "$artifact_root/timing"
timing_rc=$?
printf '%s\n' "$timing_rc" > "$artifact_root/timing_exit_code.txt"
((timing_rc == 0)) || { log "TIMING_FAILED exit=$timing_rc"; exit "$timing_rc"; }
log "TIMING_DONE"

log "PROFILE_START refresh_step=220 ordinary_step=221"
bash "$generic_runner" "${common_args[@]}" \
    --profile-modes dense,greedylore_local_svd_partial,greedylore_local_svd_full \
    --timing-modes none --measured-full-periods 2 --profile-period 2 --repeats 1 \
    --artifact-root "$artifact_root/profile"
profile_rc=$?
printf '%s\n' "$profile_rc" > "$artifact_root/profile_exit_code.txt"
((profile_rc == 0)) || { log "PROFILE_FAILED exit=$profile_rc"; exit "$profile_rc"; }

timestamp > "$artifact_root/finished_at.txt"
log "EXPERIMENT_DONE"
