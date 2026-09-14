#!/usr/bin/env bash
# Coarse 130M bucket-cap sweep for bucket-native BF16 dense/GreedyLore timing.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
artifact_root="$repo_dir/artifacts/compressed_muon/CM068-m002-gpt130m-bf16-bucket-cap-sweep-ws4-s42"
poll_seconds=300
memory_limit_mib=1024
candidate_gpus="2,3,4,5,6"
bucket_caps=(24 32 40 48 64 80)

while (($#)); do
    case "$1" in
        --artifact-root) artifact_root="$2"; shift 2 ;;
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
finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$artifact_root/controller_exit_code.txt"
    date --iso-8601=seconds > "$artifact_root/finished_at.txt"
}
trap finish_controller EXIT

CAPS="${bucket_caps[*]}" ROOT="$artifact_root" POLL="$poll_seconds" "$repo_dir/.venv/bin/python" - <<'PY' > "$artifact_root/plan.json"
import json
import os

print(json.dumps({
    "experiment_id": "CM068-m002-gpt130m-bf16-bucket-cap-sweep-ws4-s42",
    "model": {"dim": 768, "layers": 8, "heads": 12},
    "model_dtype": "bfloat16",
    "world_size": 4,
    "global_batch_size": 512,
    "device_batch_size": 128,
    "sequence_length": 256,
    "bucket_caps_mib": [int(value) for value in os.environ["CAPS"].split()],
    "timing_warmup_steps": 20,
    "measured_updates": 200,
    "measured_full_periods": 1,
    "repeats": 1,
    "training_seed": 42,
    "greedy_lore": {
        "rank": 32,
        "update_interval": 200,
        "basis_sync": "local_svd",
        "dense_aux_communication_dtype": "bucket",
    },
    "timing_modes": ["dense", "greedylore_local_svd"],
    "pairing_order": "alternated by bucket cap",
    "profile_modes": [],
    "gpu_candidates": [2, 3, 4, 5, 6],
    "gpu_idle_memory_limit_mib": 1024,
    "gpu_poll_seconds": int(os.environ["POLL"]),
    "artifact_root": os.environ["ROOT"],
}, indent=2))
PY
git -C "$repo_dir" rev-parse HEAD > "$artifact_root/git_head.txt"

select_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk -F, -v candidates="$candidate_gpus" -v limit="$memory_limit_mib" '
            BEGIN {
                n = split(candidates, values, ",")
                for (i = 1; i <= n; i++) allowed[values[i]] = 1
            }
            {
                gsub(/ /, "", $1)
                gsub(/ /, "", $2)
            }
            ($1 in allowed) && ($2 + 0 < limit + 0) {print $1}
        ' | sort -n | head -n 4 | paste -sd, -
}

gpu_list=""
while true; do
    gpu_list="$(select_gpus)"
    count=0
    [[ -n "$gpu_list" ]] && count="$(awk -F, '{print NF}' <<<"$gpu_list")"
    if ((count == 4)); then
        break
    fi
    gpu_list=""
    log "GPU_WAIT idle_count=$count required=4 poll_seconds=$poll_seconds"
    sleep "$poll_seconds"
done
printf '%s\n' "$gpu_list" > "$artifact_root/gpu_list.txt"
log "GPU_SELECTED list=$gpu_list"

for index in "${!bucket_caps[@]}"; do
    cap="${bucket_caps[$index]}"
    modes="dense,greedylore_local_svd"
    if ((index % 2 == 1)); then
        modes="greedylore_local_svd,dense"
    fi
    child="$artifact_root/bucket${cap}"
    log "CAP_START bucket_cap_mib=$cap timing_modes=$modes"
    "$launcher" \
        --world-size 4 \
        --global-batch-size 512 \
        --device-batch-size 128 \
        --model-dim 768 \
        --layers 8 \
        --heads 12 \
        --model-dtype bfloat16 \
        --sequence-length 256 \
        --bucket-cap-mb "$cap" \
        --timing-warmup-steps 20 \
        --measured-full-periods 1 \
        --repeats 1 \
        --profile-modes none \
        --timing-modes "$modes" \
        --training-seed 42 \
        --greedy-lore-rank 32 \
        --greedy-lore-update-interval 200 \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" \
        --artifact-root "$child" || exit $?
    log "CAP_DONE bucket_cap_mib=$cap"
done

ROOT="$artifact_root" CAPS="${bucket_caps[*]}" "$repo_dir/.venv/bin/python" - <<'PY' > "$artifact_root/sweep-summary.json"
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
rows = []
for cap in (int(value) for value in os.environ["CAPS"].split()):
    data = json.loads((root / f"bucket{cap}" / "timing-summary.json").read_text())
    modes = data["modes"]
    dense = modes["dense"]["mean_step_ms"]
    greedy = modes["greedylore_local_svd"]["mean_step_ms"]
    rows.append({
        "bucket_cap_mib": cap,
        "dense_step_ms": dense,
        "greedylore_step_ms": greedy,
        "greedylore_minus_dense_ms": greedy - dense,
        "greedylore_relative_percent": (greedy / dense - 1.0) * 100.0,
        "dense_peak_allocated_mib": modes["dense"]["peak_allocated_mib"],
        "greedylore_peak_allocated_mib": modes["greedylore_local_svd"]["peak_allocated_mib"],
    })
print(json.dumps({"schema_version": 1, "rows": rows}, indent=2))
PY

log "EXPERIMENT_DONE root=$artifact_root"
