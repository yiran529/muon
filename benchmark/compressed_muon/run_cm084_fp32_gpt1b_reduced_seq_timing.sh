#!/usr/bin/env bash
# Retry the CM083 FP32 comparison at reduced sequence lengths.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CM_EXPERIMENT_ID=CM084-m002-shared-score-gpt1b-fp32-reduced-seq-timing-ws4-s42
export CM_SEQUENCE_LENGTHS="128 64"
export CM_BATCH_CANDIDATES="8 4 2 1"
exec bash "$script_dir/run_cm083_fp32_gpt1b_shared_timing.sh"
