#!/usr/bin/env bash
# Run the FP32 score/interval timing comparison on GPT-350M.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CM_EXPERIMENT_ID=CM085-m002-score-gpt350m-fp32-max-batch-timing-ws4-s42
export CM_MODEL_DIM=1024
export CM_LAYERS=20
export CM_HEADS=16
export CM_SEQUENCE_LENGTHS=256
export CM_BATCH_CANDIDATES="64 48 32 24 16 8 4 2 1"
exec bash "$script_dir/run_cm083_fp32_gpt1b_shared_timing.sh"
