#!/usr/bin/env bash
# Run the FP32 score/interval timing comparison on a 718.6M-parameter GPT.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CM_EXPERIMENT_ID=CM086-m002-score-gpt720m-fp32-max-batch-timing-ws4-s42
export CM_MODEL_DIM=1280
export CM_LAYERS=30
export CM_HEADS=20
export CM_SEQUENCE_LENGTHS=256
export CM_BATCH_CANDIDATES="32 24 16 12 8 4 2 1"
exec bash "$script_dir/run_cm083_fp32_gpt1b_shared_timing.sh"
