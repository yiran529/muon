#!/usr/bin/env bash
# Run corrected CM085 and then CM086 without competing for GPUs.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$script_dir/run_cm085_fp32_gpt350m_timing.sh"
exec bash "$script_dir/run_cm086_fp32_gpt720m_timing.sh"
