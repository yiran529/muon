#!/usr/bin/env bash
# Serial dense-Muon checksum attribution launcher.  Failures are recorded and
# do not prevent subsequent isolation cells from running.
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

gpu_list="${GPU_LIST:-2,3,4,5}"
model="${MODEL:-gpt130m}"
world_size="${WORLD_SIZE:-4}"
seed="${SEED:-42}"
sequence_length="${SEQUENCE_LENGTH:-256}"
local_batch="${LOCAL_BATCH:-1}"
warmup_steps="${WARMUP_STEPS:-2}"
measure_steps="${MEASURE_STEPS:-12}"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
torchrun_bin="${TORCHRUN_BIN:-$repo_dir/.venv/bin/torchrun}"
output_root="${OUTPUT_ROOT:-$repo_dir/artifacts/compressed_muon/dense_muon_attribution}"
pythonpath="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$output_root"
overall_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for mode in rank_local_custom_hook process_group_custom_hook rank_local_default_reducer rank_local_no_triton upstream_ddp; do
    cell_dir="$output_root/$mode"
    mkdir -p "$cell_dir"
    command=(env PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu_list" "$torchrun_bin" \
        --standalone --nproc_per_node="$world_size" \
        "$repo_dir/benchmark/compressed_muon/dense_muon_attribution.py" \
        --mode "$mode" --model "$model" --world-size "$world_size" \
        --seed "$seed" --sequence-length "$sequence_length" \
        --local-batch "$local_batch" --warmup-steps "$warmup_steps" \
        --measure-steps "$measure_steps" --output "$cell_dir/result.json")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${command[@]}" >"$cell_dir/stdout.log" 2>"$cell_dir/stderr.log"
    exit_code=$?
    if [ "$exit_code" -eq 0 ]; then status=completed; else status=failed; fi
    end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ ! -f "$cell_dir/result.json" ]; then
        env PYTHONPATH="$pythonpath" "$python_bin" - "$cell_dir/result.json" "$mode" "$status" "$exit_code" <<'PY'
import json
import sys

path, mode, status, exit_code = sys.argv[1:]
json.dump({
    "schema_version": 1, "mode": mode, "status": status,
    "exit_code": int(exit_code), "correctness": None,
}, open(path, "w"), indent=2, sort_keys=True)
open(path, "a").write("\n")
PY
    fi
    env PYTHONPATH="$pythonpath" "$python_bin" - "$cell_dir/status.json" "$mode" "$status" "$exit_code" "$start" "$end" <<'PY'
import json
import sys

path, mode, status, exit_code, started, finished = sys.argv[1:]
json.dump({
    "mode": mode, "status": status, "exit_code": int(exit_code),
    "started_at": started, "finished_at": finished,
    "result_exists": __import__("pathlib").Path(path).with_name("result.json").exists(),
}, open(path, "w"), indent=2, sort_keys=True)
open(path, "a").write("\n")
PY
done

env PYTHONPATH="$pythonpath" "$python_bin" "$repo_dir/benchmark/compressed_muon/summarize_dense_muon_attribution.py" \
    "$output_root" --output "$output_root/summary.json" >"$output_root/summary.stdout.log" \
    2>"$output_root/summary.stderr.log" || true
echo "dense Muon attribution artifacts: $output_root (started $overall_start)"
