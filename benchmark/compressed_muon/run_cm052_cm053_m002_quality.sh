#!/usr/bin/env bash
# Serial paper-aligned quality runs for dense Muon and M002 GreedyLore-Muon.
set -uo pipefail

repo_dir=/home/wyr/dion
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_id=CM052-CM053-m002-paper-aligned-quality-controller
controller_dir="$artifact_base/$controller_id"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
requested_gpu_list="${GPUS:-}"
gpu_list=""
world_size=4
training_seed=1234
device_batch_size=128
global_batch=512
bucket_cap_mb=160
minimum_free_mib=18000
gpu_wait_seconds="${GPU_WAIT_SECONDS:-60}"
probe_timeout="${PROBE_TIMEOUT:-30m}"
formal_timeout="${FORMAL_TIMEOUT:-12h}"
status_log="$controller_dir/status.log"

ids=(
    CM052a-dense-muon-gpt60m-paper-aligned-ddp-ws4-s1234
    CM052b-m002-greedylore-muon-gpt60m-paper-aligned-ddp-ws4-s1234
    CM053a-dense-muon-gpt130m-paper-aligned-ddp-ws4-s1234
    CM053b-m002-greedylore-muon-gpt130m-paper-aligned-ddp-ws4-s1234
)
models=(gpt60m gpt60m gpt130m gpt130m)
modes=(dense m002 dense m002)
final_steps=(10000 10000 20000 20000)
configs=(
    "$repo_dir/configs/compressed_muon/cm052a_dense_muon_gpt60m_paper_aligned_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm052b_m002_greedy_lore_muon_gpt60m_paper_aligned_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm053a_dense_muon_gpt130m_paper_aligned_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm053b_m002_greedy_lore_muon_gpt130m_paper_aligned_s1234.yaml"
)

if [[ "${1:-}" == "--print-plan" && $# -eq 1 ]]; then
    "$python_bin" - <<'PY'
import json

print(json.dumps({
    "controller_id": "CM052-CM053-m002-paper-aligned-quality-controller",
    "cell_order": [
        "CM052a-dense-muon-gpt60m-paper-aligned-ddp-ws4-s1234",
        "CM052b-m002-greedylore-muon-gpt60m-paper-aligned-ddp-ws4-s1234",
        "CM053a-dense-muon-gpt130m-paper-aligned-ddp-ws4-s1234",
        "CM053b-m002-greedylore-muon-gpt130m-paper-aligned-ddp-ws4-s1234",
    ],
    "execution": "serial",
    "world_size": 4,
    "gpu_policy": {"selection": "any_four", "minimum_free_mib": 18000, "poll_seconds": 60},
    "dataset": "fineweb10B",
    "training_seed": 1234,
    "sequence_length": 256,
    "global_batch": 512,
    "device_batch_size": 128,
    "gradient_accumulation": 1,
    "models": {
        "gpt60m": {"steps": 10000, "tokens": 1_310_720_000, "warmup_steps": 1000},
        "gpt130m": {"steps": 20000, "tokens": 2_621_440_000, "warmup_steps": 2000},
    },
    "schedule": {"type": "cosine", "min_lr_ratio": 0.1},
    "greedy_lore": {"rank": 32, "update_interval": 200, "start_step": 1000, "basis_sync": "local_svd"},
    "quality_gate": {"after_model": "gpt60m", "max_perplexity_ratio": 1.10},
    "profiler_trace": False,
}, indent=2))
PY
    exit 0
elif (($#)); then
    printf 'usage: %s [--print-plan]\n' "$0" >&2
    exit 64
fi

timestamp() { date -Is; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_dir/controller_exit_code.txt"
    timestamp > "$controller_dir/controller_finished_at.txt"
}

capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

preflight() {
    local config
    for config in "${configs[@]}"; do
        [[ -f "$config" ]] || { log "BLOCKED missing config=$config"; return 1; }
    done
    [[ -d "$data_dir" && -x "$python_bin" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing dataset or virtualenv executables"
        return 1
    }
    "$python_bin" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)' || {
        log "BLOCKED W&B authentication unavailable"
        return 1
    }
}

select_free_gpus() {
    local -a candidates requested
    local gpu free
    candidates=()
    if [[ -n "$requested_gpu_list" ]]; then
        IFS=',' read -r -a requested <<< "$requested_gpu_list"
        [[ "${#requested[@]}" == "$world_size" ]] || return 1
        for gpu in "${requested[@]}"; do
            [[ "$gpu" =~ ^[0-9]+$ ]] || return 1
            free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
            [[ "$free" =~ ^[0-9]+$ ]] && ((free >= minimum_free_mib)) || return 1
        done
        gpu_list="$requested_gpu_list"
        return 0
    fi
    while read -r gpu free; do
        [[ "$gpu" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        ((free >= minimum_free_mib)) && candidates+=("$gpu")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    ((${#candidates[@]} >= world_size)) || return 1
    gpu_list="$(IFS=,; printf '%s' "${candidates[*]:0:$world_size}")"
}

wait_for_gpus() {
    local attempts=0
    until select_free_gpus; do
        if ((attempts == 0 || attempts % 10 == 0)); then
            log "WAITING for four GPUs with at least ${minimum_free_mib} MiB free each"
        fi
        attempts=$((attempts + 1))
        sleep "$gpu_wait_seconds"
    done
    log "GPU READY cuda_visible_devices=$gpu_list"
}

is_oom() {
    rg -qi 'CUDA out of memory|out of memory|cuda error: out of memory' "$1"
}

make_probe_config() {
    local source_config="$1" output_config="$2" mode="$3"
    cp "$source_config" "$output_config"
    sed -E -i \
        -e 's/^num_iterations:.*/num_iterations: 3/' \
        -e 's/^val_loss_every:.*/val_loss_every: 0/' \
        -e 's/^val_tokens:.*/val_tokens: 131072/' \
        -e 's/^checkpoint_freq:.*/checkpoint_freq: 0/' \
        -e 's/^warmup_steps:.*/warmup_steps: 0/' \
        -e 's/^no_wandb:.*/no_wandb: true/' \
        "$output_config"
    if [[ "$mode" == m002 ]]; then
        sed -E -i \
            -e 's/^greedy_lore_start_compress_step:.*/greedy_lore_start_compress_step: 0/' \
            -e 's/^greedy_lore_update_interval:.*/greedy_lore_update_interval: 2/' \
            "$output_config"
    fi
}

run_probe() {
    local index="$1" id model mode config entry probe_dir probe_config output rc
    id="${ids[$index]}"
    model="${models[$index]}"
    mode="${modes[$index]}"
    config="${configs[$index]}"
    entry="$repo_dir/train.py"
    [[ "$mode" == m002 ]] && entry="$repo_dir/train_greedylore.py"
    probe_dir="$controller_dir/probes/$id"
    probe_config="$probe_dir/config.yaml"
    output="$probe_dir/stdout.log"
    mkdir -p "$probe_dir"
    make_probe_config "$config" "$probe_config" "$mode"
    capture_gpu_state "$probe_dir/nvidia_smi_before.txt"
    log "PROBE START id=$id model=$model mode=$mode"
    timeout --signal=TERM --kill-after=60s "$probe_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$probe_config" --data_dir "$data_dir" \
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb" --no_wandb \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_after.txt"
    if [[ "$rc" == 0 ]] && rg -q 'Peak memory consumption:' "$output"; then
        log "PROBE PASS id=$id"
        return 0
    fi
    if is_oom "$output"; then
        log "PROBE OOM id=$id"
        return 42
    fi
    log "PROBE FAIL id=$id exit=$rc"
    return "$rc"
}

run_formal() {
    local index="$1" id model mode final_step config entry cell_dir output checkpoint_dir rc
    id="${ids[$index]}"
    model="${models[$index]}"
    mode="${modes[$index]}"
    final_step="${final_steps[$index]}"
    config="${configs[$index]}"
    entry="$repo_dir/train.py"
    [[ "$mode" == m002 ]] && entry="$repo_dir/train_greedylore.py"
    cell_dir="$artifact_base/$id"
    output="$cell_dir/stdout.log"
    checkpoint_dir="$cell_dir/checkpoints"
    if [[ -e "$output" || -e "$cell_dir/exit_code.txt" ]]; then
        log "BLOCKED refusing to overwrite formal artifact=$cell_dir"
        return 1
    fi
    mkdir -p "$cell_dir" "$checkpoint_dir"
    cp "$config" "$cell_dir/config.yaml"
    printf '%s\n' \
        "CUDA_VISIBLE_DEVICES=$gpu_list $torchrun_bin --standalone --nproc_per_node=$world_size $entry --config $config --data_dir $data_dir --training-seed $training_seed --bucket-cap-mb $bucket_cap_mb --checkpoint_dir $checkpoint_dir --wandb_job_name $id" \
        > "$cell_dir/command.txt"
    {
        printf 'experiment_id=%s\nmodel=%s\nmode=%s\n' "$id" "$model" "$mode"
        printf 'dataset=fineweb10B\nworld_size=%s\ncuda_visible_devices=%s\n' "$world_size" "$gpu_list"
        printf 'sequence_length=256\nglobal_batch=512\ndevice_batch=128\ngradient_accumulation=1\n'
        printf 'training_seed=%s\ngit_head=%s\n' "$training_seed" "$(git -C "$repo_dir" rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "FORMAL START id=$id model=$model mode=$mode"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$config" --data_dir "$data_dir" \
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb" \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$id" \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    if [[ "$rc" != 0 ]]; then
        if is_oom "$output"; then
            log "FORMAL OOM id=$id"
        else
            log "FORMAL FAIL id=$id exit=$rc"
        fi
        return "$rc"
    fi
    tr '\r' '\n' < "$output" | \
        rg "step:${final_step}/${final_step} val_loss:|Peak memory consumption:" | \
        tail -n 2 > "$cell_dir/result.txt"
    if [[ "$(wc -l < "$cell_dir/result.txt")" != 2 ]]; then
        log "FORMAL INVALID id=$id missing final validation or memory metric"
        return 3
    fi
    log "FORMAL PASS id=$id $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

check_60m_quality_gate() {
    local dense_result compressed_result gate_output
    dense_result="$artifact_base/${ids[0]}/result.txt"
    compressed_result="$artifact_base/${ids[1]}/result.txt"
    gate_output="$controller_dir/quality_gate_60m.json"
    "$python_bin" - "$dense_result" "$compressed_result" "$gate_output" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

def loss(path):
    text = Path(path).read_text()
    matches = re.findall(r"val_loss:([0-9.eE+-]+)", text)
    if not matches:
        raise SystemExit(f"missing validation loss in {path}")
    value = float(matches[-1])
    if not math.isfinite(value):
        raise SystemExit(f"non-finite validation loss in {path}")
    return value

dense = loss(sys.argv[1])
compressed = loss(sys.argv[2])
ratio = math.exp(compressed - dense)
passed = ratio <= 1.10
Path(sys.argv[3]).write_text(json.dumps({
    "dense_validation_loss": dense,
    "greedy_lore_validation_loss": compressed,
    "perplexity_ratio": ratio,
    "maximum_perplexity_ratio": 1.10,
    "passed": passed,
}, indent=2) + "\n")
raise SystemExit(0 if passed else 65)
PY
}

mkdir -p "$controller_dir"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$controller_dir/controller_started_at.txt"
cp "$0" "$controller_dir/controller.sh"
"$0" --print-plan > "$controller_dir/plan.json"
capture_gpu_state "$controller_dir/nvidia_smi_at_start.txt"
log "BEGIN CM052/CM053 M002 paper-aligned FineWeb10B serial queue"

preflight || exit 78
for index in 0 1 2 3; do
    wait_for_gpus
    run_probe "$index" || exit $?
done

wait_for_gpus
run_formal 0 || exit $?
wait_for_gpus
run_formal 1 || exit $?
if check_60m_quality_gate; then
    log "QUALITY GATE PASS model=gpt60m; continuing to GPT-130M"
else
    rc=$?
    log "QUALITY GATE STOP model=gpt60m exit=$rc; GPT-130M not started"
    exit "$rc"
fi

wait_for_gpus
run_formal 2 || exit $?
wait_for_gpus
run_formal 3 || exit $?

log "COMPLETE CM052/CM053 M002 paper-aligned FineWeb10B serial queue"
