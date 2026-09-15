#!/usr/bin/env bash
# Capture NCCL transport/bandwidth, then run a contemporaneous 1B three-arm timing.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
experiment_root="$repo_dir/artifacts/compressed_muon/CM081-m002-shared-score-gpt1b-nccl-timing-ws4-s42"
diagnostic_root="$experiment_root/nccl"
timing_root="$experiment_root/shared-only-timing"
world_size=4
minimum_free_mib=23500
poll_seconds=60
gpu_list=""
resume_after_diagnostic=0

timestamp() { date --iso-8601=seconds; }
status_log="$experiment_root/status.log"
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

if [[ -e "$experiment_root" || -L "$experiment_root" ]]; then
    if [[ -d "$experiment_root" && "$(<"$diagnostic_root/exit_code.txt")" == "0" && ! -e "$timing_root" ]]; then
        resume_after_diagnostic=1
    else
        printf 'refusing to overwrite or ambiguously resume artifact root: %s\n' "$experiment_root" >&2
        exit 73
    fi
else
    mkdir -p "$diagnostic_root"
    timestamp > "$experiment_root/started_at.txt"
fi

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$experiment_root/controller_exit_code.txt"
    timestamp > "$experiment_root/finished_at.txt"
}
trap finish_controller EXIT

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

if ((resume_after_diagnostic)); then
    gpu_list="$(<"$experiment_root/gpu_list.txt")"
    log "RESUME_AFTER_NCCL_DIAGNOSTIC gpu_list=$gpu_list timing=shared-only"
    until [[ "$gpu_list" == "$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
        awk -F, -v minimum="$minimum_free_mib" '
            {gsub(/ /, "", $1); gsub(/ /, "", $2)}
            ($2 + 0) >= minimum {print $1}
        ' | sort -n | grep -E "^($(tr ',' '|' <<<"$gpu_list"))$" | paste -sd, -
    )" ]]; do
        log "GPU_WAIT_SELECTED list=$gpu_list minimum_free_mib=$minimum_free_mib"
        sleep "$poll_seconds"
    done
else
    until select_free_gpus; do
        gpu_list=""
        log "GPU_WAIT required=$world_size minimum_free_mib=$minimum_free_mib"
        sleep "$poll_seconds"
    done
    printf '%s\n' "$gpu_list" > "$experiment_root/gpu_list.txt"
    log "GPU_SELECTED list=$gpu_list"
fi

if ((!resume_after_diagnostic)); then
    nvidia-smi > "$diagnostic_root/nvidia-smi.txt"
    nvidia-smi topo -m > "$diagnostic_root/topology.txt"
    nvidia-smi topo -p2p p > "$diagnostic_root/p2p-pcie.txt"
    "$repo_dir/.venv/bin/python" -c \
        'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version()}))' \
        > "$diagnostic_root/software.json"

    bandwidth_command=(env
        NCCL_DEBUG=INFO
        NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET
        "NCCL_DEBUG_FILE=$diagnostic_root/nccl-%h-%p.log"
        "PYTHONPATH=$repo_dir"
        "CUDA_VISIBLE_DEVICES=$gpu_list"
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size"
        "$repo_dir/benchmark/compressed_muon/benchmark_nccl_allreduce.py"
        --sizes-mib 1,32,80 --warmups 10 --iterations 50
        --output "$diagnostic_root/bandwidth.json")
    printf '%q ' "${bandwidth_command[@]}" > "$diagnostic_root/command.txt"
    printf '\n' >> "$diagnostic_root/command.txt"
    log "NCCL_DIAGNOSTIC_START gpu_list=$gpu_list"
    timeout --signal=TERM --kill-after=60 1800 "${bandwidth_command[@]}" \
        > "$diagnostic_root/stdout.log" 2> "$diagnostic_root/stderr.log"
    rc=$?
    printf '%s\n' "$rc" > "$diagnostic_root/exit_code.txt"
    ((rc == 0)) || { log "NCCL_DIAGNOSTIC_FAILED exit=$rc"; exit "$rc"; }
    log "NCCL_DIAGNOSTIC_DONE"
fi

log "TIMING_START model=gpt1b device_batch=24 global_batch=96"
"$launcher" \
    --world-size 4 --global-batch-size 96 --device-batch-size 24 \
    --model-dim 1536 --layers 30 --heads 24 \
    --model-dtype bfloat16 --sequence-length 256 --bucket-cap-mb 80 \
    --timing-warmup-steps 20 --measured-full-periods 4 --repeats 3 \
    --profile-modes none \
    --timing-modes greedylore_local_svd_shared \
    --training-seed 42 --greedy-lore-rank 32 --greedy-lore-update-interval 200 \
    --greedy-lore-dense-aux-communication-dtype bucket \
    --gpu-list "$gpu_list" --artifact-root "$timing_root"
rc=$?
printf '%s\n' "$rc" > "$experiment_root/timing_exit_code.txt"
((rc == 0)) || { log "TIMING_FAILED exit=$rc"; exit "$rc"; }
log "EXPERIMENT_DONE"
