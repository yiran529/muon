#!/usr/bin/env bash
# Shared-GPU exploratory wall-clock comparison for optimizer ARC vs DDP-hook ARC.
set -uo pipefail

repo_dir=/home/wyr/dion
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_dir="$artifact_base/CM029-CM030-hook-vs-optimizer-paperlike-wallclock"
data_dir="$repo_dir/data/fineweb10B"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
optimizer_config="$repo_dir/configs/compressed_muon/cm029_arc_optimizer_paperlike_wallclock.yaml"
hook_config="$repo_dir/configs/compressed_muon/cm029_arc_ddp_hook_paperlike_wallclock.yaml"
gpu_list="${GPUS:-4,5,6,7}"
world_size=4
global_batch=512
sequence_length=256
warmup_steps=20
measured_steps=200
num_iterations=$((warmup_steps + measured_steps))
bucket_cap_mb=160
training_seed=42
status_log="$controller_dir/status.log"

mkdir -p "$controller_dir"
cd "$repo_dir" || exit 2
: > "$status_log"

timestamp() { date -Is; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_dir/controller_exit_code.txt"
    timestamp > "$controller_dir/controller_finished_at.txt"
}
trap finish_controller EXIT

is_oom() {
    rg -qi 'CUDA out of memory|out of memory|cuda error: out of memory' "$1"
}

capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
            --format=csv,noheader 2>/dev/null || true
    } > "$output"
}

shared_gpu_preflight() {
    local gpu free
    [[ "$gpu_list" == "4,5,6,7" ]] || {
        log "BLOCKED registered shared-GPU run requires GPU 4,5,6,7; got $gpu_list"
        return 1
    }
    for gpu in 4 5 6 7; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$free" =~ ^[0-9]+$ ]] && ((free >= 4096)) || {
            log "BLOCKED gpu=$gpu has less than 4096 MiB free"
            return 1
        }
    done
    [[ -d "$data_dir" && -x "$torchrun_bin" ]] || {
        log "BLOCKED missing data or torchrun"
        return 1
    }
}

model_shape() {
    case "$1" in
        gpt60m) MODEL_DIM=512; MODEL_LAYERS=4; MODEL_HEADS=8 ;;
        gpt130m) MODEL_DIM=768; MODEL_LAYERS=8; MODEL_HEADS=12 ;;
        *) return 2 ;;
    esac
}

run_probe() {
    local model="$1" mode="$2" device_batch="$3"
    local config probe_dir output rc ga
    model_shape "$model" || return $?
    if [[ "$mode" == "hook" ]]; then
        config="$hook_config"
    else
        config="$optimizer_config"
    fi
    ga=$((global_batch / (world_size * device_batch)))
    probe_dir="$controller_dir/probes/${model}-${mode}-db${device_batch}"
    output="$probe_dir/stdout.log"
    mkdir -p "$probe_dir"
    capture_gpu_state "$probe_dir/nvidia_smi_before.txt"
    log "PROBE START model=$model mode=$mode device_batch=$device_batch ga=$ga"
    timeout --signal=TERM --kill-after=60s 30m \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" \
        "$repo_dir/train_arctopk.py" --config "$config" --data_dir "$data_dir" \
        --model_dim "$MODEL_DIM" --n_layer "$MODEL_LAYERS" --n_head "$MODEL_HEADS" \
        --sequence_length "$sequence_length" --batch_size "$global_batch" \
        --device_batch_size "$device_batch" --val_tokens 131072 \
        --num_iterations 3 --timing-warmup-steps 0 --training-seed "$training_seed" \
        --bucket-cap-mb "$bucket_cap_mb" --arc_start_compress_step 0 \
        --no_wandb --use_polar_express \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
    capture_gpu_state "$probe_dir/nvidia_smi_after.txt"
    if [[ "$rc" == 0 ]] && rg -q 'Peak memory consumption:' "$output"; then
        tr '\r' '\n' < "$output" |
            rg 'step:3/3 val_loss:|Peak memory consumption:' |
            tail -n 2 > "$probe_dir/result.txt"
        log "PROBE PASS model=$model mode=$mode device_batch=$device_batch $(tr '\n' ' ' < "$probe_dir/result.txt")"
        return 0
    fi
    if is_oom "$output"; then
        log "PROBE OOM model=$model mode=$mode device_batch=$device_batch"
        return 42
    fi
    log "PROBE FAIL model=$model mode=$mode device_batch=$device_batch exit=$rc"
    return 1
}

select_shared_batch() {
    local model="$1" candidate mode rc
    for candidate in 128 64 32; do
        for mode in hook optimizer; do
            shared_gpu_preflight || return 78
            run_probe "$model" "$mode" "$candidate"
            rc=$?
            if [[ "$rc" == 42 ]]; then
                break
            fi
            [[ "$rc" == 0 ]] || return "$rc"
        done
        if [[ "$rc" == 0 ]]; then
            SELECTED_DEVICE_BATCH="$candidate"
            SELECTED_GA=$((global_batch / (world_size * candidate)))
            printf '%s\n' "$candidate" > "$controller_dir/${model}_selected_device_batch.txt"
            log "SELECTED model=$model device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
            return 0
        fi
    done
    log "BLOCKED model=$model OOM through device_batch=32/GA4"
    return 42
}

run_cell() {
    local model="$1" mode="$2" id="$3"
    local config cell_dir output rc
    model_shape "$model" || return $?
    if [[ "$mode" == "hook" ]]; then
        config="$hook_config"
    else
        config="$optimizer_config"
    fi
    cell_dir="$artifact_base/$id"
    output="$cell_dir/stdout.log"
    if [[ -e "$output" || -e "$cell_dir/exit_code.txt" ]]; then
        log "BLOCKED refusing to overwrite $cell_dir"
        return 1
    fi
    shared_gpu_preflight || return 78
    mkdir -p "$cell_dir"
    cp "$config" "$cell_dir/config.yaml"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    {
        printf 'experiment_id=%s\nmodel=%s\nmode=%s\n' "$id" "$model" "$mode"
        printf 'shared_gpu_run=true\ngpu_list=%s\nworld_size=%s\n' "$gpu_list" "$world_size"
        printf 'sequence_length=%s\nglobal_batch=%s\ndevice_batch=%s\nga=%s\n' \
            "$sequence_length" "$global_batch" "$SELECTED_DEVICE_BATCH" "$SELECTED_GA"
        printf 'warmup_steps=%s\nmeasured_steps=%s\nbucket_cap_mb=%s\n' \
            "$warmup_steps" "$measured_steps" "$bucket_cap_mb"
        printf 'training_seed=%s\ngit_head=%s\n' "$training_seed" "$(git rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    printf '%s\n' \
        "CUDA_VISIBLE_DEVICES=$gpu_list $torchrun_bin --standalone --nproc_per_node=$world_size train_arctopk.py --config $config --data_dir $data_dir --model_dim $MODEL_DIM --n_layer $MODEL_LAYERS --n_head $MODEL_HEADS --sequence_length $sequence_length --batch_size $global_batch --device_batch_size $SELECTED_DEVICE_BATCH --val_tokens 131072 --num_iterations $num_iterations --timing-warmup-steps $warmup_steps --training-seed $training_seed --bucket-cap-mb $bucket_cap_mb --arc_start_compress_step 0 --no_wandb --use_polar_express" \
        > "$cell_dir/command.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "CELL START id=$id model=$model mode=$mode device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
    timeout --signal=TERM --kill-after=60s 2h \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" \
        "$repo_dir/train_arctopk.py" --config "$config" --data_dir "$data_dir" \
        --model_dim "$MODEL_DIM" --n_layer "$MODEL_LAYERS" --n_head "$MODEL_HEADS" \
        --sequence_length "$sequence_length" --batch_size "$global_batch" \
        --device_batch_size "$SELECTED_DEVICE_BATCH" --val_tokens 131072 \
        --num_iterations "$num_iterations" --timing-warmup-steps "$warmup_steps" \
        --training-seed "$training_seed" --bucket-cap-mb "$bucket_cap_mb" \
        --arc_start_compress_step 0 --no_wandb --use_polar_express \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    if [[ "$rc" != 0 ]]; then
        if is_oom "$output"; then
            log "CELL OOM id=$id despite probe"
        else
            log "CELL FAIL id=$id exit=$rc"
        fi
        return "$rc"
    fi
    tr '\r' '\n' < "$output" |
        rg "step:${num_iterations}/${num_iterations} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$cell_dir/result.txt"
    if [[ "$(wc -l < "$cell_dir/result.txt")" != 2 ]]; then
        log "CELL INVALID id=$id missing final timing or memory metric"
        return 3
    fi
    log "CELL PASS id=$id $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

if [[ -e "$controller_dir/controller_started_at.txt" ]]; then
    log "BLOCKED refusing to overwrite prior controller"
    exit 73
fi
timestamp > "$controller_dir/controller_started_at.txt"
cp "$0" "$controller_dir/controller.sh"
git rev-parse HEAD > "$controller_dir/git_commit.txt"
git status --short > "$controller_dir/git_status.txt"
capture_gpu_state "$controller_dir/nvidia_smi_start.txt"
log "BEGIN shared-GPU hook-vs-optimizer wall-clock queue gpu_list=$gpu_list"

select_shared_batch gpt60m || exit $?
run_cell gpt60m optimizer CM029a-arc-optimizer-muon-gpt60m-paperlike-wallclock-ws4-s42 || exit $?
run_cell gpt60m hook CM029b-arc-ddp-hook-muon-gpt60m-paperlike-wallclock-ws4-s42 || exit $?

select_shared_batch gpt130m || exit $?
run_cell gpt130m hook CM030b-arc-ddp-hook-muon-gpt130m-paperlike-wallclock-ws4-s42 || exit $?
run_cell gpt130m optimizer CM030a-arc-optimizer-muon-gpt130m-paperlike-wallclock-ws4-s42 || exit $?

capture_gpu_state "$controller_dir/nvidia_smi_end.txt"
log "COMPLETE shared-GPU hook-vs-optimizer wall-clock queue"
