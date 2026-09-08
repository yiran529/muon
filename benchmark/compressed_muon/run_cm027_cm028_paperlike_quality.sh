#!/usr/bin/env bash
# Serial paper-scale quality runs for dense Muon and optimizer-side ARC-TopK.
set -uo pipefail

repo_dir=/home/wyr/dion
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_dir="$artifact_base/CM027-CM028-paperlike-quality-controller"
data_dir="$repo_dir/data/fineweb10B"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
python_bin="$repo_dir/.venv/bin/python"
gpus="${GPUS:-2,3,4,5}"
world_size=4
status_log="$controller_dir/status.log"
probe_timeout="${PROBE_TIMEOUT:-20m}"
formal_timeout="${FORMAL_TIMEOUT:-12h}"

mkdir -p "$controller_dir"
cd "$repo_dir" || exit 2
: > "$status_log"

timestamp() {
    date -Is
}

log() {
    printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"
}

finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_dir/controller_exit_code.txt"
    timestamp > "$controller_dir/controller_finished_at.txt"
}
trap finish_controller EXIT

is_oom() {
    rg -qi 'CUDA out of memory|out of memory|cuda error: out of memory' "$1"
}

gpu_preflight() {
    local state
    state="$({
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
            awk -F', ' '$1 >= 2 && $1 <= 5 {count += 1; if ($2 >= 1024) busy = 1} END {print (count == 4 && !busy) ? "ok" : "busy"}'
    })"
    [[ "$gpus" == "2,3,4,5" ]] || {
        log "BLOCKED this registered run requires the audited GPU set 2,3,4,5; got $gpus"
        return 1
    }
    [[ "$state" == "ok" ]] || {
        log "BLOCKED GPU 2-5 are not all below 1 GiB usage"
        return 1
    }
    [[ -d "$data_dir" ]] || {
        log "BLOCKED missing dataset $data_dir"
        return 1
    }
    [[ -x "$torchrun_bin" && -x "$python_bin" ]] || {
        log "BLOCKED missing repository virtualenv executables"
        return 1
    }
    "$python_bin" -c 'import sys, wandb; sys.exit(0 if wandb.api.api_key else 1)' || {
        log "BLOCKED W&B authentication unavailable"
        return 1
    }
}

make_probe_config() {
    local source_config="$1"
    local output_config="$2"
    local val_tokens="$3"
    cp "$source_config" "$output_config"
    sed -E -i \
        -e 's/^num_iterations:.*/num_iterations: 3/' \
        -e 's/^val_loss_every:.*/val_loss_every: 0/' \
        -e "s/^val_tokens:.*/val_tokens: $val_tokens/" \
        -e 's/^no_wandb:.*/no_wandb: true/' \
        -e 's/^arc_start_compress_step:.*/arc_start_compress_step: 0/' \
        "$output_config"
}

run_probe() {
    local model="$1"
    local mode="$2"
    local source_config="$3"
    local device_batch="$4"
    local entry probe_dir probe_config output val_tokens rc
    if [[ "$mode" == "arc" ]]; then
        entry="$repo_dir/train_arctopk.py"
    else
        entry="$repo_dir/train.py"
    fi
    probe_dir="$controller_dir/probes/${model}-${mode}-db${device_batch}"
    probe_config="$probe_dir/config.yaml"
    output="$probe_dir/stdout.log"
    val_tokens=$((device_batch * 256 * world_size))
    mkdir -p "$probe_dir"
    make_probe_config "$source_config" "$probe_config" "$val_tokens"
    log "PROBE START model=$model mode=$mode device_batch=$device_batch ga=$((512 / (world_size * device_batch)))"
    timeout --signal=TERM --kill-after=60s "$probe_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        CUDA_VISIBLE_DEVICES="$gpus" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$probe_config" --data_dir "$data_dir" \
        --device_batch_size "$device_batch" --batch_size 512 \
        --training-seed 42 --no_wandb --use_polar_express \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$probe_dir/exit_code.txt"
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
    local model="$1"
    local dense_config="$2"
    local arc_config="$3"
    local candidate rc
    for candidate in 128 64 32 16; do
        run_probe "$model" arc "$arc_config" "$candidate"
        rc=$?
        if [[ "$rc" == 42 ]]; then
            continue
        fi
        [[ "$rc" == 0 ]] || return "$rc"
        run_probe "$model" dense "$dense_config" "$candidate"
        rc=$?
        if [[ "$rc" == 42 ]]; then
            continue
        fi
        [[ "$rc" == 0 ]] || return "$rc"
        printf '%s\n' "$candidate" > "$controller_dir/${model}_selected_device_batch.txt"
        SELECTED_DEVICE_BATCH="$candidate"
        SELECTED_GA=$((512 / (world_size * candidate)))
        log "SELECTED model=$model shared_device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA effective_local_batch=128"
        return 0
    done
    log "BLOCKED model=$model OOM through device_batch=16/GA8"
    return 42
}

run_formal() {
    local model="$1"
    local mode="$2"
    local id="$3"
    local config="$4"
    local final_step="$5"
    local entry cell_dir output rc
    if [[ "$mode" == "arc" ]]; then
        entry="$repo_dir/train_arctopk.py"
    else
        entry="$repo_dir/train.py"
    fi
    cell_dir="$artifact_base/$id"
    output="$cell_dir/stdout.log"
    if [[ -e "$output" || -e "$cell_dir/exit_code.txt" ]]; then
        log "BLOCKED refusing to overwrite existing formal artifact $cell_dir"
        return 1
    fi
    mkdir -p "$cell_dir"
    cp "$config" "$cell_dir/config.yaml"
    printf '%s\n' \
        "CUDA_VISIBLE_DEVICES=$gpus $torchrun_bin --standalone --nproc_per_node=$world_size $entry --config $config --data_dir $data_dir --device_batch_size $SELECTED_DEVICE_BATCH --batch_size 512 --training-seed 42 --wandb_job_name $id --use_polar_express" \
        > "$cell_dir/command.txt"
    {
        printf 'experiment_id=%s\nmodel=%s\nmode=%s\n' "$id" "$model" "$mode"
        printf 'dataset=fineweb10B\nworld_size=%s\ncuda_visible_devices=%s\n' "$world_size" "$gpus"
        printf 'sequence_length=256\nglobal_batch=512\neffective_local_batch=128\n'
        printf 'device_batch=%s\ngradient_accumulation=%s\n' "$SELECTED_DEVICE_BATCH" "$SELECTED_GA"
        printf 'training_seed=42\ngit_head=%s\n' "$(git rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "FORMAL START id=$id model=$model mode=$mode device_batch=$SELECTED_DEVICE_BATCH ga=$SELECTED_GA"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        CUDA_VISIBLE_DEVICES="$gpus" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$config" --data_dir "$data_dir" \
        --device_batch_size "$SELECTED_DEVICE_BATCH" --batch_size 512 \
        --training-seed 42 --wandb_job_name "$id" --use_polar_express \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    if [[ "$rc" != 0 ]]; then
        if is_oom "$output"; then
            log "FORMAL OOM id=$id despite successful probe"
        else
            log "FORMAL FAIL id=$id exit=$rc"
        fi
        return "$rc"
    fi
    tr '\r' '\n' < "$output" |
        rg "step:${final_step}/${final_step} val_loss:|Peak memory consumption:" |
        tail -n 2 > "$cell_dir/result.txt"
    if [[ "$(wc -l < "$cell_dir/result.txt")" != 2 ]]; then
        log "FORMAL INVALID id=$id missing final validation or memory metric"
        return 3
    fi
    log "FORMAL PASS id=$id $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

run_model_pair() {
    local model="$1"
    local dense_id="$2"
    local arc_id="$3"
    local dense_config="$4"
    local arc_config="$5"
    local final_step="$6"
    gpu_preflight || return 78
    select_shared_batch "$model" "$dense_config" "$arc_config" || return $?
    gpu_preflight || return 78
    run_formal "$model" dense "$dense_id" "$dense_config" "$final_step" || return $?
    gpu_preflight || return 78
    run_formal "$model" arc "$arc_id" "$arc_config" "$final_step" || return $?
}

timestamp > "$controller_dir/controller_started_at.txt"
cp "$0" "$controller_dir/controller.sh"
log "BEGIN CM027/CM028 paper-like FineWeb10B quality queue"

run_model_pair \
    gpt60m \
    CM027a-muon-dense-gpt60m-paperlike-train-ddp-ws4-s42 \
    CM027b-m001-arc-optimizer-muon-gpt60m-paperlike-train-ddp-ws4-s42 \
    "$repo_dir/configs/compressed_muon/cm027a_dense_muon_gpt60m_paperlike.yaml" \
    "$repo_dir/configs/compressed_muon/cm027b_arc_optimizer_muon_gpt60m_paperlike.yaml" \
    8393 || exit $?

run_model_pair \
    gpt130m \
    CM028a-muon-dense-gpt130m-paperlike-train-ddp-ws4-s42 \
    CM028b-m001-arc-optimizer-muon-gpt130m-paperlike-train-ddp-ws4-s42 \
    "$repo_dir/configs/compressed_muon/cm028a_dense_muon_gpt130m_paperlike.yaml" \
    "$repo_dir/configs/compressed_muon/cm028b_arc_optimizer_muon_gpt130m_paperlike.yaml" \
    16785 || exit $?

log "COMPLETE CM027/CM028 paper-like quality queue"
