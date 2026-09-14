#!/usr/bin/env bash
# Run targeted BF16 profiles, then six paper-aligned BF16 quality cells serially.
set -uo pipefail

repo_dir=/home/wyr/dion
artifact_base="$repo_dir/artifacts/compressed_muon"
controller_id=CM069-CM070-m002-bf16-profile-and-quality-controller
controller_dir="$artifact_base/$controller_id"
launcher="$repo_dir/benchmark/compressed_muon/run_greedy_lore_profiler.sh"
data_dir="$repo_dir/data/fineweb10B"
python_bin="$repo_dir/.venv/bin/python"
torchrun_bin="$repo_dir/.venv/bin/torchrun"
world_size=4
training_seed=1234
minimum_free_mib=18000
gpu_wait_seconds=300
formal_timeout=12h
gpu_list=""
status_log="$controller_dir/status.log"

ids=(
    CM070a-bf16-dense-muon-gpt60m-paper-aligned-ddp-ws4-s1234
    CM070b-bf16-m002-greedylore-muon-gpt60m-paper-aligned-ddp-ws4-s1234
    CM070c-bf16-dense-muon-gpt130m-paper-aligned-ddp-ws4-s1234
    CM070d-bf16-m002-greedylore-muon-gpt130m-paper-aligned-ddp-ws4-s1234
    CM070e-bf16-m002-greedylore-muon-gpt60m-paper-aligned-r128-ddp-ws4-s1234
    CM070f-bf16-m002-greedylore-muon-gpt130m-paper-aligned-r256-ddp-ws4-s1234
)
models=(gpt60m gpt60m gpt130m gpt130m gpt60m gpt130m)
modes=(dense m002 dense m002 m002 m002)
final_steps=(10000 10000 20000 20000 10000 20000)
configs=(
    "$repo_dir/configs/compressed_muon/cm052a_dense_muon_gpt60m_paper_aligned_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm052b_m002_greedy_lore_muon_gpt60m_paper_aligned_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm053a_dense_muon_gpt130m_paper_aligned_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm053b_m002_greedy_lore_muon_gpt130m_paper_aligned_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm052c_m002_greedy_lore_muon_gpt60m_paper_aligned_r128_s1234.yaml"
    "$repo_dir/configs/compressed_muon/cm053c_m002_greedy_lore_muon_gpt130m_paper_aligned_r256_s1234.yaml"
)

timestamp() { date --iso-8601=seconds; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$status_log"; }
finish_controller() {
    local rc=$?
    printf '%s\n' "$rc" > "$controller_dir/controller_exit_code.txt"
    timestamp > "$controller_dir/finished_at.txt"
}
capture_gpu_state() {
    local output="$1"
    {
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true
    } > "$output"
}
select_free_gpus() {
    local -a candidates=()
    local gpu free
    while read -r gpu free; do
        [[ "$gpu" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        ((free >= minimum_free_mib)) && candidates+=("$gpu")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    ((${#candidates[@]} >= world_size)) || return 1
    gpu_list="$(IFS=,; printf '%s' "${candidates[*]:0:$world_size}")"
}
wait_for_gpus() {
    until select_free_gpus; do
        log "GPU_WAIT required=$world_size minimum_free_mib=$minimum_free_mib poll_seconds=$gpu_wait_seconds"
        sleep "$gpu_wait_seconds"
    done
    log "GPU_SELECTED list=$gpu_list"
}
run_profile_cap() {
    local cap="$1"
    local root="$artifact_base/CM069-m002-gpt130m-bf16-bucket${cap}-targeted-profile-ws4-s42"
    log "PROFILE_START bucket_cap_mib=$cap root=$root"
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
        --profile-modes dense,greedylore_local_svd \
        --timing-modes none \
        --training-seed 42 \
        --greedy-lore-rank 32 \
        --greedy-lore-update-interval 200 \
        --greedy-lore-dense-aux-communication-dtype bucket \
        --gpu-list "$gpu_list" \
        --artifact-root "$root" || return $?
    log "PROFILE_DONE bucket_cap_mib=$cap root=$root"
}
run_formal() {
    local index="$1"
    local id="${ids[$index]}" model="${models[$index]}" mode="${modes[$index]}"
    local final_step="${final_steps[$index]}" config="${configs[$index]}"
    local entry="$repo_dir/train.py" cell_dir="$artifact_base/$id" output checkpoint_dir rc
    [[ "$mode" == m002 ]] && entry="$repo_dir/train_greedylore.py"
    output="$cell_dir/stdout.log"
    checkpoint_dir="$cell_dir/checkpoints"
    if [[ -e "$cell_dir" ]]; then
        log "BLOCKED refusing_to_overwrite=$cell_dir"
        return 73
    fi
    mkdir -p "$cell_dir" "$checkpoint_dir"
    cp "$config" "$cell_dir/source-config.yaml"
    printf '%q ' env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        "PYTHONPATH=$repo_dir" "CUDA_VISIBLE_DEVICES=$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$config" --data_dir "$data_dir" --model_dtype bfloat16 \
        --training-seed "$training_seed" --bucket-cap-mb 160 \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$id" \
        > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    {
        printf 'experiment_id=%s\nsource_experiment=%s\nmodel=%s\nmode=%s\n' \
            "$id" "$(basename "$config" .yaml)" "$model" "$mode"
        printf 'model_dtype=bfloat16\ndataset=fineweb10B\nworld_size=%s\ncuda_visible_devices=%s\n' \
            "$world_size" "$gpu_list"
        printf 'training_seed=%s\ngit_head=%s\n' "$training_seed" "$(git -C "$repo_dir" rev-parse HEAD)"
    } > "$cell_dir/environment.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_before.txt"
    timestamp > "$cell_dir/started_at.txt"
    log "FORMAL_START id=$id source=$(basename "$config")"
    timeout --signal=TERM --kill-after=60s "$formal_timeout" \
        env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE \
        PYTHONPATH="$repo_dir" CUDA_VISIBLE_DEVICES="$gpu_list" \
        "$torchrun_bin" --standalone "--nproc_per_node=$world_size" "$entry" \
        --config "$config" --data_dir "$data_dir" --model_dtype bfloat16 \
        --training-seed "$training_seed" --bucket-cap-mb 160 \
        --checkpoint_dir "$checkpoint_dir" --wandb_job_name "$id" \
        > "$output" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$cell_dir/exit_code.txt"
    timestamp > "$cell_dir/finished_at.txt"
    capture_gpu_state "$cell_dir/nvidia_smi_after.txt"
    ((rc == 0)) || { log "FORMAL_FAIL id=$id exit=$rc"; return "$rc"; }
    tr '\r' '\n' < "$output" | \
        rg "step:${final_step}/${final_step} val_loss:|Peak memory consumption:" | \
        tail -n 2 > "$cell_dir/result.txt"
    [[ "$(wc -l < "$cell_dir/result.txt")" == 2 ]] || {
        log "FORMAL_INVALID id=$id missing_final_metrics"
        return 65
    }
    log "FORMAL_DONE id=$id $(tr '\n' ' ' < "$cell_dir/result.txt")"
}

if [[ -e "$controller_dir" ]]; then
    printf 'refusing to overwrite existing controller artifact: %s\n' "$controller_dir" >&2
    exit 73
fi
mkdir -p "$controller_dir"
trap finish_controller EXIT
: > "$status_log"
timestamp > "$controller_dir/started_at.txt"
cp "$0" "$controller_dir/controller.sh"
git -C "$repo_dir" rev-parse HEAD > "$controller_dir/git_head.txt"
capture_gpu_state "$controller_dir/nvidia_smi_at_start.txt"

log "CONTROLLER_START phase=profile_then_quality"
wait_for_gpus
if run_profile_cap 40; then
    printf '0\n' > "$controller_dir/profile_bucket40_exit_code.txt"
else
    rc=$?
    printf '%s\n' "$rc" > "$controller_dir/profile_bucket40_exit_code.txt"
    log "PROFILE_ALLOWED_FAILURE bucket_cap_mib=40 exit=$rc"
fi
wait_for_gpus
if run_profile_cap 80; then
    printf '0\n' > "$controller_dir/profile_bucket80_exit_code.txt"
else
    rc=$?
    printf '%s\n' "$rc" > "$controller_dir/profile_bucket80_exit_code.txt"
    log "PROFILE_ALLOWED_FAILURE bucket_cap_mib=80 exit=$rc"
fi

for index in "${!ids[@]}"; do
    wait_for_gpus
    if run_formal "$index"; then
        printf '0\n' > "$controller_dir/${ids[$index]}_queue_exit_code.txt"
    else
        rc=$?
        printf '%s\n' "$rc" > "$controller_dir/${ids[$index]}_queue_exit_code.txt"
        log "FORMAL_ALLOWED_FAILURE id=${ids[$index]} exit=$rc; continuing_queue=true"
    fi
done
log "CONTROLLER_DONE"
