#!/usr/bin/env bash
# Serial, resumable M001 ARC-TopK scale-out through GPT-1B.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; ARTIFACT_ROOT="${SCALE_TO_1B_ARTIFACT_ROOT:-$ROOT/artifacts/compressed_muon}"
PYTHON="$ROOT/.venv/bin/python"; TORCHRUN="$ROOT/.venv/bin/torchrun"; GPUS="2,3,4,5"; RUN_TAG="$(date +%Y%m%dT%H%M%S%z)"
MANIFEST="${SCALE_TO_1B_MANIFEST:-$ARTIFACT_ROOT/scale-to-1b-manifest.jsonl}"; STATUS_LOG="${SCALE_TO_1B_STATUS_LOG:-$ARTIFACT_ROOT/scale-to-1b-status-$RUN_TAG.log}"; VALIDATOR="$ARTIFACT_ROOT/validate_scale_to_1b.py"; DRY_RUN=0; RECONCILE_ONLY=0
MODELS=(gpt130m gpt350m gpt1b); TRANSPORTS=(normal p2p_disabled)
cd "$ROOT" || exit 2

log() { printf '%s %s\n' "$(date -Is)" "$1" | tee -a "$STATUS_LOG"; }
event() { "$PYTHON" -c 'import json,sys; print(json.dumps(dict(zip(sys.argv[1::2],sys.argv[2::2])),sort_keys=True))' "$@" >> "$MANIFEST"; }
init_manifest() {
  mkdir -p "$ARTIFACT_ROOT"; [[ -s "$MANIFEST" ]] && return
  printf '%s\n' '{"schema_version":1,"kind":"m001-arc-topk-scale-to-1b","status":"created"}' > "$MANIFEST"
  event event planned run_tag "$RUN_TAG" started_at "$(date -Is)"
  for model in "${MODELS[@]}"; do for transport in "${TRANSPORTS[@]}"; do
    [[ "$transport" == normal ]] && case "$model" in gpt130m) base=CM004;; gpt350m) base=CM006;; gpt1b) base=CM008;; esac || case "$model" in gpt130m) base=CM005;; gpt350m) base=CM007;; gpt1b) base=CM009;; esac
    for suffix in a b c d; do case "$suffix" in a) id="${base}a-adamw-dense-$model-ddp-ws4-s42";; b) id="${base}b-m001-adamw-arc-$model-ddp-ws4-s42";; c) id="${base}c-muon-dense-$model-ddp-ws4-s42";; d) id="${base}d-m001-muon-arc-$model-ddp-ws4-s42";; esac; event event planned id "$id" model "$model" transport "$transport"; done
  done; done
}
manifest_has() {
  "$PYTHON" -c 'import json,sys; path,event,key,value=sys.argv[1:]; found=False
for line in open(path):
    if not line.strip(): continue
    row=json.loads(line)
    if row.get("event")==event and row.get(key)==value: found=True; break
print("1" if found else "0")' "$MANIFEST" "$1" "$2" "$3" | grep -qx 1
}
manifest_has_id() {
  "$PYTHON" -c 'import json,sys; path,value=sys.argv[1:]; found=False
for line in open(path):
    if not line.strip(): continue
    row=json.loads(line)
    if row.get("id")==value or row.get("stale_id")==value: found=True; break
print("1" if found else "0")' "$MANIFEST" "$1" | grep -qx 1
}
reconcile_manifest() {
  local model transport suffix corrected stale
  # Run every mapping on every resume. Each exact event is checked independently,
  # so an interrupted/partially reconciled manifest converges without duplicates.
  for model in "${MODELS[@]}"; do for transport in "${TRANSPORTS[@]}"; do for suffix in a b c d; do
    corrected="$(cell_id "$model" "$transport" "$suffix")"
    stale="${corrected:0:5}-${corrected:5}"
    if ! manifest_has planned id "$corrected"; then
      event event planned id "$corrected" model "$model" transport "$transport"
    fi
    if manifest_has_id "$stale" && ! manifest_has superseded stale_id "$stale"; then
      event event superseded stale_id "$stale" corrected_id "$corrected" reason "corrected suffix mapping"
    fi
  done; done; done
}
cell_event() { event event cell id "$1" status "$2" path "${3:-}" message "${4:-}"; }
oom_sentinel() { printf '%s/scale-to-1b-oom-%s-%s' "$ARTIFACT_ROOT" "$1" "$2"; }
mode_oom() { [[ -e "$(oom_sentinel "$1" "$2")" ]]; }
record_oom() { : > "$(oom_sentinel "$1" "$2")"; log "OOM gate optimizer=$1 sync=$2; independent modes continue"; }
write_partial() {
  local model="$1" output="$ARTIFACT_ROOT/${model}-partial.json"
  "$PYTHON" -c 'import json,sys; out,model,manifest=sys.argv[1:]; events=[json.loads(x) for x in open(manifest) if x.strip()]; rows={}; [rows.setdefault(e.get("id"),[]).append(e) for e in events if e.get("event")=="cell" and model in e.get("id","") and e.get("id","")[5:6] != "-"]; classes={};
for k,v in rows.items():
    statuses={e.get("status") for e in v}; classes[k]="invalid" if statuses & {"invalid","failed"} else "oom" if "oom" in statuses else "skipped" if "skipped" in statuses else "completed" if {"timing_valid","profile_valid"} <= statuses else "pending"
json.dump({"schema_version":1,"model":model,"status":"partial" if any(x != "completed" for x in classes.values()) else "completed","status_counts":{s:list(classes.values()).count(s) for s in sorted(set(classes.values()))},"cells":rows,"classification":classes},open(out,"w"),indent=2); open(out,"a").write("\n")' "$output" "$model" "$MANIFEST"
}
attempt_marker() { printf '%s/.%s-r%s-attempted' "$2" "$1" "$3"; }

preflight() {
  local info selected disk_avail
  if ! info="$(nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader 2>&1)"; then log "BLOCKED GPU preflight: nvidia-smi failed: $info"; event event preflight status blocked reason nvidia-smi-unavailable; return 78; fi
  selected="$(printf '%s\n' "$info" | awk -F', ' '$1 == 2 || $1 == 3 || $1 == 4 || $1 == 5 {print}')"
  if [[ "$(printf '%s\n' "$selected" | sed '/^$/d' | wc -l)" -ne 4 ]]; then log "BLOCKED GPU preflight: expected physical GPUs 2,3,4,5"; event event preflight status blocked reason gpu-selection; return 78; fi
  log "GPU preflight selected physical GPUs 2,3,4,5"; printf '%s\n' "$selected" >> "$STATUS_LOG"; event event preflight status verified gpu_info "$selected"
  read -r _ disk_avail < <(df -Pk "$ROOT" | awk 'NR == 2 {print $2, $4}'); log "Disk preflight: ${disk_avail} KB available"; event event disk available_kb "$disk_avail"
  [[ "$disk_avail" -ge 20971520 ]] || { log "BLOCKED disk preflight: less than 20 GiB available"; return 78; }
}

cell_dir() { printf '%s/%s' "$ARTIFACT_ROOT" "$1"; }
write_metadata() {
  local id="$1" optimizer="$2" sync="$3" model="$4" transport="$5" dir; dir="$(cell_dir "$id")"; mkdir -p "$dir/profiler"; touch "$dir/stdout.log" "$dir/stderr.log"
  if [[ ! -e "$dir/config.yaml" ]]; then printf '%s\n' "experiment_id: $id" "optimizer: $optimizer" "sync: $sync" "model: $model" "transport: $transport" 'world_size: 4' "cuda_visible_devices: \"$GPUS\"" 'seed: 42' 'warmup_steps: 20' 'measure_steps: 100' 'local_batch: 1' 'sequence_length: 256' 'gradient_accumulation: 1' 'dtype: bfloat16' 'compile_model: false' 'arc_ratio: 0.2' 'arc_projection_rank: 4' 'arc_eta: 0.1' 'arc_start_compress_step: 0' 'validation: false' 'wandb: false' 'checkpoint: false' > "$dir/config.yaml"; fi
  if [[ ! -e "$dir/environment.txt" ]]; then { echo "experiment_id=$id"; echo "created_at=$(date -Is)"; echo "hostname=$(hostname)"; echo "git_head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"; echo "python=$PYTHON"; "$PYTHON" -c 'import torch; print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda or ""))' 2>&1 || true; echo "cuda_visible_devices=$GPUS"; echo "dtype=bfloat16"; echo "transport=$transport"; [[ "$transport" == normal ]] && echo relevant_nccl_env=unset || { echo NCCL_P2P_DISABLE=1; echo NCCL_SHM_DISABLE=0; }; } > "$dir/environment.txt"; fi
}

valid_timing() { "$PYTHON" "$VALIDATOR" --path "$1" --root "$ROOT" --kind timing --timing-samples 100 --experiment-id "$2" --optimizer "$3" --sync "$4" --model "$5" --transport "$6" >/dev/null 2>&1; }
valid_profile() { "$PYTHON" "$VALIDATOR" --path "$1" --root "$ROOT" --kind profile --experiment-id "$2" --optimizer "$3" --sync "$4" --model "$5" --transport "$6" >/dev/null 2>&1; }
find_valid() {
  local kind="$1" id="$2" optimizer="$3" sync="$4" model="$5" transport="$6" rep="$7" dir pattern path; dir="$(cell_dir "$id")"; [[ "$kind" == timing ]] && pattern="$dir/timing-r${rep}*.json" || pattern="$dir/profiler/profile-r${rep}*-summary.json"; shopt -s nullglob
  for path in $pattern; do
  if [[ "$kind" == timing ]]; then valid_timing "$path" "$id" "$optimizer" "$sync" "$model" "$transport"; else valid_profile "$path" "$id" "$optimizer" "$sync" "$model" "$transport"; fi
    [[ "$?" == 0 ]] && { shopt -u nullglob; printf '%s' "$path"; return 0; }
  done
  shopt -u nullglob; return 1
}
new_path() { local kind="$1" id="$2" rep="$3" dir base trace; dir="$(cell_dir "$id")"; [[ "$kind" == timing ]] && base="$dir/timing-r${rep}.json" || base="$dir/profiler/profile-r${rep}-summary.json"; trace="${base%-summary.json}-rank0.json"; [[ ! -e "$base" && ( "$kind" == timing || ! -e "$trace" ) ]] && printf '%s' "$base" || printf '%s' "${base%.json}-retry-${RUN_TAG}.json"; }
remember_command() { local dir="$1" cmd="$2"; touch "$dir/command.txt"; grep -Fqx -- "$cmd" "$dir/command.txt" || printf '%s\n' "$cmd" >> "$dir/command.txt"; }

run_one() {
  local id="$1" optimizer="$2" sync="$3" kind="$4" rep="$5" dir="$6" output="$7" command="$8" out="$dir/.${kind}-r${rep}-${RUN_TAG}.stdout" err="$dir/.${kind}-r${rep}-${RUN_TAG}.stderr" rc
  remember_command "$dir" "$command"; log "START $id $kind-r$rep"; [[ "$DRY_RUN" == 1 ]] && { log "DRY-RUN $command"; return 0; }
  bash -c "$command" > "$out" 2> "$err"; rc=$?; cat "$out" >> "$dir/stdout.log"; cat "$err" >> "$dir/stderr.log"
  if [[ "$rc" != 0 ]]; then if grep -Eiq 'CUDA out of memory|out of memory|cuda error: out of memory' "$out" "$err"; then record_oom "$optimizer" "$sync"; cell_event "$id" oom "$output" "actual CUDA OOM during $kind-r$rep"; else cell_event "$id" failed "$output" "$kind-r$rep exit=$rc"; log "FAIL $id $kind-r$rep exit=$rc"; fi; return "$rc"; fi
  log "END $id $kind-r$rep exit=0"; return 0
}

run_timing() {
  local id="$1" optimizer="$2" sync="$3" model="$4" transport="$5" rep="$6" dir output prefix command existing rc
  dir="$(cell_dir "$id")"; write_metadata "$id" "$optimizer" "$sync" "$model" "$transport"
  if mode_oom "$optimizer" "$sync"; then cell_event "$id" skipped "" "corresponding mode OOM"; return; fi
  if existing="$(find_valid timing "$id" "$optimizer" "$sync" "$model" "$transport" "$rep")"; then cell_event "$id" timing_valid "$existing" reused; log "SKIP $id timing-r$rep already valid"; return; fi
  if [[ -e "$(attempt_marker timing "$dir" "$rep")" || -e "$dir/timing-r${rep}.json" || -e "$dir/timing-r${rep}-retry-"* ]]; then : > "$(attempt_marker timing "$dir" "$rep")"; cell_event "$id" invalid "$dir/timing-r${rep}.json" "existing launched attempt failed validation; no retry"; return; fi
  output="$(new_path timing "$id" "$rep")"; [[ "$transport" == p2p_disabled ]] && prefix="env -u NCCL_DEBUG CUDA_VISIBLE_DEVICES=$GPUS NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0" || prefix="env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE CUDA_VISIBLE_DEVICES=$GPUS"
  [[ "$DRY_RUN" == 0 ]] && : > "$(attempt_marker timing "$dir" "$rep")"
  command="$prefix $TORCHRUN --standalone --nproc-per-node=4 --module benchmark.compressed_muon.benchmark_arc_2x2 --experiment-id $id --optimizer $optimizer --sync $sync --model $model --warmup-steps 20 --measure-steps 100 --seed 42 --world-size 4 --local-batch 1 --sequence-length 256 --gradient-accumulation 1 --transport $transport --no-compile-model --output $output"; run_one "$id" "$optimizer" "$sync" timing "$rep" "$dir" "$output" "$command"; rc=$?; if [[ "$DRY_RUN" == 0 && "$rc" == 0 ]]; then if [[ ! -s "$output" ]]; then cell_event "$id" invalid "$output" "successful launch produced no JSON"; elif valid_timing "$output" "$id" "$optimizer" "$sync" "$model" "$transport"; then cell_event "$id" timing_valid "$output"; else cell_event "$id" invalid "$output" "timing JSON failed validation; no retry"; fi; fi
}
run_profile() {
  local id="$1" optimizer="$2" sync="$3" model="$4" transport="$5" rep="$6" dir output trace prefix command existing rc
  dir="$(cell_dir "$id")"; write_metadata "$id" "$optimizer" "$sync" "$model" "$transport"
  if mode_oom "$optimizer" "$sync"; then cell_event "$id" skipped "" "corresponding mode OOM"; return; fi
  if existing="$(find_valid profile "$id" "$optimizer" "$sync" "$model" "$transport" "$rep")"; then cell_event "$id" profile_valid "$existing" reused; log "SKIP $id profile-r$rep already valid"; return; fi
  if [[ -e "$(attempt_marker profile "$dir" "$rep")" || -e "$dir/profiler/profile-r${rep}-summary.json" || -e "$dir/profiler/profile-r${rep}-summary-retry-"* ]]; then cell_event "$id" invalid "$dir/profiler/profile-r${rep}-summary.json" "existing launched attempt failed validation; no retry"; return; fi
  output="$(new_path profile "$id" "$rep")"; trace="${output%-summary.json}-rank0.json"; [[ "$transport" == p2p_disabled ]] && prefix="env -u NCCL_DEBUG CUDA_VISIBLE_DEVICES=$GPUS NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0" || prefix="env -u NCCL_DEBUG -u NCCL_P2P_DISABLE -u NCCL_SHM_DISABLE CUDA_VISIBLE_DEVICES=$GPUS"
  [[ "$DRY_RUN" == 0 ]] && : > "$(attempt_marker profile "$dir" "$rep")"
  command="$prefix $TORCHRUN --standalone --nproc-per-node=4 --module benchmark.compressed_muon.benchmark_arc_2x2 --experiment-id $id --optimizer $optimizer --sync $sync --model $model --warmup-steps 3 --measure-steps 5 --smoke --profile --seed 42 --world-size 4 --local-batch 1 --sequence-length 256 --gradient-accumulation 1 --transport $transport --output $output --profile-output $trace"; run_one "$id" "$optimizer" "$sync" profile "$rep" "$dir" "$output" "$command"; rc=$?; if [[ "$DRY_RUN" == 0 && "$rc" == 0 ]]; then if [[ ! -s "$output" ]]; then cell_event "$id" invalid "$output" "successful launch produced no profiler JSON"; elif valid_profile "$output" "$id" "$optimizer" "$sync" "$model" "$transport"; then cell_event "$id" profile_valid "$output"; else cell_event "$id" invalid "$output" "profiler JSON failed validation; no retry"; fi; fi
}

cell_id() { local model="$1" transport="$2" suffix="$3" base; [[ "$transport" == normal ]] && case "$model" in gpt130m) base=CM004;; gpt350m) base=CM006;; gpt1b) base=CM008;; esac || case "$model" in gpt130m) base=CM005;; gpt350m) base=CM007;; gpt1b) base=CM009;; esac; case "$suffix" in a) echo "${base}a-adamw-dense-$model-ddp-ws4-s42";; b) echo "${base}b-m001-adamw-arc-$model-ddp-ws4-s42";; c) echo "${base}c-muon-dense-$model-ddp-ws4-s42";; d) echo "${base}d-m001-muon-arc-$model-ddp-ws4-s42";; esac; }
run_matrix() {
  local kind="$1" model="$2" transport="$3" rep suffix id optimizer sync; for rep in 1 2 3; do case "$rep" in 1) order=(a b c d);; 2) order=(d c b a);; 3) order=(b c d a);; esac; for suffix in "${order[@]}"; do id="$(cell_id "$model" "$transport" "$suffix")"; case "$suffix" in a) optimizer=adamw; sync=dense;; b) optimizer=adamw; sync=arc;; c) optimizer=muon; sync=dense;; d) optimizer=muon; sync=arc;; esac; [[ "$kind" == timing ]] && run_timing "$id" "$optimizer" "$sync" "$model" "$transport" "$rep" || run_profile "$id" "$optimizer" "$sync" "$model" "$transport" "$rep"; done; done
}

run_probe() {
  local optimizer="$1" sync="$2" world="$3" suffix="$4" id="CM008-probe${suffix}-${optimizer}-${sync}-gpt1b-ddp-ws${world}-s42" dir="$ARTIFACT_ROOT/probes/gpt1b-${optimizer}-${sync}-ws${world}" output command out err rc marker="$ARTIFACT_ROOT/probes/.${optimizer}-${sync}-ws${world}-attempted"; mkdir -p "$dir"; touch "$dir/stdout.log" "$dir/stderr.log"
  if mode_oom "$optimizer" "$sync"; then cell_event "$id" skipped "" "corresponding mode OOM sentinel"; return; fi
  if [[ -s "$dir/probe.json" ]] && "$PYTHON" "$VALIDATOR" --path "$dir/probe.json" --root "$ROOT" --kind timing --timing-samples 1 --expected-world-size "$world" --experiment-id "$id" --optimizer "$optimizer" --sync "$sync" --model gpt1b --transport normal >/dev/null 2>&1; then cell_event "$id" probe_valid "$dir/probe.json" reused; return; fi
  if [[ -e "$marker" ]]; then cell_event "$id" invalid "$dir/probe.json" "existing launched probe failed validation; no retry"; return; fi
  output="$dir/probe.json"; [[ "$world" == 1 ]] && command="env -u NCCL_DEBUG CUDA_VISIBLE_DEVICES=2 $PYTHON -m benchmark.compressed_muon.benchmark_arc_2x2 --experiment-id $id --optimizer $optimizer --sync $sync --model gpt1b --warmup-steps 1 --measure-steps 1 --smoke --seed 42 --world-size 1 --local-batch 1 --sequence-length 256 --gradient-accumulation 1 --transport normal --no-compile-model --output $output" || command="env -u NCCL_DEBUG CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --standalone --nproc-per-node=4 --module benchmark.compressed_muon.benchmark_arc_2x2 --experiment-id $id --optimizer $optimizer --sync $sync --model gpt1b --warmup-steps 1 --measure-steps 1 --smoke --seed 42 --world-size 4 --local-batch 1 --sequence-length 256 --gradient-accumulation 1 --transport normal --no-compile-model --output $output"; remember_command "$dir" "$command"; log "START probe $id"; [[ "$DRY_RUN" == 1 ]] && { log "DRY-RUN $command"; return; }; [[ "$DRY_RUN" == 0 ]] && : > "$marker"; out="$dir/.probe-${RUN_TAG}.stdout"; err="$dir/.probe-${RUN_TAG}.stderr"; bash -c "$command" > "$out" 2> "$err"; rc=$?; cat "$out" >> "$dir/stdout.log"; cat "$err" >> "$dir/stderr.log"; if [[ "$rc" != 0 ]]; then grep -Eiq 'CUDA out of memory|out of memory|cuda error: out of memory' "$out" "$err" && { record_oom "$optimizer" "$sync"; cell_event "$id" oom "$output" "actual CUDA OOM during probe"; } || cell_event "$id" failed "$output" "probe exit=$rc"; return "$rc"; fi; "$PYTHON" "$VALIDATOR" --path "$output" --root "$ROOT" --kind timing --timing-samples 1 --expected-world-size "$world" --experiment-id "$id" --optimizer "$optimizer" --sync "$sync" --model gpt1b --transport normal >/dev/null 2>&1 && cell_event "$id" probe_valid "$output" || cell_event "$id" failed "$output" "probe JSON failed validation"; log "END probe $id exit=0"
}

summary_inputs() { local kind="$1" id="$2" optimizer="$3" sync="$4" model="$5" transport="$6" rep path; for rep in 1 2 3; do [[ "$kind" == timing ]] && path="$(find_valid timing "$id" "$optimizer" "$sync" "$model" "$transport" "$rep" || true)" || path="$(find_valid profile "$id" "$optimizer" "$sync" "$model" "$transport" "$rep" || true)"; [[ -n "$path" ]] || return 1; printf '%s\n' "$path"; done; }
make_summary() {
  local model="$1" optimizer="$2" transport="$3" dense_id arc_id output timing_text profile_text rc
  if [[ "$optimizer" == adamw ]]; then dense_id="$(cell_id "$model" "$transport" a)"; arc_id="$(cell_id "$model" "$transport" b)"; else dense_id="$(cell_id "$model" "$transport" c)"; arc_id="$(cell_id "$model" "$transport" d)"; fi
  output="$ARTIFACT_ROOT/${model}-${optimizer}-${transport}-summary.json"; [[ -s "$output" ]] && return
  timing_text="$(summary_inputs timing "$dense_id" "$optimizer" dense "$model" "$transport"; summary_inputs timing "$arc_id" "$optimizer" arc "$model" "$transport")" || { log "PARTIAL summary unavailable $model $optimizer $transport"; return; }
  profile_text="$(summary_inputs profile "$dense_id" "$optimizer" dense "$model" "$transport"; summary_inputs profile "$arc_id" "$optimizer" arc "$model" "$transport")" || { log "PARTIAL profiler summary unavailable $model $optimizer $transport"; return; }
  mapfile -t timing <<< "$timing_text"; mapfile -t profile <<< "$profile_text"
  [[ "${#timing[@]}" == 6 && "${#profile[@]}" == 6 ]] || { log "PARTIAL summary requires exactly 6 timing + 6 profiler inputs: $model $optimizer $transport"; return; }
  "$PYTHON" -m benchmark.compressed_muon.summarize_arc_2x2 "${timing[@]}" --profiles "${profile[@]}" --output "$output"; rc=$?; [[ "$rc" == 0 || "$rc" == 2 ]] && log "SUMMARY $output exit=$rc (CV evidence only)" || log "SUMMARY FAILED $output exit=$rc"
}
make_comparison() { local model="$1" optimizer="$2" output="$ARTIFACT_ROOT/${model}-${optimizer}-transport-comparison-summary.json" normal="$ARTIFACT_ROOT/${model}-${optimizer}-normal-summary.json" restricted="$ARTIFACT_ROOT/${model}-${optimizer}-p2p_disabled-summary.json"; [[ -s "$output" || ! -s "$normal" || ! -s "$restricted" ]] && return; "$PYTHON" -c 'import json,sys; o,m,p,n,r=sys.argv[1:]; json.dump({"schema_version":1,"model":m,"optimizer":p,"transport":"comparison","normal":json.load(open(n)),"p2p_disabled":json.load(open(r))},open(o,"w"),indent=2); open(o,"a").write("\n")' "$output" "$model" "$optimizer" "$normal" "$restricted"; }
finalize_state() {
  local state
  for model in "${MODELS[@]}"; do write_partial "$model"; done
  state="$($PYTHON -c 'import json,sys; planned=[json.loads(x).get("id") for x in open(sys.argv[1]) if x.strip() and json.loads(x).get("event")=="planned"]; events=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; rows={}; [rows.setdefault(e.get("id"),set()).add(e.get("status")) for e in events if e.get("event")=="cell"]; ids=[x for x in planned if x and x[5:6] != "-"]; print("completed" if len(ids)==24 and all({"timing_valid","profile_valid"} <= rows.get(x,set()) and not rows.get(x,set()) & {"invalid","failed","oom","skipped"} for x in ids) else "partial")' "$MANIFEST")"
  event event launcher status "$state"; [[ "$state" == completed ]] && { : > "$ARTIFACT_ROOT/scale-to-1b-complete"; log "COMPLETE scale-out launcher"; } || log "PARTIAL scale-out launcher: see per-model partial manifests"
}

main() {
  while [[ "$#" -gt 0 ]]; do case "$1" in --dry-run) DRY_RUN=1;; --resume) :;; --reconcile-only) RECONCILE_ONLY=1;; *) echo "unknown argument: $1" >&2; return 2;; esac; shift; done
  init_manifest; reconcile_manifest; [[ "$RECONCILE_ONLY" == 1 ]] && return 0; event event launcher status running status_log "$STATUS_LOG"; [[ "$DRY_RUN" == 1 ]] && { log "DRY-RUN requested"; run_matrix timing gpt130m normal; run_matrix profile gpt130m normal; return 0; }
  preflight || return $?; for model in gpt130m gpt350m; do for transport in "${TRANSPORTS[@]}"; do run_matrix timing "$model" "$transport"; done; done
  for world in 1 4; do for suffix in a b c d; do case "$suffix" in a) optimizer=adamw; sync=dense;; b) optimizer=adamw; sync=arc;; c) optimizer=muon; sync=dense;; d) optimizer=muon; sync=arc;; esac; run_probe "$optimizer" "$sync" "$world" "$suffix" || true; done; done
  for transport in "${TRANSPORTS[@]}"; do run_matrix timing gpt1b "$transport"; done; preflight || return $?; for model in "${MODELS[@]}"; do for transport in "${TRANSPORTS[@]}"; do run_matrix profile "$model" "$transport"; done; done
  for model in "${MODELS[@]}"; do for optimizer in adamw muon; do for transport in "${TRANSPORTS[@]}"; do make_summary "$model" "$optimizer" "$transport"; done; make_comparison "$model" "$optimizer"; done; done
  finalize_state; event event launcher finished_at "$(date -Is)"
}
main "$@"
