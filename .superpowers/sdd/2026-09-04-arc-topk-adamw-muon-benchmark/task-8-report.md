# Task 8 report: serial ARC-TopK scale-out

Status: `DONE_WITH_CONCERNS`

## Implementation

- Added `artifacts/compressed_muon/scale_to_1b_launcher.sh`, a serial and resumable launcher for CM004–CM009.
- Added `artifacts/compressed_muon/validate_scale_to_1b.py` for identity/configuration, timing-sample, finite-value, checksum/signature, profiler-category, trace, byte-accounting, and unattributed-NCCL validation.
- Registered the 24 formal rows in `docs/compressed_muon/EXPERIMENTS.md` as `running`.
- Formal values are BF16, local batch 1, sequence length 256, gradient accumulation 1, seed 42, no compile, 20 warmup + 100 measured steps; ARC ratio/rank/eta/start are 0.2/4/0.1/0.
- The launcher rotates order `(a,b,c,d)`, `(d,c,b,a)`, `(b,c,d,a)`, uses fresh torchrun processes, preserves valid JSON/traces, records failed/OOM attempts, gates only the corresponding optimizer/sync mode on actual CUDA OOM, and emits six summary JSONs per model when all six timing and six profiler inputs are valid.
- GPT-1B probes run as single-process and four-rank smoke cells before GPT-1B formal cells.

## Validation and launch

- `bash -n artifacts/compressed_muon/scale_to_1b_launcher.sh`: passed.
- `python -m py_compile artifacts/compressed_muon/validate_scale_to_1b.py`: passed.
- `git diff --check`: passed.
- Existing benchmark `parse_args` accepted all 24 formal IDs under both transport environments.
- `./artifacts/compressed_muon/scale_to_1b_launcher.sh --dry-run`: exit 0; verified rotated commands, 12 timing and 12 profiler invocations in the exercised matrix, exact formal step settings, and GPU list `2,3,4,5`.
- Detached tmux launch: `m001_arc_topk_scale_to_1b`.
- Launch status log: `artifacts/compressed_muon/scale-to-1b-status-20260904T222524+0800.log`.
- Machine-readable event manifest: `artifacts/compressed_muon/scale-to-1b-manifest.jsonl`.
- At report time, CM004a timing-r1 and CM004b timing-r1 exited 0; CM004c timing-r1 was running.

## GPU and disk evidence

Preflight recorded physical GPUs 2–5 and their UUIDs:

```text
2 GPU-e6622753-895a-f8fc-6082-ac71bbfa0037
3 GPU-f7d3c0fb-aed7-235f-7332-68966b63e0c5
4 GPU-0bc7caf4-d72c-c6b9-a9c1-60d2ff5779c4
5 GPU-76169292-9c3b-c2e1-682b-bacd97a2f23c
```

Free disk at launch was `67,082,040 KB` (about 64 GB). A concurrent `nvidia-smi` check showed the benchmark process on GPUs 2–5; GPUs 0/1 retained their pre-existing processes and were not selected.

## Concerns

- The detached benchmark is intentionally still running; final timing/profile summaries and any OOM outcomes are not available in this handoff. Resume with the same launcher if the tmux process is interrupted.
- Raw artifacts, traces, event manifest, sentinels, and timestamped logs are runtime outputs under the ignored `artifacts/` tree and are not committed.

## Review round 1 fix (2026-09-04)

- Paused only tmux session `m001_arc_topk_scale_to_1b`; no other process was stopped.
- Root cause confirmed for GPT-130M Muon-dense: all six timing JSONs (CM004c/CM005c, r1–r3) have `parameter_checksum_agreement=false` while collective signatures match. The validator now rejects these artifacts; evidence remains untouched and the launcher records the cells as `invalid` without changing optimizer/benchmark math.
- Fixed launcher success handling: a zero exit with missing/invalid JSON emits an explicit machine-readable `invalid` event and does not retry an already-launched repetition. Existing invalid outputs and traces are preserved.
- Added stale-manifest superseded/corrected events for the old `CM004-a` style IDs, per-model partial JSON output, ROOT working-directory setup, explicit `NCCL_DEBUG` unsetting, metadata creation before OOM skips, and probe OOM sentinels/attempt markers.
- Profile retry naming now detects an existing rank-0 trace and chooses a retry summary/trace pair, so a valid trace cannot be overwritten.
- Fresh checks: shell syntax, Python compilation, whitespace check, strict invalid-artifact rejection, corrected-event emission, and dry-run all passed. The dry-run emitted no torchrun processes and retained the required rotated order.
- The original live outputs already used the intended three launches per timing repetition; resume will not add replacement attempts for invalid Muon-dense repetitions.
- Additional bounded-state fix: completion now requires both `timing_valid` and `profile_valid` for every corrected formal ID and rejects any cell with invalid/failed/OOM/skipped history; per-model partial manifests classify each cell as `completed`, `skipped`, `oom`, `invalid`, or `pending`.
- Final fix checks: `bash -n`, validator `py_compile`, `git diff --check`, strict rejection of a Muon-dense artifact (exit 2), outside-repository `--dry-run` (exit 0), corrected supersession events (24), and no retry JSONs were observed.
- Resumed status log: `artifacts/compressed_muon/scale-to-1b-status-20260904T224243+0800.log`; at the latest check CM006b GPT-350M timing-r1 was running after valid CM004/CM005 outputs were reused.

## Review round 2 fix (2026-09-04)

- Paused only `m001_arc_topk_scale_to_1b` while fixing manifest migration; no other process was stopped.
- Reworked `reconcile_manifest` to independently ensure all 24 exact corrected `planned` IDs and to add each exact stale-ID → corrected-ID `superseded` mapping at most once. Removed the global superseded-event early return, so stale-only and partially reconciled manifests converge on resume. Existing raw artifacts and event history are preserved.
- Added `--reconcile-only` plus `SCALE_TO_1B_ARTIFACT_ROOT`, `SCALE_TO_1B_MANIFEST`, and `SCALE_TO_1B_STATUS_LOG` overrides solely to support isolated, reproducible migration checks without touching the live manifest.
- Focused stale-only fixture: corrected planned=24, exact superseded mappings=24, second reconciliation unchanged at 73 lines.
- Focused partially reconciled fixture (one corrected planned row and one mapping pre-existing): corrected planned=24, exact superseded mappings=24, second reconciliation unchanged at 73 lines.
- Live manifest reconciliation on resume will add missing corrected planned rows and exact mappings; malformed historical `superseded_id` evidence is retained.

## Review round 3 fix (2026-09-04)

- Left the running `m001_arc_topk_scale_to_1b` tmux/GPU job untouched.
- Supersession deduplication now recognizes an exact legacy event with `id=<stale ID>` as well as the canonical `stale_id=<stale ID>`, while requiring the matching exact `corrected_id`; missing canonical mappings are still appended.
- Legacy-ID fixture check: 24 corrected planned rows and 24 existing legacy mappings were preserved, no duplicate canonical mappings were added, and the second reconciliation was unchanged at 73 lines.
- `bash -n` and `git diff --check` passed.
