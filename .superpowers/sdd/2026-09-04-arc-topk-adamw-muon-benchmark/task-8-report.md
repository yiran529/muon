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
