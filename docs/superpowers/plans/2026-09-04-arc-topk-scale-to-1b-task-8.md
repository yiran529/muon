# ARC-TopK GPT-1B Scale-Out Implementation Plan

> **For agentic workers:** This plan is executed inline by the Task 8 worker; the task brief remains the source of exact benchmark values.

**Goal:** Add a guarded, resumable serial launcher for the GPT-130M, GPT-350M, and GPT-1B ARC-TopK 2x2 matrices.

**Architecture:** A tracked shell launcher owns preflight, ordered independent torchrun jobs, metadata, valid-output skipping, OOM mode gating, manifest/sentinel state, and summarization. A small tracked Python validator checks benchmark JSON identity, timing/profiler correctness, and trace/observer accounting. Raw artifacts remain untracked under `artifacts/compressed_muon/`.

**Tech Stack:** Bash, the worktree venv's Python/torchrun, existing `benchmark.compressed_muon.benchmark_arc_2x2` and `summarize_arc_2x2` interfaces, tmux.

**Spec:** `.superpowers/sdd/2026-09-04-arc-topk-adamw-muon-benchmark/task-8-brief.md`

## Global Constraints

- Use only physical GPUs 2,3,4,5; GPU 0/1 processes are never touched.
- Formal cells use BF16, local batch 1, sequence length 256, gradient accumulation 1, seed 42, no compile, and 20 warmup + 100 measured steps.
- ARC defaults are ratio 0.2, rank 4, eta 0.1, start 0.
- Each non-OOM cell gets exactly three fresh timing and three fresh profiler launches; profiler schedule is 3 wait + 3 warmup + 5 active.
- CV above 5% is evidence only and never stops later cells.
- Only actual CUDA OOM skips a corresponding optimizer/sync mode; all independent modes continue.
- Normal transport unsets P2P/SHM overrides; restricted transport sets `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0`.

### Task 1: Guarded launcher and validator

**Files:**
- Create: `artifacts/compressed_muon/scale_to_1b_launcher.sh`
- Create: `artifacts/compressed_muon/validate_scale_to_1b.py`

- [ ] Add preflight GPU UUID/free-space capture, resumable metadata, rotated serial order, probe gating, safe output paths, and partial manifests.
- [ ] Add validator checks for timing samples, finite/checksum/signature correctness, expected profiler categories, trace existence, byte agreement, and unattributed NCCL kernels.
- [ ] Run shell syntax and dry-run checks before launch.

### Task 2: Experiment registration and launch report

**Files:**
- Modify: `docs/compressed_muon/EXPERIMENTS.md`
- Create: `.superpowers/sdd/2026-09-04-arc-topk-adamw-muon-benchmark/task-8-report.md`

- [ ] Register all 24 planned cells as `running` with exact IDs/configuration.
- [ ] Launch the guarded script in tmux and record session, manifest, status-log, and concerns.
- [ ] Commit only reproducible scripts/docs.
