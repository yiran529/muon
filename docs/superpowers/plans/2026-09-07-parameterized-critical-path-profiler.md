# Parameterized GPT Critical-Path Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parameterized, production-training-path profiler and use three exclusive GPUs to identify the approximate Dense Muon versus optimizer-side ARC critical-path hotspots without changing training semantics.

**Architecture:** The shared training loop gains an opt-in profiler that starts immediately before the final gradient-accumulation micro-step of one selected optimizer step and stops after `optimizer.step()`. Existing ARC and Muon ranges are refined to distinguish local compute, collective launch, and host wait. A reusable launcher accepts world size, global/device batch, GPU list or dynamic selection, model dimensions, and profile window parameters, then runs Dense and ARC serially and summarizes every rank's trace offline.

**Tech Stack:** Python 3.10, PyTorch 2.11 Kineto profiler, DDP/NCCL, pytest, Bash, existing FineWeb10B GPT/Muon entries.

**Spec:** `docs/worklog/M001-arc-topk-ef21m-muon.md` section “下一步判断” plus the approved 2026-09-07 three-GPU exploratory profiling design.

## Global Constraints

- Profiler runs are attribution evidence only; CM022 remains the unperturbed wall-clock source.
- Default exploratory configuration is world size 3, global batch 768, device batch 1, gradient accumulation 256, GPT-350M, sequence length 1024, seed 42, and Polar Express on both sides.
- A future four-GPU run must require only CLI changes (`--world-size 4 --global-batch-size 1024`), not script edits.
- Capture all ranks, but only the final micro-step plus optimizer of one post-compile training step.
- Do not enable `--time_optimizer`, profiler memory tracking, Python stacks, or tensor shape recording.
- Preserve collective order and tensor values; instrumentation may add named ranges and profiler-only synchronization at capture boundaries.
- Never run on a GPU with at least 1024 MiB allocated; selected GPUs are locked for both serial cells.

---

### Task 1: Add the opt-in targeted training profiler

**Files:**
- Create: `benchmark/compressed_muon/training_profiler.py`
- Create: `tests/test_training_profiler.py`
- Modify: `train.py`

**Interfaces:**
- Produces: `TrainingProfileConfig(output_dir: Path, profile_step: int)`
- Produces: `validate_training_profile_request(profile_step, num_iterations) -> None`
- Produces: `TrainingProfileCapture.maybe_start(step, micro_step, grad_accum_steps) -> bool`
- Produces per-rank `rank-<rank>.json` Chrome traces.

- [x] Write tests that reject invalid profile steps, verify capture triggers only on the selected step's final micro-step, and verify rank-specific trace paths.
- [x] Run `uv run --frozen --extra dev pytest tests/test_training_profiler.py -v` and observe RED because the module and CLI do not exist.
- [x] Implement the minimal controller and add `--profile-output-dir` / `--profile-step` to `train.py`.
- [x] Start profiling after a CUDA synchronization immediately before the selected final micro-step; add `train/profile_window`, `train/final_microstep`, `train/final_forward`, `train/final_backward`, and `train/optimizer` ranges; synchronize and export after the optimizer.
- [x] Run the focused tests and existing DDP sync tests; expect PASS.

### Task 2: Complete ARC/Muon hotspot ranges and offline attribution

**Files:**
- Modify: `dion/arc_topk.py`
- Modify: `dion/megabatch_base.py`
- Modify: `benchmark/compressed_muon/profiler_trace.py`
- Create: `tests/test_training_profiler_trace.py`

**Interfaces:**
- Produces named CPU ranges for ARC state stacking, projection, sketch compute/launch/wait, TopK, gather, selected-values launch/wait, scatter, EF21M, state copy, and Muon result collective wait.
- Produces: `summarize_training_trace(path) -> dict` with named CPU range durations, NCCL union/overlap/exposed time, collective attribution, and unattributed NCCL fraction.

- [x] Write synthetic trace tests proving default-DDP NCCL inside `train/final_backward` is classified as `ddp_gradient`, wait ranges are reported separately, and interval unions do not double count overlap.
- [x] Run the focused trace tests and observe RED on absent fields/classification.
- [x] Add only named ranges around existing operations; do not move operations or waits.
- [x] Extend offline attribution and run existing profiler trace tests plus the new tests; expect PASS.

### Task 3: Add the reusable serial launcher and summary

**Files:**
- Create: `benchmark/compressed_muon/run_training_critical_path_profiler.sh`
- Create: `benchmark/compressed_muon/summarize_training_profiles.py`
- Create: `tests/test_training_profiler_launcher.py`

**Interfaces:**
- Launcher arguments include `--artifact-root`, `--world-size`, `--global-batch-size`, `--device-batch-size`, `--gpu-list`, `--exclude-gpus`, `--model-dim`, `--layers`, `--heads`, `--sequence-length`, `--profile-step`, `--num-iterations`, `--repeats`, and `--print-plan`.
- `--gpu-list` omitted means dynamically select the lowest-numbered idle GPUs matching `--world-size`.
- Produces two serial cell directories per repeat and a `summary.json` containing per-rank and cross-repeat hotspot metrics.

- [x] Write launcher contract tests for the default 3-card plan, a 4-card override, batch divisibility rejection, GPU selection, and absence of `--time_optimizer`.
- [x] Run the launcher tests and observe RED because the launcher does not exist.
- [x] Implement argument parsing, fail-closed preflight, serial execution, durable artifacts, and offline summary.
- [x] Run launcher tests, `bash -n`, and both 3-card and 4-card `--print-plan`; expect PASS without GPU work.

### Task 4: Verify and launch the three-GPU exploratory profile

**Files:**
- Runtime output: `artifacts/compressed_muon/CM023-gpt350m-critical-path-profiler-ws3/`
- Append after completion: `docs/worklog/M001-arc-topk-ef21m-muon.md`

- [x] Run all focused CPU tests and existing ARC/Muon distributed tests.
- [x] Run a low-cost three-GPU smoke capture and require both rank traces, expected collective categories, finite loss, and zero exit status before the full pair.
- [x] Launch the parameterized script in tmux/systemd with `--world-size 3 --global-batch-size 768 --device-batch-size 1 --repeats 1` on three exclusive GPUs.
- [x] Validate all six rank traces, compute approximate hotspot accounting, and append a clearly exploratory result to the existing M001 worklog without changing `RESULTS.md`.
