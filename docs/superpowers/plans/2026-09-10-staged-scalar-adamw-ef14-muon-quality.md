# Staged Scalar AdamW, EF14, and Muon Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple auxiliary AdamW hyperparameters from Muon, then run three ordered GPT-60M dense/ARC pairs that separately introduce the scalar recipe, EF14, and warmup/cosine/clipping.

**Architecture:** Add explicit scalar-group hyperparameters to the shared training configuration and construct Muon auxiliary groups through one shared helper used by dense and ARC entry points. Preserve existing behavior when the new fields are unset. A fail-closed controller runs CM037, CM038, and CM039 sequentially; every stage contains a contemporaneous dense/ARC pair and must succeed before the next stage starts.

**Tech Stack:** Python, PyTorch Muon/AdamW, DDP communication hooks, pytest, YAML, Bash/torchrun.

**Spec:** User-approved ordered experiment sequence in chat on 2026-09-10.

## Global Constraints

- Work directly in `/home/wyr/dion`; do not create a worktree.
- Every formal cell is GPT-60M, BF16, compile enabled, 4-GPU DDP, FineWeb10B, sequence length 256, global batch 512, seed 42, 8393 updates, and 10,485,760 validation tokens.
- Muon matrix settings remain `lr=0.02`, `mu=0.95`, Nesterov enabled, `adjust_lr=spectral_norm`, and weight decay `0.01`.
- Auxiliary embedding/lm_head AdamW settings are `scalar_lr=0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `0`.
- ARC uses the all-2D DDP hook, ratio `0.2`, projection rank `4`, eta `1.0`, seed `42`, compression after step `1000`, and bucket cap `160 MiB`.
- CM037 uses EF21M, no LR warmup, the existing final-20% linear decay, and no clipping.
- CM038 changes only the ARC cell from EF21M to EF14; its dense cell is rerun contemporaneously.
- CM039 retains EF14 and adds exact 1000-step linear LR warmup, cosine decay to zero, and global grad-norm clipping at `1.0` to both cells.
- Each stage probes one common physical device batch/GA for its two cells, then runs dense before ARC. A failed probe or formal cell stops the controller before later stages.
- Never overwrite an existing artifact root, never fall back after a non-OOM probe error, and record command/config/commit/GPU state/exit code for every cell.

---

### Task 1: Correct and decouple auxiliary AdamW parameter groups

**Files:**
- Modify: `train.py`
- Modify: `train_arctopk.py`
- Modify: `tests/test_train_arctopk.py`
- Modify: `tests/test_train_factories.py`

**Interfaces:**
- Produces: `Hyperparameters.scalar_lr: Optional[float]`, `scalar_adam_beta1: Optional[float]`, `scalar_adam_beta2: Optional[float]`, `scalar_adam_eps: Optional[float]`, and `scalar_weight_decay: float`.
- Produces: `build_muon_param_groups(model, hp) -> list[dict]`, shared by dense Muon and ARC Muon.
- Preserves: when `scalar_lr is None`, auxiliary groups use `hp.lr`; legacy Adam moments remain Muon defaults `(0.9, 0.95)` and epsilon `1e-8` unless explicitly overridden.

- [x] **Step 1: Add a failing optimizer-group behavior test**

  Construct a stub GPT with transformer, embedding, and lm_head parameters. Set `lr=0.02`, `scalar_lr=0.001`, `scalar_adam_beta1=0.9`, `scalar_adam_beta2=0.999`, `scalar_adam_eps=1e-8`, and `scalar_weight_decay=0`. Build both dense and ARC Muon optimizers and assert their actual auxiliary param groups expose `lr=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, and `weight_decay=0`; assert the matrix group remains Muon at LR `0.02` and weight decay `0.01`.

- [x] **Step 2: Run the new test and confirm RED**

  Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_train_arctopk.py tests/test_train_factories.py`

  Expected: FAIL because the scalar fields/helper do not exist and current mixed groups incorrectly set the unused `betas` key instead of Muon's consumed `beta1`/`beta2` keys.

- [x] **Step 3: Add configuration fields, CLI arguments, validation, and shared grouping**

  Add the five fields to `Hyperparameters`, add matching CLI options, and implement `build_muon_param_groups`. For AdamW auxiliary groups, write Muon's consumed keys `beta1`, `beta2`, and `epsilon`; do not write a dead `betas` key. Use the helper from both `train.init_optimizer` and `train_arctopk.init_arc_topk_optimizer`. Validate finite positive LR/epsilon, betas in `[0,1)`, and nonnegative weight decay before optimizer construction.

- [x] **Step 4: Verify GREEN and legacy behavior**

  Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_train_arctopk.py tests/test_train_factories.py tests/test_train_adamw_builder.py`

  Expected: all pass; pure AdamW continues to use global `adam_beta1/adam_beta2/adam_eps`, not the mixed-Muon scalar fields.

- [x] **Step 5: Commit Task 1**

  ```bash
  git add train.py train_arctopk.py tests/test_train_arctopk.py tests/test_train_factories.py
  git commit -m "feat: decouple Muon auxiliary AdamW settings"
  ```

### Task 2: Register the three ordered 2-cell GPT-60M stages

**Files:**
- Create: `configs/compressed_muon/cm037a_dense_muon_scalar_adamw.yaml`
- Create: `configs/compressed_muon/cm037b_ef21m_muon_scalar_adamw.yaml`
- Create: `configs/compressed_muon/cm038a_dense_muon_scalar_adamw_repeat.yaml`
- Create: `configs/compressed_muon/cm038b_ef14_muon_scalar_adamw.yaml`
- Create: `configs/compressed_muon/cm039a_dense_muon_scalar_adamw_warmup_cosine_clip.yaml`
- Create: `configs/compressed_muon/cm039b_ef14_muon_scalar_adamw_warmup_cosine_clip.yaml`
- Create: `tests/test_cm037_cm039_staged_muon_quality_launcher.py`

**Interfaces:**
- CM037 pair: scalar recipe plus dense/EF21M.
- CM038 pair: exact CM037 settings plus ARC error feedback EF14; repeat dense.
- CM039 pair: exact CM038 settings plus `warmup_steps=1000`, `lr_schedule=cosine`, `warmdown_ratio=0`, and `grad_clip_norm=1.0` on both cells.

- [x] **Step 1: Write failing configuration differential tests**

  Parse all six YAML files and assert literal common model/training/Muon/scalar values. Assert CM037b is CM037a plus ARC EF21M fields, CM038a equals CM037a except identity, CM038b differs from CM037b only in identity and `arc_error_feedback=ef14`, and CM039 adds only the declared schedule/clipping fields to the corresponding CM038 mode.

- [x] **Step 2: Run tests and confirm RED**

  Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_cm037_cm039_staged_muon_quality_launcher.py`

  Expected: FAIL because the stage configs do not exist.

- [x] **Step 3: Add the six literal YAML configurations**

  Use `scalar_lr: 0.001`, `scalar_adam_beta1: 0.9`, `scalar_adam_beta2: 0.999`, `scalar_adam_eps: 1e-8`, and `scalar_weight_decay: 0.0` in every file. Keep compression start at 1000 in CM039 so the third stage changes only schedule/clipping.

- [x] **Step 4: Run config tests and confirm GREEN**

  Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_cm037_cm039_staged_muon_quality_launcher.py`

  Expected: configuration differential tests pass.

### Task 3: Build the fail-closed CM037–CM039 serial controller

**Files:**
- Create: `benchmark/compressed_muon/run_cm037_cm039_staged_muon_quality.sh`
- Modify: `tests/test_cm037_cm039_staged_muon_quality_launcher.py`

**Interfaces:**
- Produces: `--print-plan` JSON listing exactly three ordered stages and two cells per stage.
- Produces: artifact root `artifacts/compressed_muon/CM037-CM039-m001-staged-scalar-adamw-ef14-muon-gpt60m-ws4-s42/`, with one subdirectory per formal cell and a parent summary after all six succeed.

- [x] **Step 1: Add failing executable controller tests**

  Execute `--print-plan` and assert stage order `CM037 -> CM038 -> CM039`, exact two-cell membership, common batch/token/seed/scalar settings, EF21M only in CM037 ARC, EF14 in CM038/CM039 ARC, and schedule/clipping only in CM039. Execute a read-only synthetic summary mode against temporary cell results and assert loss/PPL/timing arithmetic and rejection of missing or non-finite metrics.

- [x] **Step 2: Run controller tests and confirm RED**

  Run: `PYTHONPATH=. .venv/bin/pytest -q tests/test_cm037_cm039_staged_muon_quality_launcher.py`

  Expected: FAIL because the controller and summary interface do not exist.

- [x] **Step 3: Implement the controller**

  Adapt the established CM035 paired-run lifecycle. For each stage in order: probe dense and ARC at the same candidate DB, select a common DB/GA, run dense formal, require exit 0 and final validation/timing/memory records, then run ARC formal and require the same. Only then advance. Install the EXIT trap after the artifact-root overwrite guard and directory creation. Record `git status`; require a clean tracked tree before formal launch.

- [x] **Step 4: Verify controller behavior**

  Run:

  ```bash
  bash -n benchmark/compressed_muon/run_cm037_cm039_staged_muon_quality.sh
  PYTHONPATH=. .venv/bin/pytest -q tests/test_cm037_cm039_staged_muon_quality_launcher.py
  benchmark/compressed_muon/run_cm037_cm039_staged_muon_quality.sh --print-plan
  ```

  Expected: syntax and tests pass; print-plan performs no GPU, W&B, data, or artifact writes.

- [ ] **Step 5: Commit Tasks 2–3**

  ```bash
  git add configs/compressed_muon/cm037*.yaml configs/compressed_muon/cm038*.yaml configs/compressed_muon/cm039*.yaml benchmark/compressed_muon/run_cm037_cm039_staged_muon_quality.sh tests/test_cm037_cm039_staged_muon_quality_launcher.py docs/superpowers/plans/2026-09-10-staged-scalar-adamw-ef14-muon-quality.md
  git commit -m "exp: add staged scalar AdamW and EF14 Muon runs"
  ```

### Task 4: Verification and launch

**Files:**
- Modify after results: `docs/worklog/M001-arc-topk-ef21m-muon.md`
- Modify after results: `docs/compressed_muon/RESULTS.md`

- [ ] **Step 1: Run repository-relevant CPU/Gloo regressions**

  Run all ARC primitive/sync/hook/future/checkpoint tests, Muon/AdamW optimizer tests, shared training parser/factory/schedule tests, and CM033–CM039 launcher tests. Record the exact pass/fail counts.

- [ ] **Step 2: Run a two-GPU NCCL smoke on idle GPUs**

  Run `tests/test_train_arctopk_ddp_hook_nccl.py -m multi_gpu` on two GPUs that are not occupied by another user or experiment. Require the EF21M Muon and EF14 Muon modes to complete.

- [ ] **Step 3: Commit any verification-only documentation changes and require a clean tree**

  Run `git diff --check`, `bash -n` on the controller, and require `git status --porcelain` to be empty before launch.

- [ ] **Step 4: Launch the detached serial controller once**

  Start the controller with its stdout/stderr redirected to a stable log, record PID/start identity, and confirm once that the process is alive. Do not start a second controller for the same artifact root.

- [ ] **Step 5: After all stages finish, record results**

  Add all six final validation losses/PPL/step times/throughput/memory values and paired deltas to the worklog and `RESULTS.md`. Interpret CM037 as the paired EF21M result under the new scalar recipe (comparison to CM027/CM033 is cross-run context, not a clean scalar-recipe causal estimate), CM038 versus CM037 as the EF14 effect with a repeated dense control, and CM039 versus CM038 as the warmup/cosine/clip effect with another repeated dense control; retain the shared-GPU and single-seed limitations.
