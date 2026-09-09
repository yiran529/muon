# CM035 Paper AdamW Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a paired GPT-60M dense/EF21M all-2D hook experiment that changes CM034 only to the public paper-like AdamW recipe.

**Architecture:** Parameterize AdamW betas/epsilon, exact-step warmup, scheduler kind, and gradient clipping in the shared training path while preserving all old defaults. Add CM035 dense and ARC configs plus a fail-closed serial launcher derived from CM034; ARC remains EF21M with unchanged compression settings.

**Tech Stack:** Python, PyTorch AdamW/DDP, pytest, YAML, Bash/torchrun.

**Spec:** User-approved chat design from 2026-09-10; this phase intentionally stops after launching the paired run.

## Global Constraints

- Preserve existing defaults and CM034 reproducibility.
- CM035 recipe is AdamW betas `(0.9, 0.999)`, epsilon `1e-8`, gradient clipping norm `1.0`, exactly 1000 warmup steps, cosine decay, and weight decay `0.0`.
- CM035 uses learning rate `0.001`; no LR sweep in this phase.
- ARC remains EF21M, all `ndim == 2`, ratio `0.2`, projection rank `4`, eta `1.0`, seed `42`, compression after step `1000`, bucket cap `160 MiB`.
- Dense and ARC must use the same device batch and gradient accumulation selected before either formal cell.
- Do not implement EF14, boundary ablations, or follow-on experiments in this phase.
- After background launch is confirmed, stop without monitoring training.

---

### Task 1: Parameterize the paper-like AdamW training recipe

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train_adamw_builder.py`
- Create: `tests/test_train_schedule_and_clipping.py`

**Interfaces:**
- Produces: `Hyperparameters.adam_beta1`, `adam_beta2`, `adam_eps`, `warmup_steps`, `lr_schedule`, and `grad_clip_norm`.
- Produces: shared LR multiplier and gradient-norm/clipping helpers used by the training loop.

- [ ] Write failing tests for configurable AdamW groups, exact 1000-step cosine warmup/decay, old linear defaults, and clipping disabled/enabled behavior.
- [ ] Run focused tests and confirm failures are caused by missing configuration behavior.
- [ ] Implement the minimal shared configuration and helpers, preserving old defaults.
- [ ] Run focused and existing training factory/config tests.
- [ ] Commit the production and test changes.

### Task 2: Add CM035 paired configs and serial launcher

**Files:**
- Create: `configs/compressed_muon/cm035a_dense_adamw_gpt60m_paper_recipe.yaml`
- Create: `configs/compressed_muon/cm035b_all2d_hook_adamw_gpt60m_paper_recipe.yaml`
- Create: `benchmark/compressed_muon/run_cm035_adamw_paper_recipe_quality.sh`
- Create: `tests/test_cm035_adamw_paper_recipe_launcher.py`
- Modify: `docs/worklog/M001-arc-topk-ef21m-muon.md`

**Interfaces:**
- Consumes: Task 1 hyperparameter fields.
- Produces: a two-cell dense-then-ARC controller with common OOM fallback and recorded summary.

- [ ] Write failing config/launcher contract tests proving the exact recipe, unchanged ARC fields, paired batch selection, token budget, and side-effect-free `--print-plan`.
- [ ] Run the tests and confirm the CM035 files are absent.
- [ ] Add configs and adapt the CM034 controller with a new experiment ID and explicit recipe fields.
- [ ] Run launcher tests, YAML/config parsing, `bash -n`, `--print-plan`, and `git diff --check`.
- [ ] Record the launch design and commit.

### Task 3: Integrate and launch, then pause

**Files:**
- No source changes expected after merge.
- Runtime output: `artifacts/compressed_muon/CM035-*`.

**Interfaces:**
- Consumes: Task 2 launcher.
- Produces: one detached background serial controller.

- [ ] Run the relevant CPU regression suite on the feature branch.
- [ ] Fast-forward merge the feature branch into `main` and verify static checks on the merged tree.
- [ ] Confirm the registered artifact root is absent, W&B authentication is available, data/venv paths exist, and the selected four GPUs satisfy the launcher's free-memory gate.
- [ ] Start the serial launcher using `nohup setsid`, verify only that the controller process initially exists, and report PID/artifact path.
- [ ] Stop work immediately; do not poll, summarize, implement EF14, or launch another experiment.
