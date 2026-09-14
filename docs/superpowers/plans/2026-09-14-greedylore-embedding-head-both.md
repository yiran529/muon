# GreedyLore Embedding/Head Both Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one opt-in switch that compresses both token embedding and LM-head gradients with GreedyLore while retaining their AdamW optimizer groups, then run a paired timing experiment and a GPT-60M full training experiment when four GPUs are free.

**Architecture:** The training entry point owns the communication-role decision independently of optimizer ownership. Enabling `greedy_lore_compress_embedding_lm_head` adds exactly `transformer.wte.weight` and `lm_head.weight` to the existing compressed-parameter identity set; the generic DDP hook then allocates and checkpoints their compressor state. Very wide matrices use a left-Gram eigendecomposition for refresh so no full `Vh` is materialized. A fail-fast controller waits for four GPUs, runs CM074 timing, and only then starts CM075 full training in the same tmux session.

**Tech Stack:** Python, PyTorch DDP/NCCL, pytest, Bash, YAML, tmux.

**Spec:** `docs/compressed_muon/RESULTS.md` section “CM073 mixed-bucket 组成审计与后续方向”, amended by the user's decision to expose one combined both switch rather than separate embedding/head switches.

## Global Constraints

- Work directly on `main` as explicitly authorized; preserve the existing unrelated `docs/compressed_muon/PAPER_NOTES.md` modification.
- The switch defaults to `false`; baseline and existing M002 configurations retain their current communication roles.
- Optimizer ownership does not change: transformer matrices remain Muon, embedding and LM head remain AdamW.
- Experiment scripts do not receive dedicated automated tests; validate them with `bash -n` and `--print-plan`/configuration inspection.
- Never interrupt other users' GPU processes. Long experiments run in tmux and write under `artifacts/compressed_muon/`.

---

### Task 1: Combined communication-role switch

**Files:**
- Modify: `train_greedylore.py`
- Modify: `tests/test_train_greedylore.py`

**Interfaces:**
- Consumes: `train.build_muon_param_groups(model, hp)` and stable model parameter names.
- Produces: `GreedyLoreHyperparameters.greedy_lore_compress_embedding_lm_head: bool` and an all-three-groups compression-role set when enabled.

- [ ] Add a failing entry-point test that enables the switch and expects block, `transformer.wte.weight`, and `lm_head.weight` specs to have role `matrix`, while asserting optimizer groups 1 and 2 still use AdamW.
- [ ] Run the focused test and confirm it fails because the hyperparameter/switch behavior is absent.
- [ ] Add the dataclass field, CLI argument, logging, and role-set union; keep the default false.
- [ ] Run the focused test and existing train-greedylore tests.

### Task 2: Wide-matrix refresh without full Vh

**Files:**
- Modify: `dion/greedy_lore.py`
- Modify: `tests/test_greedy_lore.py`

**Interfaces:**
- Consumes: `refresh_basis(global_corrected: Tensor, rank: int)`.
- Produces: the same `(basis, projector, support)` contract, using FP32 `eigh(X @ X.T)` for sufficiently wide canonical matrices and the existing SVD path otherwise.

- [ ] Add a failing wide-matrix test that monkeypatches SVD to reject the wide case and verifies an orthonormal, deterministically signed left basis and initial rank support.
- [ ] Run the focused test and confirm it fails through the current SVD path.
- [ ] Implement the minimal aspect-ratio dispatch and left-Gram eigendecomposition, sorting eigenvectors by descending eigenvalue and reusing sign canonicalization.
- [ ] Run tensor-level and DDP-hook CPU tests.

### Task 3: CM074 timing and CM075 full-training controller

**Files:**
- Create: `configs/compressed_muon/cm075_m002_greedylore_both_gpt60m_bf16_s1234.yaml`
- Create: `benchmark/compressed_muon/run_cm074_cm075_greedylore_both.sh`
- Modify: `docs/compressed_muon/EXPERIMENTS.md`
- Modify after execution: `docs/worklog/M002-greedy-lore-muon.md`
- Modify after execution when comparative results exist: `docs/compressed_muon/RESULTS.md`

**Interfaces:**
- Consumes: `benchmark/compressed_muon/run_greedy_lore_profiler.sh`, the CM070b GPT-60M BF16 paper-aligned recipe, and the new CLI switch.
- Produces: CM074 profiler-off GPT-130M dense/both timing artifacts and CM075 GPT-60M BF16 10,000-step GreedyLore-both training artifacts.

- [ ] Extend the generic GreedyLore launcher so M002 cells can forward the combined switch; do not apply it to dense cells.
- [ ] Create CM075 by copying the CM052b paper-aligned recipe, adding BF16 model storage, the combined switch, and CM075 W&B identity.
- [ ] Create a controller that waits for four GPUs with at least 18 GiB free, runs CM074 as three rotated dense/both pairs with 20 warmup plus 800 measured updates at GPT-130M/bucket80/rank32/interval200/seed42, then runs CM075 for 10,000 updates at GPT-60M/bucket160/seed1234.
- [ ] Validate shell syntax and inspect both printed plans/config parsing; do not add script unit tests.
- [ ] Run focused Python tests, then a two-rank NCCL smoke if two free GPUs are available without disturbing other jobs.
- [ ] Register CM074/CM075 as planned, commit code/config/controller/docs, launch the controller in tmux, and let the script own all waiting and sequencing.
- [ ] After completion, summarize timing/training artifacts and append evidence-backed results and worklog entries.
