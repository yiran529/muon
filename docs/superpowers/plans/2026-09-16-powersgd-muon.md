# PowerSGD-Muon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checkpointable, asynchronously pipelined PowerSGD DDP gradient compressor in front of unchanged ordinary Muon.

**Architecture:** Pure tensor operations live in `dion/power_sgd.py`; stable layout identity lives in `dion/power_sgd_layout.py`; the DDP hook owns per-parameter state, lifecycle, packed collectives, CUDA streams, and checkpointing. `train_powersgd.py` installs the hook and returns the existing `GradientSyncRuntime` contract.

**Tech Stack:** Python 3.10+, PyTorch distributed/Futures/CUDA streams, pytest, OmegaConf training configuration.

**Spec:** `docs/superpowers/specs/2026-09-16-powersgd-muon-design.md`

## Global Constraints

- Implement directly on the current `main` branch as explicitly requested.
- Preserve all pre-existing dirty worktree changes and exclude them from feature commits.
- DDP only; do not modify dense Muon, ARC-TopK, or GreedyLore behavior.
- Default compression scope is the two-dimensional Muon matrix parameter group.
- Never key persistent state only by DDP bucket index.
- Do not call `Work.wait()`, `torch.cuda.synchronize()`, or tensor `.item()` from hook callbacks.
- Use TDD for every production behavior and run fresh verification before each commit.

---

### Task 1: Tensor-level PowerSGD foundations

**Files:**
- Create: `dion/power_sgd.py`
- Create: `tests/test_power_sgd.py`

**Interfaces:**
- Produces: `PowerSGDConfig`, `should_compress`, `compressed_phase`, `derive_power_sgd_seed`, `make_random_factor`, `orthogonalize`, `corrected_gradient`, `compute_left_factor`, `compute_right_factor`, and `reconstruct`.

- [ ] **Step 1: Write failing configuration and eligibility tests**

```python
def test_should_compress_counts_both_factors():
    assert should_compress(8, 16, rank=2, min_compression_rate=2.0)
    assert not should_compress(4, 4, rank=2, min_compression_rate=2.0)
```

- [ ] **Step 2: Run the focused test and confirm import failure**

Run: `pytest -q tests/test_power_sgd.py`
Expected: collection fails because `dion.power_sgd` does not exist.

- [ ] **Step 3: Implement minimal validated config and pure functions**

```python
@dataclass(frozen=True)
class PowerSGDConfig:
    rank: int = 1
    start_compress_step: int = 1000
    min_compression_rate: float = 2.0
    error_feedback: Literal["ef14", "none"] = "ef14"
    warm_start: bool = True
    seed: int = 42
    orthogonalization_epsilon: float = 1e-8
    seed_scheme_version: int = 1
```

- [ ] **Step 4: Add failing numerical tests, then implement each operation**

Use literal small matrices to verify deterministic random state, column
orthonormality, factor shapes, reconstruction, full-rank exactness, and
`error = corrected - reconstructed`.

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_power_sgd.py`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add dion/power_sgd.py tests/test_power_sgd.py
git commit -m "feat: add PowerSGD tensor foundations"
```

### Task 2: Stable parameter layout and fingerprint

**Files:**
- Create: `dion/power_sgd_layout.py`
- Create: `tests/test_power_sgd_layout.py`

**Interfaces:**
- Consumes: `PowerSGDConfig`.
- Produces: `PowerSGDParameterDescriptor`, `canonical_power_sgd_fingerprint`, and `validate_power_sgd_fingerprint_across_ranks`.

- [ ] **Step 1: Write failing deterministic-layout tests**

Assert that identical descriptors produce identical SHA-256 fingerprints and
that changes to rank, role, shape, dtype, stable ID, or process-group ranks
change the fingerprint.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_power_sgd_layout.py`
Expected: collection fails because the layout module is absent.

- [ ] **Step 3: Implement canonical JSON hashing and distributed validation**

The canonical payload contains config via `asdict`, ordered group ranks, and
ordered parameter descriptors. Cross-rank validation uses fixed-size digest
tensors and fails with a PowerSGD-specific message.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest -q tests/test_power_sgd_layout.py`
Expected: all tests pass.

```bash
git add dion/power_sgd_layout.py tests/test_power_sgd_layout.py
git commit -m "feat: add stable PowerSGD parameter layout"
```

### Task 3: DDP state lifecycle and checkpoint

**Files:**
- Create: `dion/power_sgd_ddp_hook.py`
- Create: `tests/test_power_sgd_ddp_checkpoint.py`

**Interfaces:**
- Consumes: Task 1 tensor functions and Task 2 fingerprint.
- Produces: `PowerSGDDDPParameterSpec`, `PowerSGDParameterState`, and `PowerSGDDDPState` with `begin_step`, `note_bucket`, `finish_step`, `commit_step`, `checkpoint_metadata`, `validate_checkpoint_metadata`, `state_dict`, and `load_state_dict`.

- [ ] **Step 1: Write failing lifecycle tests**

Cover double begin, note without begin, duplicate/missing parameter coverage,
finish with an in-flight Future, commit before finish, and checkpoint during an
active step.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_power_sgd_ddp_checkpoint.py`
Expected: import or missing-class failure.

- [ ] **Step 3: Implement preallocated per-parameter state and lifecycle**

Initialize error and `Q` by stable parameter identity. Classify compressibility
at construction and reject unused parameters, unsupported shapes, invalid
dtypes, or optimizer/layout mismatch before hook registration.

- [ ] **Step 4: Add failing checkpoint round-trip and schema tests**

Round-trip nonzero error/Q/initialized/step values and reject mismatched world
size, config fingerprint, parameter table, missing ranks, field names, shapes,
and dtypes before copying payloads.

- [ ] **Step 5: Implement checkpoint metadata and payload methods**

Store only shared schema/step plus rank-local error, Q, and initialized tensors;
exclude Futures, streams, events, and transient packed buffers.

- [ ] **Step 6: Verify and commit**

Run: `pytest -q tests/test_power_sgd_ddp_checkpoint.py`
Expected: all tests pass.

```bash
git add dion/power_sgd_ddp_hook.py tests/test_power_sgd_ddp_checkpoint.py
git commit -m "feat: add PowerSGD DDP state lifecycle"
```

### Task 4: Correct two-stage DDP hook

**Files:**
- Modify: `dion/power_sgd_ddp_hook.py`
- Create: `tests/test_power_sgd_ddp_hook.py`
- Create: `tests/test_power_sgd_ddp_hook_distributed.py`

**Interfaces:**
- Consumes: `PowerSGDDDPState.note_bucket` and Task 1 math.
- Produces: `power_sgd_ddp_hook(state, bucket) -> torch.futures.Future[Tensor]`.

- [ ] **Step 1: Write failing fake-bucket tests for dense warmup and packing**

Assert one dense collective during warmup and the exact first/second packed
buffer sizes for a mixed compressed bucket.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_power_sgd_ddp_hook.py`
Expected: hook symbol or behavior failure.

- [ ] **Step 3: Implement conservative Future-only hook**

Implement dense averaging and the sequence `P+dense AllReduce -> Q AllReduce
-> reconstruction/error`. Use callbacks and bridge Futures without callback
waits or device synchronization.

- [ ] **Step 4: Write failing two-rank Gloo numerical tests**

Use hand-derived rank-local gradients to assert dense auxiliary averages,
identical reconstructed gradients, the local EF14 recurrence, deterministic
warm start, and observer sequence `powersgd_hook/p_plus_aux` then
`powersgd_hook/q`.

- [ ] **Step 5: Implement distributed correctness fixes**

Ensure reduced P is orthogonalized identically, Q is divided exactly once,
corrected matrix views survive until reconstruction, and errors are updated
before the bucket view is overwritten.

- [ ] **Step 6: Verify and commit**

Run: `pytest -q tests/test_power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook_distributed.py`
Expected: all tests pass.

```bash
git add dion/power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook_distributed.py
git commit -m "feat: add PowerSGD DDP communication hook"
```

### Task 5: CUDA stream pipeline

**Files:**
- Modify: `dion/power_sgd_ddp_hook.py`
- Modify: `tests/test_power_sgd_ddp_hook.py`
- Create: `tests/test_power_sgd_ddp_hook_nccl.py`

**Interfaces:**
- Extends: Task 4 hook without changing its numerical or collective contract.
- Produces: preparation, collective, and reconstruction stream ordering with separate collective and completion tails.

- [ ] **Step 1: Write failing Future-order tests**

Assert that preparation can precede the prior bucket's DDP completion, bucket B
collectives wait for bucket A's Q collective, bucket B can launch before bucket
A reconstruction completes, and `finish_step` waits for all reconstructions.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_power_sgd_ddp_hook.py`
Expected: ordering assertions fail under the conservative chain.

- [ ] **Step 3: Implement split tails and CUDA event visibility**

Record bucket-ready and preparation-done events, retain and `record_stream`
asynchronous buffers, release the collective tail after Q, and complete the
DDP Future only after reconstruction/error writes.

- [ ] **Step 4: Add optional two-GPU NCCL tests**

Exercise rank-skewed delays, allocator churn, returned-Future visibility, and
cross-rank collective signature equality. Mark them `multi_gpu` and skip when
fewer than two exclusive CUDA devices are available.

- [ ] **Step 5: Verify and commit**

Run: `pytest -q tests/test_power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook_distributed.py`
Expected: all CPU/Gloo tests pass.

Run when available: `pytest -q -m multi_gpu tests/test_power_sgd_ddp_hook_nccl.py`
Expected: all selected NCCL tests pass.

```bash
git add dion/power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook_nccl.py
git commit -m "perf: pipeline PowerSGD DDP buckets"
```

### Task 6: Training integration and configuration

**Files:**
- Create: `train_powersgd.py`
- Modify: `dion/__init__.py`
- Create: `configs/compressed_muon/m005_power_sgd_muon_ddp.yaml`
- Create: `tests/test_train_powersgd.py`

**Interfaces:**
- Consumes: `PowerSGDConfig`, layout helpers, `PowerSGDDDPState`, `power_sgd_ddp_hook`, `train.build_muon_param_groups`, and `train.GradientSyncRuntime`.
- Produces: `PowerSGDHyperparameters`, parser validation, hook installation, and `init_power_sgd_optimizer`.

- [ ] **Step 1: Write failing factory tests**

Assert DDP-only behavior, rejection of explicit optimizer-owned sync, exact
Muon parameter role selection, unchanged `Muon` construction, hook
registration, and checkpoint state name `power_sgd_compressor`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_train_powersgd.py`
Expected: import failure because the training entry point is absent.

- [ ] **Step 3: Implement entry point and public exports**

Follow `train_greedylore.py` integration while exposing only PowerSGD-specific
arguments. Return ordinary Muon plus `GradientSyncRuntime` with the compressor
lifecycle and checkpoint state.

- [ ] **Step 4: Add a runnable YAML configuration**

Use the current compressed-Muon GPT-130M DDP schema, BF16 model dtype, rank 4,
warm start and EF14 enabled, and compression starting after optimizer step
1000. Record all method-specific values explicitly.

- [ ] **Step 5: Verify and commit**

Run: `pytest -q tests/test_train_powersgd.py tests/test_train_factories.py`
Expected: all tests pass.

```bash
git add train_powersgd.py dion/__init__.py configs/compressed_muon/m005_power_sgd_muon_ddp.yaml tests/test_train_powersgd.py
git commit -m "feat: integrate PowerSGD with Muon training"
```

### Task 7: Research records and full verification

**Files:**
- Create: `docs/compressed_muon/methods/M005_power_sgd_muon.md`
- Create: `docs/worklog/M005-power-sgd-muon.md`
- Modify: `docs/compressed_muon/METHOD_INDEX.md`

**Interfaces:**
- Documents: method semantics, implementation paths, verified scope, payload formula, evidence, and remaining experiment gates.

- [ ] **Step 1: Add method and worklog records**

Record M005 as `testing`, distinguish gradient compression from Muon result
communication, and state that no convergence or speedup claim exists before
formal experiments.

- [ ] **Step 2: Run formatting and focused verification**

Run: `python -m compileall -q dion train_powersgd.py`
Expected: exit 0.

Run: `pytest -q tests/test_power_sgd.py tests/test_power_sgd_layout.py tests/test_power_sgd_ddp_checkpoint.py tests/test_power_sgd_ddp_hook.py tests/test_power_sgd_ddp_hook_distributed.py tests/test_train_powersgd.py`
Expected: all tests pass.

- [ ] **Step 3: Run the broader regression suite**

Run: `pytest -q`
Expected: all non-environment-specific tests pass; any GPU skips are reported.

- [ ] **Step 4: Inspect the final diff and commit documentation**

Confirm no pre-existing modifications in `RESULTS.md`, `EXPERIMENTS.md`, the
M002 worklog, or untracked CM083 scripts enter the feature commit.

```bash
git add docs/compressed_muon/METHOD_INDEX.md docs/compressed_muon/methods/M005_power_sgd_muon.md docs/worklog/M005-power-sgd-muon.md docs/superpowers/specs/2026-09-16-powersgd-muon-design.md docs/superpowers/plans/2026-09-16-powersgd-muon.md
git commit -m "docs: record PowerSGD-Muon method"
```

