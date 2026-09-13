# GreedyLore Bucket-Native BF16 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in full-model BF16 DDP mode and make GreedyLore persistent state and ordinary communication follow the DDP bucket dtype.

**Architecture:** A shared `model_dtype` setting controls model materialization for both dense Muon and GreedyLore while retaining FP32 defaults. GreedyLore removes its FP32-only state assumptions, temporarily upcasts only SVD-sensitive work, and relies on existing dtype-bearing layouts/checkpoints to reject incompatible restores.

**Tech Stack:** Python, PyTorch DDP, torch.distributed, pytest, YAML, Bash

**Spec:** `docs/superpowers/specs/2026-09-13-greedylore-bucket-native-bf16-design.md`

## Global Constraints

- `model_dtype` accepts exactly `float32` and `bfloat16`; its default is `float32`.
- Apply BF16 to all trainable GPT parameters, including transformer, embedding, and LM head.
- Explicit BF16 model parameters are initially DDP-only; reject BF16 with a device mesh.
- GreedyLore error, basis, score, factor, dense auxiliary payload, and basis broadcast follow bucket dtype by default.
- SVD and sign canonicalization use temporary FP32 compute and store the result back in bucket dtype.
- Preserve the packed `score+dense_aux` diagnostic override and its `bucket` default.
- Preserve all unrelated working-tree changes, especially `docs/compressed_muon/PAPER_NOTES.md`.

---

### Task 1: Shared full-model parameter dtype

**Files:**
- Modify: `train.py`
- Create: `tests/test_train_model_dtype.py`

**Interfaces:**
- Produces: `Hyperparameters.model_dtype: str`
- Produces: `resolve_model_dtype(name: str) -> torch.dtype`
- Produces: `validate_model_dtype_parallelism(name: str, device_mesh: Optional[DeviceMesh]) -> None`
- Produces: `materialize_and_initialize_model(model: GPT, *, device: str, dtype: torch.dtype) -> None`

- [ ] **Step 1: Write failing configuration and materialization tests**

```python
def test_model_dtype_defaults_to_float32():
    assert train.Hyperparameters().model_dtype == "float32"

def test_model_dtype_cli_accepts_bfloat16(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--model_dtype", "bfloat16"])
    hp = train.override_args_from_cli(train.Hyperparameters(), train.parse_cli_args())
    assert hp.model_dtype == "bfloat16"

@pytest.mark.parametrize("name, expected", [("float32", torch.float32), ("bfloat16", torch.bfloat16)])
def test_resolve_model_dtype(name, expected):
    assert train.resolve_model_dtype(name) is expected

def test_materialize_initializes_every_parameter_as_bfloat16():
    with torch.device("meta"):
        model = GPT(GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=1, n_embd=8))
    train.materialize_and_initialize_model(model, device="cpu", dtype=torch.bfloat16)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}

def test_bfloat16_model_dtype_rejects_device_mesh():
    with pytest.raises(ValueError, match="DDP only"):
        train.validate_model_dtype_parallelism("bfloat16", object())
```

- [ ] **Step 2: Run tests and verify the new API is absent**

Run: `pytest -q tests/test_train_model_dtype.py`

Expected: FAIL because `model_dtype` and helper functions do not exist.

- [ ] **Step 3: Implement the shared dtype configuration and model materialization**

Add the dataclass field and CLI argument, implement an exact string-to-dtype mapping, validate the DDP-only BF16 constraint immediately after distributed initialization, and replace the inline `to_empty/init_weights` sequence with:

```python
model_dtype = resolve_model_dtype(hp.model_dtype)
materialize_and_initialize_model(model, device="cuda", dtype=model_dtype)
```

The helper must call `to_empty`, convert the entire module to `dtype`, then call `init_weights`, so weights are initialized directly in their selected storage dtype.

- [ ] **Step 4: Run focused and shared-entry regression tests**

Run: `pytest -q tests/test_train_model_dtype.py tests/test_configs.py tests/test_train_factories.py tests/test_train_adamw_builder.py`

Expected: PASS.

- [ ] **Step 5: Commit the shared dtype feature**

```bash
git add train.py tests/test_train_model_dtype.py
git commit -m "feat: add configurable model parameter dtype"
```

### Task 2: BF16 Muon and scalar optimizer state contracts

**Files:**
- Modify: `tests/test_train_model_dtype.py`

**Interfaces:**
- Consumes: BF16 model parameters produced by Task 1.
- Produces: BF16 momentum/variance state for matrix, embedding, and LM-head parameters, with FP32 optimizer control tensors.

- [ ] **Step 1: Add BF16 state and one-step optimizer tests**

Build a tiny BF16 GPT, use `train.build_muon_param_groups`, and instantiate Muon with `use_triton=False` and a deterministic test orthogonalizer. Parameterize scalar groups over Lion and AdamW. After backward and one optimizer step, assert matrix momentum, scalar momentum, and AdamW variance match parameter dtype; assert `step_dev` and persistent learning-rate tensors are FP32 and all parameters remain finite BF16 tensors.

- [ ] **Step 2: Run the optimizer tests and capture any unsupported operation**

Run: `pytest -q tests/test_train_model_dtype.py -k optimizer`

Expected: PASS with existing bucket-native state allocation, or FAIL at the first concrete BF16-incompatible update operation.

- [ ] **Step 3: Confirm no optimizer compatibility source change is needed**

The expected implementation is the existing bucket-native `zeros_like(parameter)` allocation. If Step 2 instead exposes a concrete defect, stop this task, invoke systematic debugging, and revise the plan around that observed operation; do not speculate or introduce persistent FP32 copies.

- [ ] **Step 4: Run Muon and scalar optimizer regressions**

Run: `pytest -q tests/test_train_model_dtype.py tests/test_optimizers.py tests/test_megabatch_empty_shard.py tests/test_normuon_split_lr.py`

- [ ] **Step 5: Commit the optimizer contracts**

```bash
git add tests/test_train_model_dtype.py
git commit -m "test: cover BF16 Muon optimizer state"
```

### Task 3: Bucket-native GreedyLore state and computation

**Files:**
- Modify: `dion/greedy_lore.py`
- Modify: `dion/greedy_lore_ddp_hook.py`
- Modify: `tests/test_greedy_lore.py`
- Modify: `tests/test_greedy_lore_ddp_state.py`
- Modify: `tests/test_greedy_lore_ddp_hook.py`

**Interfaces:**
- Modify: `make_random_vectors(*, rows: int, columns: int, seed: int, device: torch.device, dtype: torch.dtype) -> Tensor`
- Preserve: `refresh_basis(global_corrected: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]`, returning basis/projector in the input dtype after FP32 SVD.
- Produce: `GreedyLoreDDPState` support for FP32 and BF16 matrix parameters.

- [ ] **Step 1: Add failing tensor-foundation tests**

Parameterize BF16 and FP32 tests asserting `corrected_gradient`, random vectors, score, factor, reconstruction, and `refresh_basis` outputs match input/state dtype. Monkeypatch `torch.linalg.svd` to assert its input is FP32 while the stored/output basis returns to BF16.

- [ ] **Step 2: Add failing DDP-state allocation tests**

Construct BF16 parameter specs and assert their `error` and `basis` are BF16, `last_support` remains int64, and unsupported non-floating or mixed layout dtypes fail with a precise error.

- [ ] **Step 3: Run the new foundation/state tests**

Run: `pytest -q tests/test_greedy_lore.py tests/test_greedy_lore_ddp_state.py`

Expected: FAIL on the current FP32-only validation and allocations.

- [ ] **Step 4: Implement bucket-native tensor foundations and state**

Use `error.dtype`/parameter dtype for corrected gradients and persistent allocation. Generate deterministic random vectors in FP32 and cast them to the requested bucket dtype before returning. In `refresh_basis`, save `original_dtype`, run SVD and sign canonicalization on FP32, then cast the basis/projector back before returning. Accept exactly FP32 and BF16 matrix parameters.

- [ ] **Step 5: Add failing hook payload-dtype tests**

For FP32 and BF16 fake buckets, exercise compressed steps and observe collectives. Assert packed score+dense auxiliary payload uses the bucket dtype under `bucket`, factor payload always uses bucket dtype, and explicit packed-payload overrides affect only `score+dense_aux`. Assert refresh basis broadcast uses bucket dtype.

- [ ] **Step 6: Implement bucket-native packing and profiling**

Remove the hard-coded FP32 dense auxiliary staging, construct empty/packed buffers using the resolved communication dtype, and keep signed scores/factors in bucket dtype. Replace hard-coded four-byte score/factor accounting with element sizes derived from their effective dtypes. Preserve final copies into DDP gradient views.

- [ ] **Step 7: Run focused GreedyLore tests**

Run: `pytest -q tests/test_greedy_lore.py tests/test_greedy_lore_ddp_state.py tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_profiler_trace.py`

Expected: PASS.

- [ ] **Step 8: Commit GreedyLore dtype propagation**

```bash
git add dion/greedy_lore.py dion/greedy_lore_ddp_hook.py tests/test_greedy_lore.py tests/test_greedy_lore_ddp_state.py tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_profiler_trace.py
git commit -m "feat: make GreedyLore state bucket-native"
```

### Task 4: Distributed and checkpoint dtype guarantees

**Files:**
- Modify: `tests/test_greedy_lore_ddp_hook_distributed.py`
- Modify: `tests/test_greedy_lore_ddp_hook_nccl.py`
- Modify: `tests/test_greedy_lore_ddp_checkpoint.py`

**Interfaces:**
- Consumes: bucket-native GreedyLore implementation from Task 3.
- Produces: same-dtype FP32/BF16 checkpoint round trips and fail-closed cross-dtype loads.

- [ ] **Step 1: Add a two-rank BF16 reconstruction test**

Run warmup, refresh, and compressed steps through real DDP using BF16 parameters. Assert gradients and persistent state remain BF16, ranks reconstruct identical averaged gradients within BF16 tolerances, and the collective observer records BF16 score, factor, and basis payloads.

- [ ] **Step 2: Run the distributed CPU/Gloo test**

Run: `pytest -q tests/test_greedy_lore_ddp_hook_distributed.py -k bfloat16`

Expected: PASS after Task 3; otherwise expose a rank ordering or dtype mismatch to fix before proceeding.

- [ ] **Step 3: Add checkpoint dtype tests**

Parameterize existing checkpoint-resume coverage over FP32 and BF16. Add a test that saves BF16 metadata/state, builds an FP32 runtime with the same shapes/names, and asserts metadata validation rejects the parameter table/fingerprint or tensor schema before payload copy.

- [ ] **Step 4: Run checkpoint tests and implement only required schema fixes**

Run: `pytest -q tests/test_greedy_lore_ddp_checkpoint.py`

Expected: PASS because current parameter fingerprints and tensor schemas already contain dtype. If not, repair metadata validation without adding implicit casts.

- [ ] **Step 5: Run NCCL smoke tests when GPUs are available**

First inspect `nvidia-smi` without disturbing existing processes. If two suitable GPUs are free, run: `pytest -q tests/test_greedy_lore_ddp_hook_nccl.py -k bfloat16`. Otherwise record that NCCL verification was not run.

- [ ] **Step 6: Commit distributed/checkpoint coverage**

```bash
git add tests/test_greedy_lore_ddp_hook_distributed.py tests/test_greedy_lore_ddp_hook_nccl.py tests/test_greedy_lore_ddp_checkpoint.py
git commit -m "test: verify GreedyLore BF16 distributed resume"
```

### Task 5: Strictly paired four-cell experiment setup and documentation

**Files:**
- Create: `configs/compressed_muon/cm066a_dense_muon_gpt130m_fp32_dtype_matrix_s1234.yaml`
- Create: `configs/compressed_muon/cm066b_m002_greedy_lore_muon_gpt130m_fp32_dtype_matrix_s1234.yaml`
- Create: `configs/compressed_muon/cm066c_dense_muon_gpt130m_bf16_dtype_matrix_s1234.yaml`
- Create: `configs/compressed_muon/cm066d_m002_greedy_lore_muon_gpt130m_bf16_dtype_matrix_s1234.yaml`
- Create: `benchmark/compressed_muon/run_cm066_greedylore_model_dtype_matrix.sh`
- Create: `tests/test_cm066_greedylore_model_dtype_matrix_launcher.py`
- Modify: `docs/compressed_muon/EXPERIMENTS.md`
- Modify: `docs/worklog/M002-greedy-lowrank-muon.md`
- Do not modify for setup alone: `docs/compressed_muon/RESULTS.md`, `docs/compressed_muon/PAPER_NOTES.md`

**Interfaces:**
- Consumes: `model_dtype` from Task 1 and GreedyLore bucket-native mode from Task 3.
- Produces: a serial, reproducible `FP32/BF16 x dense/GreedyLore` run matrix with newly assigned CM IDs.

- [ ] **Step 1: Confirm CM066 is free and select the CM053 paper-aligned model recipe**

Use `rg -n 'CM066' docs/compressed_muon/EXPERIMENTS.md configs/compressed_muon benchmark/compressed_muon` and require no matches before creating the files. Base all four cells on the CM053a/CM053b GPT-130M paper-aligned recipe; keep seed, data, model, batch, schedule, optimizer hyperparameters, and bucket size identical.

- [ ] **Step 2: Write a failing launcher/config test**

Assert the four configs form the Cartesian product `{muon, greedy_lore_muon} x {float32, bfloat16}`, use the correct training entry (`train.py` or `train_greedylore.py`), share all non-method/non-dtype fields, and have distinct CM IDs and artifact/W&B names.

- [ ] **Step 3: Run the launcher test and verify files are absent**

Run the exact new test file with `pytest -q`.

Expected: FAIL because the four configs and launcher do not exist.

- [ ] **Step 4: Add configs and a fail-fast serial launcher**

The launcher must capture the git revision, resolved config, command, environment, stdout, and result path per cell. It must stop on a failed cell, must not pair against old artifacts, and must accept a no-W&B smoke mode without changing the formal defaults.

- [ ] **Step 5: Register the planned experiment and implementation worklog**

Add four planned rows to `EXPERIMENTS.md`. Append a Chinese M002 worklog entry describing the dtype design, code/config identifiers, validations actually run, and the strict pairing rule. Do not edit results or paper claims before runs complete.

- [ ] **Step 6: Run launcher/config tests**

Run: `pytest -q tests/test_cm066_greedylore_model_dtype_matrix_launcher.py tests/test_configs.py tests/test_greedy_lore_profiler_launcher.py`

Expected: PASS.

- [ ] **Step 7: Commit experiment setup**

```bash
git add configs/compressed_muon benchmark/compressed_muon tests docs/compressed_muon/EXPERIMENTS.md docs/worklog/M002-greedy-lowrank-muon.md
git commit -m "exp: add paired GreedyLore model dtype matrix"
```

Stage only the four configs, launcher, relevant tests, experiment registry, and M002 worklog; explicitly exclude `PAPER_NOTES.md`.

### Task 6: Full verification and optional formal launch

**Files:**
- Modify only after completed runs: `docs/compressed_muon/RESULTS.md`
- Preserve: `docs/compressed_muon/PAPER_NOTES.md`

**Interfaces:**
- Consumes: all preceding code, tests, configs, and launcher.
- Produces: verified implementation; formal results only if the four-cell run is explicitly launched and completes.

- [ ] **Step 1: Run the focused regression suite**

Run all training-entry, Muon, GreedyLore, layout, profiler, distributed, and checkpoint tests touched by Tasks 1-5. Record the exact command and pass/skip totals.

- [ ] **Step 2: Run formatting and repository hygiene checks**

Run: `git diff --check HEAD~5..HEAD` and `git status --short`.

Expected: no whitespace errors; the unrelated `PAPER_NOTES.md` edit remains present and unstaged.

- [ ] **Step 3: Inspect GPUs before any long run**

Run `nvidia-smi` once and inspect existing processes. Do not stop or alter any process. If resources are unsuitable, stop at a verified launcher and report the formal matrix as planned.

- [ ] **Step 4: If resources are suitable and the user-requested run fits the current session, launch the four cells through the serial launcher**

Use tmux or the launcher's background mode and preserve logs under each `artifacts/compressed_muon/CM066[a-d]-*/` directory; do not use an agent polling loop. Otherwise provide `bash benchmark/compressed_muon/run_cm066_greedylore_model_dtype_matrix.sh` for later execution.

- [ ] **Step 5: Update result documents only from completed artifacts**

After all four cells complete, compute within-dtype dense-versus-GreedyLore comparisons and FP32-versus-BF16 effects. Append evidence-backed results to `RESULTS.md`, `EXPERIMENTS.md`, and the M002 worklog. Do not overwrite or stage the user's unrelated `PAPER_NOTES.md` changes.

- [ ] **Step 6: Final verification commit if result documents changed**

```bash
git add docs/compressed_muon/RESULTS.md docs/compressed_muon/EXPERIMENTS.md docs/worklog/M002-greedy-lowrank-muon.md
git commit -m "docs: record paired GreedyLore dtype results"
```
