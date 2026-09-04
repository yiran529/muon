# Shared ARC-TopK AdamW/Muon Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one DDP-only ARC-TopK/EF21M gradient synchronization layer shared by pure AdamW and Muon, then run a fair 4-GPU 2x2 communication-limited benchmark under normal and P2P-disabled NCCL transports.

**Architecture:** Keep the tensor compressor and EF21M recurrence in `dion/arc_topk.py`; add a small orchestration module that owns deterministic shape grouping, rank-symmetric missing-gradient handling, optimizer-state initialization, dense fallback, and communication-byte accounting. `ArcTopKMuon` and a new `ArcTopKAdamW` both consume the synchronizer output, while a standalone benchmark owns synthetic GPT workloads, CUDA-event timing, profiler capture, repeated runs, and JSON artifacts.

**Tech Stack:** Python 3.10+, PyTorch distributed/DDP/NCCL, CUDA events, `torch.profiler`, pytest, Gloo distributed tests, YAML/JSONL, tmux.

**Spec:** `docs/worklog/M001-arc-topk-ef21m-muon.md`, section `### 推荐给后续 agent 的测试设置`; repository constraints in `AGENTS.md` and `docs/compressed_muon/RESEARCH_GUIDE.md`.

## Global Constraints

- Preserve the original `dion/muon.py` baseline and all unrelated user changes.
- Do not install or upgrade PyTorch, CUDA, NCCL, Triton, or any other dependency.
- First implementation remains DDP-only; reject `DeviceMesh`, FSDP, HSDP, TP, and CUDA Graph capture for the new AdamW ARC path.
- The compressed tensor set is exactly `model.transformer.h.parameters()`; `transformer.wte` and `lm_head` use dense All-Reduce in both ARC optimizers.
- Both ARC optimizers use identical parameter order, `(shape, dtype)` grouping, zero substitution for missing local gradients, seed derivation, compression start semantics, and ARC collective calls.
- ARC settings are `ratio=0.2`, projection rank `4`, `eta=0.1`, seed `42`, and compression start `0`; step 1 remains the required dense EF21M initialization and is excluded from timing.
- Benchmark settings are world size `4`, local batch `1`, sequence length `256`, gradient accumulation `1`, BF16 parameters and compute, at least 20 stable warmup steps, at least 100 measured steps, and at least 3 independent process launches per configuration.
- Run each 60M-scale 2x2 on the same four idle GPUs, first with normal NCCL, then with `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=0`.
- Never stop, alter, or attach profilers to processes not started for this experiment.
- Do not place scripts in the repository root. Reusable code belongs in `benchmark/compressed_muon/`; raw outputs belong in `artifacts/compressed_muon/<experiment-id>/`.
- Do not enable W&B, validation, checkpointing, or synchronous per-step logging inside the measured interval.
- Use TDD: every production behavior starts with a focused failing test, and the implementing agent records the expected RED failure before writing implementation code.

---

## File Structure

- Create `dion/arc_topk_sync.py`: common state, deterministic batching, dense/ARC synchronization, phase timing hooks, and logical communication accounting.
- Modify `dion/muon_arctopk.py`: delegate ARC state initialization and gradient synchronization to the shared module; retain Muon momentum, orthogonalization, and parameter update locally.
- Create `dion/adamw_arctopk.py`: DDP-only optimizer that delegates gradient synchronization to the same shared module and applies fused AdamW updates.
- Modify `dion/__init__.py`: export `ArcTopKAdamW`.
- Create `benchmark/compressed_muon/benchmark_arc_2x2.py`: construct GPT, DDP, optimizer variants, run warmup/measurement, collect CUDA-event and profiler metrics, and emit one rank-0 JSON result.
- Create `benchmark/compressed_muon/profiler_trace.py`: attribute NCCL kernels to DDP gradient buckets, ARC collectives, and Muon result communication from profiler traces, and estimate overlap by interval intersection.
- Create `benchmark/compressed_muon/summarize_arc_2x2.py`: validate repetitions and compute means, standard deviations, coefficients of variation, and reduction ratios.
- Create `tests/test_arc_topk_sync.py`: local orchestration and byte-accounting tests.
- Create `tests/test_arc_topk_sync_distributed.py`: two-rank Gloo tests for collective ordering and dense/ARC numerical behavior.
- Create `tests/test_adamw_arctopk.py` and `tests/test_adamw_arctopk_distributed.py`: state, update, restore, and rank-consistency tests.
- Create `tests/test_benchmark_arc_2x2.py`: CLI/config/result-schema and summary-math tests without requiring CUDA.
- Create experiment-owned `config.yaml`, `command.txt`, `environment.txt`, `stdout.log`, `metrics.json`, and `profiler/` under `CM002a-d` and `CM003a-d` artifact directories.
- Modify `docs/compressed_muon/EXPERIMENTS.md` and append to `docs/worklog/M001-arc-topk-ef21m-muon.md` only after runs have an observed status.

---

### Task 1: Common deterministic ARC synchronization layer

**Files:**
- Create: `dion/arc_topk_sync.py`
- Test: `tests/test_arc_topk_sync.py`
- Test: `tests/test_arc_topk_sync_distributed.py`

**Interfaces:**
- Consumes: `arc_topk_ef21m_async()` and `validate_arc_topk_config()` from `dion.arc_topk`.
- Produces:

```python
@dataclass(frozen=True)
class ArcTopKSyncConfig:
    ratio: float = 0.2
    projection_rank: int = 4
    eta: float = 0.1
    seed: int = 42
    start_compress_step: int = 0

@dataclass(frozen=True)
class ArcTopKLogicalBytes:
    dense_gradient: int
    arc_seed: int
    arc_sketch: int
    arc_selected_values: int
    uncompressed: int

def initialize_arc_state_(state: dict, param: Tensor) -> None

def group_parameters_by_shape_dtype(
    params: Iterable[Tensor],
) -> list[list[Tensor]]

def synchronize_arc_batch_async(
    *, params: list[Tensor], states: list[dict], process_group: Optional[ProcessGroup],
    config: ArcTopKSyncConfig, step: int, task_index: int,
) -> Generator[None, None, list[Tensor]]

def average_gradients_async(
    gradients: list[Tensor], process_group: Optional[ProcessGroup],
) -> Generator[None, None, list[Tensor]]

def estimate_arc_logical_bytes(
    *, compressed_batches: list[list[Tensor]], uncompressed_params: list[Tensor],
    config: ArcTopKSyncConfig, step: int,
) -> ArcTopKLogicalBytes
```

- `group_parameters_by_shape_dtype()` preserves first-occurrence order and includes every parameter regardless of local `grad` presence.
- `synchronize_arc_batch_async()` substitutes `zeros_like(param)` for a missing local gradient and passes state keys `arc_h_local`, `arc_g_local`, and `arc_g_global` to the existing primitive.
- Logical bytes mean collective input payload per rank, before transport-level ring amplification. Step 1 and steps through `start_compress_step` count the full compressed-set gradient payload as `dense_gradient`; compressed steps count an 8-byte seed broadcast plus sketch and selected-value All-Reduce inputs. `uncompressed` always counts dense inputs from the explicitly supplied parameters.

- [ ] **Step 1: Write local RED tests for state and stable grouping**

Add tests using literal tensors, including two same-shaped parameters separated by a different shape. Assert that grouping order is `[[p0, p2], [p1]]`, every ARC state tensor is zero, distinct, and shape/dtype/device-matched, and calling initialization twice preserves existing values.

```python
def test_grouping_is_shape_dtype_stable():
    p0 = torch.nn.Parameter(torch.zeros(4, 3))
    p1 = torch.nn.Parameter(torch.zeros(2, 3))
    p2 = torch.nn.Parameter(torch.ones(4, 3))
    groups = group_parameters_by_shape_dtype([p0, p1, p2])
    assert groups == [[p0, p2], [p1]]
```

- [ ] **Step 2: Run the local tests and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest tests/test_arc_topk_sync.py -v
```

Expected: collection fails because `dion.arc_topk_sync` does not exist.

- [ ] **Step 3: Implement config validation, state initialization, and grouping**

Use ordinary dictionaries whose insertion order is deterministic. Call `validate_arc_topk_config()` from `ArcTopKSyncConfig.__post_init__`; do not duplicate validation rules.

- [ ] **Step 4: Write RED tests for byte accounting**

Use one BF16 compressed batch with shape `(2, 10, 8)`, `ratio=0.2`, `r=4`, and one BF16 uncompressed tensor `(7, 8)`. Hand-derived compressed-step values are:

```text
dense_gradient      = 2 * 10 * 8 * 2 = 320 bytes
arc_seed            = 8 bytes
arc_sketch          = 2 * 10 * 4 * 2 = 160 bytes
arc_selected_values = 2 * ceil(10 * 0.2) * 8 * 2 = 64 bytes
uncompressed        = 7 * 8 * 2 = 112 bytes
```

Also assert step 1 reports `dense_gradient=320`, zero sketch/selected bytes, and `uncompressed=112`.

- [ ] **Step 5: Run byte tests and verify RED, then implement the estimator**

Run the individual tests with `pytest ...::test_name -v`. Compute `k=max(1, ceil(rows*ratio))`; use tensor `element_size()` rather than a hard-coded dtype table.

- [ ] **Step 6: Write two-rank RED tests for the shared synchronization call**

Spawn two Gloo ranks with rank-local gradients and assert:

1. Step 1 returns their dense average and initializes all three states correctly.
2. Step 2 with `ratio=1`, `eta=1` returns the hand-computed dense average.
3. One rank having `grad=None` still completes the same collective sequence and treats that local gradient as zero.
4. Two shape groups receive task indices `0, 1` in the same order on both ranks.

- [ ] **Step 7: Run distributed tests and verify RED, then implement synchronization**

Run:

```bash
uv run --frozen --extra dev pytest tests/test_arc_topk_sync_distributed.py -v
```

Expected RED reason: `synchronize_arc_batch_async` is absent or incomplete. Implement it as a thin adapter around `arc_topk_ef21m_async`; do not copy projection, Top-K, scatter, or EF21M math.

- [ ] **Step 8: Verify Task 1 and commit**

```bash
uv run --frozen --extra dev pytest tests/test_arc_topk.py tests/test_arc_topk_distributed.py tests/test_arc_topk_sync.py tests/test_arc_topk_sync_distributed.py -v
git diff --check
git add dion/arc_topk_sync.py tests/test_arc_topk_sync.py tests/test_arc_topk_sync_distributed.py
git commit -m "refactor: share ARC gradient synchronization"
```

---

### Task 2: Migrate ArcTopKMuon to the shared synchronization layer

**Files:**
- Modify: `dion/muon_arctopk.py`
- Modify: `tests/test_muon_arctopk.py`
- Modify: `tests/test_muon_arctopk_distributed.py`

**Interfaces:**
- Consumes: all Task 1 interfaces.
- Produces: unchanged public `ArcTopKMuon` constructor and checkpoint keys; `_create_ortho_tasks()` now uses the shared grouping and synchronization adapter.

- [ ] **Step 1: Add a RED contract test that observes the shared boundary**

Patch only `dion.muon_arctopk.synchronize_arc_batch_async` with a generator fake that returns supplied sentinel gradients, then execute one optimizer step with a no-op Newton–Schulz function. Assert Muon momentum consumes the sentinel and that the call receives `ArcTopKSyncConfig(0.2, 4, 0.1, 42, 0)` and stable task indices. This test guards the module boundary; existing distributed tests continue to exercise the real compressor.

- [ ] **Step 2: Run the new test and verify RED**

```bash
uv run --frozen --extra dev pytest tests/test_muon_arctopk.py -v
```

Expected: failure because `ArcTopKMuon` still calls `arc_topk_ef21m_async` directly.

- [ ] **Step 3: Refactor Muon without changing its post-sync update**

Replace local grouping and ARC state initialization with Task 1 helpers. Keep these operations in their current order after synchronization:

```text
Muon momentum/Nesterov -> megabatch orthogonalize -> LR adjustment -> weight update
```

Move `average_gradients_async()` to the shared module and update dense embedding/lm-head AdamW/Lion callers to import it. Preserve old checkpoint migration: absent `arc_start_compress_step` becomes `0`.

- [ ] **Step 4: Run all M001 optimizer tests**

```bash
uv run --frozen --extra dev pytest tests/test_muon_arctopk.py tests/test_muon_arctopk_distributed.py tests/test_train_arctopk.py -v
```

Expected: all pass, including state-dict round trip, warmup semantics, missing gradients, and two-rank parameter equality.

- [ ] **Step 5: Commit the migration**

```bash
git diff --check
git add dion/muon_arctopk.py tests/test_muon_arctopk.py tests/test_muon_arctopk_distributed.py
git commit -m "refactor: route ARC Muon through shared synchronizer"
```

---

### Task 3: Add pure AdamW with shared ARC synchronization

**Files:**
- Create: `dion/adamw_arctopk.py`
- Modify: `dion/__init__.py`
- Create: `tests/test_adamw_arctopk.py`
- Create: `tests/test_adamw_arctopk_distributed.py`

**Interfaces:**
- Consumes: Task 1 synchronization APIs and `adamw_update_foreach_async()` from `dion.scalar_opts`.
- Produces:

```python
class ArcTopKAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        process_group: Optional[ProcessGroup] = None,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        arc_topk_ratio: float = 0.2,
        arc_projection_rank: int = 4,
        arc_eta: float = 0.1,
        arc_seed: int = 42,
        arc_start_compress_step: int = 0,
    )
```

- Param groups carry `arc_compress: bool`; compressed groups must contain only 2D parameters. The GPT factory passes `arc_compress=True` only for transformer block matrices and `False` for embedding/lm-head.
- Every parameter has AdamW `momentum`, `variance`, and FP32 device `step_dev` state. Compressed parameters additionally have the three ARC state tensors.
- `step()` increments one optimizer-wide integer `_arc_step`, schedules compressed shape batches in Task 1 order, dense-averages uncompressed gradients, and then calls fused AdamW with the synchronized gradients. It never replaces `param.grad` permanently.
- `state_dict()` stores `_arc_step` in each param group's `arc_step` field; `load_state_dict()` restores it and migrates missing `arc_start_compress_step` to `0`.

- [ ] **Step 1: Write constructor/state RED tests**

Test invalid ARC arguments, a non-2D compressed parameter, state prepopulation, correct absence of ARC states on dense parameters, and export via `from dion import ArcTopKAdamW`.

- [ ] **Step 2: Run tests and verify RED**

```bash
uv run --frozen --extra dev pytest tests/test_adamw_arctopk.py -v
```

Expected: import failure because `ArcTopKAdamW` does not exist.

- [ ] **Step 3: Implement the smallest optimizer shell and state initialization**

Follow `DistributedOrthoBase._get_or_initialize_state()` for state tensor dtype/device rules, but do not subclass `DistributedOrthoBase`: AdamW must not initialize Newton–Schulz machinery. Reject `DeviceMesh` explicitly if passed instead of a `ProcessGroup`.

- [ ] **Step 4: Write local numerical RED tests**

For world size 1 and `ratio=1`, `eta=1`, compare two steps against `torch.optim.AdamW` using copied FP32 parameters and literal gradients. Set identical LR, betas, epsilon, and weight decay. Assert parameter, first moment, and second moment agreement within explicit tolerances.

- [ ] **Step 5: Implement optimizer step and verify local GREEN**

Use `AsyncTask`/`AsyncRuntime(max_concurrent_tasks=3)` with tasks emitted in stable shape-group order, matching `ArcTopKMuon`. Do not create a second scheduler or execute AdamW collectives in a different order.

- [ ] **Step 6: Write distributed RED tests**

Two Gloo ranks must verify:

1. Different local transformer gradients lead to identical parameters after step 1 and compressed step 2.
2. Dense embedding/lm-head groups average gradients and remain rank-identical.
3. A missing local gradient cannot change collective order.
4. `state_dict()` round trip resumes with the same next-step update.
5. With `ratio=1`, `eta=1`, both parameters and AdamW states match dense-gradient AdamW.

- [ ] **Step 7: Implement distributed behavior and run all AdamW tests**

```bash
uv run --frozen --extra dev pytest tests/test_adamw_arctopk.py tests/test_adamw_arctopk_distributed.py -v
```

- [ ] **Step 8: Commit AdamW ARC**

```bash
git diff --check
git add dion/adamw_arctopk.py dion/__init__.py tests/test_adamw_arctopk.py tests/test_adamw_arctopk_distributed.py
git commit -m "feat: add shared ARC Top-K AdamW"
```

---

### Task 4: Implement the reproducible 2x2 benchmark and result summarizer

**Files:**
- Create: `benchmark/compressed_muon/benchmark_arc_2x2.py`
- Create: `benchmark/compressed_muon/profiler_trace.py`
- Create: `benchmark/compressed_muon/summarize_arc_2x2.py`
- Create: `tests/test_benchmark_arc_2x2.py`

**Interfaces:**
- Consumes: `models.gpt_model.GPT`, `ArcTopKAdamW`, `ArcTopKMuon`, and Task 1 byte estimator.
- Produces one JSON object with schema:

```json
{
  "schema_version": 1,
  "experiment_id": "CM002a-adamw-dense-gpt60m-ddp-ws4-s42",
  "optimizer": "adamw",
  "sync_mode": "dense",
  "transport": "normal",
  "model": {"label": "gpt60m", "dim": 512, "layers": 4, "heads": 8, "parameters": 64094208},
  "workload": {"world_size": 4, "local_batch": 1, "sequence_length": 256, "gradient_accumulation": 1, "dtype": "bfloat16"},
  "arc": {"ratio": 0.2, "projection_rank": 4, "eta": 0.1, "seed": 42, "start_compress_step": 0},
  "timing_ms": {"step_samples": [], "fwd_bwd_samples": [], "optimizer_samples": [], "step_mean": 0.0},
  "throughput": {"tokens_per_second": 0.0},
  "memory": {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0},
  "communication": {"dense_gradient_bytes": 0, "arc_seed_bytes": 0, "arc_sketch_bytes": 0, "arc_selected_values_bytes": 0, "uncompressed_bytes": 0},
  "profiler": {"trace_path": null, "nccl_kernel_time_ms": null, "collectives": []},
  "environment": {"git_commit": "", "torch": "", "cuda": "", "nccl": "", "gpu_names": []}
}
```

- Exact model presets:

```python
MODEL_PRESETS = {
    "gpt60m":  dict(model_dim=512,  n_layer=4,  n_head=8),
    "gpt130m": dict(model_dim=768,  n_layer=8,  n_head=12),
    "gpt350m": dict(model_dim=1024, n_layer=20, n_head=16),
    "gpt1b":   dict(model_dim=1536, n_layer=30, n_head=24),
}
```

- Required CLI:

```text
--experiment-id --optimizer {adamw,muon} --sync {dense,arc}
--model {gpt60m,gpt130m,gpt350m,gpt1b} --warmup-steps --measure-steps
--seed --output --profile-output --compile-model/--no-compile-model
```

- [ ] **Step 1: Write RED tests for presets, CLI validation, and JSON schema**

Tests must instantiate configuration only; CUDA is not required. Reject `sync=arc` with an unsupported optimizer, warmup below 20, measure steps below 100 in formal mode, world size other than 4 in formal mode, or an experiment ID inconsistent with optimizer/sync/model.

- [ ] **Step 2: Run benchmark tests and verify RED**

```bash
uv run --frozen --extra dev pytest tests/test_benchmark_arc_2x2.py -v
```

Expected: import failure because the benchmark module does not exist.

- [ ] **Step 3: Implement optimizer factories with a single compressed-set selector**

Use this partition for all variants:

```python
compressed = list(model.transformer.h.parameters())
uncompressed = [*model.transformer.wte.parameters(), *model.lm_head.parameters()]
```

Dense AdamW uses ordinary `torch.optim.AdamW` with standard DDP synchronization. Dense Muon uses existing `Muon` with standard DDP synchronization. Both ARC variants execute every backward inside `DDP.no_sync()` and synchronize gradients inside their optimizers. For Muon ARC, embedding/lm-head stay on its existing dense optimizer-side path.

- [ ] **Step 4: Implement synchronized CUDA-event timing**

Use separate event pairs for forward/backward, optimizer, and full step. Record events on every measured step, synchronize once after the measured interval, materialize samples, then reduce rank-level summary statistics with `dist.all_reduce(op=MAX)` so reported critical-path time is the slowest rank. Do not call `.item()`, write files, update tqdm, or print per step inside the interval.

- [ ] **Step 5: Add deterministic synthetic data and warmup semantics**

Set Python and torch seeds to `42 + rank`; initialize identical model weights by setting seed `42` before model construction on every rank. Pre-generate rank-local integer token/target batches outside measurement. Run compile-trigger steps first, reset optimizer/model to a freshly constructed identical instance, run at least 20 warmup steps, reset peak-memory counters, then measure at least 100 steps. This avoids including compilation while preventing compilation steps from mutating measured optimizer state.

- [ ] **Step 6: Add profiler mode separate from timing mode**

Profiler mode runs 3 wait + 3 warmup + 5 active steps and exports a Chrome trace from rank 0. Add `record_function` ranges named `benchmark/forward_backward`, `benchmark/optimizer`, `arc/projection`, `arc/topk`, `arc/selected_values`, `arc/ef21m`, `muon/newton_schulz`, and `muon/result_collective`. If a range spans generator yields, place ranges around the actual submitted work rather than around the whole generator.

- [ ] **Step 7: Write RED trace-attribution tests**

Create a minimal synthetic Chrome trace fixture with literal timestamps for a DDP bucket All-Reduce, ARC sketch All-Reduce, ARC selected-values All-Reduce, Muon result collective, and compute kernels on another stream. Assert exact NCCL kernel counts/durations per category, message sizes copied from recorded collective metadata, union duration, compute-overlap duration, and exposed duration. A kernel without a recognized parent range must be reported as `unattributed`, not silently discarded.

- [ ] **Step 8: Implement profiler trace attribution**

Correlate CPU launch events/user ranges and GPU NCCL kernels using trace correlation/external IDs. Compute interval unions before overlap so concurrent NCCL kernels are not double-counted. Define `exposed = nccl_union - intersection(nccl_union, non_nccl_compute_union)` and label it as a trace-derived estimate in JSON. Preserve raw total, overlap, and exposed values separately.

- [ ] **Step 9: Write RED summary-math tests**

Using literal repetition means `[10.0, 11.0, 9.0]`, assert mean `10.0`, sample standard deviation `1.0`, and CV `0.1`. Using dense `10.0` and ARC `7.5`, assert `R_step=0.25`. Add equivalent hand-derived assertions for `R_bytes` and `R_grad_comm`; reject missing or mismatched transport/model/workload fields.

- [ ] **Step 10: Implement the summarizer**

The summarizer reads exactly three or more result JSON files per cell, verifies invariant fields, reports mean/sample standard deviation/CV for step, optimizer, NCCL, throughput, and memory, and computes:

```text
R_bytes     = 1 - ARC communication bytes / Dense communication bytes
R_grad_comm = 1 - ARC NCCL gradient-sync time / Dense NCCL gradient-sync time
R_step      = 1 - ARC full-step time / Dense full-step time
```

Exit nonzero when any step-time CV exceeds `0.05`; preserve the summary JSON even when exiting nonzero so the instability is inspectable.

- [ ] **Step 11: Run benchmark unit tests and commit**

```bash
uv run --frozen --extra dev pytest tests/test_benchmark_arc_2x2.py -v
uv run --frozen --extra dev python -m compileall -q benchmark/compressed_muon dion tests
git diff --check
git add benchmark/compressed_muon/benchmark_arc_2x2.py benchmark/compressed_muon/profiler_trace.py benchmark/compressed_muon/summarize_arc_2x2.py tests/test_benchmark_arc_2x2.py
git commit -m "bench: add ARC AdamW Muon 2x2 harness"
```

---

### Task 5: Run CPU/Gloo regression gate and 4-GPU smoke matrix

**Files:**
- Create after GPU selection: `artifacts/compressed_muon/smoke-m001-arc-2x2-gpt60m-ws4-s42/command.txt`
- Create after GPU selection: `artifacts/compressed_muon/smoke-m001-arc-2x2-gpt60m-ws4-s42/environment.txt`
- Create after GPU selection: `artifacts/compressed_muon/smoke-m001-arc-2x2-gpt60m-ws4-s42/stdout.log`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: regression evidence and eight short smoke outcomes: 2 optimizers × 2 sync modes × 2 transports.

- [ ] **Step 1: Run focused and related regression tests**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk.py tests/test_arc_topk_distributed.py \
  tests/test_arc_topk_sync.py tests/test_arc_topk_sync_distributed.py \
  tests/test_muon_arctopk.py tests/test_muon_arctopk_distributed.py \
  tests/test_adamw_arctopk.py tests/test_adamw_arctopk_distributed.py \
  tests/test_train_arctopk.py tests/test_train_factories.py \
  tests/test_configs.py tests/test_state_prepopulation.py tests/test_optimizers.py -v
```

Record passed/failed/skipped counts and the first root-cause traceback for any failure.

- [ ] **Step 2: Inspect GPUs without disturbing existing jobs**

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Select four idle GPUs. If fewer than four are idle, do not launch; record the blocker and re-check later. Never terminate an existing PID.

- [ ] **Step 3: Run eight 5-step smoke cases outside formal mode**

Use `--warmup-steps 2 --measure-steps 5 --smoke` and one fresh `torchrun` per cell. Enable `NCCL_DEBUG=INFO` for only the first normal and first P2P-disabled case, saving stdout. Confirm normal logs show the actual selected transport and P2P-disabled logs do not use P2P while SHM remains enabled.

- [ ] **Step 4: Validate smoke outputs**

For every cell assert exit code 0, finite loss, finite parameters, four-rank parameter checksum agreement, expected sync mode, nonzero relevant byte counters, identical collective signature across ranks, and no validation/W&B/checkpoint activity.

- [ ] **Step 5: Run one short profiler smoke**

Capture a rank-0 trace for P2P-disabled Muon ARC, open/parse its event table, and verify the trace contains ARC ranges, NCCL kernels, Muon Newton–Schulz, and full-step ranges before launching formal repeats.

---

### Task 6: Run formal 60M experiments under normal and P2P-disabled NCCL

**Files:**
- Create: `artifacts/compressed_muon/CM002a-adamw-dense-gpt60m-ddp-ws4-s42/`
- Create: `artifacts/compressed_muon/CM002b-m001-adamw-arc-gpt60m-ddp-ws4-s42/`
- Create: `artifacts/compressed_muon/CM002c-muon-dense-gpt60m-ddp-ws4-s42/`
- Create: `artifacts/compressed_muon/CM002d-m001-muon-arc-gpt60m-ddp-ws4-s42/`
- Create: `artifacts/compressed_muon/CM003a-adamw-dense-gpt60m-ddp-ws4-s42/`
- Create: `artifacts/compressed_muon/CM003b-m001-adamw-arc-gpt60m-ddp-ws4-s42/`
- Create: `artifacts/compressed_muon/CM003c-muon-dense-gpt60m-ddp-ws4-s42/`
- Create: `artifacts/compressed_muon/CM003d-m001-muon-arc-gpt60m-ddp-ws4-s42/`

**Interfaces:**
- Consumes: verified Task 5 harness.
- Produces: three independent timing results per experiment, representative P2P-disabled profiler traces, and normal/P2P-disabled summary JSON files.

- [ ] **Step 1: Register eight experiments as `planned`**

Add rows to `docs/compressed_muon/EXPERIMENTS.md`. CM002a–d are normal NCCL; CM003a–d set `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`. Each row points at its exact artifact directory.

- [ ] **Step 2: Write exact artifact metadata before launch**

Each directory gets the fully resolved benchmark arguments in `config.yaml`, exact `torchrun` command in `command.txt`, `git rev-parse HEAD`, `git status --short`, `pip/uv` environment identity, torch/CUDA/NCCL versions, GPU UUIDs, relevant NCCL environment, hostname, and start timestamp. Do not record secrets or the full environment variable set.

- [ ] **Step 3: Launch normal-transport repetitions in tmux**

For each CM002 cell, run three independent `torchrun --standalone --nproc-per-node=4` launches with fresh processes, 20 warmup steps, and 100 measured steps. Use the exact same four `CUDA_VISIBLE_DEVICES` values and rotate cell order by repetition to reduce thermal/order bias:

```text
repeat 1: AdamW Dense -> AdamW ARC -> Muon Dense -> Muon ARC
repeat 2: Muon ARC -> Muon Dense -> AdamW ARC -> AdamW Dense
repeat 3: AdamW ARC -> Muon Dense -> Muon ARC -> AdamW Dense
```

- [ ] **Step 4: Summarize normal transport and enforce CV gate**

Run `summarize_arc_2x2.py`. If any cell's step-time CV exceeds 5%, inspect GPU contention, clock/power variation, and outlier samples, then add two fresh independent repetitions for every affected cell. Never silently delete an outlier.

- [ ] **Step 5: Launch P2P-disabled repetitions in tmux**

Repeat Step 3 in CM003 directories with these exact environment variables in every launch:

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0
```

Keep `NCCL_DEBUG` unset during formal timing.

- [ ] **Step 6: Capture profiler measurements outside timed repetitions**

For each CM003 cell, run profiler mode three times with fresh processes and the same GPUs/transport. Store all three profiler summaries and retain at least the first full Chrome trace under `profiler/`; keeping the other full traces is optional if their summary JSON and checksums are retained. Do not mix profiled step samples into timing means. Use these three profiler summaries for `R_grad_comm` variability; do not treat the three timing repetitions as communication-time repetitions.

- [ ] **Step 7: Summarize P2P-disabled transport and compute decisions**

Report AdamW and Muon `R_bytes`, `R_grad_comm`, and `R_step` with repetition variability. Apply these precommitted rules:

```text
abs(Muon R_bytes - AdamW R_bytes) <= 0.05
    => approximately equal theoretical gradient-byte reduction

abs(Muon R_grad_comm - AdamW R_grad_comm) <= 0.05
and repetition uncertainty does not reverse the conclusion
    => approximately equal measured gradient-communication benefit

R_grad_comm values close, but Muon R_step materially lower
    => investigate Muon compute, non-gradient communication, ARC overhead, or overlap;
       do not claim ARC fails to compress Muon

R_step values close and both positive
    => similar wall-clock benefit in this benchmark setting
```

- [ ] **Step 8: Mark experiment status and commit metadata/docs**

Mark every run `completed`, `failed`, or `stopped` according to observed outcome. Do not label an environment-blocked run as a method failure.

---

### Task 7: Record results and decide whether to scale beyond 60M

**Files:**
- Modify: `docs/worklog/M001-arc-topk-ef21m-muon.md`
- Modify: `docs/compressed_muon/EXPERIMENTS.md`
- Modify only if evidence is stable: `docs/compressed_muon/RESULTS.md`

**Interfaces:**
- Consumes: Task 6 raw and summarized results.
- Produces: auditable conclusions and a scale-up decision.

- [ ] **Step 1: Append a dated worklog entry in Chinese**

Include purpose/hypothesis, code commit, exact eight experiment IDs, GPU identities, normal and P2P-disabled settings, dtype, warmup/measurement/repetition counts, communication bytes, collective counts/sizes, NCCL time, exposed optimizer time, ARC phase time, forward/backward, Newton–Schulz, full step, throughput, memory, CVs, profiler paths, failures, and interpretation limits.

- [ ] **Step 2: Separate observations from inferences**

State explicitly that a synthetic 100-step benchmark does not establish convergence or time-to-quality. If profiler attribution cannot isolate overlap precisely, report the directly measured ranges and label exposed communication time as an inference rather than a direct measurement.

- [ ] **Step 3: Apply the scale-up gate**

Proceed to 130M only if all eight 60M cells are correct, all required metrics exist, CV is at most 5% after permitted reruns, and ARC mode actually reduces logical bytes. Proceed from 130M to 350M under the same gate. Attempt 1B only after a single-process memory probe and a four-rank smoke fit without OOM. Use new experiment numbers for each model size; do not reuse CM002/CM003.

- [ ] **Step 4: Run final verification before claiming completion**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk.py tests/test_arc_topk_distributed.py \
  tests/test_arc_topk_sync.py tests/test_arc_topk_sync_distributed.py \
  tests/test_muon_arctopk.py tests/test_muon_arctopk_distributed.py \
  tests/test_adamw_arctopk.py tests/test_adamw_arctopk_distributed.py \
  tests/test_benchmark_arc_2x2.py tests/test_train_arctopk.py \
  tests/test_train_factories.py tests/test_configs.py tests/test_state_prepopulation.py tests/test_optimizers.py -v
uv run --frozen --extra dev python -m compileall -q dion benchmark/compressed_muon tests train.py train_arctopk.py
git diff --check
```

- [ ] **Step 5: Commit the research record**

Commit reusable code, tests, and research-document updates. Do not commit large Chrome traces or raw logs unless repository policy explicitly tracks them; keep their paths and checksums in the worklog.

---

## Completion Criteria

Implementation is complete only when:

1. Both ARC optimizers call the same synchronization adapter and existing M001 tests still pass.
2. `ArcTopKAdamW` passes local, two-rank, missing-gradient, dense-equivalence, and state-resume tests.
3. The benchmark reports synchronized CUDA-event timing, logical bytes, collective signatures, profiler NCCL time, throughput, and peak memory.
4. The 60M smoke matrix proves all eight paths before formal measurement.
5. CM002a–d and CM003a–d each contain at least three independent valid runs or a documented failure/blocker.
6. The experiment registry and M001 worklog match the raw artifacts and do not overstate short-benchmark evidence.
