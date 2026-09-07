# ARC-TopK Local Deterministic Seed and Full DDP Bucket Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Follow strict red-green-refactor and stop at every review checkpoint.

**Goal:** Remove per-task seed synchronization from optimizer-side ARC-TopK, then move complete ARC-TopK/EF21M data-parallel gradient synchronization into a real asynchronous DDP communication hook so compressed communication can overlap backward.

**Architecture:** Keep `ArcTopKMuon` as the corrected optimizer-side comparison path and add `arc_seed_mode=broadcast|local_deterministic`, with `broadcast` remaining the compatibility default. The new hook path registers one stateful DDP bucket hook, maps bucket views to stable parameter identities, runs complete EF21M dense/full-support or sparse sketch/selected-value synchronization, returns a `Future[Tensor]`, and then lets ordinary `Muon` consume synchronized gradients. A single cross-bucket tail Future fixes collective order in the first production version. This plan deliberately skips the selective-dense intermediate hook and never uses per-parameter autograd hooks.

**Tech Stack:** Python 3.10, PyTorch 2.11+ DDP/GradBucket/Future, Gloo, NCCL, DCP, pytest, Bash, existing GPT-350M training and profiler entries.

**Spec:** `docs/superpowers/specs/2026-09-06-arc-topk-ddp-bucket-hook-design.md`

## Global constraints and acceptance boundary

- Do not change production behavior before the corresponding focused test fails for the intended reason.
- Preserve low-level/optimizer API `arc_seed_mode=broadcast` semantics, including the existing behavior in which rank 0's configured base seed wins. The dedicated training entry uses an unset seed-mode default and derives `optimizer -> broadcast`, `ddp_hook -> local_deterministic`.
- `local_deterministic` must use fixed integer arithmetic only. It must not use Python `hash()`, global RNG state, rank, a transient bucket index, or callback arrival order.
- `local_deterministic` performs one initialization/resume fingerprint validation, but performs no seed collective, device-to-host `.item()`, or seed tensor allocation in a training step.
- `arc_sync_mode=ddp_hook` requires `arc_seed_mode=local_deterministic`; the compatibility `broadcast` seed path is supported only by optimizer-side ARC.
- The full hook owns only data-parallel gradient synchronization. Ordinary Muon still owns momentum, orthogonalization, Muon result communication, and parameter updates.
- With gradient accumulation, only the final micro-batch enables the reducer/hook. The first `N-1` micro-batches keep forward and backward together under `DDP.no_sync()`.
- Hook mode requires DDP, `find_unused_parameters=False`, a static participating parameter set, and exactly one synchronization role for every optimizer parameter.
- The first hook implementation globally orders complete bucket chains through one tail Future. Do not add cross-bucket concurrency until profiler evidence shows the tail is the next bottleneck.
- A blocking hook may be used only inside an isolated test prototype. The production hook must return an actual completion Future and must not call `Work.wait()` on the callback critical path.
- Do not implement a selective-dense hook or per-parameter autograd hook.
- Do not claim speedup from API shape or collective byte counts. Acceptance requires a profiler trace showing ARC communication GPU kernels overlapping later genuine backward compute kernels plus repeated end-to-end wall-clock improvement.

---

### Task 1: Add explicit seed modes without changing the compatibility path

**Files:**
- Modify: `dion/arc_topk.py`
- Modify: `dion/arc_topk_sync.py`
- Modify: `dion/muon_arctopk.py`
- Modify: `dion/adamw_arctopk.py`
- Modify: `train_arctopk.py`
- Modify: `tests/test_arc_topk.py`
- Modify: `tests/test_arc_topk_distributed.py`
- Modify: `tests/test_arc_topk_sync.py`
- Modify: `tests/test_muon_arctopk.py`
- Modify: `tests/test_train_arctopk.py`

**Interfaces:**

```python
ArcSeedMode = Literal["broadcast", "local_deterministic"]

def validate_arc_seed_mode(mode: str) -> None: ...

def derive_arc_seed(*, base_seed: int, step: int, stable_task_id: int) -> int: ...

class ArcTopKSyncConfig:
    seed_mode: ArcSeedMode = "broadcast"
    seed_scheme_version: int = 1

class ArcTopKMuon(Muon):
    def __init__(..., arc_seed_mode: ArcSeedMode = "broadcast",
                 arc_parameter_names: Mapping[Parameter, str] | None = None): ...
```

`derive_arc_seed` uses the current arithmetic `base_seed + step * 1_000_003 + stable_task_id`, normalized into the range accepted by `torch.Generator.manual_seed`. This preserves the existing projection sequence for ordinary non-negative inputs when all ranks already agree, while making overflow/negative behavior explicit. `ArcTopKMuon` stores `arc_seed_mode` in each Muon group and its state dict. The CLI/config field is nullable `arc_seed_mode`; only the low-level constructor defaults directly to `broadcast`.

- [ ] **Step 1: Write failing validation and deterministic-derivation tests**

Add tests asserting that unsupported modes raise, identical triples produce the same integer across processes, changing any component changes the result, standard positive inputs exactly match the historical formula, large and negative base seeds remain valid, and calls do not mutate `torch.random.get_rng_state()`.

- [ ] **Step 2: Write the failing local-mode primitive test**

Instrument `dist.broadcast` and `Tensor.item` around a compressed `arc_topk_ef21m_async` step. Assert that `broadcast` mode retains one seed broadcast, while `local_deterministic` performs neither a seed broadcast nor `.item()` and produces the same projection on two independent invocations. With identical rank base seeds and stable task IDs, run at least three lossy EF21M steps and require local mode to match the broadcast path's support, three compressor states, synchronized gradient, momentum, and parameter trajectory within dtype tolerance.

- [ ] **Step 3: Run RED**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk.py \
  tests/test_arc_topk_sync.py \
  tests/test_muon_arctopk.py \
  tests/test_train_arctopk.py -v
```

Expected: fail because `ArcSeedMode`, `derive_arc_seed`, `seed_mode`, and the CLI field do not exist.

- [ ] **Step 4: Implement the minimal dual path**

Keep the current seed tensor/broadcast/`.item()` code under `broadcast`. Under `local_deterministic`, compute the seed on the host before projection construction and bypass that entire block. Pass `stable_task_id` through `synchronize_arc_batch_async`; for the optimizer route it is the frozen first-seen shape/dtype task index. Distributed local mode requires `arc_parameter_names`, supplied by `train_arctopk.init_arc_topk_optimizer` from `raw_model.named_parameters()`; direct distributed callers that omit it fail before the first step.

- [ ] **Step 5: Preserve state-dict compatibility**

Old checkpoints missing `arc_seed_mode` load as `broadcast`. New state dicts persist the selected mode. Add an explicit migration assertion to `tests/test_muon_arctopk.py`.

- [ ] **Step 6: Run GREEN and the existing optimizer regressions**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk.py \
  tests/test_arc_topk_distributed.py \
  tests/test_arc_topk_sync.py \
  tests/test_arc_topk_sync_distributed.py \
  tests/test_muon_arctopk.py \
  tests/test_muon_arctopk_distributed.py \
  tests/test_adamw_arctopk.py \
  tests/test_adamw_arctopk_distributed.py \
  tests/test_train_arctopk.py -v
```

- [ ] **Step 7: Review checkpoint**

Confirm from a profiler/observer unit fixture that local mode reports `arc_seed_bytes == 0` and no `arc/seed` range. Do not yet infer wall-clock improvement.

---

### Task 2: Add one-time canonical layout fingerprint validation

**Files:**
- Create: `dion/arc_topk_layout.py`
- Create: `tests/test_arc_topk_layout.py`
- Create: `tests/test_arc_topk_layout_distributed.py`
- Modify: `dion/muon_arctopk.py`
- Modify: `train_arctopk.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ArcParameterDescriptor:
    stable_name: str
    stable_id: int
    shape: tuple[int, ...]
    dtype: str
    role: Literal["arc_matrix", "dense_aux"]

@dataclass(frozen=True)
class ArcOptimizerTaskDescriptor:
    group_id: int
    task_id: int
    ordered_parameter_names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    config: ArcTopKSyncConfig

def canonical_arc_fingerprint(
    *, base_seed: int, config: ArcTopKSyncConfig,
    group_ranks: Sequence[int], parameters: Sequence[ArcParameterDescriptor],
    optimizer_tasks: Sequence[ArcOptimizerTaskDescriptor] | None = None,
) -> str: ...

def validate_arc_fingerprint_across_ranks(
    fingerprint: str, process_group: ProcessGroup,
) -> None: ...
```

The canonical payload uses sorted JSON with an explicit schema version and SHA-256. Cross-rank validation uses one initialization-time `all_gather_object` or fixed-size digest all-gather and reports all mismatching ranks before failing.

- [ ] **Step 1: Write failing canonicalization tests**

Cover stable equality, name/order/shape/dtype/role/config changes, independence from object identity, and rejection of duplicate stable names or IDs.

- [ ] **Step 2: Write a failing two-rank mismatch test**

Use real Gloo. Equal descriptors pass; differing seed, parameter role, or ordered parameter table makes both ranks raise the same `ArcTopKLayoutMismatch` without hanging.

- [ ] **Step 3: Run RED**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk_layout.py tests/test_arc_topk_layout_distributed.py -v
```

- [ ] **Step 4: Implement canonicalization and collective validation**

Define two different canonical layouts; do not compare their digests to each other:

- optimizer layout: the exact runtime order of `(group_id, task_id, ordered_parameter_names, shape, dtype, per-group ARC config)` after param-group traversal and first-seen shape/dtype batching;
- hook layout: the stable ordered parameter descriptors used by the hook, independent of transient bucket IDs.

Build the optimizer layout once from `raw_model.named_parameters()` and the constructed optimizer groups. Cache the verified fingerprint and frozen task table on the optimizer; do not rebuild or validate per optimizer step. Reject `add_param_group` after freezing. On load, verify that restored param groups reproduce the frozen task table before accepting the compressor step.

- [ ] **Step 5: Run GREEN and verify no per-step validation**

Add an observer count over three optimizer steps: the layout validation collective occurs once, while seed collectives remain zero. Add a two-rank case with the same model parameter table but different optimizer group boundaries/order; both ranks must fail before launching the first training collective.

---

### Task 3: Extract reusable batched EF21M prepare/finalize primitives

**Files:**
- Modify: `dion/arc_topk.py`
- Modify: `dion/arc_topk_sync.py`
- Create: `tests/test_arc_topk_ef21m_primitives.py`
- Modify: `tests/test_arc_topk_distributed.py`

**Interfaces:**

```python
@dataclass
class ArcPreparedBatch:
    tracker_batch: Tensor
    local_estimate_batch: Tensor
    global_estimate_batch: Tensor
    delta_batch: Tensor | None
    projection_batch: Tensor | None
    local_sketch_batch: Tensor | None

def prepare_arc_batch(
    gradient_batch: Tensor, tracker_batch: Tensor, local_estimate_batch: Tensor,
    global_estimate_batch: Tensor, *, config: ArcTopKSyncConfig,
    step: int, projection_batch: Tensor | None,
) -> ArcPreparedBatch: ...

def finalize_arc_full_support_(prepared: ArcPreparedBatch,
                               averaged_tracker_batch: Tensor) -> Tensor: ...

def finalize_arc_sparse_(prepared: ArcPreparedBatch, indices: Tensor,
                         local_selected: Tensor,
                         averaged_selected: Tensor) -> Tensor: ...
```

These functions accept both singleton batches and the existing 3D shape batches, perform no distributed collective, and do not copy state through temporary Python dictionaries. The optimizer adapter must preserve the existing single batched projection generation, BMM, membership/order, and collective sequence; it must not turn a same-shape group into per-parameter projection calls or small kernels. The hook may use singleton batches with parameter-level seeds.

- [ ] **Step 1: Write a hand-calculated multi-step oracle**

Cover step 1, final warmup step, first compressed step, subsequent compressed step, `ratio=1`, `eta<1`, BF16 tolerances, and local/global estimate divergence.

- [ ] **Step 2: Run RED, implement, and route the old path through the primitives**

```bash
uv run --frozen --extra dev pytest tests/test_arc_topk_ef21m_primitives.py -v
```

Then implement the minimum functions and replace duplicated state math inside `arc_topk_ef21m_async` without changing projection generation, batch layout, BMM shape, or collective order.

- [ ] **Step 3: Prove optimizer-side behavior did not regress**

Run the full ARC optimizer suite from Task 1. For `broadcast`, require the old distributed literal test to retain rank-0 seed semantics. With identical base seeds and layout, require optimizer `local_deterministic` to match the broadcast projection/support/state trajectory; the hook's parameter-level seed scheme remains a separate, versioned trajectory.

---

### Task 4: Build stable hook metadata and lifecycle state

**Files:**
- Create: `dion/arc_topk_ddp_hook.py`
- Create: `tests/test_arc_topk_ddp_hook_state.py`
- Modify: `dion/__init__.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ArcTopKDDPParameterSpec:
    parameter: torch.nn.Parameter
    stable_name: str
    stable_id: int
    role: Literal["arc_matrix", "dense_aux"]

class ArcTopKDDPState:
    def begin_step(self) -> int: ...
    def parameter_state(self, parameter: Parameter) -> ArcParameterState: ...
    def note_bucket(self, bucket: dist.GradBucket) -> BucketContext: ...
    def finish_step(self) -> None: ...
    def commit_step(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
```

`BucketContext` captures the bucket buffer/views, the hook-entry CUDA stream, and a bucket-ready CUDA event recorded immediately when DDP invokes the hook. Construction receives the DDP process group, verified hook fingerprint, ordered specs, ARC config, and compression start. Runtime lookup uses parameter object identity; persistence uses stable name. `begin_step()` reserves `committed_step + 1` without committing it and resets per-step coverage/tail state. `finish_step()` requires all expected parameters to have appeared exactly once and the tail to be complete. `commit_step()` runs only after successful `optimizer.step()` and advances the persisted counter.

- [ ] **Step 1: Write failing metadata tests**

Cover duplicate/missing roles, unsupported matrix rank, parameter not owned by optimizer, optimizer parameter absent from model, stable lookup after simulated bucket reorder, and state preservation after reorder.

- [ ] **Step 2: Write failing lifecycle tests**

Require exactly one `begin_step`, no second step while a tail is in flight, consistent step capture across all buckets, exact coverage at `finish_step`, and deterministic errors for unsupported unused-parameter configuration.

- [ ] **Step 3: Implement only state and metadata**

Do not implement the compression hook yet. Use a completed CUDA-aware `torch.futures.Future(devices=[device])` as the initial tail, record the bucket-ready event at hook entry, and keep active `BucketContext` objects strongly referenced until their completion Future fires. Add an early DCP schema smoke fixture here: a fresh zero-initialized state must preallocate the full stable-name tensor schema before load.

- [ ] **Step 4: Run GREEN**

```bash
uv run --frozen --extra dev pytest tests/test_arc_topk_ddp_hook_state.py -v
```

---

### Task 5: Implement and prove the cross-bucket Future sequencer

**Files:**
- Modify: `dion/arc_topk_ddp_hook.py`
- Create: `tests/test_arc_topk_ddp_hook_future.py`
- Create: `tests/test_arc_topk_ddp_hook_future_distributed.py`

**Interfaces:**

```python
def bridge_future(source: torch.futures.Future,
                  destination: torch.futures.Future,
                  transform: Callable[[Any], Tensor]) -> None: ...

def enqueue_bucket_chain(state: ArcTopKDDPState, context: BucketContext,
                         launch: Callable[[BucketContext], Future[Tensor]]) \
                         -> Future[Tensor]: ...
```

- [ ] **Step 1: Write failing Future-shape and exception tests**

Verify the outer Future resolves to one tensor, not `Future[Future[Tensor]]`; exceptions from a collective callback reach the DDP-facing Future and poison later buckets; contexts remain live until completion and are released afterward. Install the new destination tail before attaching callbacks, and cover already-completed source Futures so inline callback re-entry cannot corrupt the chain or deadlock on a held lock.

- [ ] **Step 2: Write a failing real two-rank, multi-bucket order test**

Use a tiny DDP model with a small `bucket_cap_mb`; run enough iterations for DDP bucket rebuild, then assert that at least two real bucket callbacks occurred before applying injected rank-specific callback delays. The launch observer must show the identical global signature sequence on both ranks. The test must have a process-group timeout and unconditional `destroy_process_group()`.

- [ ] **Step 3: Run RED and implement the minimal dummy sequencer**

Use `Work.get_future()` and explicit CUDA-aware destination completion. Never depend on `.then()` flattening. Before delayed local prepare, make its execution stream wait on the bucket-ready event as well as the previous tail; tail ordering alone is insufficient. Do not use a host wait. Call `set_result(bucket.buffer())` from the stream that enqueued final scatter so the destination Future records the correct CUDA event. Avoid `from __future__ import annotations` on the registered hook or otherwise ensure its runtime annotations are exactly `GradBucket -> Future[Tensor]`, as required by DDP registration.

- [ ] **Step 4: Add an NCCL completion smoke test**

Mark it `pytest.mark.multi_gpu`; verify a non-default CUDA stream consumer sees completed bucket data and repeated iterations do not leak contexts or deadlock. Add a stronger stale-read test: delay writing a later bucket on a different CUDA stream while the preceding tail completes early, then require the hook to read the delayed value rather than an older buffer value.

---

### Task 6: Implement full-support and mixed dense bucket synchronization

**Files:**
- Modify: `dion/arc_topk_ddp_hook.py`
- Create: `tests/test_arc_topk_ddp_hook.py`
- Create: `tests/test_arc_topk_ddp_hook_distributed.py`

**Interfaces:**

```python
def arc_topk_ddp_hook(state: ArcTopKDDPState,
                      bucket: dist.GradBucket) -> Future[Tensor]: ...
```

- [ ] **Step 1: Write failing single-rank bucket reconstruction tests**

Use fake GradBucket fixtures to cover mixed shapes within one dtype/device bucket, `arc_matrix`/`dense_aux` roles, offsets, and exact output buffer shape/device/dtype. Test dtype handling with separate realistic buckets; do not assume DDP places mixed dtypes in one bucket.

- [ ] **Step 2: Write failing two-rank full-support tests**

With real DDP/Gloo, cover first step, warmup, `ratio=1`, `eta=1`, and general `eta`. Compare ARC slices to an EF21M tracker oracle and dense slices to exact DDP average. Verify rank-identical `.grad` and parameters after ordinary Muon step.

- [ ] **Step 3: Implement one packed full-support all-reduce per bucket**

Before enqueue, replace ARC slices with the updated tracker and leave dense slices as gradients. Divide the completed packed buffer by world size, update ARC global/local state correctly, restore bucket views, and complete the DDP Future.

- [ ] **Step 4: Prove no double gradient synchronization**

Register the observer and assert hook mode has hook-owned gradient collectives only. Ordinary Muon may still emit its result communication; classify it separately and do not suppress it.

---

### Task 7: Implement the complete sparse ARC sketch/value hook

**Files:**
- Modify: `dion/arc_topk_ddp_hook.py`
- Modify: `tests/test_arc_topk_ddp_hook.py`
- Modify: `tests/test_arc_topk_ddp_hook_distributed.py`
- Create: `tests/test_arc_topk_ddp_hook_nccl.py`

- [ ] **Step 1: Write failing sparse oracle tests**

Use `ratio<1`, multiple differently-shaped ARC parameters in one bucket, mixed dense parameters, at least three EF21M steps, and rank-distinct gradients. Assert `h_local`, `g_local`, `g_global`, support, reconstructed bucket values, and post-Muon parameters against a per-parameter oracle.

- [ ] **Step 2: Write failing collective-signature tests**

For each sparse bucket require a fixed signature: optional packed dense all-reduce, packed ARC sketch all-reduce, then packed selected-values all-reduce. Verify there is no `arc/seed` collective and no whole ARC-gradient dense all-reduce after compression starts.

- [ ] **Step 3: Implement local prepare and packed sketch**

Call `prepare_arc_batch` with a singleton batch per ARC view and a projection derived from `derive_arc_seed(... stable_id ...)`. Flatten sketches into one bucket buffer, enqueue async SUM, and average only in its completion callback.

- [ ] **Step 4: Implement TopK, local/averaged selected buffers, and finalize**

After sketch completion, compute support, gather rank-local values, clone a distinct averaged-values buffer, enqueue async SUM, and finalize local/global estimates after completion. Keep all projections, offsets, supports, and packing buffers alive in `BucketContext`.

- [ ] **Step 5: Run Gloo and NCCL stress tests**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk_ddp_hook.py \
  tests/test_arc_topk_ddp_hook_distributed.py -v
```

Then, on at least two idle GPUs:

```bash
uv run --frozen --extra dev pytest tests/test_arc_topk_ddp_hook_nccl.py -v
```

Repeat multiple iterations with small buckets, callback delays, allocator churn, and a process timeout. Any hang, rank divergence, unfinished Future, or retained bucket context fails the task.

- [ ] **Step 6: Review checkpoint**

Inspect the implementation for all forbidden synchronization points: `Work.wait`, `torch.cuda.synchronize`, Tensor `.item()`, Python polling, and device-to-host copies inside the hook. Any occurrence needs a documented removal or an explicit proof that it is outside the training-step path.

---

### Task 8: Integrate the hook with the formal training entry

**Files:**
- Modify: `train_arctopk.py`
- Modify: `train.py`
- Modify: `tests/test_train_arctopk.py`
- Create: `tests/test_train_arctopk_ddp_hook.py`
- Modify: `configs/compressed_muon/m001_arc_topk_muon_ddp.yaml`

**Interfaces:**

```python
ArcSyncMode = Literal["optimizer", "ddp_hook"]

@dataclass
class GradientSyncRuntime:
    optimizer_owns_gradient_sync: bool
    begin_step: Callable[[], None] | None = None
    finish_step: Callable[[], None] | None = None
    commit_step: Callable[[], None] | None = None
    checkpoint_state: Stateful | None = None

def init_arc_topk_optimizer(...) -> tuple[Optimizer, GradientSyncRuntime]: ...

def resolve_arc_sync_policy(*, arc_sync_mode: ArcSyncMode,
                            legacy_replicate_mesh_grad_sync: bool | None,
                            arc_seed_mode: ArcSeedMode) -> bool: ...
```

The shared `train.main` normalizes legacy factories returning only an optimizer to `GradientSyncRuntime(optimizer_owns_gradient_sync=hp.replicate_mesh_grad_sync)`. The ARC factory returns:

- `optimizer`: `ArcTopKMuon` + runtime `optimizer_owns_gradient_sync=True`;
- `ddp_hook`: ordinary `Muon` + registered `ArcTopKDDPState` runtime with `optimizer_owns_gradient_sync=False`.

In the dedicated ARC hyperparameters, make both the legacy `replicate_mesh_grad_sync` and `arc_seed_mode` nullable so “unset” differs from an explicit conflict. `arc_sync_mode` derives gradient ownership and the seed default: optimizer mode derives `broadcast`, hook mode derives `local_deterministic`. Explicit incompatible legacy ownership or explicit `ddp_hook + broadcast` fails, while setting only `arc_sync_mode=ddp_hook` works.

The shared loop calls `runtime.begin_step()` immediately before the first training micro-batch, passes the runtime boolean to `forward_backward_micro_step`, calls `runtime.finish_step()` after backward and before gradient norm/optimizer step, calls `runtime.commit_step()` only after `optimizer.step()` succeeds, and exposes `runtime.checkpoint_state` to checkpointing. This first version supports the current unscaled, no-retry training loop: compressor and Muon steps must be equal after commit, and checkpointing is legal only at that boundary. AMP GradScaler skip/retry behavior is out of scope and must fail fast if later introduced without a runtime abort protocol.

- [ ] **Step 1: Write failing factory and context-policy tests**

Assert exact optimizer class, hook registration count, runtime policy, DDP-only rejection, no `ArcTopKMuon` in hook mode, no hook in optimizer mode, `ddp_hook + broadcast` rejection, setting only the new mode successfully overrides the nullable legacy default, and startup failure for explicitly ambiguous combinations.

- [ ] **Step 2: Write a failing real two-rank formal-loop test**

For two accumulation micro-batches, optimizer mode produces zero DDP hook calls; hook mode produces one call per ready bucket only on the final micro-batch. Verify the hook consumes accumulated gradients, not final-micro-batch-only gradients.

- [ ] **Step 3: Implement the runtime adapter and training-loop calls**

Keep ordinary optimizer factories source-compatible. Do not use `hp.replicate_mesh_grad_sync` directly once a runtime is created. Ensure validation forward passes never call `begin_step` or the hook. Add GA=256 on a tiny model and assert each ARC tracker advances once per optimizer step, never once per micro-batch or bucket.

- [ ] **Step 4: Run the formal-entry suite**

```bash
uv run --frozen --extra dev pytest \
  tests/test_train_ddp_sync.py \
  tests/test_train_arctopk.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_no_sync_diagnostic.py -v
```

---

### Task 9: Add rank-local compressor checkpointing and deterministic compressor resume

**Files:**
- Modify: `dion/arc_topk_ddp_hook.py`
- Modify: `train.py`
- Create: `tests/test_arc_topk_ddp_checkpoint.py`
- Modify: `tests/test_train_arctopk_ddp_hook.py`

**Interfaces:**

```python
class CheckpointManager:
    def __init__(..., extra_stateful: Mapping[str, Stateful] | None = None): ...
```

The hook runtime is saved under `arc_compressor`. Shared metadata contains schema version, world size, group ranks, fingerprint, seed-scheme version, committed compressor step, and ordered parameter table. Rank-local `h_local/g_local` live under `rank_<global_rank>/<stable_name>`; replicated `g_global` is either stored once with validated equality or stored explicitly and checked on load. This task guarantees compressor/optimizer continuation under a fixed LR schedule; it does not claim exact formal-training resume because the existing `CheckpointManager` does not persist the LR scheduler. Repairing general scheduler resume is a separate training-system task.

- [ ] **Step 1: Write a failing real DCP two-rank round-trip**

Create deliberately different local state on each rank, save, destroy the old optimizer/hook objects, rebuild a fresh zero-state DDP/optimizer/hook with a changed bucket layout, load, and compare every state tensor by stable name. Continue two fixed-LR compressed steps and compare to an uninterrupted oracle.

- [ ] **Step 2: Write failing incompatibility tests**

Reject schema mismatch, world-size/rank-membership change, fingerprint mismatch, seed-scheme mismatch, compressor/Muon committed-step mismatch, missing/duplicate state, wrong role/shape/dtype, and snapshots before optimizer commit or while a Future is in flight. The first version has no partial compressor-reset escape hatch: incompatible state fails closed because resetting only compressor state while retaining Muon state makes their step semantics inconsistent.

- [ ] **Step 3: Implement generic extra checkpoint state and hook serialization**

Keep non-ARC checkpoints unchanged. Fresh hook construction must preallocate every stable-name state tensor before `CheckpointManager._get_state_dict()` creates the DCP destination schema. Validate the current runtime fingerprint against separately read checkpoint metadata before loading tensor payloads; never overwrite the runtime fingerprint first and then compare it to itself. Use the real DCP planner in tests; do not substitute a plain `torch.save` mock.

- [ ] **Step 4: Run checkpoint and regression tests**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk_ddp_checkpoint.py \
  tests/test_train_arctopk_ddp_hook.py -v
```

---

### Task 10: Extend profiler instrumentation and the parameterized launcher

**Files:**
- Modify: `dion/collective_observer.py`
- Modify: `dion/arc_topk_ddp_hook.py`
- Modify: `benchmark/compressed_muon/training_profiler.py`
- Modify: `benchmark/compressed_muon/profiler_trace.py`
- Modify: `benchmark/compressed_muon/summarize_training_profiles.py`
- Modify: `benchmark/compressed_muon/run_training_critical_path_profiler.sh`
- Modify: `train.py`
- Modify: `tests/test_training_profiler.py`
- Modify: `tests/test_training_profiler_trace.py`
- Modify: `tests/test_training_profiler_launcher.py`

**Required ranges and metrics:**

- `arc_hook/local_prepare`, `arc_hook/dense`, `arc_hook/sketch`, `arc_hook/topk`, `arc_hook/selected_values`, `arc_hook/finalize`;
- bucket-ready timestamp, final Future completion, bucket bytes, ARC bytes, dense bytes, bucket count;
- first ARC collective relative to backward start and last ARC completion relative to backward end;
- time-interval intersection between ARC communication kernels and genuine backward compute kernels, excluding hook prepare/TopK/finalize kernels;
- exposed ARC communication tail after the last genuine backward compute kernel;
- absence/presence of `arc/seed` by mode.

- [ ] **Step 1: Write failing summary and launcher contract tests**

Extend cells to `dense`, `arc_optimizer`, and `arc_ddp_hook`; rotate all six cell orders across paired repeats rather than always favoring one mode. Add CLI options `--world-size`, `--global-batch-size`, `--gpu-list`, `--exclude-gpus`, `--bucket-cap-mb`, `--arc-seed-mode`, `--timing-warmup-steps`, measured step count, and artifact root. No GPU IDs are hard-coded. Replace `train.py`'s hard-coded step-10 timer reset with `timing_warmup_steps` while keeping 10 as the default; set `num_iterations = timing_warmup_steps + measured_steps` so the terminal validation reports exactly the requested measured window.

- [ ] **Step 2: Implement trace ranges and fail-closed summarization**

Teach `profiler_trace.py` to classify hook dense/sketch/selected-values NCCL explicitly instead of falling through to `ddp_gradient`. Correlate callback-thread ranges with GPU kernels and compute overlap from GPU time intervals, not containment inside the CPU `backward` range. Add a counterexample trace in which ARC communication is entirely inside the backward CPU range but begins after the last backward compute kernel; its computed overlap must be zero. Reject missing rank traces, rank-divergent hook counts/signatures, missing final timings, OOM/timeout, incomplete cells, or local mode containing a seed collective. Keep raw traces as source of truth.

- [ ] **Step 3: Run CPU contracts**

```bash
uv run --frozen --extra dev pytest \
  tests/test_training_profiler.py \
  tests/test_training_profiler_trace.py \
  tests/test_training_profiler_launcher.py -v
bash -n benchmark/compressed_muon/run_training_critical_path_profiler.sh
benchmark/compressed_muon/run_training_critical_path_profiler.sh --print-plan
```

The formal-entry timing test must use a non-default warmup value and assert that the reported divisor and sample window exclude exactly that many completed optimizer steps.

---

### Task 11: Run correctness gates, profiler attribution, and wall-clock comparison

**Files:**
- Create: `configs/compressed_muon/cm024a_dense_muon_gpt350m.yaml`
- Create: `configs/compressed_muon/cm024b_arc_optimizer_local_seed_gpt350m.yaml`
- Create: `configs/compressed_muon/cm024c_arc_ddp_hook_local_seed_gpt350m.yaml`
- Modify: `docs/compressed_muon/EXPERIMENTS.md`
- Modify: `docs/compressed_muon/RESULTS.md`
- Modify: `docs/worklog/M001-arc-topk-ef21m-muon.md`

- [ ] **Step 1: Run the complete CPU correctness gate**

Run all new tests plus the existing ARC, Muon, training, no-sync, and launcher suites. Record the command and exact pass/skip counts. Any failure blocks GPU work.

- [ ] **Step 2: Run a two-or-three GPU NCCL hook gate**

Use idle GPUs selected dynamically. Run the NCCL stress test and a 3-step tiny formal-entry smoke test for all three modes. Require identical parameters across ranks within dtype tolerance and zero unexpected collective signatures.

- [ ] **Step 3: Run a short bucket-size profiler scan**

Use GPT-350M, BF16, compile enabled, sequence length 1024, device batch 1, ratio 0.2, projection rank 4, eta 0.1, compression start 0, seed 42, and local deterministic seed for both ARC routes. Run `bucket_cap_mb` values 5, 25, and 50 with the available exclusive GPU count. This phase is attribution-only and may use fewer than four cards if all three cells share the identical topology and batch geometry.

Success requires the hook trace to show a positive GPU-time intersection between an ARC collective kernel and a later genuine backward compute kernel, plus a smaller exposed gradient-sync tail than optimizer-side ARC. Merely falling inside the host backward range is insufficient. If the condition fails, stop and diagnose before the longer run.

- [ ] **Step 4: Run the formal four-GPU wall-clock comparison when four exclusive GPUs are available**

Use global batch 1024 so all modes have gradient accumulation 256. Run an initial three complete paired blocks, rotating all cell orders, with 20 warmup steps and 100 measured optimizer steps per cell, normal NCCL, and the best pre-registered bucket size from Step 3. Do not use `--time_optimizer` for primary timing. If the observed hook-vs-optimizer ARC difference is below 2% or its run-level interval crosses zero, extend to at least ten paired blocks before making any acceleration claim.

- [ ] **Step 5: Apply result acceptance rules**

Treat each complete paired block—not each correlated step sample—as the statistical unit. Report paired run ratios/differences, medians, means, standard deviations, CVs, throughput, and a 95% paired bootstrap or randomization interval. A wall-clock acceleration claim requires:

1. the run-level hook-vs-optimizer ARC interval excludes zero in the improvement direction;
2. cell order shows no unresolved systematic drift and CV is at most 5% per cell;
3. traces prove overlap rather than missing work;
4. loss/gradient/parameter consistency gates pass;
5. no shared GPU and no topology/config mismatch.

If the high-accumulation end-to-end gain remains tiny while the final-microstep trace improves, report exactly that: overlap is restored locally, but Amdahl dilution prevents a material training-step gain at GA=256.

- [ ] **Step 6: Update research records**

Pre-register CM024a/b/c before launch, preserve commands/configs/environment/exit codes/raw logs/traces, and update `RESULTS.md` plus the M001 worklog after artifacts validate whether the result is positive, null, or negative. Keep CM019, CM020, CM023, and CM024 interpretations separate.

---

### Task 12: Final review and integration boundary

- [ ] **Step 1: Run the complete repository-relevant verification**

```bash
uv run --frozen --extra dev pytest \
  tests/test_arc_topk.py \
  tests/test_arc_topk_distributed.py \
  tests/test_arc_topk_sync.py \
  tests/test_arc_topk_sync_distributed.py \
  tests/test_muon_arctopk.py \
  tests/test_muon_arctopk_distributed.py \
  tests/test_train_ddp_sync.py \
  tests/test_train_arctopk.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_arc_topk_layout.py \
  tests/test_arc_topk_layout_distributed.py \
  tests/test_arc_topk_ef21m_primitives.py \
  tests/test_arc_topk_ddp_hook_state.py \
  tests/test_arc_topk_ddp_hook_future.py \
  tests/test_arc_topk_ddp_hook_future_distributed.py \
  tests/test_arc_topk_ddp_hook.py \
  tests/test_arc_topk_ddp_hook_distributed.py \
  tests/test_arc_topk_ddp_checkpoint.py \
  tests/test_training_profiler.py \
  tests/test_training_profiler_trace.py \
  tests/test_training_profiler_launcher.py -v
```

- [ ] **Step 2: Independent code review**

Request review focused on collective ordering, Future completion/exception behavior, tensor lifetime, EF21M state semantics, accumulation policy, checkpoint rank locality, forbidden synchronization points, and whether profiler evidence supports the claim.

- [ ] **Step 3: Integration decision**

Do not remove optimizer-side ARC. If correctness passes but wall-clock acceptance fails, keep hook behind `arc_sync_mode=ddp_hook`, document the measured boundary, and decide from profiler evidence whether bucket scheduling—not another seed or collective micro-optimization—is the next research problem.
