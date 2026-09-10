# GreedyLore-Muon DDP Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Repository baseline:** Revised 2026-09-10 against commit `b81849a` after
> ARC all-2D, shared Muon parameter-group, scalar AdamW, scheduler/clipping, and
> compressor-step lifecycle changes.

> **Author-code reference:** Read-only snapshot at `~/greedy_lore`, which has
> no Git metadata. Use the SHA-256-pinned files and source hierarchy in the spec;
> never add this external directory as a runtime or test dependency.

**Goal:** Add a paper-faithful GreedyLore low-rank DDP gradient synchronization path that feeds reconstructed gradients into the unchanged Muon optimizer, with deterministic state, checkpointing, and end-to-end profiling.

**Architecture:** Keep dense Muon and ARC-TopK unchanged. Add pure GreedyLore tensor primitives, a dedicated stateful asynchronous DDP bucket hook, and a narrow training entry that constructs ordinary Muon; reuse the existing gradient-sync lifecycle, no-sync accumulation policy, observer, profiler capture, and checkpoint manager contracts. The default basis path performs identical local SVD on homogeneous ranks with deterministic sign canonicalization, while an explicit full-basis broadcast mode remains available for strong-consistency diagnosis.

**Tech Stack:** Python 3.10, PyTorch 2.11+ DDP/GradBucket/Future/DCP, CUDA/NCCL, Gloo, pytest, Bash, existing Muon and compressed-Muon profiling utilities.

**Spec:** docs/superpowers/specs/2026-09-09-greedylore-muon-design.md

## Global Constraints

- Compression owns only data-parallel gradient synchronization; do not modify dion/muon.py, Muon momentum, orthogonalization, result communication, or parameter updates.
- Treat paper Algorithms 2-3 as authoritative for recurrence and score semantics. Use `~/greedy_lore` only to cross-check dimensionally valid orientation, batching, packing, and reconstruction patterns; do not copy its bucket-index state, blocking waits/synchronization, full-gradient fake communication, incomplete score helper, or raw-gradient refresh discrepancy.
- Preserve dion/arc_topk_ddp_hook.py, all ARC behavior, and existing arc_compressor checkpoint names.
- The first implementation supports DDP, find_unused_parameters=False, a static participating parameter set, fixed world size, and two-dimensional Muon matrix parameters.
- Define matrix membership from the first group returned by train.build_muon_param_groups(), not from ndim alone. Unlike current ARC all-2D mode, embedding and lm-head parameters remain dense_aux in M002; expanding that scope is a separate ablation.
- Reuse train.build_muon_param_groups() and inherited scalar optimizer, warmup, learning-rate schedule, and gradient-clipping configuration; do not duplicate their defaults in train_greedylore.py.
- local_svd is an experimental assumption validated only for a specific homogeneous PyTorch/CUDA/linear-algebra environment; broadcast is the strong-consistency reference.
- Muon matrix buckets, compression state, low-rank arithmetic, and matrix communication are FP32 in the first implementation; fail at factory construction for a non-FP32 Muon matrix parameter.
- Reduce signed lambda values before squaring the average; update local error from the untouched local factor before factor All-Reduce.
- The first compressed step is a refresh: phase = step - start_compress_step - 1, and refresh occurs when phase % update_interval == 0.
- Compressor committed_step is owned by the begin/finish/commit lifecycle and must not inspect or compare optimizer-internal step counters.
- Use stable parameter names and IDs for state, seeds, and checkpoints; never persist a bucket index or callback arrival position.
- Keep one global cross-bucket tail Future in the first version; every rank must launch the same collective sequence.
- The hook path must not call Work.wait(), torch.cuda.synchronize(), tensor .item(), perform device-to-host copies, or poll completion.
- Do not infer speedup from logical bytes or asynchronous API shape. Performance acceptance requires profiler-disabled, repeated, complete-period wall-clock improvement plus trace evidence and correctness/quality gates.

---

### Task 1: Implement configuration, orientation, deterministic seeds, and SVD canonicalization

**Files:**
- Create: dion/greedy_lore.py
- Create: tests/test_greedy_lore.py
- Reference (read-only): ~/greedy_lore/comm_hooks/subspace_hook.py:105-178
- Reference (read-only): ~/greedy_lore/comm_hooks/fake_subspace_hook.py:108-168

**Interfaces:**
- Consumes: torch.Tensor, a stable parameter ID, and the configuration values from the spec.
- Produces:

~~~python
@dataclass(frozen=True)
class GreedyLoreConfig:
    rank: int = 32
    update_interval: int = 200
    seed: int = 42
    start_compress_step: int = 1000
    basis_sync: Literal["local_svd", "broadcast"] = "local_svd"
    seed_scheme_version: int = 1

@dataclass(frozen=True)
class MatrixOrientation:
    original_shape: tuple[int, int]
    compressed_shape: tuple[int, int]
    transposed: bool

def matrix_orientation(shape: Sequence[int]) -> MatrixOrientation: ...
def orient_matrix(tensor: Tensor, orientation: MatrixOrientation) -> Tensor: ...
def unorient_matrix(tensor: Tensor, orientation: MatrixOrientation) -> Tensor: ...
def compressed_phase(step: int, start_compress_step: int) -> int | None: ...
def is_refresh_step(step: int, config: GreedyLoreConfig) -> bool: ...
def derive_greedy_lore_seed(*, base_seed: int, phase: int,
                            stable_parameter_id: int) -> int: ...
def make_random_vectors(*, rows: int, columns: int, seed: int,
                        device: torch.device) -> Tensor: ...
def canonicalize_svd_basis(basis: Tensor) -> Tensor: ...
~~~

- [ ] **Step 1: Write failing validation and phase tests**

~~~python
@pytest.mark.parametrize("kwargs", [
    {"rank": 0}, {"update_interval": 0}, {"start_compress_step": -1},
    {"basis_sync": "unknown"}, {"seed_scheme_version": 0},
    {"seed_scheme_version": 2},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GreedyLoreConfig(**kwargs)

def test_first_compressed_step_is_refresh_independent_of_absolute_step():
    config = GreedyLoreConfig(start_compress_step=7, update_interval=3)
    assert [compressed_phase(step, 7) for step in range(7, 12)] == [None, 0, 1, 2, 3]
    assert [is_refresh_step(step, config) for step in range(7, 12)] == [False, True, False, False, True]
~~~

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

~~~bash
uv run --frozen --extra dev pytest tests/test_greedy_lore.py -v
~~~

Expected: collection fails because dion.greedy_lore does not exist.

- [ ] **Step 3: Implement configuration and exact phase semantics**

Validate booleans separately from integers, require positive rank/update interval, and reject every seed_scheme_version other than 1. Implement:

~~~python
def compressed_phase(step, start_compress_step):
    if step <= start_compress_step:
        return None
    return step - start_compress_step - 1

def is_refresh_step(step, config):
    phase = compressed_phase(step, config.start_compress_step)
    return phase is not None and phase % config.update_interval == 0
~~~

- [ ] **Step 4: Write failing orientation round-trip tests**

~~~python
@pytest.mark.parametrize("shape,expected,transposed", [
    ((3, 5), (3, 5), False),
    ((5, 3), (3, 5), True),
    ((4, 4), (4, 4), False),
])
def test_orientation_round_trip(shape, expected, transposed):
    tensor = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    orientation = matrix_orientation(shape)
    assert orientation.compressed_shape == expected
    assert orientation.transposed is transposed
    assert torch.equal(unorient_matrix(orient_matrix(tensor, orientation), orientation), tensor)
~~~

Also require matrix_orientation((2, 3, 4)) to raise ValueError mentioning two-dimensional input.

Add self-contained parity cases for both branches used by the author snapshot:
for rows < columns, compare normalized-orientation projection with selecting
columns from `U`; for rows >= columns, compare it with selecting rows from `Vh`
and right-multiplying. Copy only the small equations and literal tensors into
the test—never import the external snapshot.

- [ ] **Step 5: Implement orientation with view operations only**

Use tensor.mT for both directions when transposed=True; do not allocate a persistent transposed copy.

- [ ] **Step 6: Write failing seed, RNG-isolation, and sign-canonicalization tests**

~~~python
def test_random_vectors_are_local_and_reproducible():
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    seed = derive_greedy_lore_seed(
        base_seed=42, phase=5, stable_parameter_id=9,
    )
    first = make_random_vectors(rows=3, columns=4, seed=seed, device=torch.device("cpu"))
    second = make_random_vectors(rows=3, columns=4, seed=seed, device=torch.device("cpu"))
    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), before)

def test_canonicalize_svd_basis_uses_smallest_maximum_index_as_positive_pivot():
    basis = torch.tensor([[-0.5, 0.0], [-0.5, -1.0]])
    result = canonicalize_svd_basis(basis)
    assert result[0, 0] > 0
    assert result[1, 1] > 0
    assert torch.allclose(result.T @ result, basis.T @ basis)
~~~

- [ ] **Step 7: Implement local generator and canonicalization**

For seed scheme version one, compute (base_seed + phase * 1_000_003 + stable_parameter_id) % (2**63 - 1). Construct a device-local torch.Generator, call manual_seed on it, and pass it explicitly to torch.randn with dtype=torch.float32. For each SVD column, choose argmax(abs(column)), where argmax supplies the smallest-index tie break, and multiply the column by a sign that makes the pivot non-negative.

- [ ] **Step 8: Run GREEN and commit**

~~~bash
uv run --frozen --extra dev pytest tests/test_greedy_lore.py -v
git add dion/greedy_lore.py tests/test_greedy_lore.py
git commit -m "feat: add GreedyLore tensor foundations"
~~~

---

### Task 2: Implement the paper recurrence as collective-free tensor primitives

**Files:**
- Modify: dion/greedy_lore.py
- Modify: tests/test_greedy_lore.py
- Create: tests/test_greedy_lore_oracle.py
- Reference (read-only): ~/greedy_lore/comm_hooks/fake_subspace_hook.py:235-359
- Reference (read-only): ~/greedy_lore/comm_hooks/lore_hook.py:190-430

**Interfaces:**
- Consumes: the Task 1 orientation, seed, and basis helpers.
- Produces:

~~~python
def corrected_gradient(gradient: Tensor, error: Tensor,
                       orientation: MatrixOrientation) -> Tensor: ...
def refresh_basis(global_corrected: Tensor, rank: int) \
        -> tuple[Tensor, Tensor, Tensor]: ...  # basis, projector, support
def approximate_signed_lambda(corrected: Tensor, basis: Tensor,
                              random_vectors: Tensor) -> Tensor: ...
def select_projector(basis: Tensor, averaged_lambda: Tensor, rank: int) \
        -> tuple[Tensor, Tensor]: ...
def compress_local(corrected: Tensor, projector: Tensor) \
        -> tuple[Tensor, Tensor]: ...
def reconstruct_global(projector: Tensor, averaged_factor: Tensor) -> Tensor: ...
~~~

- [ ] **Step 1: Write a literal two-rank, three-step reference oracle**

Use two 2 x 3 rank-local gradient sequences, rank=1, update_interval=2, and literal random-vector tensors. Assert at every step the corrected gradients, signed lambdas, squared averaged scores, support, local factors, local errors, averaged factor, and reconstructed global gradient.

~~~python
scores = ((lambda_rank0 + lambda_rank1) / 2).square()
support = torch.argsort(scores, descending=True, stable=True)[:1]
local_r0 = projector.T @ corrected_rank0
expected_e0 = corrected_rank0 - projector @ local_r0
global_r = (local_r0 + local_r1) / 2
expected = projector @ global_r
~~~

Include a case where rank lambdas have opposite signs so mean(lambda.square()) chooses a different column from mean(lambda).square().

Add two source-cross-check cases. First, show that ranking
`abs(mean(signed_lambda))` from the snapshot's `sigma_type=1` branch agrees
with ranking `mean(signed_lambda).square()` when there are no ties, while the
implementation retains the paper's square and stable tie rule. Second, start a
refresh with nonzero old error and prove the oracle reduces `gradient + error`
before resetting error, explicitly guarding against the snapshot's raw-gradient
EF14 refresh behavior.

- [ ] **Step 2: Run the oracle and verify RED**

~~~bash
uv run --frozen --extra dev pytest tests/test_greedy_lore_oracle.py -v
~~~

Expected: imports of the recurrence functions fail.

- [ ] **Step 3: Implement corrected gradient, refresh SVD, and stable Top-r**

Cast the oriented gradient to FP32 before adding error. Use:

~~~python
basis, singular_values, _ = torch.linalg.svd(global_corrected, full_matrices=False)
basis = canonicalize_svd_basis(basis)
support = torch.arange(rank, device=basis.device, dtype=torch.int64)
projector = basis.index_select(1, support)
~~~

For non-refresh selection, compute scores = averaged_lambda.square() and use stable descending argsort; never use torch.topk without an explicit tie policy.

- [ ] **Step 4: Implement score, local error, and reconstruction primitives**

~~~python
projected_rows = basis.mT @ corrected
signed_lambda = (projected_rows * random_vectors).sum(dim=1)
local_factor = projector.mT @ corrected
next_error = corrected - projector @ local_factor
reconstructed = projector @ averaged_factor
~~~

Return a fresh local factor; collective code later reduces a distinct packing buffer.

The author snapshot's approximate branch uses uniform random values, but the
paper requires independent standard-normal vectors. Keep `torch.randn` from
Task 1; add a test that would fail if the implementation switches to a
nonnegative-only distribution.

- [ ] **Step 5: Add boundary tests**

Cover r=1, r=a, tau=1, zero corrected gradient, square and transposed matrices, deterministic score ties, and FP32 state. The r=a test must call select_projector, compress_local, and reconstruct_global and assert residual and reconstruction with atol=1e-5 and rtol=1e-5 using a nontrivial SVD basis; it must not branch to a dense shortcut.

- [ ] **Step 6: Run GREEN and commit**

~~~bash
uv run --frozen --extra dev pytest \
  tests/test_greedy_lore.py tests/test_greedy_lore_oracle.py -v
git add dion/greedy_lore.py tests/test_greedy_lore.py tests/test_greedy_lore_oracle.py
git commit -m "feat: implement GreedyLore recurrence primitives"
~~~

---

### Task 3: Add canonical layout validation and compressor lifecycle state

**Files:**
- Create: dion/greedy_lore_layout.py
- Create: dion/greedy_lore_ddp_hook.py
- Create: tests/test_greedy_lore_layout.py
- Create: tests/test_greedy_lore_layout_distributed.py
- Create: tests/test_greedy_lore_ddp_state.py

**Interfaces:**
- Consumes: GreedyLoreConfig, MatrixOrientation, model parameters, the frozen optimizer-owned parameter identity set, and a DDP process group. The state does not retain the optimizer or inspect its step counters.
- Produces:

~~~python
@dataclass(frozen=True)
class GreedyLoreParameterDescriptor:
    stable_name: str
    stable_id: int
    shape: tuple[int, ...]
    dtype: str
    role: Literal["matrix", "dense_aux"]

def canonical_greedy_lore_fingerprint(*, config: GreedyLoreConfig,
        group_ranks: Sequence[int],
        parameters: Sequence[GreedyLoreParameterDescriptor]) -> str: ...
def validate_greedy_lore_fingerprint_across_ranks(
        fingerprint: str, process_group: ProcessGroup) -> None: ...

@dataclass(frozen=True)
class GreedyLoreDDPParameterSpec:
    parameter: Parameter
    stable_name: str
    stable_id: int
    role: Literal["matrix", "dense_aux"]

@dataclass
class GreedyLoreParameterState:
    spec: GreedyLoreDDPParameterSpec
    orientation: MatrixOrientation
    error: Tensor
    basis: Tensor
    last_support: Tensor

@dataclass
class BucketContext:
    context_id: int
    bucket: dist.GradBucket
    buffer: Tensor
    gradients: tuple[Tensor, ...]
    parameters: tuple[Parameter, ...]
    parameter_states: tuple[GreedyLoreParameterState | None, ...]
    step: int
    phase: int | None
    entry_stream: torch.cuda.Stream | None
    bucket_ready_event: torch.cuda.Event | None
    previous_tail: torch.futures.Future
    completion_future: torch.futures.Future
    retained: list[Any]

class GreedyLoreDDPState:
    def begin_step(self) -> int: ...
    def note_bucket(self, bucket: dist.GradBucket) -> BucketContext: ...
    def finish_step(self) -> None: ...
    def commit_step(self) -> None: ...
    def parameter_state(self, parameter: Parameter) -> GreedyLoreParameterState: ...
    def validate_replicated_basis_across_ranks(
            self, *, atol: float = 1e-6, rtol: float = 1e-5) -> None: ...
~~~

- [ ] **Step 1: Write failing canonical-layout tests**

Assert stable equality and sensitivity to seed, rank, interval, start step, basis mode, group membership, parameter order/name/shape/dtype/role, duplicate names, and duplicate IDs. Object identity must not enter the JSON payload.

- [ ] **Step 2: Write a failing two-rank mismatch test**

Spawn two real Gloo ranks. Equal layouts pass; different basis_sync, rank, or parameter order makes both ranks raise GreedyLoreLayoutMismatch within a 30-second process-group timeout. Always destroy the group in finally.

- [ ] **Step 3: Implement sorted-JSON SHA-256 fingerprints and one-time validation**

Mirror the value-only structure of dion/arc_topk_layout.py without importing ARC-specific types. Observe the initialization collective as greedylore/layout_validation; do not validate per step.

- [ ] **Step 4: Write failing state construction tests**

~~~python
def test_state_preallocates_matrix_state_in_compressed_orientation():
    parameter = torch.nn.Parameter(torch.zeros(5, 3))
    state = make_state(parameter, rank=2)
    item = state.parameter_state(parameter)
    assert item.error.shape == (3, 5)
    assert item.error.dtype == torch.float32
    assert item.basis.shape == (3, 3)
    assert torch.equal(item.basis, torch.eye(3))
    assert item.last_support.tolist() == [0, 1]
~~~

Also reject missing/duplicate ownership, matrix rank larger than min(shape), non-2D matrix roles, non-FP32 matrix parameters, and find_unused_parameters=True. Freeze the optimizer parameter identity table at construction; later bucket coverage outside that table fails deterministically.

- [ ] **Step 5: Write failing lifecycle and bucket-coverage tests**

Require one active step, exact once-only parameter coverage, no next step while the tail is in flight, no commit before finish, no checkpoint inside a step, compressor-owned committed-step progression, and stable identity lookup after simulated bucket reorder. Add a regression proving construction and checkpointing work with an optimizer whose parameter groups expose no shared step field.

- [ ] **Step 6: Implement state without compression launches**

Initialize a completed CUDA-aware tail Future, preallocate all matrix tensors for DCP, record a bucket-ready CUDA event at callback entry, and retain active contexts until their DDP-facing Futures complete. Implement committed-boundary basis validation by gathering bases in stable-name order, checking supports exactly and bases with the interface tolerances, and rejecting disagreement; this diagnostic is called by correctness gates, never by the timed hook path. Use the current ARC compressor-owned, one-based begin/finish/commit contract without retaining an optimizer reference, but keep all class names and errors GreedyLore-specific.

- [ ] **Step 7: Run GREEN and commit**

~~~bash
uv run --frozen --extra dev pytest \
  tests/test_greedy_lore_layout.py \
  tests/test_greedy_lore_layout_distributed.py \
  tests/test_greedy_lore_ddp_state.py -v
git add dion/greedy_lore_layout.py dion/greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_layout.py \
  tests/test_greedy_lore_layout_distributed.py \
  tests/test_greedy_lore_ddp_state.py
git commit -m "feat: add GreedyLore DDP state and layout"
~~~

---

### Task 4: Implement the asynchronous cross-bucket Future sequencer

**Files:**
- Modify: dion/greedy_lore_ddp_hook.py
- Create: tests/test_greedy_lore_ddp_future.py
- Create: tests/test_greedy_lore_ddp_future_distributed.py
- Create: tests/test_greedy_lore_ddp_hook_nccl.py

**Interfaces:**
- Consumes: BucketContext and the tail Future created in Task 3.
- Produces:

~~~python
def bridge_future(source: torch.futures.Future,
                  destination: torch.futures.Future,
                  transform: Callable[[Any], Tensor]) -> None: ...
def enqueue_bucket_chain(state: GreedyLoreDDPState, context: BucketContext,
                         launch: Callable[[BucketContext], torch.futures.Future]) \
        -> torch.futures.Future: ...
~~~

- [ ] **Step 1: Write failing Future value, exception, and retention tests**

Assert the hook-facing result is one tensor rather than Future[Future[Tensor]]; an exception in a source callback reaches the current destination and poisons later buckets; an already-completed previous Future cannot deadlock through inline callback re-entry; active context retention ends only after completion.

- [ ] **Step 2: Write a failing real multi-bucket order test**

Use a two-rank Gloo DDP model with bucket_cap_mb=0.0005, run enough iterations for bucket rebuild, and first prove at least two real callbacks occur. Add different callback delays on the two ranks and require identical observer signature order. Use a process timeout and unconditional group teardown.

- [ ] **Step 3: Port the proven ARC sequencing pattern under new names**

Install the destination tail before attaching callbacks. On CUDA, make the execution stream wait for both bucket_ready_event and the preceding tail. Launch collectives through Work.get_future(), bridge completion explicitly, and never depend on .then() flattening.

- [ ] **Step 4: Prove final GPU work is part of completion**

Add an NCCL-marked dummy-chain test that delays a final write on the compressor stream. A consumer on a non-default stream must observe the new value after the returned Future completes. The callback stream must wait for the compressor stream before set_result(bucket.buffer()).

- [ ] **Step 5: Run GREEN and commit**

~~~bash
uv run --frozen --extra dev pytest \
  tests/test_greedy_lore_ddp_future.py \
  tests/test_greedy_lore_ddp_future_distributed.py -v
uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook_nccl.py -v
git add dion/greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_future.py \
  tests/test_greedy_lore_ddp_future_distributed.py \
  tests/test_greedy_lore_ddp_hook_nccl.py
git commit -m "feat: sequence GreedyLore DDP bucket futures"
~~~

---

### Task 5: Implement dense warmup and refresh synchronization

**Files:**
- Modify: dion/greedy_lore_ddp_hook.py
- Create: tests/test_greedy_lore_ddp_hook.py
- Create: tests/test_greedy_lore_ddp_hook_distributed.py
- Reference (read-only): ~/greedy_lore/comm_hooks/lore_hook.py:190-285
- Reference (read-only): ~/greedy_lore/comm_hooks/subspace_hook.py:197-285

**Interfaces:**
- Consumes: Tasks 1-4 state, recurrence primitives, and Future sequencer.
- Produces:

~~~python
def greedy_lore_ddp_hook(state: GreedyLoreDDPState,
                         bucket: dist.GradBucket) \
        -> torch.futures.Future[Tensor]: ...
~~~

- [ ] **Step 1: Write failing single-rank bucket reconstruction tests**

Use fake GradBucket fixtures for matrix and dense roles, differently shaped FP32 matrices in one bucket, exact offsets, square and transposed orientations, and dense-only buckets. Assert returned buffer shape/device/dtype and that state tensors remain FP32.

- [ ] **Step 2: Write failing two-rank dense-warmup tests**

With real Gloo DDP, supply rank-distinct gradients before compression starts. Assert the returned matrix and auxiliary slices equal the exact global average and the observer records one greedylore_hook/dense All-Reduce per real bucket.

- [ ] **Step 3: Implement one packed dense warmup All-Reduce**

Use the FP32 bucket buffer directly for matrix and same-bucket auxiliary slices. Reject non-FP32 matrix buckets at factory/state construction rather than inside an asynchronous callback. Dense-only auxiliary buckets retain their native dtype and take one exact dense All-Reduce.

- [ ] **Step 4: Write failing local-SVD refresh tests**

Use two ranks with a nonzero old error. Assert the refresh input is gradient + old_error, output is its exact global average, error resets to zero, the complete canonicalized basis and first-rank support are stored, no factor All-Reduce occurs, and no basis broadcast occurs in local_svd mode.

This test is also the explicit paper-versus-snapshot guard: the expected output
must differ from an All-Reduce of the raw gradient alone.

Add zero, repeated, and near-repeated singular-value matrices. Compare basis, support, and reconstructed output separately across ranks. Also manually rotate a repeated-singular-value basis on one rank and require validate_replicated_basis_across_ranks() to reject it. Passing these tests validates only the named homogeneous environment; it does not turn local_svd into a general consistency guarantee.

- [ ] **Step 5: Implement local-SVD refresh on the compressor stream**

After dense completion, call torch.linalg.svd with full_matrices=False independently on every rank, canonicalize signs, store complete basis/support, zero the local error, scatter the averaged corrected gradient, and complete the bucket Future only after the scatter is visible.

- [ ] **Step 6: Write and implement broadcast-mode refresh tests**

Require only process-group rank zero to call SVD, followed by one full-basis broadcast per matrix in stable parameter order. All ranks store the broadcast basis. Observer bytes equal a*a*4 per matrix and use category greedylore_hook/basis_broadcast.

- [ ] **Step 7: Run GREEN and commit**

~~~bash
uv run --frozen --extra dev pytest \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py -v
git add dion/greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py
git commit -m "feat: add GreedyLore dense and refresh hook paths"
~~~

---

### Task 6: Implement signed-score and low-rank-factor compressed synchronization

**Files:**
- Modify: dion/greedy_lore_ddp_hook.py
- Modify: tests/test_greedy_lore_ddp_hook.py
- Modify: tests/test_greedy_lore_ddp_hook_distributed.py
- Modify: tests/test_greedy_lore_ddp_hook_nccl.py
- Reference (read-only): ~/greedy_lore/comm_hooks/subspace_hook.py:287-439
- Reference (read-only): ~/greedy_lore/comm_hooks/fake_subspace_hook.py:266-359

**Interfaces:**
- Consumes: approximate_signed_lambda, select_projector, compress_local, and reconstruct_global from Task 2.
- Produces: complete compressed-step behavior from the registered greedy_lore_ddp_hook.

- [ ] **Step 1: Write a failing two-rank compressed oracle test**

Run one refresh plus at least three compressed steps with rank-distinct gradients, multiple differently shaped matrix parameters, and dense auxiliary parameters. Compare basis, signed local/averaged lambda, squared score, support, untouched local factor, next error, averaged factor, reconstructed bucket values, and collective signature to the Task 2 oracle.

- [ ] **Step 2: Write a regression that catches the two critical ordering bugs**

Construct lambdas whose signs cancel across ranks and prove support is selected from mean(lambda).square(). Hold local_factor, launch reduction on a distinct copied factor buffer, and prove error equals corrected - projector @ local_factor, not a value derived from the averaged factor.

- [ ] **Step 3: Implement score packing and the first All-Reduce**

Group matrix views by compressed shape for batched local math, derive seeds from stable parameter IDs and the exact phase stored in BucketContext, and generate random vectors through local generators. Because every mixed matrix bucket is FP32, pack its signed lambda vectors and FP32 dense auxiliary values into the same first reduction in stable parameter order. A dense-only bucket of another dtype takes its single exact dense reduction and stops.

Use the snapshot only as a layout cross-check for same-shape batching and
contiguous factor storage. Unlike the snapshot, never key persistent buffers by
bucket index, never derive seeds with global RNG plus `.item()`, and include the
score vector in logical-byte/profitability reporting.

- [ ] **Step 4: Implement deterministic support and local error update**

After averaging signed lambdas, square them and use stable descending argsort. Gather basis columns, compute each local factor, update error immediately from the local factor, then copy factors into one packed reduction buffer. Retain corrected matrices, projectors, local factors, supports, and packing offsets in the bucket context.

- [ ] **Step 5: Implement factor All-Reduce and reconstruction**

Average the packed factor buffer, reconstruct each matrix as projector @ averaged_factor, unorient it, cast to the bucket view dtype, and copy it into the original view. A dense-only bucket never launches the factor collective.

- [ ] **Step 6: Add signature and forbidden-synchronization tests**

Require zero seed collectives, zero full matrix All-Reduce on compressed steps, and stable per-bucket order:

~~~text
mixed matrix/aux bucket: score_plus_aux_allreduce -> factor_allreduce
dense-only bucket: dense_allreduce
~~~

Scan the registered hook path for Work.wait, torch.cuda.synchronize, .item(, and CPU tensor transfers. Any occurrence inside a training-step callback fails the test.

- [ ] **Step 7: Add real NCCL stress and completion tests**

On two idle GPUs, cover repeated iterations, small buckets, bucket rebuild, gradient accumulation, allocator churn, delayed callbacks, non-default stream consumption, zero/repeated-spectrum refresh, and exception propagation. Set a process-group timeout and fail on unfinished Futures, retained contexts, rank-divergent parameters, or differing observer signatures.

- [ ] **Step 8: Run GREEN and commit**

~~~bash
uv run --frozen --extra dev pytest \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py -v
uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook_nccl.py -v
git add dion/greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py \
  tests/test_greedy_lore_ddp_hook_nccl.py
git commit -m "feat: implement compressed GreedyLore DDP synchronization"
~~~

---

### Task 7: Integrate ordinary Muon through a dedicated training entry

**Files:**
- Create: train_greedylore.py
- Create: tests/test_train_greedylore.py
- Modify: train.py:104-112
- Modify: train.py:1187-1192
- Modify: dion/__init__.py
- Create: configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml
- Modify: tests/test_train_arctopk_ddp_hook.py
- Modify: tests/test_configs.py

**Interfaces:**
- Consumes: ordinary Muon, GreedyLoreDDPState, greedy_lore_ddp_hook, and the shared training main loop.
- Produces:

~~~python
@dataclass
class GreedyLoreHyperparameters(train.Hyperparameters):
    optimizer: str = "greedy_lore_muon"
    greedy_lore_rank: int = 32
    greedy_lore_update_interval: int = 200
    greedy_lore_seed: int = 42
    greedy_lore_start_compress_step: int = 1000
    greedy_lore_basis_sync: Literal["local_svd", "broadcast"] = "local_svd"

def init_greedy_lore_optimizer(...) \
        -> tuple[torch.optim.Optimizer, train.GradientSyncRuntime]: ...

@dataclass
class GradientSyncRuntime:
    optimizer_owns_gradient_sync: bool
    begin_step: Callable[[], Any] | None = None
    finish_step: Callable[[], Any] | None = None
    commit_step: Callable[[], Any] | None = None
    checkpoint_state: Any = None
    checkpoint_state_name: str | None = None
~~~

- [ ] **Step 1: Write failing factory-boundary tests**

Assert DDP-only construction, exact ordinary Muon type, exactly one registered GreedyLore hook, matrix roles derived from the first group returned by train.build_muon_param_groups() rather than merely ndim, exact dense auxiliary roles (including two-dimensional embedding/lm-head parameters), stable names/IDs, and rejection of FSDP, missing DDP, find_unused_parameters=True, unsupported scalar optimizer, and explicit legacy optimizer-owned sync. Assert independent scalar AdamW settings are preserved in the ordinary Muon parameter groups.

- [ ] **Step 2: Write a failing backward-compatible checkpoint-name test**

Existing ARC runtimes with a checkpoint state and no explicit name still register arc_compressor. GreedyLore registers greedy_lore_compressor. Optimizer-only factories with no state keep extra_stateful empty.

- [ ] **Step 3: Implement the minimal shared training change**

~~~python
extra_stateful = None
if gradient_sync_runtime.checkpoint_state is not None:
    name = gradient_sync_runtime.checkpoint_state_name or "arc_compressor"
    extra_stateful = {name: gradient_sync_runtime.checkpoint_state}
~~~

Do not rename the ARC checkpoint or require changes in train_arctopk.py.

- [ ] **Step 4: Implement the dedicated factory and exports**

Call train.build_muon_param_groups(model, hp), use its first group's parameter identities to assign matrix roles, construct ordinary Muon from all returned groups, build and validate the GreedyLore layout once, register the hook, and return runtime callbacks with checkpoint_state_name="greedy_lore_compressor". Do not copy train_arctopk.py's current ndim-based all-2D role rule. Export only configuration, spec, state, and hook types needed by callers.

- [ ] **Step 5: Write a real CPU factory/helper integration test**

Construct the real factory on two Gloo CPU ranks and drive forward_backward_micro_step plus the begin/finish/commit lifecycle directly through two accumulation microbatches, matching the existing ARC test boundary. Assert hooks run only on the final microbatch, the first compressed step is refresh, each compressor state advances once per completed training update without consulting optimizer counters, gradients, Muon momentum, and parameters agree within the declared tolerances, supports and collective signatures agree exactly, and ordinary Muon records result communication separately. Do not describe this as executing train.main: that entry requires CUDA/NCCL and is exercised by the Task 10 smoke test.

Add a focused ordering test for the current common loop contract: finish the hook before compute_and_clip_grad_norm_, then run ordinary Muon, then commit the compressor. With clipping enabled, assert clipping sees reconstructed gradients and the stored GreedyLore error is unchanged by clipping.

- [ ] **Step 6: Add r=a dense equivalence through Muon**

Run refresh and compressed steps for dense Muon and GreedyLore-Muon from identical initial parameters and gradients. Require the GreedyLore path to execute score and factor collectives and match dense gradients, Muon momentum, and parameters within FP32/BF16 operation tolerances.

- [ ] **Step 7: Add the formal M002 configuration**

Keep this as a method default/smoke configuration rather than an experiment result. Base its model and batching geometry on the current GPT-350M ARC/dense profiling pair, set rank=32, update_interval=200, seed=42, start_compress_step=1000, and basis_sync=local_svd. Keep optimizer, batch, Muon, scalar optimizer settings, warmup/schedule/clipping policy, and model settings explicit. Task 10 must create separately numbered CM configs by pairing GreedyLore with the selected current dense baseline; do not reuse an older run name as new evidence.

- [ ] **Step 8: Run GREEN and commit**

~~~bash
uv run --frozen --extra train --extra dev pytest \
  tests/test_train_greedylore.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_train_ddp_sync.py \
  tests/test_configs.py -v
git add train_greedylore.py train.py dion/__init__.py \
  configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml \
  tests/test_train_greedylore.py tests/test_train_arctopk_ddp_hook.py \
  tests/test_configs.py
git commit -m "feat: integrate GreedyLore gradient sync with Muon"
~~~

---

### Task 8: Add rank-local DCP checkpoint and deterministic continuation

**Files:**
- Modify: dion/greedy_lore_ddp_hook.py
- Create: tests/test_greedy_lore_ddp_checkpoint.py
- Modify: tests/test_train_greedylore.py

**Interfaces:**
- Consumes: the shared CheckpointManager extra-stateful contract and preallocated Task 3 state.
- Produces:

~~~python
class GreedyLoreDDPState:
    def checkpoint_metadata(self) -> dict[str, Any]: ...
    def validate_checkpoint_metadata(self, metadata: dict[str, Any]) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
~~~

- [ ] **Step 1: Write a failing real two-rank DCP round trip**

Run through a refresh and compressed step, create deliberately different local errors, save, destroy every old model/optimizer/hook object, rebuild with a different bucket cap, load into fresh preallocated state, and compare errors, bases, supports, and compressor committed step by stable name. Verify optimizer state through its public state_dict round trip and subsequent parameter updates, not by assuming a shared optimizer group step field.

- [ ] **Step 2: Continue across the next refresh boundary**

Run fixed gradients through both uninterrupted and restored instances until after the next refresh. Assert reconstructed gradients, new bases/supports/errors, Muon momentum, and parameters match within tolerance.

- [ ] **Step 3: Write metadata and payload incompatibility tests**

The JSON metadata contains the exact stable-name schema for every error, basis, and support tensor. Before dcp.load, reject schema, world size, group ranks, fingerprint, seed scheme, rank, interval, start step, basis mode, parameter name/order/shape/dtype/role, declared tensor schema, or an invalid compressor committed step. Do not inspect or compare optimizer-internal step counters before or after DCP: the current shared lifecycle deliberately decouples compressor progress from optimizer-specific state layouts. Instead require the restored compressor committed step to equal the saved compressor metadata, then prove alignment by deterministic continuation through the next refresh and matching optimizer public state/parameters. Saving or loading with an active step or unfinished tail also fails before DCP mutation. Separately corrupt a real DCP payload and assert DCP reports failure; do not claim rollback or untouched destination tensors for corruption that contradicts already validated JSON metadata.

- [ ] **Step 4: Implement metadata and stable-name tensor state**

Store rank-local values under rank_<global_rank>/<stable_name>/{error,basis,last_support} and shared scalar step/schema tensors under shared. Emit names, shapes, and dtypes for those exact tensors in checkpoint_metadata(), and validate that JSON before DCP loads tensor payloads. After load, verify replicated basis/support equality across ranks and fail closed on disagreement. Do not add a generic staging load or restructure CheckpointManager in this task.

- [ ] **Step 5: Run GREEN and commit**

~~~bash
uv run --frozen --extra train --extra dev pytest \
  tests/test_greedy_lore_ddp_checkpoint.py \
  tests/test_train_greedylore.py -v
git add dion/greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_checkpoint.py tests/test_train_greedylore.py
git commit -m "feat: checkpoint GreedyLore compressor state"
~~~

---

### Task 9: Extend profiler attribution and add a dedicated comparison launcher

**Files:**
- Modify: benchmark/compressed_muon/profiler_trace.py
- Modify: benchmark/compressed_muon/summarize_training_profiles.py
- Create: benchmark/compressed_muon/run_greedy_lore_profiler.sh
- Modify: dion/greedy_lore_ddp_hook.py
- Create: tests/test_greedy_lore_profiler_trace.py
- Create: tests/test_greedy_lore_profiler_launcher.py
- Modify: tests/test_training_profiler_trace.py

**Interfaces:**
- Consumes: existing Chrome traces, record_function, and CollectiveObserver.
- Produces: preserved ARC fields plus collective operation types, compressor-local GPU durations, collective-only and full critical-path tails, collective signatures, byte totals, overlap, and launcher plan metadata.

- [ ] **Step 1: Write failing synthetic trace classification tests**

Include base ranges and parameterized /payload bytes=N ranges for dense, basis broadcast, score All-Reduce, and factor All-Reduce. Require exact logical bytes, operation values, and collective order. Add launch-correlated hook-local SVD/score/Top-r/factor/error/reconstruction kernels and require exact per-category GPU milliseconds while proving they are excluded from genuine backward compute overlap.

- [ ] **Step 2: Cover all parser routing points**

Extend _CATEGORY_NAMES, _COLLECTIVE_CATEGORIES, _GRADIENT_COLLECTIVES, hook-local category sets, and _range_category prefix handling. Replace the arc_seed-only operation conditional with an explicit category-to-operation mapping in which arc_seed and greedylore_hook_basis_broadcast are broadcasts. Add gpu_ranges_ms by propagating CPU launch-range correlation to GPU kernels, without deleting or renaming current arc_* fields. A counterexample whose collective is inside the CPU backward range but after the last genuine backward GPU kernel reports zero compute overlap.

Add both exposed_gradient_sync_tail_ms, which remains collective-only for backward compatibility, and compressor_critical_path_tail_ms, which runs from the last genuine backward GPU kernel to the latest of the final GreedyLore local GPU kernel, final GreedyLore collective GPU kernel, or greedylore_hook/future_complete timestamp. Return None when the trace has no genuine backward GPU kernel. In a synthetic trace where factor All-Reduce ends first and reconstruction finishes later, the second metric must include reconstruction while the first does not; a dense-only trace must still end at its collective GPU completion.

- [ ] **Step 3: Add precise hook annotations**

Record bucket bytes, matrix bytes, dense auxiliary bytes, score bytes, factor bytes, optional basis bytes, parameter count, and refresh/compressed phase. Every collective gets a nested payload range carrying actual tensor bytes and an observer event with the same category. Every local GPU stage gets its own launch-correlatable range; CPU range durations remain separately available in cpu_ranges_ms.

- [ ] **Step 4: Write the launcher contract test**

Require --print-plan to produce rotated paired cells for dense, greedylore_local_svd, and greedylore_broadcast; parameterize world size, global/device batch, model geometry, bucket cap, warmup, measured full periods, rank, interval, GPU allow/exclude lists, and artifact root. No GPU ID is hard-coded.

- [ ] **Step 5: Implement the dedicated launcher**

Reuse the current safe artifact layout and fail-closed summary behavior. Profile one known refresh step and one known compressed step per mode, then run profiler-disabled timing for an integer number of complete update intervals. Save plan, commands, environment, stdout/stderr, exit codes, per-rank raw traces, and summary under one CM experiment directory.

- [ ] **Step 6: Run CPU contracts and commit**

~~~bash
uv run --frozen --extra dev pytest \
  tests/test_greedy_lore_profiler_trace.py \
  tests/test_greedy_lore_profiler_launcher.py \
  tests/test_training_profiler_trace.py -v
bash -n benchmark/compressed_muon/run_greedy_lore_profiler.sh
benchmark/compressed_muon/run_greedy_lore_profiler.sh --print-plan
git add benchmark/compressed_muon/profiler_trace.py \
  benchmark/compressed_muon/summarize_training_profiles.py \
  benchmark/compressed_muon/run_greedy_lore_profiler.sh \
  dion/greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_profiler_trace.py \
  tests/test_greedy_lore_profiler_launcher.py \
  tests/test_training_profiler_trace.py
git commit -m "perf: profile GreedyLore Muon communication"
~~~

---

### Task 10: Register M002 and run correctness, profiling, and quality gates

**Files:**
- Create: docs/compressed_muon/methods/M002_greedy_lore_muon.md
- Create: docs/worklog/M002-greedy-lore-muon.md
- Modify: docs/compressed_muon/METHOD_INDEX.md
- Modify: docs/compressed_muon/EXPERIMENTS.md
- Modify: docs/compressed_muon/RESULTS.md
- Modify: docs/compressed_muon/PAPER_NOTES.md
- Reference (read-only): ~/greedy_lore/run_c4_llama60m_lore_fp32.slurm
- Reference (read-only): ~/greedy_lore/run_c4_llama60m_lore_bf16.slurm

**Interfaces:**
- Consumes: all preceding implementation, tests, profiler launcher, and repository research conventions.
- Produces: an auditable M002 method record and evidence that distinguishes correctness, logical communication, profiler attribution, end-to-end performance, and training quality.

- [ ] **Step 1: Register the method before formal experiments**

Add M002 with status testing, describe compression as DDP gradient-input compression before unchanged Muon, state that MSGD/Adam theory does not transfer, and document local-SVD versus broadcast consistency/cost. Create one append-only worklog with links to the spec, plan, implementation, config, and subsequent artifacts.

Record the local author-code snapshot path and pinned hashes, the reusable
orientation/batching ideas, and every intentional divergence listed in the
spec. Do not call the snapshot an executable oracle: its real greedy hook has
an incomplete score helper and its fake hook communicates the dense tensor.

- [ ] **Step 2: Run the complete CPU correctness gate**

~~~bash
uv run --frozen --extra train --extra dev pytest \
  tests/test_greedy_lore.py \
  tests/test_greedy_lore_oracle.py \
  tests/test_greedy_lore_layout.py \
  tests/test_greedy_lore_layout_distributed.py \
  tests/test_greedy_lore_ddp_state.py \
  tests/test_greedy_lore_ddp_future.py \
  tests/test_greedy_lore_ddp_future_distributed.py \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py \
  tests/test_greedy_lore_ddp_checkpoint.py \
  tests/test_train_greedylore.py \
  tests/test_train_ddp_sync.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_greedy_lore_profiler_trace.py \
  tests/test_greedy_lore_profiler_launcher.py \
  tests/test_training_profiler_trace.py \
  tests/test_configs.py -v
~~~

Record the exact pass/skip counts in the M002 worklog. A skip in a GreedyLore or training-integration test is an unmet gate rather than a pass; only tests explicitly marked multi_gpu are excluded from this CPU command. Any failure blocks GPU profiling.

- [ ] **Step 3: Run the multi-GPU correctness gate**

On at least two exclusive idle GPUs, run the NCCL stress test and execute the real train.main entry in a tiny training smoke for dense Muon, local-SVD GreedyLore, and broadcast GreedyLore. Cover both unclipped execution and one grad_clip_norm-enabled case; the latter must preserve finish-hook -> clip reconstructed gradient -> Muon step -> compressor commit ordering. Immediately after a local-SVD refresh in the preflight, call validate_replicated_basis_across_ranks() outside the timed path. Require basis agreement at atol=1e-6 and rtol=1e-5, exact support equality, reconstructed-gradient agreement at atol=1e-5 and rtol=1e-4, exact collective-signature equality, and Muon momentum/parameter agreement at the dtype-appropriate tolerance. A mismatch blocks local-SVD performance claims and makes broadcast mode the correctness baseline; passing validates only the recorded homogeneous environment.

- [ ] **Step 4: Profile refresh and compressed critical paths**

Pre-register a CM experiment ID. Use the same model, batch geometry, accumulation, bucket cap, seed, and hardware for all three modes. Require traces to account for all collectives, separate Muon result communication, report score/SVD/factor/error/reconstruction GPU time, and calculate communication overlap only against genuine backward kernels.

- [ ] **Step 5: Measure complete-period wall-clock and memory**

Run at least three rotated paired blocks without profiler, each covering an integer number of update_interval periods after warmup. Report paired ratios/differences, mean, median, standard deviation, CV, throughput, peak allocated memory, local-SVD versus broadcast difference, and a 95% paired bootstrap or randomization interval. Extend to at least ten blocks if the interval crosses zero or the difference is below 2%.

- [ ] **Step 6: Run a short quality gate before long training**

Compare dense Muon and local-SVD GreedyLore from identical initialization/data order for multiple seeds. Record training/validation loss, gradient norms, NaN/Inf checks, and residual norms. Do not start a long paper-scale run unless the short run is stable and the performance evidence justifies it.

For the first paper-oriented quality recipe, use the repository's current
GPT-60M geometry corresponding to the paper's LLaMA-60M setup: global batch
512, device batch 128 on four ranks, sequence length 256, 10,000 updates,
1,000 warmup updates, rank 32, update interval 200, and Adam-style scalar
settings. Keep this separate from the short gate. Record two recipe deltas
explicitly: the inspected author launcher sets `grad_clipping=0` while the
current CM039 staged Muon recipe uses `grad_clip_norm=1.0`; moreover, the
snapshot's 60M launcher uses 200 scheduler-warmup steps even though paper Table
VII reports 1,000. The snapshot omits the referenced C4 training source, so its
cosine endpoint cannot be verified locally. Treat clipping as a paired ablation,
use Dion's existing cosine-to-zero behavior explicitly, and label these as
paper-oriented comparisons rather than exact reproductions.

- [ ] **Step 7: Apply claim boundaries and update research records**

Classify the result as positive, null, or negative. State separately: logical payload reduction, measured collective time, exposed tail, complete-period throughput, peak memory, and quality. Do not describe GreedyLore-Muon as theoretically convergent from the paper or as equivalent to dense Muon. Preserve commands, configuration, raw logs, traces, environment, and summary paths in EXPERIMENTS.md, RESULTS.md, the M002 worklog, and PAPER_NOTES.md.

- [ ] **Step 8: Run final relevant regression and commit research records**

~~~bash
uv run --frozen --extra train --extra dev pytest \
  tests/test_greedy_lore.py \
  tests/test_greedy_lore_oracle.py \
  tests/test_greedy_lore_layout.py \
  tests/test_greedy_lore_layout_distributed.py \
  tests/test_greedy_lore_ddp_state.py \
  tests/test_greedy_lore_ddp_future.py \
  tests/test_greedy_lore_ddp_future_distributed.py \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py \
  tests/test_greedy_lore_ddp_checkpoint.py \
  tests/test_train_greedylore.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_optimizers.py \
  tests/test_training_profiler_trace.py \
  tests/test_configs.py -v
git add docs/compressed_muon docs/worklog/M002-greedy-lore-muon.md
git commit -m "docs: record GreedyLore Muon evaluation"
~~~

---

## Execution Handoff

Implement tasks in order. Tasks 1-2 freeze algorithm semantics before any collective code; Tasks 3-6 establish distributed safety before formal training; Tasks 7-9 integrate and measure the method; Task 10 is the evidence and research record gate. Stop at every task commit for review, and do not launch formal GPU experiments until the preceding CPU and NCCL correctness gates pass.
