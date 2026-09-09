# GreedyLore-Muon DDP Gradient Compression Design

## Goal

Implement the GreedyLore method from arXiv:2507.08784v4 as an optional DDP
gradient synchronization path in front of the existing Muon optimizer. The
implementation must preserve dense Muon as the baseline, keep ARC-TopK
unchanged, and isolate the new algorithm in dedicated modules, tests,
configuration, profiling, and research records.

## Research interpretation

GreedyLore compresses the data-parallel gradient before the optimizer sees it.
For this project, every rank first reconstructs the same approximate global
gradient and ordinary `Muon` then performs its existing momentum, Nesterov,
orthogonalization, result communication, and parameter update.

This ordering defines a new approximate Muon variant:

```text
rank-local accumulated gradient
  -> GreedyLore DDP communication hook
  -> rank-consistent reconstructed gradient within declared tolerances
  -> ordinary Muon momentum and orthogonalization
  -> existing Muon result communication
  -> parameter update
```

The MSGD/Adam convergence results in the GreedyLore paper do not automatically
apply because Muon applies a nonlinear orthogonalization after momentum. The
method documentation and experiment reports must state this limitation.

## Supported scope

The first implementation supports:

- PyTorch DDP with one fixed process group;
- `find_unused_parameters=False` and a static participating parameter set;
- two-dimensional Muon matrix parameters;
- exact dense synchronization for embedding, output-head, scalar-optimizer,
  and other non-Muon parameters;
- gradient accumulation through the existing final-microbatch `DDP.no_sync()`
  policy;
- homogeneous ranks running the same PyTorch, CUDA, and linear-algebra stack;
- checkpoint resume with the same world size, group membership, parameter
  layout, configuration, and compressor phase.
- FP32 DDP buckets for Muon matrix parameters. Dense-only auxiliary buckets
  may retain their native dtype and use exact dense synchronization.

FSDP/HSDP, DDP join, unused parameters, dynamic parameter groups, world-size
changing resume, three-dimensional matrix batches, AMP GradScaler skip/retry,
and heterogeneous accelerator stacks are out of scope and must fail early
where they can otherwise produce silent errors.

## Per-parameter representation

For an original matrix gradient with shape `m x n`, define:

```text
a = min(m, n)
b = max(m, n)
transposed = m > n
```

`orient(G)` returns `G.T` when `transposed` is true and otherwise returns `G`,
so all compressed matrices have shape `a x b`. `unorient` applies the inverse
operation. This follows the prose in Appendix D of the paper; the displayed
Algorithm 4 reverses the final transpose branches and cannot be followed
literally because the resulting shapes do not match the parameter. Algorithm 3
defines the refresh/error/output recurrence; Appendix D is used only for its
matrix-orientation rule because its displayed Algorithm 4 also omits the main
algorithm's refresh-specific dense output and error reset.

Each Muon matrix parameter owns:

- a rank-local error buffer `error` with shape `a x b`;
- a complete replicated left basis `basis` with shape `a x a`;
- the last selected support of `rank` basis columns for diagnostics;
- stable name and stable integer ID used for checkpointing and local seed
  derivation.

The first implementation performs compressor state, low-rank arithmetic, and
matrix communication in FP32, matching the repository's current FP32 parameter
training path and keeping SVD behavior explicit. A non-FP32 matrix bucket fails
at factory construction. Supporting lower-precision matrix parameters, error
state, or factor communication requires a separate quality/performance
ablation.

## Configuration and phase

```python
@dataclass(frozen=True)
class GreedyLoreConfig:
    rank: int = 32
    update_interval: int = 200
    seed: int = 42
    start_compress_step: int = 1000
    basis_sync: Literal["local_svd", "broadcast"] = "local_svd"
    seed_scheme_version: int = 1
```

`rank` and `update_interval` are positive integers;
`seed_scheme_version` must equal `1`; `start_compress_step` is a non-negative
integer. Each matrix requires `rank <= min(m, n)`.

The lifecycle uses one-based optimizer steps, matching the existing ARC hook:

```text
step <= start_compress_step        -> dense warmup
phase = step - start_compress_step - 1
phase % update_interval == 0       -> refresh
otherwise                          -> compressed
```

Consequently, the first compressed step is always a refresh, independent of
the absolute training-step number. The committed step and this derived phase
are validated during resume.

## Refresh step

For rank `i`, let `g_i` be the rank-local accumulated gradient and `E_i` the
previous error buffer:

```text
H_i = orient(g_i).float() + E_i
H_bar = AllReduceSUM(H_i) / world_size
U, singular_values, _ = svd(H_bar, full_matrices=False)
U = canonicalize_svd_basis(U)
P = U[:, :rank]
E_i = 0
output = unorient(H_bar)
```

Because `a <= b`, `full_matrices=False` still returns the required complete
`a x a` left basis. The refresh output is the averaged corrected gradient, not
the averaged raw gradient when an old residual exists.

The paper's main Algorithm 3 resets error to zero and uses `H_bar` directly on
refresh. Computing and reducing `R = P.T @ H` on that step has no consumer and
is omitted as a semantics-preserving dead-computation removal.

### Basis consistency

The default `basis_sync="local_svd"` has every rank run SVD on the identical
all-reduced `H_bar`. It removes full-basis communication. Each column is sign
canonicalized by making its largest-absolute-value element positive, breaking
ties by the smallest row index. Singular vectors remain ordered by descending
singular value.

This removes ordinary column-sign ambiguity but cannot mathematically choose a
unique rotation inside an exactly repeated singular-value subspace. Therefore
`local_svd` is an experimental assumption limited to a tested homogeneous
environment, not a general rank-consistency guarantee. Preflight tests compare
basis, support, reconstructed gradient, Muon momentum, and parameters across
ranks. A committed-boundary diagnostic gathers the bases and fails on a
manually rotated repeated-singular-value subspace; it is required by the NCCL
correctness gate but excluded from timed production runs.

`basis_sync="broadcast"` is the strong-consistency fallback: process-group
rank zero computes and canonicalizes `U`, then broadcasts the full `a x a`
basis. Its payload is reported separately and is never silently included in a
paper-faithful communication claim.

## Compressed step

For every matrix parameter:

```text
H_i = orient(g_i).float() + E_i
Z_i = U.T @ H_i
lambda_i[j] = dot(Z_i[j, :], v_j)
lambda_bar = AllReduceSUM(lambda_i) / world_size
scores = lambda_bar.square()
J = stable_top_r(scores)
P = U[:, J]
R_i = P.T @ H_i
E_i = H_i - P @ R_i
R_bar = AllReduceSUM(copy(R_i)) / world_size
output = unorient(P @ R_bar)
```

The random vectors `v_j ~ Normal(0, I_b)` are produced with a local generator.
For schema version one, the exact seed is
`(base_seed + phase * 1_000_003 + stable_parameter_id) % (2**63 - 1)`, where
`phase = step - start_compress_step - 1`. The generator must not touch global
RNG state and no seed collective or tensor-to-host `.item()` is allowed.

Algorithm correctness depends on these orderings:

- reduce signed `lambda_i` first, then square `lambda_bar`;
- select Top-r with deterministic index tie-breaking;
- calculate `E_i` from rank-local `R_i` before any in-place collective could
  overwrite it;
- reduce a separate factor buffer and reconstruct from `R_bar`.

The `P R^T` expression printed in Algorithm 3 is dimensionally inconsistent;
the implementation follows equations (9)-(10) and uses `P @ R`.

## Bucket and collective protocol

Runtime state is keyed by parameter identity; persistence and seeds use stable
names and IDs. Bucket indices and callback arrival order are never persistent
identities.

Before compression starts, one packed dense All-Reduce synchronizes the bucket.
On a refresh step, corrected matrix gradients and dense auxiliary gradients are
packed into one dense All-Reduce. `local_svd` performs no second collective;
`broadcast` performs a basis broadcast for each matrix in stable parameter
order.

On a compressed step, one bucket chain performs:

```text
packed dense auxiliary values + signed lambda vectors: All-Reduce
  -> stable Top-r and local factor/error computation
packed matrix factors: All-Reduce
  -> reconstruction and bucket scatter
```

A dense-only bucket stops after its packed dense All-Reduce. Every rank must
launch an identical collective signature. The first production version uses a
single cross-bucket tail Future, matching the safe ARC pattern, rather than
attempting cross-bucket concurrency.

## Asynchronous safety

The hook records a CUDA event as soon as DDP marks the bucket ready. The
compressor execution stream waits for both that event and the preceding bucket
tail. Collective operations use `Work.get_future()` and explicit destination
Futures; nested `Future[Future[Tensor]]` is forbidden.

The DDP-facing Future completes only after the final reconstruction/scatter GPU
work is visible to the callback stream. Exceptions poison the active chain and
propagate to the Future returned to DDP. Bucket contexts and all packing,
factor, projection, and reconstruction tensors stay strongly referenced until
completion.

The hook path must not call `Work.wait()`, `torch.cuda.synchronize()`, tensor
`.item()`, perform device-to-host copies, or poll work completion.

## Training integration and checkpointing

`train_greedylore.py` mirrors the narrow factory structure of
`train_arctopk.py`. It constructs the existing Muon parameter groups, creates
ordinary `Muon`, validates one canonical GreedyLore layout fingerprint,
registers exactly one communication hook, and returns the existing
`GradientSyncRuntime` callbacks.

`GradientSyncRuntime` gains an optional `checkpoint_state_name`. The shared
training loop continues to use `"arc_compressor"` when the field is absent, so
existing ARC checkpoints and factories do not change. GreedyLore sets
`"greedy_lore_compressor"`.

Checkpoint metadata contains schema version, process-group ranks, world size,
configuration fingerprint, seed scheme, committed step, and ordered parameter
table. It also contains the exact stable-name tensor schema for every error,
basis, and support tensor, including shape and dtype. Rank-local error buffers
are stored under the global rank. Replicated bases and last supports are stored
with validation. JSON metadata incompatibility fails before `dcp.load` mutates
preallocated tensors. A storage payload that contradicts already validated
metadata is allowed to fail inside DCP and is not promised to roll back partial
destination writes.

## Profiling and communication accounting

The new hook records:

```text
greedylore_hook/bucket_ready
greedylore_hook/dense
greedylore_hook/svd
greedylore_hook/basis_broadcast
greedylore_hook/score
greedylore_hook/score_allreduce
greedylore_hook/topr
greedylore_hook/factor
greedylore_hook/factor_allreduce
greedylore_hook/error_feedback
greedylore_hook/reconstruct
greedylore_hook/future_complete
```

The trace parser must classify both base ranges and `/payload ...` ranges,
include GreedyLore in collective and hook-local sets, and preserve all existing
ARC metrics. New compressor-generic metrics report bucket bytes, logical dense
bytes, score bytes, factor bytes, optional basis bytes, collective counts,
per-category local GPU kernel time, communication/backward-compute overlap,
collective-only tail, and the complete compressor critical-path tail through
final reconstruction/Future completion. GPU local time is attributed by launch
correlation; CPU `record_function` duration is reported separately and never
substituted for GPU time. Basis payload ranges are classified as broadcast,
not All-Reduce.

For one `a x b` matrix with local SVD, the paper-faithful average logical
payload per compressed-period step is:

```text
ab / update_interval
  + (1 - 1 / update_interval) * (a + rank * b)
```

Broadcast mode adds `a*a/update_interval`. Dense auxiliary bytes are added
separately. These are logical payloads, not claimed wall-clock savings.

Profiler traces separately capture refresh and compressed steps. Performance
claims use repeated, profiler-disabled wall-clock runs covering complete update
intervals; the one-step Kineto window is attribution evidence, not throughput
evidence.

## Verification and acceptance

Correctness requires:

- a hand-computed multi-step oracle for the exact recurrence;
- signed-lambda reduction before square, deterministic ties, `m > n`, `r=1`,
  `r=a`, `tau=1`, zero, repeated, and near-repeated singular values;
- global RNG isolation and zero seed collectives;
- real two-rank Gloo agreement through refresh and compressed steps;
- real NCCL completion, stream visibility, exception propagation, allocator
  churn, and bucket rebuild/reorder coverage;
- `r=a` dense equivalence through the actual projection path, not a shortcut;
- gradients, bases, Muon momentum, and parameters agree across ranks within
  their declared numerical tolerances, while supports and collective
  signatures agree exactly;
- exactly one compressor advance per optimizer step under large gradient
  accumulation;
- deterministic DCP continuation across a refresh boundary.

Performance acceptance requires reporting refresh and compressed traces,
full-period wall-clock, peak allocated memory, score/SVD/factor/reconstruction
GPU time, logical and observed communication, overlap with genuine backward
kernels, exposed tail, throughput, and training loss. No acceleration claim is
made unless repeated complete-period runs improve end-to-end wall-clock while
the correctness and short-training quality gates pass.

## File boundary

Create:

- `dion/greedy_lore.py`
- `dion/greedy_lore_layout.py`
- `dion/greedy_lore_ddp_hook.py`
- `train_greedylore.py`
- focused `tests/test_greedy_lore*.py` files
- `configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml`
- `benchmark/compressed_muon/run_greedy_lore_profiler.sh`
- `docs/compressed_muon/methods/M002_greedy_lore_muon.md`
- `docs/worklog/M002-greedy-lore-muon.md`

Modify narrowly:

- `dion/__init__.py` for public exports;
- `train.py` for a backward-compatible checkpoint state name;
- profiler trace/summarizer modules and their tests for GreedyLore categories;
- research indexes and result records when implementation or experiments reach
  the corresponding stage.

Do not refactor `dion/arc_topk_ddp_hook.py`, subclass its ARC-specific state,
or change `dion/muon.py` in the first implementation. Small generic Future
helpers may be extracted only in a later change after both implementations are
stable and their true common boundary is demonstrated.
