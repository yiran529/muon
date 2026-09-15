# PowerSGD-Muon DDP Gradient Compression Design

## Goal

Implement PowerSGD as an optional DDP gradient synchronization path in front
of the existing Muon optimizer. Every rank reconstructs the same approximate
global gradient before ordinary Muon performs momentum, Nesterov,
orthogonalization, result communication, and the parameter update.

The method is provisionally M005 once implementation artifacts are added. It
is an approximate Muon variant, not a communication-equivalent implementation
of dense Muon, because Muon's orthogonalization is nonlinear.

## Source hierarchy

Algorithm semantics follow the local PowerSGD implementation at
`/home/wyr/greedy_lore/comm_hooks/powerSGD_hook.py`. Dion's existing
GreedyLore hook is the reference for lifecycle, checkpoint, Future, CUDA
stream, bucket-rebuild, parameter-layout, and profiling behavior. The local
PowerSGD source is read-only reference code and is not a runtime dependency.

The implementation deliberately does not copy reference behaviors that are
unsafe in Dion: state keyed only by bucket index, iteration advancement through
`bucket.is_last()`, callback-side `Work.wait()`, device-wide synchronization,
and random-number progression coupled to bucket arrival order.

## Scope

The first version supports:

- DDP with one fixed process group;
- `find_unused_parameters=False` and a static participating parameter set;
- two-dimensional parameters selected by the Muon parameter group;
- exact dense synchronization for embedding, LM head, scalar-optimizer, vector,
  and unprofitable matrix parameters;
- DDP gradient accumulation through the existing final-microbatch policy;
- FP32 and BF16 parameter/bucket dtypes;
- EF14 error feedback, warm-started right factors, and dense warmup;
- checkpoint resume at a committed step with the same process group, world
  size, parameter layout, dtype, and compressor configuration.

FSDP/HSDP, DDP join, unused parameters, dynamic parameter groups,
world-size-changing resume, three-dimensional matrix batches, and AMP
GradScaler skip/retry are out of scope and fail early where silent corruption
would otherwise be possible.

## Algorithm

For rank `i`, a compressible matrix gradient `G_i` and its local error `E_i`
form the corrected gradient:

```text
H_i = G_i + E_i
```

The first compressed step initializes `Q` from a deterministic standard normal
distribution shared by all ranks. Later steps reuse the preceding averaged
`Q` when warm start is enabled. A PowerSGD step is:

```text
Q = orthogonalize(Q)
P_i = H_i @ Q
P = orthogonalize(AllReduceSUM(P_i))
Q_i = H_i.T @ P
Q_bar = AllReduceSUM(Q_i) / world_size
G_hat = P @ Q_bar.T
E_i = H_i - G_hat
```

The scale of the reduced `P` does not affect its orthonormal basis, so it is
not divided by world size before orthogonalization. The reconstructed
`G_hat` is identical across ranks. Error remains rank-local and is defined
against the global reconstruction, matching the reference hook.

Version one preserves each parameter's native two-dimensional orientation to
stay close to the reference PowerSGD implementation. Canonical transposition
to reduce orthogonalization cost is a later, separately measured optimization.

When error feedback is disabled, `H_i = G_i` and no error tensor is updated.
When warm start is disabled, `Q` is regenerated deterministically on every
compressed step. The seed is derived from the base seed, zero-based compressed
phase, and stable parameter ID, independent of bucket arrival order and without
touching global RNG state.

## Compression eligibility and dtype

A matrix with shape `m x n` and effective rank
`r = min(config.rank, m, n)` is compressed only when:

```text
min_compression_rate * r * (m + n) < m * n
```

The decision is part of the frozen parameter layout and is identical across
ranks. Non-matrix parameters and rejected matrices use exact dense averaging.
The default scope contains only parameters owned by Muon's matrix group;
compressing embedding and LM-head matrices is a later explicit ablation.

Communication buffers use the DDP bucket dtype by default. Orthogonalization
accumulates in FP32 and casts its result back to the communication dtype. This
avoids relying on half-precision QR while keeping payload accounting aligned
with actual bucket-native communication.

## State and checkpoint

Each compressible parameter owns:

- stable name and integer ID;
- an EF14 error tensor with the parameter's shape and dtype;
- a warm-start `Q` tensor with shape `[columns, effective_rank]` and dtype;
- a boolean tensor indicating whether `Q` has been initialized.

`P`, corrected-gradient views, packed buffers, CUDA events, and Futures are
transient bucket context, not checkpoint state.

The state object owns the one-based lifecycle
`begin_step -> note_bucket -> finish_step -> commit_step`. It validates exact
parameter coverage, tracks an aggregate completion tail, and rejects
checkpoint/save/load while a step is active. Metadata validates schema version,
world size, group membership, configuration fingerprint, seed scheme, ordered
parameter table, committed step, and tensor schema before payload copying.

## Bucket and collective protocol

Warmup and dense-only buckets use one exact dense All-Reduce. A compressed
bucket packs exact dense entries and all local `P` factors into its first
buffer:

```text
dense auxiliary + P factors All-Reduce
  -> copy averaged dense entries
  -> orthogonalize reduced P factors
  -> compute and pack local Q factors
Q factors All-Reduce
  -> divide Q by world size
  -> reconstruct matrices and update local errors
  -> write approximate gradients into DDP bucket
```

Preparation starts on a dedicated CUDA stream as soon as the bucket is ready.
A single collective tail preserves the same cross-rank order:

```text
P(A) -> Q(A) -> P(B) -> Q(B)
```

After `Q(A)` completes, the collective tail releases bucket B while bucket A
reconstructs on a separate stream. The DDP-facing Future for A completes only
after reconstruction and error writes are visible. Step finish waits for the
aggregate of every bucket completion, not only the last-arriving bucket.

Failure before or during a collective poisons both the bucket completion and
the collective tail. Failure after the second collective fails that bucket's
DDP Future but does not suppress already-valid later collective submission.

## Integration

`train_powersgd.py` builds the existing Muon parameter groups and an unchanged
`dion.Muon`, freezes stable parameter roles, validates the layout fingerprint
across ranks, registers the PowerSGD hook, and returns a
`train.GradientSyncRuntime`. The runtime declares that the optimizer does not
own gradient synchronization and exposes lifecycle and checkpoint state.

No behavior in dense Muon, ARC-TopK, GreedyLore, Rand-K, or Top-K is changed.

## Observability

Every collective is reported through Dion's collective observer. Profiler
ranges distinguish bucket readiness, preparation, first-factor All-Reduce,
second-factor All-Reduce, orthogonalization, reconstruction/error, Future
completion, and chain wait. Logical and physical payload summaries include
both factors and every dense fallback; they do not present `r(m+n)` as an
end-to-end speedup claim.

## Verification

CPU tensor tests cover validation, eligibility, deterministic random factors,
orthogonalization, reconstruction, full-rank exactness, warm start, and EF14.
Two-rank Gloo tests cover dense warmup, mixed buckets, rank-consistent output,
local error recurrence, collective order, bucket rebuild, and checkpoint
resume. Existing fake-bucket Future tests cover lifecycle and failure behavior.
When exclusive GPUs are available, NCCL tests cover final stream visibility,
rank-skewed delays, allocator churn, and collective signature agreement.

Performance evaluation begins with profiler-off paired timing on GPT-130M and
GPT-1B using ranks 1, 4, 8, 16, and 32. Dense Muon, M002, and M005 are compared
both at matched rank and approximately matched logical payload. Quality runs
start only after correctness and timing gates, and must report that PowerSGD's
SGD convergence results do not automatically apply to nonlinear Muon.

