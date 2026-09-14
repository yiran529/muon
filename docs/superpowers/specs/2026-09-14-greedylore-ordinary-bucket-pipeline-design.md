# GreedyLore Ordinary Bucket Pipeline Design

## Goal

Allow an ordinary compressed GreedyLore bucket to prepare its score as soon as
DDP marks the bucket ready, and allow the next bucket's first collective to run
while the preceding bucket reconstructs its gradient. Preserve the existing
per-bucket collective order and all numerical semantics.

## Scope

The pipeline applies only to ordinary compressed steps. Warmup and refresh
steps retain the existing globally serialized bucket chain. No new process
group, communicator, persistent workspace, formal benchmark, or configuration
switch is introduced.

## State and context model

`GreedyLoreDDPState` owns long-lived parameter state and two independent tails:

- `collective_tail` orders the first collective of every bucket. For an
  ordinary compressed bucket it completes after the bucket's factor AllReduce;
  for conservative paths it completes with the full bucket Future.
- `completion_tail` aggregates all DDP-facing bucket completions without using
  that aggregate to schedule collectives. Step finish and checkpoint boundaries
  require this aggregate to be complete.

Each `BucketContext` captures the preceding collective tail, owns a placeholder
for its own collective completion, owns the DDP-facing completion Future, and
retains any asynchronously prepared tensors and CUDA events.

## CUDA flow

At hook entry, an ordinary compressed matrix bucket records its existing bucket
ready event, then queues corrected-gradient, score, and packing work on a
per-device preparation stream. A `prepare_done` event connects this work to the
communication execution stream.

The bucket's first collective waits for both its preceding collective Future
and its own `prepare_done` event. Score reduction, projector selection, local
factor/error computation, and factor reduction retain their current order.

After factor reduction, the callback records an exact factor-ready event. It
completes the bucket's collective placeholder so the next bucket can launch,
and independently queues reconstruction on a per-device reconstruction stream.
Only reconstruction completion resolves the DDP-facing Future.

The global collective sequence remains:

```text
score(A) -> factor(A) -> score(B) -> factor(B)
```

The new overlap opportunity is:

```text
factor(A) -> reconstruct(A)
          -> score(B)
```

## Lifetime and failure rules

Prepared buffers remain retained by their context through DDP completion.
Preparation or collective failures fail both the collective placeholder and
the DDP completion. A reconstruction failure fails the DDP completion but must
not suppress already-valid later collective submissions, which could otherwise
make ranks disagree about collective order.

## Verification

CPU Future tests prove that preparation happens before the preceding DDP
completion, that the next bucket is released by collective rather than
reconstruction completion, that finish waits for every reconstruction, and
that failures propagate. Existing hook/oracle tests prove numerical behavior
and collective signatures. CUDA/NCCL smoke tests, when run, prove final stream
visibility and rank-consistent order. Formal timing experiments are out of
scope for this implementation pass.
