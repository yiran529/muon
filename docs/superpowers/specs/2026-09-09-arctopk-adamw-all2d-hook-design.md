# ARC-TopK AdamW All-2D DDP Hook Design

## Goal

Add a production training path for pure AdamW whose DDP gradients use the
existing ARC-TopK hook for every two-dimensional parameter. Dense AdamW remains
the baseline, the AdamW update rule and hyperparameters remain identical
between dense and compressed runs, and existing ARC-TopK Muon behavior remains
unchanged.

The intended comparison is:

```text
dense:  DDP dense gradient synchronization -> torch.optim.AdamW
ARC:    DDP all-2D ARC-TopK hook           -> torch.optim.AdamW
```

The existing optimizer-side `ArcTopKAdamW` remains available as a benchmark
control. It is not replaced or expanded into the primary quality-training
path.

## Architectural boundary

ARC-TopK is gradient synchronization, not an optimizer update rule. The hook
must reconstruct the gradients visible through DDP bucket views before the
ordinary optimizer step. AdamW then consumes those gradients without knowing
whether they came from dense DDP or ARC-TopK.

The hook owns:

- ARC-TopK projection, support selection, and selected-value collectives;
- EF21M local and global trackers;
- all-2D compression eligibility;
- stable parameter identity and layout validation;
- asynchronous bucket/Future/stream lifetime;
- compressor step lifecycle and checkpoint state.

The optimizer owns:

- AdamW momentum and variance;
- learning rate, betas, epsilon, and weight decay;
- parameter updates and optimizer checkpoint state.

The training loop is the only coordinator between them:

```text
compressor.begin_step()
  -> accumulated backward, with DDP hook on the final microbatch
compressor.finish_step()
  -> gradient norm
optimizer.step()
compressor.commit_step()
```

An optimizer exception prevents `commit_step()`, so compressor progress cannot
silently advance beyond a failed update.

## Supported configurations

`train_arctopk.py` continues to use `arc_sync_mode` as the single owner of
gradient synchronization and accepts these combinations:

| Optimizer setting | Sync mode | Update implementation | Gradient synchronization |
| --- | --- | --- | --- |
| `arc_topk_muon` | `optimizer` | `ArcTopKMuon` | optimizer-side ARC for Muon matrices |
| `arc_topk_muon` | `ddp_hook` | ordinary `Muon` | all-2D ARC DDP hook |
| `arc_topk_adamw` | `ddp_hook` | `torch.optim.AdamW` | all-2D ARC DDP hook |

`arc_topk_adamw` with `arc_sync_mode=optimizer` is deliberately unsupported in
the formal training entry point. The existing standalone benchmark remains the
place for optimizer-side `ArcTopKAdamW` comparisons. Unsupported combinations
must fail before model compilation or training.

Dense AdamW continues to use `train.py` with `optimizer=adamw` and ordinary DDP
synchronization.

## AdamW parity

Dense and hook-side AdamW must use one shared builder so their update semantics
cannot drift. The builder receives the existing parameter groups and applies:

```text
lr = hp.lr for every group
betas = (0.9, 0.95) for every group
weight_decay = hp.weight_decay
eps = the torch.optim.AdamW default unless an existing explicit setting is added
```

The scalar-optimizer setting and Muon-specific embedding/output-head learning
rate scaling are ignored for pure AdamW, matching the current dense AdamW path.
The builder returns `torch.optim.AdamW`; it does not use the repository's
custom foreach AdamW kernel.

Refactoring the current dense AdamW branch into this helper must be behavior
preserving. Existing dense configurations and CLI values must continue to
produce the same parameter groups and hyperparameters.

## All-2D compression policy

The AdamW hook path uses exactly the same parameter-role rule as the current
CM033 Muon hook path:

```python
role = "arc_matrix" if parameter.ndim == 2 else "dense_aux"
```

This includes Transformer matrices, token embeddings, and the language-model
head. Biases, normalization scales, and every non-two-dimensional parameter use
packed dense all-reduce inside the hook.

Compression eligibility is independent of optimizer parameter groups. Every
model parameter must occur exactly once in the canonical hook table and exactly
once across optimizer parameter groups. A mismatch fails during factory
construction.

## Optimizer-independent hook lifecycle

`ArcTopKDDPState` currently accepts an optional optimizer and inspects
`optimizer.param_groups[*]["step"]`. That contract is specific to the existing
Muon implementation and is incompatible with standard AdamW, which stores step
counters in per-parameter state.

The hook state will stop inspecting optimizer internals. Its committed-boundary
rules become:

- no active step;
- the bucket tail Future is complete;
- `finish_step()` covered the entire frozen parameter table before commit;
- only `commit_step()` advances `committed_step`;
- save/load occurs only at a committed boundary.

The existing `optimizer_parameters` argument remains required for ownership
validation. The optional optimizer reference and Muon-specific step mismatch
messages are removed. Checkpoint atomicity follows the training control flow:
the checkpoint manager saves the optimizer and compressor together only after
both have completed the same training iteration.

Existing ARC-TopK Muon hook checkpoints retain their serialized compressor
schema and remain loadable because optimizer step inspection is runtime-only;
the persisted parameter table, configuration, committed step, and tracker
tensors do not change.

## Training factory structure

`train_arctopk.py` separates three responsibilities:

1. Construct the existing model parameter groups.
2. Select the ordinary or optimizer-integrated update implementation.
3. When `arc_sync_mode=ddp_hook`, install one shared ARC hook runtime.

A private hook installation helper takes the model, DDP wrapper, optimizer,
and ARC configuration. It builds the canonical all-2D parameter specs,
validates the cross-rank fingerprint, registers `arc_topk_ddp_hook` exactly
once, and returns the existing `GradientSyncRuntime` callbacks plus checkpoint
state.

Both Muon-hook and AdamW-hook branches call this helper. There must not be an
AdamW-specific copy of the hook setup.

## Gradient accumulation and synchronization ownership

Hook-side AdamW sets `optimizer_owns_gradient_sync=False`. Existing training
logic therefore uses `DDP.no_sync()` for every non-final accumulation
microbatch and invokes the hook only on the final backward. The final bucket
views contain the accumulated local gradient, and the hook replaces DDP's
dense synchronization.

Optimizer-side `ArcTopKMuon` continues to set
`optimizer_owns_gradient_sync=True`, disabling all DDP gradient synchronization
so its optimizer can own ARC communication. No configuration may enable both
optimizer-side synchronization and the DDP hook.

## Checkpoint compatibility

Hook-side AdamW checkpoints contain two independent stateful objects:

- the standard AdamW optimizer state through the existing checkpoint manager;
- the ARC compressor state under the existing `arc_compressor` extra-state key.

The ARC state retains stable names, stable IDs, configuration fingerprint,
process-group membership, committed step, rank-local trackers, and replicated
global trackers. Resume requires the same world size, group membership,
parameter layout, dtype, and ARC configuration.

The implementation does not promise compatibility between optimizer-side
`ArcTopKAdamW` checkpoints and hook-side AdamW checkpoints because their ARC
state ownership and serialization layouts differ.

## Validation and tests

Implementation follows test-driven development in these layers.

### Hook state tests

- Construct hook state for parameters owned by standard `torch.optim.AdamW`.
- Save and load without a group-level `step` field.
- Reject save/load during an active or in-flight step.
- Preserve existing parameter ownership, layout, and tracker validation.
- Confirm an optimizer failure cannot be followed by compressor commit in the
  training control flow.

### Factory tests

- `arc_topk_adamw + ddp_hook` constructs `torch.optim.AdamW` and registers one
  hook.
- AdamW uses the same LR, betas, epsilon, weight decay, parameter order, and
  scheduler-visible groups as dense AdamW.
- embedding and language-model-head matrices are `arc_matrix`.
- bias and normalization parameters are `dense_aux`.
- `arc_topk_adamw + optimizer` fails early with a migration-quality message.
- Existing Muon optimizer and hook modes remain unchanged.

### Distributed correctness tests

- Two-rank Gloo with full support (`ratio=1`) matches dense AdamW updates.
- Sparse ARC keeps post-step parameters identical across ranks.
- Missing gradients follow the existing rank-symmetric hook policy.
- Gradient accumulation invokes hook communication only on the final
  microbatch.
- Interrupted-and-resumed AdamW hook training matches an uninterrupted run for
  model, optimizer, compressor step, and ARC tracker state.

### CUDA/NCCL smoke tests

- Every DDP bucket completes through the Future chain.
- all-2D parameters enter ARC and non-2D parameters enter dense communication.
- Future and CUDA stream lifetimes complete without a global synchronize in
  the hook.
- Cross-rank parameter checksums agree after multiple steps.

The existing ARC hook, Muon ARC, AdamW ARC, train integration, and checkpoint
test suites must all pass. A short GPT-60M smoke run must complete before a
paper-scale quality run is launched.

## Experiment artifacts

Add paired GPT-60M configurations for approximately 1.1B training tokens:

- dense DDP AdamW through `train.py`;
- all-2D ARC-hook AdamW through `train_arctopk.py`.

They must share model, data, tokenizer, sequence length, global batch, device
batch, gradient accumulation, training seed, AdamW hyperparameters, LR
schedule, validation tokens, and checkpoint cadence. The ARC run alone adds
ARC ratio, projection rank, eta, seed, and compression start step.

The launcher records final validation loss, `exp(loss)` perplexity, average
step time, tokens per second, peak memory, and the exact configs and commit.
Validation uses the repository's fixed-length FineWeb batches, for which the
existing equal-batch average is also a token-weighted average. Any future
padded evaluator must accumulate loss by non-padding token count instead.

Running the full experiment is a separate operational action from implementing
and verifying the code and launcher.

## Non-goals

- Rewriting the ARC-TopK or EF21M mathematical primitives.
- Removing optimizer-side `ArcTopKAdamW` or its benchmarks.
- Compressing non-two-dimensional gradients.
- Adding FSDP/HSDP support.
- Supporting world-size-changing resume.
- Changing Muon update ownership or CM033 behavior.
- Claiming quality or speed until the formal paired experiment completes.

