# GreedyLore Bucket-Native BF16 Design

## Goal

Add an opt-in full-model BF16 parameter mode shared by dense Muon and
GreedyLore-Muon, while preserving FP32 as the default. In GreedyLore, persistent
compression state and ordinary collectives follow the DDP bucket dtype; only
numerically sensitive operations use temporary FP32 compute.

This enables fair, newly-run `FP32/BF16 x dense/GreedyLore` comparisons. A BF16
GreedyLore result must not be paired with an older FP32 dense result.

## Scope and isolation

- Add `model_dtype: {float32,bfloat16}` to the shared training configuration and
  CLI. The default is `float32`, so existing configurations retain their current
  behavior.
- Apply the selected dtype to every trainable GPT parameter before DDP wrapping:
  transformer matrices, token embedding, and LM head. The current GPT uses
  parameter-free functional RMSNorm, so there are no norm parameters to cast.
- Initially support explicit BF16 parameter mode only on the DDP path used by
  dense Muon and GreedyLore. Reject BF16 with a device mesh because FSDP already
  has a distinct mixed-precision policy with FP32 reductions.
- Do not change ARC-TopK, Sparse-K, Dion, or other compressor semantics. They
  continue to use FP32 unless their caller explicitly selects the shared BF16
  model mode.

## Model and optimizer dtype flow

The selected dtype is established before weight initialization so all trainable
parameters are born in that dtype. BF16 parameters naturally produce BF16
gradients and BF16 DDP buckets. BF16 autocast remains enabled for forward passes
in both parameter modes, preserving the existing FP32-default training recipe.

Muon and scalar optimizer momentum/variance buffers continue to be allocated
with `zeros_like(parameter)`, so they follow parameter dtype. Optimizer step,
learning-rate, and other control scalars remain FP32. If an operation requires
greater precision, it may use a temporary FP32 value and cast the result back;
the implementation must not introduce persistent FP32 parameter replicas.

## GreedyLore dtype flow

For each matrix parameter, `error` and `basis` are allocated in the parameter
and bucket dtype. Corrected gradients, random projections, signed scores,
projectors, low-rank factors, reconstruction, and the corresponding score,
factor, and basis collectives use that dtype by default.

SVD and sign canonicalization operate on a temporary FP32 tensor. The refreshed
basis is cast back into the bucket-native persistent basis. Integer supports and
seed/control state keep their existing integer types.

The existing `greedy_lore_dense_aux_communication_dtype` option remains as an
explicit diagnostic override for the packed `score+dense_aux` All-Reduce. Its
default `bucket` supplies the bucket-native behavior. `float32` and `bfloat16`
continue to override only that packed collective; factor, error, and basis stay
bucket-native.

Warmup and refresh-phase dense gradient All-Reduce also naturally use the DDP
bucket dtype. A refresh computes the global corrected gradient in bucket dtype,
temporarily upcasts it for SVD, and stores/broadcasts the resulting basis in
bucket dtype.

## Checkpoint behavior

The existing GreedyLore metadata and parameter fingerprint record actual tensor
dtypes. FP32 and BF16 checkpoints must each round-trip and resume deterministically
within their own dtype. Cross-dtype restore is rejected before tensor loading;
there is no implicit checkpoint precision conversion.

## Validation

- Training configuration tests cover the FP32 default, BF16 parsing, invalid
  values, full-model parameter dtype, and the BF16/device-mesh rejection.
- Optimizer tests cover BF16 Muon matrix state plus Lion and AdamW scalar state
  allocation and one-step updates, while retaining FP32 control scalars.
- GreedyLore unit tests parameterize FP32/BF16 across warmup, refresh, and
  compressed phases and assert bucket-native error, basis, score, factor,
  dense-aux, and basis-broadcast dtypes.
- Two-rank tests verify BF16 collective payloads, rank-consistent reconstruction,
  explicit packed-collective dtype overrides, and same-dtype checkpoint resume.
- Cross-dtype checkpoint tests verify fail-closed behavior.
- Run the focused training-entry, Muon, GreedyLore, layout, and checkpoint suites,
  followed by NCCL smoke tests when suitable GPUs are available.

## Experiment follow-up

Create new, strictly paired configurations and a serial launcher for
`FP32/BF16 x dense Muon/GreedyLore`, with identical model, seed, batch, schedule,
and optimizer hyperparameters apart from the intended method/dtype dimensions.
Assign new CM identifiers and record actual runs in `EXPERIMENTS.md` and the M002
worklog. Update `RESULTS.md` and paper conclusions only after completed runs.

The unrelated, pre-existing `PAPER_NOTES.md` working-tree edit must be preserved
and excluded from task commits unless a later result requires a safe append.
