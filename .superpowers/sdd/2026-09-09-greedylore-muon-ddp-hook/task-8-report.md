# Task 8 Report: GreedyLore DCP Checkpoint State

## Implementation

Added rank-local GreedyLore DDP checkpoint support without changing the shared `CheckpointManager` lifecycle.

- `dion/greedy_lore_ddp_hook.py`
  - Added `checkpoint_metadata()`, `validate_checkpoint_metadata()`, and `load_state_dict()` to `GreedyLoreDDPState`.
  - Stored tensor payloads at `rank_<global_rank>/<stable_name>/{error,basis,last_support}`.
  - Stored shared scalar tensors at `shared/schema_version` and `shared/committed_step`.
  - Emitted value-only JSON metadata containing schema version, world size, group ranks, fingerprint, config, seed scheme, committed step, ordered parameter table, and exact tensor schema.
  - Rejected incompatible metadata before DCP load via the existing `CheckpointManager` extra-stateful validation hook.
  - Rejected incompatible tensor payload structure, dtype, shape, schema scalar, or committed step before copying into preallocated tensors.
  - Remembered the validated metadata committed step and required the loaded shared committed-step tensor to match it.
  - Validated replicated basis/support after load; `state_dict()` also validates replicated basis/support before save.
- `tests/test_greedy_lore_ddp_checkpoint.py`
  - Added local metadata/payload compatibility tests.
  - Added a real two-rank DCP round trip that saves after a refresh and compressed steps, rebuilds all model/optimizer/hook objects with a different bucket cap, restores, and continues through the next refresh boundary.
  - Compared errors, bases, supports, reconstructed gradients, parameters, and public optimizer `state_dict()` snapshots; no optimizer internal step fields are inspected.
  - Added a real corrupted-DCP-payload case; truncating a `.distcp` shard raises PyTorch DCP `CheckpointException`.
- `tests/test_train_greedylore.py`
  - Added a factory-facing assertion that the runtime checkpoint object exposes the GreedyLore compressor metadata/load contract under `greedy_lore_compressor`.

## RED Evidence

Initial RED run after adding the checkpoint tests:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py -v
```

Output summary:

```text
collected 27 items
27 failed
AttributeError: 'GreedyLoreDDPState' object has no attribute 'checkpoint_metadata'
AttributeError: 'GreedyLoreDDPState' object has no attribute 'load_state_dict'
ProcessRaisedException: stateful.load_state_dict(state_dict[name])
AttributeError: 'GreedyLoreDDPState' object has no attribute 'load_state_dict'
```

Corrupted payload test first attempt, before catching PyTorch DCP's explicit failure type:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py -v
```

Output summary:

```text
collected 28 items
27 passed, 1 failed
torch.distributed.checkpoint.api.CheckpointException: CheckpointException ranks:dict_keys([0, 1])
EOFError
```

PyTorch DCP exception inheritance check:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev python - <<'PY'
from torch.distributed.checkpoint.api import CheckpointException
print(CheckpointException.__mro__)
PY
```

Output:

```text
(<class 'torch.distributed.checkpoint.api.CheckpointException'>, <class 'BaseException'>, <class 'object'>)
```

Committed-step metadata/payload mismatch RED:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py::test_load_requires_payload_committed_step_to_match_validated_metadata -v
```

Output summary:

```text
collected 1 item
FAILED tests/test_greedy_lore_ddp_checkpoint.py::test_load_requires_payload_committed_step_to_match_validated_metadata
Failed: DID NOT RAISE ValueError
```

## GREEN Evidence

Focused checkpoint suite after implementing the main contract:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py -v
```

Output summary:

```text
28 passed, 14 warnings in 28.01s
```

Checkpoint suite after adding explicit parameter name/order metadata cases:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py -v
```

Output summary:

```text
30 passed, 14 warnings in 27.91s
```

Committed-step mismatch GREEN:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py::test_load_requires_payload_committed_step_to_match_validated_metadata -v
```

Output summary:

```text
1 passed, 14 warnings in 3.57s
```

Task 3 regressions:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_layout.py tests/test_greedy_lore_layout_distributed.py tests/test_greedy_lore_ddp_state.py -v
```

Output summary:

```text
41 passed, 14 warnings in 38.33s
```

Task 7 regressions:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_train_greedylore.py tests/test_train_arctopk_ddp_hook.py tests/test_train_ddp_sync.py tests/test_configs.py -v
```

Output summary:

```text
46 passed, 14 warnings in 106.09s (0:01:46)
```

Exact Task 8 suite after final code changes:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py tests/test_train_greedylore.py -v
```

Output summary:

```text
40 passed, 14 warnings in 65.33s (0:01:05)
```

Full required Task 8 plus affected Task 3/7 regression set:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest tests/test_greedy_lore_ddp_checkpoint.py tests/test_train_greedylore.py tests/test_greedy_lore_layout.py tests/test_greedy_lore_layout_distributed.py tests/test_greedy_lore_ddp_state.py tests/test_train_arctopk_ddp_hook.py tests/test_train_ddp_sync.py tests/test_configs.py -v
```

Output summary:

```text
118 passed, 14 warnings in 166.08s (0:02:46)
```

Whitespace check:

```bash
git diff --check
```

Output:

```text
```

Formatting command:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev black dion/greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_checkpoint.py tests/test_train_greedylore.py
```

Output summary:

```text
Warning: Python 3.10 cannot parse code formatted for Python 3.15...
reformatted tests/test_greedy_lore_ddp_checkpoint.py
reformatted tests/test_train_greedylore.py
reformatted dion/greedy_lore_ddp_hook.py
All done
```

Unrelated Black formatting in `tests/test_train_greedylore.py` and one existing hook loop was reverted by hand to keep the final diff scoped.

## Self-Review

- Metadata fail-closed coverage includes schema, world size, group ranks, fingerprint, seed scheme, config rank/update interval/start step/basis mode, parameter name/order/shape/dtype/role, tensor schema names/shapes/dtypes, and invalid committed step.
- `CheckpointManager` was not restructured; the existing extra-stateful metadata validation path is used unchanged.
- State load copies into existing preallocated `error`, `basis`, and `last_support` tensors only after structure and tensor schema checks pass.
- Save/load both require a committed step boundary; active step and finish-before-commit tail states are rejected.
- The real DCP round trip destroys old model/optimizer/runtime/hook objects before rebuilding with a different DDP bucket cap.
- Deterministic continuation is proven with public optimizer `state_dict()` equality, reconstructed gradient equality, parameter equality, and GreedyLore basis/support/error equality through the next refresh.
- Replicated basis/support validation runs after load and in the test path after later refreshes.
- ARC and ordinary Muon paths were not modified.

## Concerns

- The tests use CPU/Gloo two-rank DCP. No CUDA/NCCL checkpoint smoke was run for this task.
- The corrupted-payload assertion relies on current PyTorch DCP raising `CheckpointException`, which in this environment inherits directly from `BaseException`.
