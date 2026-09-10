# Task 7 Report: GreedyLore Muon Training Entry

## Implementation

Adopted and audited the inherited draft, then kept it because it satisfied the task contract without further production churn.

- Created `train_greedylore.py` with `GreedyLoreHyperparameters`, CLI parser extension, validation, and `init_greedy_lore_optimizer`.
- `init_greedy_lore_optimizer` rejects FSDP/device-mesh mode, missing DDP, explicit legacy optimizer-owned sync, unsupported GreedyLore optimizer names, unsupported scalar optimizer settings, and `find_unused_parameters=True`.
- The factory calls `train.build_muon_param_groups(model, hp)` and constructs the unchanged ordinary `dion.Muon` from all returned groups.
- Matrix membership is identity-based from the first Muon group only; two-dimensional embeddings and lm-head parameters remain `dense_aux`.
- GreedyLore layout/fingerprint validation runs once before registering exactly one `greedy_lore_ddp_hook`.
- The returned `GradientSyncRuntime` owns the GreedyLore begin/finish/commit lifecycle and checkpoints as `greedy_lore_compressor`.
- Updated `train.GradientSyncRuntime` with optional `checkpoint_state_name` and added `extra_stateful_from_gradient_sync_runtime`, preserving unnamed ARC state as `arc_compressor`.
- Exported `GreedyLoreConfig`, `GreedyLoreDDPParameterSpec`, and `GreedyLoreDDPState` from `dion/__init__.py`.
- Added `configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml`, based on the current GPT-350M ARC/dense profiling geometry with explicit optimizer, batch, schedule, clipping, model, scalar optimizer, and GreedyLore settings.
- Added `tests/test_train_greedylore.py` and focused ARC/config regression coverage.

Files:

- `train_greedylore.py`
- `train.py`
- `dion/__init__.py`
- `configs/compressed_muon/m002_greedy_lore_muon_ddp.yaml`
- `tests/test_train_greedylore.py`
- `tests/test_train_arctopk_ddp_hook.py`
- `tests/test_configs.py`

## Inherited Draft Assessment

The inherited draft had no report or commit, but its production shape matched the plan:

- It used ordinary `Muon`, not a compressed optimizer.
- It delegated param grouping to `train.build_muon_param_groups`.
- It used first-group parameter identity for GreedyLore matrix roles, not `ndim`.
- It preserved scalar AdamW group settings.
- It preserved ARC checkpoint naming through an unnamed-state fallback.
- It did not touch `dion/muon.py` or alter ARC hook behavior.

I inspected the draft tests, production code, config, shared training loop call order, ARC factory pattern, Muon param-group behavior, and compressed-Muon research guide. I found no required repair beyond retaining the draft and documenting it.

## RED Evidence

Because the draft already passed in-place, I recovered RED at the base commit in a disposable worktree:

```bash
tmp_root=$(mktemp -d /tmp/greedylore-red.XXXXXX)
git worktree add --detach "$tmp_root/base" 5887b91368eb6b5890bda1ed0011b66a37d0e465
cp tests/test_train_greedylore.py "$tmp_root/base/tests/test_train_greedylore.py"
cp tests/test_train_arctopk_ddp_hook.py "$tmp_root/base/tests/test_train_arctopk_ddp_hook.py"
cp tests/test_configs.py "$tmp_root/base/tests/test_configs.py"
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest \
  tests/test_train_greedylore.py::test_factory_builds_ordinary_muon_with_one_greedylore_hook_and_muon_group_roles \
  tests/test_train_arctopk_ddp_hook.py::test_hook_mode_builds_ordinary_muon_and_registers_exactly_one_hook \
  tests/test_configs.py::test_m002_greedylore_config_is_recognized_by_dedicated_entry -v
```

Output summary:

```text
collected 3 items
FAILED tests/test_train_greedylore.py::test_factory_builds_ordinary_muon_with_one_greedylore_hook_and_muon_group_roles
FAILED tests/test_train_arctopk_ddp_hook.py::test_hook_mode_builds_ordinary_muon_and_registers_exactly_one_hook
FAILED tests/test_configs.py::test_m002_greedylore_config_is_recognized_by_dedicated_entry

ModuleNotFoundError: No module named 'train_greedylore'
AttributeError: 'GradientSyncRuntime' object has no attribute 'checkpoint_state_name'
ModuleNotFoundError: No module named 'train_greedylore'
```

This is genuine inherited RED for the new factory/config path and the checkpoint-name compatibility change. The RED is necessarily broad for `tests/test_train_greedylore.py` because `train_greedylore.py` did not exist at base.

The disposable worktree was removed after the RED run.

## GREEN Evidence

Initial attempt without the task-required environment failed during collection:

```bash
uv run --frozen --extra train --extra dev pytest \
  tests/test_train_greedylore.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_train_ddp_sync.py \
  tests/test_configs.py -v
```

Output summary:

```text
ERROR tests/test_train_ddp_sync.py
ModuleNotFoundError: No module named 'train'
```

Rerun with the brief's required `PYTHONPATH=.`:

```bash
PYTHONPATH=. uv run --frozen --extra train --extra dev pytest \
  tests/test_train_greedylore.py \
  tests/test_train_arctopk_ddp_hook.py \
  tests/test_train_ddp_sync.py \
  tests/test_configs.py -v
```

Output summary:

```text
46 passed, 14 warnings in 105.69s (0:01:45)
```

Focused Task 6 CPU regressions:

```bash
PYTHONPATH=. uv run --frozen --extra dev pytest \
  tests/test_greedy_lore_ddp_hook.py \
  tests/test_greedy_lore_ddp_hook_distributed.py -v
```

Output summary:

```text
12 passed, 14 warnings in 34.20s
```

## Self-Review

- DDP-only: yes; device mesh and missing DDP are rejected in the factory.
- Unsupported legacy sync: yes; explicit `replicate_mesh_grad_sync` use is rejected.
- Ordinary Muon: yes; `type(optimizer) is Muon` is asserted.
- Matrix roles: yes; identity membership in `build_muon_param_groups(...)[0]["params"]`, with embeddings/lm-head as `dense_aux`.
- Scalar AdamW fields: yes; scalar group LR/betas/epsilon/weight decay are asserted.
- Hook registration: yes; the factory registers one GreedyLore DDP hook and returns state lifecycle callbacks.
- Checkpoint naming: yes; unnamed ARC remains `arc_compressor`, GreedyLore is `greedy_lore_compressor`, optimizer-only runtime produces no extra state.
- Training-loop order: yes; tests exercise finish hook before gradient clipping, ordinary Muon step, then compressor commit; clipping sees reconstructed gradients and leaves GreedyLore error unchanged.
- Dense equivalence: yes; full-rank GreedyLore-Muon matches dense Muon gradients, momentum, and parameters while executing GreedyLore score/factor collectives.
- ARC preservation: yes; Task 7 suite and focused Task 6 regressions pass; `dion/muon.py` was not modified.

## Concerns

- No GPU/NCCL smoke was run for this task; Task 7 brief only required CPU/Gloo factory and focused regressions, while Task 10 owns real `train.main` CUDA/NCCL smoke coverage.
- The first GREEN command must be run with `PYTHONPATH=.` in this worktree, matching the task note; without it, `tests/test_train_ddp_sync.py` cannot import `train`.
