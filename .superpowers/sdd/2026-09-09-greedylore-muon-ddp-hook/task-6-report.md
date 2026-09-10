# Task 6 Report: Compressed GreedyLore Synchronization

## RED

Command:

```bash
PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook.py::test_compressed_step_reduces_signed_scores_before_square_and_updates_local_error_before_factor_average tests/test_greedy_lore_ddp_hook.py::test_dense_only_compressed_bucket_launches_one_dense_reduction_and_stops tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_compressed_hook_matches_three_step_oracle_and_signature -v
```

Output summary:

```text
3 failed, 14 warnings in 8.41s
NotImplementedError: GreedyLore compressed non-refresh hook path is not implemented yet
RuntimeError: Unable to cast GreedyLore compressed non-refresh hook path is not implemented yet to Tensor
```

An earlier draft of the first unit regression failed before reaching the hook because it patched a helper the hook had not imported. I fixed the test setup and reran RED; the retained RED above failed on the intended missing compressed path.

## Implementation

Files changed:

- `dion/greedy_lore_ddp_hook.py`
- `tests/test_greedy_lore_ddp_hook.py`
- `tests/test_greedy_lore_ddp_hook_distributed.py`
- `tests/test_greedy_lore_ddp_hook_nccl.py`

Implemented compressed non-refresh bucket synchronization:

- dense-only compressed buckets use the existing exact dense all-reduce and stop;
- mixed buckets pack signed lambda vectors and FP32 dense auxiliary gradients into one `score_plus_aux_allreduce`;
- scores are averaged while signed, then squared inside `select_projector`;
- seeds are derived from `GreedyLoreConfig.seed`, `BucketContext.phase`, and stable parameter IDs;
- local factors and next errors are computed from untouched local corrected matrices before launching factor reduction;
- packed factor reduction uses a distinct contiguous factor buffer;
- final reconstruction runs from the factor Future completion and writes back into the original bucket gradient views.

Added tests for:

- signed-score cancellation and error-before-factor-reduction ordering;
- dense-only compressed buckets launching only one dense collective;
- forbidden synchronization strings in the registered hook module;
- two-rank Gloo refresh plus three compressed steps against the Task 2 recurrence oracle and collective signature;
- real NCCL repeated compressed iterations with small buckets, bucket rebuild pressure, gradient accumulation, allocator churn, delayed callbacks, non-default stream consumption, zero-input refresh coverage, retained-context checks, rank agreement, and signature agreement.

## GREEN

Command:

```bash
PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py -v
```

Output:

```text
12 passed, 14 warnings in 35.07s
```

Command:

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook_nccl.py -v
```

Output:

```text
3 passed, 14 warnings in 22.54s
```

Command:

```bash
PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_future.py tests/test_greedy_lore_ddp_future_distributed.py -v
```

Output:

```text
6 passed, 14 warnings in 8.43s
```

Additional checks:

```bash
PYTHONPATH=. uv run --frozen --extra dev python -m py_compile dion/greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py tests/test_greedy_lore_ddp_hook_nccl.py
git diff --check
rg "Work\\.wait|torch\\.cuda\\.synchronize|\\.item\\(|\\.cpu\\(|\\.to\\([\\\"']cpu" dion/greedy_lore_ddp_hook.py
```

All exited 0 except the `rg` scan, which exited 1 because it found no forbidden strings. I also tried `uv run --frozen --extra dev ruff check ...`; it failed to spawn `ruff` because `ruff` is not installed in this frozen dev environment.

## Self-Review

- Collective order is one score+aux all-reduce followed by one factor all-reduce for mixed compressed buckets.
- Dense-only compressed buckets do not launch score or factor collectives.
- No seed or full-matrix collective is launched on compressed steps.
- The callback path contains no `Work.wait`, `torch.cuda.synchronize`, `.item(`, `.cpu(`, or explicit CPU `.to(...)`.
- The local error update happens after local factor computation and before factor all-reduce launch.
- Packed factor reduction uses a fresh contiguous buffer built from cloned local factor views.
- Existing warmup, local-SVD refresh, broadcast refresh, and future sequencing regressions still pass.

## Concerns

- The frozen dev environment does not provide `ruff`, so lint verification is limited to `py_compile`, `git diff --check`, and pytest.
