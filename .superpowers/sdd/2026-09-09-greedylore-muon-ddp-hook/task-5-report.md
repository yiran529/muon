# Task 5 Report: GreedyLore Dense Warmup and Refresh Synchronization

## Implementation

Implemented the registered `greedy_lore_ddp_hook` paths for Task 5 in `dion/greedy_lore_ddp_hook.py`.

- Dense warmup uses one packed `greedylore_hook/dense` All-Reduce per bucket and averages the returned bucket buffer.
- Refresh steps pack `gradient + old_error` for matrix parameters into the bucket buffer before the dense All-Reduce, leave dense auxiliary slices uncorrected, return the exact averaged corrected bucket, reset matrix errors to zero, and store full FP32 bases plus first-rank support.
- `basis_sync="local_svd"` runs canonicalized local SVD on each rank without basis communication.
- `basis_sync="broadcast"` runs SVD only on `state.group_ranks[0]` and broadcasts each full basis in stable parameter-name order with `greedylore_hook/basis_broadcast` observer events.
- Unsupported compressed non-refresh steps now fail loudly pending Task 6, instead of silently densifying.

## Files

- Modified: `dion/greedy_lore_ddp_hook.py`
- Added: `tests/test_greedy_lore_ddp_hook.py`
- Added: `tests/test_greedy_lore_ddp_hook_distributed.py`

## RED Evidence

Initial RED for the requested Task 5 suite:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py -v
collected 0 items / 2 errors
ERROR tests/test_greedy_lore_ddp_hook.py
ImportError: cannot import name 'greedy_lore_ddp_hook' from 'dion.greedy_lore_ddp_hook'
ERROR tests/test_greedy_lore_ddp_hook_distributed.py
ImportError: cannot import name 'greedy_lore_ddp_hook' from 'dion.greedy_lore_ddp_hook'
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
======================== 14 warnings, 2 errors in 3.75s ========================
```

RED for the self-review guard against silently densifying Task 6's unsupported path:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook.py::test_non_refresh_compressed_step_is_not_silently_densified -v
collected 1 item
tests/test_greedy_lore_ddp_hook.py::test_non_refresh_compressed_step_is_not_silently_densified FAILED [100%]
E       Failed: DID NOT RAISE NotImplementedError
======================== 1 failed, 14 warnings in 3.49s ========================
```

## GREEN Evidence

Requested Task 5 suite:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py -v
collected 9 items
tests/test_greedy_lore_ddp_hook.py::test_warmup_reconstructs_mixed_shape_and_role_offsets_exactly PASSED [ 11%]
tests/test_greedy_lore_ddp_hook.py::test_refresh_reduces_corrected_matrices_and_resets_error_in_exact_offsets PASSED [ 22%]
tests/test_greedy_lore_ddp_hook.py::test_dense_only_warmup_bucket_retains_native_dtype PASSED [ 33%]
tests/test_greedy_lore_ddp_hook.py::test_non_refresh_compressed_step_is_not_silently_densified PASSED [ 44%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_dense_warmup_averages_real_ddp_bucket_once PASSED [ 55%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_local_svd_refresh_averages_corrected_gradient_and_replicates_basis[zero] PASSED [ 66%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_local_svd_refresh_averages_corrected_gradient_and_replicates_basis[repeated] PASSED [ 77%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_local_svd_refresh_averages_corrected_gradient_and_replicates_basis[near_repeated] PASSED [ 88%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_broadcast_refresh_runs_rank_zero_svd_and_broadcasts_stable_order PASSED [100%]
======================= 9 passed, 14 warnings in 29.60s ========================
```

Task 3/4 regression suite:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_state.py tests/test_greedy_lore_ddp_future.py tests/test_greedy_lore_ddp_future_distributed.py -v
collected 25 items
tests/test_greedy_lore_ddp_state.py::* PASSED
tests/test_greedy_lore_ddp_future.py::* PASSED
tests/test_greedy_lore_ddp_future_distributed.py::test_two_rank_rebuilt_ddp_buckets_share_one_global_launch_order PASSED [100%]
======================= 25 passed, 14 warnings in 8.41s ========================
```

Additional checks:

```text
$ PYTHONPATH=. uv run --frozen --extra dev black --target-version py310 --check dion/greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py
All done! ✨ 🍰 ✨
3 files would be left unchanged.

$ rg -n "\.wait\(|cuda\.synchronize|\.item\(|\.cpu\(|to\(['\"]cpu|Work\.wait" dion/greedy_lore_ddp_hook.py
<no matches>

$ PYTHONPATH=. uv run --frozen --extra dev python -m compileall -q dion/greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py
<exit 0, no output>
```

`ruff` was not available in the frozen dev environment:

```text
$ uv run --frozen --extra dev ruff check dion/greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py
error: Failed to spawn: `ruff`
  Caused by: No such file or directory (os error 2)
```

## Self-Review

- Verified no changes to ARC or Muon files.
- Verified matrix FP32 rejection remains at state construction from Task 3.
- Verified hook callback path contains no `Work.wait`, `torch.cuda.synchronize`, `.item()`, `.cpu()`, or explicit CPU transfers.
- Verified local-SVD refresh uses `gradient + old_error`, not raw gradient, and the distributed test asserts the raw-gradient average differs.
- Verified broadcast refresh emits only dense plus basis-broadcast collectives and orders basis broadcasts by stable parameter name.

## Concerns

- Task 6's compressed non-refresh score/factor path is intentionally not implemented yet; the hook raises `NotImplementedError` for that phase.
- Validation here is CPU/Gloo plus existing NCCL Future sequencer coverage; this task did not add a CUDA end-to-end GreedyLore hook test.

## Fix Round 1: Broadcast Refresh Stream Export

### Review Finding

Fixed the Critical review finding that `basis_sync="broadcast"` refresh returned `context.completion_future` directly and completed it from `_on_bucket_execution_stream()` after basis broadcasts. That bypassed the Task 4 result/export bridge used by dense and local-SVD paths.

### Changed Files

- Modified: `dion/greedy_lore_ddp_hook.py`
- Modified: `tests/test_greedy_lore_ddp_hook_nccl.py`
- Modified: `.superpowers/sdd/2026-09-09-greedylore-muon-ddp-hook/task-5-report.md`

### Implementation

Broadcast refresh now returns an intermediate launched Future. After dense All-Reduce, it still launches the full-basis broadcasts in stable parameter-name order, but `torch.futures.collect_all(futures)` now finalizes through `_on_bucket_execution_stream_result()` and then `bridge_future()`. Because the launched Future is no longer `context.completion_future`, `enqueue_bucket_chain()` applies its normal `export_completion()` bridge before completing the hook Future.

### RED Evidence

Focused NCCL regression before the fix:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook_nccl.py::test_broadcast_refresh_future_exports_final_compressor_stream_write -v
collected 1 item
tests/test_greedy_lore_ddp_hook_nccl.py::test_broadcast_refresh_future_exports_final_compressor_stream_write FAILED [100%]
E       torch.multiprocessing.spawn.ProcessRaisedException:
E       -- Process 1 terminated with the following error:
E         File ".../tests/test_greedy_lore_ddp_hook_nccl.py", line 184, in _broadcast_refresh_worker
E           assert export_calls == 1
E       AssertionError
======================== 1 failed, 14 warnings in 9.67s ========================
```

An earlier visibility-only version of the same NCCL probe passed on this PyTorch build before the fix, so the regression was tightened to assert that broadcast refresh completion uses the Task 4 result/export helper while still exercising real NCCL broadcast collectives and a delayed compressor-stream write.

### GREEN Evidence

Focused NCCL regression after the fix:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook_nccl.py::test_broadcast_refresh_future_exports_final_compressor_stream_write -v
collected 1 item
tests/test_greedy_lore_ddp_hook_nccl.py::test_broadcast_refresh_future_exports_final_compressor_stream_write PASSED [100%]
======================== 1 passed, 14 warnings in 9.42s ========================
```

Requested Task 5 coverage:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_distributed.py -v
collected 9 items
tests/test_greedy_lore_ddp_hook.py::test_warmup_reconstructs_mixed_shape_and_role_offsets_exactly PASSED [ 11%]
tests/test_greedy_lore_ddp_hook.py::test_refresh_reduces_corrected_matrices_and_resets_error_in_exact_offsets PASSED [ 22%]
tests/test_greedy_lore_ddp_hook.py::test_dense_only_warmup_bucket_retains_native_dtype PASSED [ 33%]
tests/test_greedy_lore_ddp_hook.py::test_non_refresh_compressed_step_is_not_silently_densified PASSED [ 44%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_dense_warmup_averages_real_ddp_bucket_once PASSED [ 55%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_local_svd_refresh_averages_corrected_gradient_and_replicates_basis[zero] PASSED [ 66%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_local_svd_refresh_averages_corrected_gradient_and_replicates_basis[repeated] PASSED [ 77%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_local_svd_refresh_averages_corrected_gradient_and_replicates_basis[near_repeated] PASSED [ 88%]
tests/test_greedy_lore_ddp_hook_distributed.py::test_two_rank_broadcast_refresh_runs_rank_zero_svd_and_broadcasts_stable_order PASSED [100%]
======================= 9 passed, 14 warnings in 29.18s ========================
```

Task 4 Future coverage:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_future.py tests/test_greedy_lore_ddp_future_distributed.py -v
collected 6 items
tests/test_greedy_lore_ddp_future.py::test_bridge_future_resolves_to_a_tensor_not_a_nested_future PASSED [ 16%]
tests/test_greedy_lore_ddp_future.py::test_bridge_future_propagates_transform_and_source_exceptions PASSED [ 33%]
tests/test_greedy_lore_ddp_future.py::test_bucket_context_remains_live_until_async_launch_completion PASSED [ 50%]
tests/test_greedy_lore_ddp_future.py::test_new_tail_is_installed_before_completed_source_callback_runs_inline PASSED [ 66%]
tests/test_greedy_lore_ddp_future.py::test_failed_bucket_poison_propagates_without_launching_later_bucket PASSED [ 83%]
tests/test_greedy_lore_ddp_future_distributed.py::test_two_rank_rebuilt_ddp_buckets_share_one_global_launch_order PASSED [100%]
======================== 6 passed, 14 warnings in 8.66s ========================
```

Full GreedyLore NCCL hook coverage:

```text
$ PYTHONPATH=. uv run --frozen --extra dev pytest tests/test_greedy_lore_ddp_hook_nccl.py -v
collected 2 items
tests/test_greedy_lore_ddp_hook_nccl.py::test_returned_future_exports_final_compressor_stream_write PASSED [ 50%]
tests/test_greedy_lore_ddp_hook_nccl.py::test_broadcast_refresh_future_exports_final_compressor_stream_write PASSED [100%]
======================= 2 passed, 14 warnings in 15.74s ========================
```

Additional checks:

```text
$ PYTHONPATH=. uv run --frozen --extra dev black --target-version py310 --check dion/greedy_lore_ddp_hook.py tests/test_greedy_lore_ddp_hook_nccl.py
All done! ✨ 🍰 ✨
2 files would be left unchanged.

$ rg -n "\.wait\(|cuda\.synchronize|\.item\(|\.cpu\(|to\(['\"]cpu|Work\.wait" dion/greedy_lore_ddp_hook.py
<no matches>
```

### Self-Review

- Confirmed collective order is unchanged: dense All-Reduce first, then stable-order basis broadcasts.
- Confirmed broadcast refresh no longer returns `context.completion_future` as its launched Future, so `enqueue_bucket_chain()` applies the same export bridge as Task 4.
- Confirmed no changes to ARC or Muon files.

### Concerns

- The focused NCCL regression verifies the stream-export discipline and exercises real NCCL collectives; the raw CUDA race was not reproducible on this PyTorch build without the explicit export-path assertion.
