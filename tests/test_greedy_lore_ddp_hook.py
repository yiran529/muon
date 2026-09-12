"""Single-rank reconstruction tests for the GreedyLore DDP communication hook."""

import inspect

import torch
import pytest

from dion.collective_observer import CollectiveObserver, set_active_observer
from dion.greedy_lore import (
    GreedyLoreConfig,
    approximate_signed_lambda,
    canonicalize_svd_basis,
    compress_local,
    corrected_gradient,
    derive_greedy_lore_seed,
    make_random_vectors,
    orient_matrix,
    reconstruct_global,
    select_projector,
)
import dion.greedy_lore_ddp_hook as hook_module
from dion.greedy_lore_ddp_hook import (
    GreedyLoreDDPParameterSpec,
    GreedyLoreDDPState,
    greedy_lore_ddp_hook,
)


class FakeGradBucket:
    def __init__(self, parameters, gradient_values):
        self._parameters = tuple(parameters)
        self._buffer = torch.cat([value.reshape(-1) for value in gradient_values])
        self._gradients = []
        offset = 0
        for parameter in parameters:
            view = self._buffer[offset : offset + parameter.numel()].view_as(parameter)
            self._gradients.append(view)
            offset += parameter.numel()

    def parameters(self):
        return list(self._parameters)

    def gradients(self):
        return list(self._gradients)

    def buffer(self):
        return self._buffer


def _state(parameters_and_roles, *, start_compress_step=1, update_interval=2):
    parameters = [item[0] for item in parameters_and_roles]
    return GreedyLoreDDPState(
        process_group=None,
        fingerprint="e" * 64,
        parameter_specs=[
            GreedyLoreDDPParameterSpec(parameter, name, index, role)
            for index, (parameter, name, role) in enumerate(parameters_and_roles)
        ],
        optimizer_parameters=parameters,
        config=GreedyLoreConfig(
            rank=1,
            start_compress_step=start_compress_step,
            update_interval=update_interval,
        ),
    )


def test_warmup_reconstructs_mixed_shape_and_role_offsets_exactly():
    wide = torch.nn.Parameter(torch.zeros(2, 3))
    dense = torch.nn.Parameter(torch.zeros(2))
    tall = torch.nn.Parameter(torch.zeros(3, 2))
    square = torch.nn.Parameter(torch.zeros(2, 2))
    gradients = [
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([7.0, 8.0]),
        torch.tensor([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]),
        torch.tensor([[15.0, 16.0], [17.0, 18.0]]),
    ]
    bucket = FakeGradBucket([wide, dense, tall, square], gradients)
    state = _state(
        [
            (wide, "wide", "matrix"),
            (dense, "dense", "dense_aux"),
            (tall, "tall", "matrix"),
            (square, "square", "matrix"),
        ]
    )

    state.begin_step()
    result = greedy_lore_ddp_hook(state, bucket).wait()
    state.finish_step()

    expected = torch.cat([gradient.flatten() for gradient in gradients])
    assert result is bucket.buffer()
    assert result.shape == expected.shape
    assert result.dtype == torch.float32
    assert result.device == torch.device("cpu")
    torch.testing.assert_close(result, expected)
    for parameter in (wide, tall, square):
        parameter_state = state.parameter_state(parameter)
        assert parameter_state.error.dtype == torch.float32
        assert parameter_state.basis.dtype == torch.float32
        torch.testing.assert_close(
            parameter_state.error, torch.zeros_like(parameter_state.error)
        )


def test_refresh_reduces_corrected_matrices_and_resets_error_in_exact_offsets():
    wide = torch.nn.Parameter(torch.zeros(2, 3))
    dense = torch.nn.Parameter(torch.zeros(2))
    tall = torch.nn.Parameter(torch.zeros(3, 2))
    wide_gradient = torch.tensor([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dense_gradient = torch.tensor([3.0, 5.0])
    tall_gradient = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    bucket = FakeGradBucket(
        [wide, dense, tall], [wide_gradient, dense_gradient, tall_gradient]
    )
    state = _state(
        [
            (wide, "wide", "matrix"),
            (dense, "dense", "dense_aux"),
            (tall, "tall", "matrix"),
        ],
        start_compress_step=0,
    )
    wide_state = state.parameter_state(wide)
    tall_state = state.parameter_state(tall)
    wide_state.error.copy_(torch.tensor([[-4.0, 0.0, 0.0], [0.0, 3.0, 0.0]]))
    tall_state.error.copy_(torch.tensor([[1.0, -1.0, 0.5], [2.0, -2.0, 1.5]]))

    expected_wide = wide_gradient + wide_state.error
    expected_tall_oriented = tall_gradient.mT + tall_state.error
    expected_tall = expected_tall_oriented.mT
    expected = torch.cat(
        [expected_wide.flatten(), dense_gradient, expected_tall.flatten()]
    )

    state.begin_step()
    result = greedy_lore_ddp_hook(state, bucket).wait()
    state.finish_step()

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(wide_state.error, torch.zeros_like(wide_state.error))
    torch.testing.assert_close(tall_state.error, torch.zeros_like(tall_state.error))
    torch.testing.assert_close(wide_state.last_support, torch.tensor([0]))
    torch.testing.assert_close(tall_state.last_support, torch.tensor([0]))
    for parameter, expected_matrix in (
        (wide, expected_wide),
        (tall, expected_tall),
    ):
        parameter_state = state.parameter_state(parameter)
        expected_basis, _, _ = torch.linalg.svd(
            orient_matrix(expected_matrix, parameter_state.orientation),
            full_matrices=False,
        )
        expected_basis = canonicalize_svd_basis(expected_basis)
        torch.testing.assert_close(parameter_state.basis, expected_basis)


def test_dense_only_warmup_bucket_retains_native_dtype():
    dense = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    gradient = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    bucket = FakeGradBucket([dense], [gradient])
    state = _state([(dense, "dense", "dense_aux")])

    state.begin_step()
    result = greedy_lore_ddp_hook(state, bucket).wait()
    state.finish_step()

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, gradient)


def test_compressed_step_reduces_signed_scores_before_square_and_updates_local_error_before_factor_average():
    matrix = torch.nn.Parameter(torch.zeros(2, 3))
    dense = torch.nn.Parameter(torch.zeros(2))
    matrix_gradient = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
    dense_gradient = torch.tensor([5.0, 7.0])
    bucket = FakeGradBucket([matrix, dense], [matrix_gradient, dense_gradient])
    state = _state(
        [(matrix, "matrix", "matrix"), (dense, "dense", "dense_aux")],
        start_compress_step=0,
        update_interval=100,
    )
    state.committed_step = 1
    state.world_size = 2
    parameter_state = state.parameter_state(matrix)
    parameter_state.basis.copy_(torch.eye(2))

    corrected = corrected_gradient(
        matrix_gradient,
        parameter_state.error,
        parameter_state.orientation,
    )
    local_lambda = approximate_signed_lambda(
        corrected,
        parameter_state.basis,
        make_random_vectors(
            rows=2,
            columns=3,
            seed=derive_greedy_lore_seed(
                base_seed=state.config.seed,
                phase=1,
                stable_parameter_id=0,
            ),
            device=torch.device("cpu"),
        ),
    )
    averaged_lambda = torch.tensor([0.0, 4.0])
    projector, support = select_projector(parameter_state.basis, averaged_lambda, 1)
    local_factor, expected_error = compress_local(corrected, projector)
    other_factor = torch.tensor([[4.0, 6.0, 8.0]])
    averaged_factor = (local_factor + other_factor) / 2
    expected_reconstructed = reconstruct_global(projector, averaged_factor)
    expected_dense = torch.tensor([11.0, 13.0])

    original_all_reduce = hook_module._all_reduce_future
    calls = []

    def fake_all_reduce(current_state, tensor, category):
        calls.append((category, tensor.clone()))
        future = torch.futures.Future()
        if category == "greedylore_hook/score_plus_aux_allreduce":
            torch.testing.assert_close(tensor[:2], local_lambda)
            torch.testing.assert_close(tensor[2:], dense_gradient)
            score_and_dense_sum = torch.cat(
                [averaged_lambda * 2, expected_dense * 2]
            )
            tensor.copy_(score_and_dense_sum)
        elif category == "greedylore_hook/factor_allreduce":
            torch.testing.assert_close(parameter_state.error, expected_error)
            torch.testing.assert_close(tensor, local_factor.flatten())
            tensor.copy_((local_factor + other_factor).flatten())
        else:
            raise AssertionError(f"unexpected collective category {category!r}")
        future.set_result(tensor)
        return future

    hook_module._all_reduce_future = fake_all_reduce
    try:
        state.begin_step()
        result = greedy_lore_ddp_hook(state, bucket).wait()
        state.finish_step()
    finally:
        hook_module._all_reduce_future = original_all_reduce

    assert result is bucket.buffer()
    assert [category for category, _ in calls] == [
        "greedylore_hook/score_plus_aux_allreduce",
        "greedylore_hook/factor_allreduce",
    ]
    assert torch.equal(support, torch.tensor([1]))
    assert torch.equal(parameter_state.last_support, torch.tensor([1]))
    torch.testing.assert_close(parameter_state.error, expected_error)
    torch.testing.assert_close(bucket.gradients()[0], expected_reconstructed)
    torch.testing.assert_close(bucket.gradients()[1], expected_dense)


def test_compressed_step_batches_compatible_oriented_shapes():
    first = torch.nn.Parameter(torch.zeros(3, 2))
    different = torch.nn.Parameter(torch.zeros(2, 4))
    second = torch.nn.Parameter(torch.zeros(2, 3))
    parameters = [first, different, second]
    gradients = [
        torch.tensor([[8.0, 1.0], [3.0, 7.0], [2.0, 9.0]]),
        torch.tensor([[9.0, 2.0, 8.0, 1.0], [1.0, 7.0, 3.0, 6.0]]),
        torch.tensor([[4.0, 12.0, 10.0], [5.0, 13.0, 6.0]]),
    ]
    bucket = FakeGradBucket(parameters, gradients)
    state = _state(
        [
            (first, "first", "matrix"),
            (different, "different", "matrix"),
            (second, "second", "matrix"),
        ],
        start_compress_step=0,
        update_interval=100,
    )
    state.committed_step = 1

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        state.begin_step()
        result = greedy_lore_ddp_hook(state, bucket).wait()
        state.finish_step()

    operator_counts = {event.key: event.count for event in profiler.key_averages()}
    assert result is bucket.buffer()
    assert torch.isfinite(result).all()
    # The (3, 2) and (2, 3) matrices share the canonical (2, 3) orientation.
    # Each canonical-shape group performs batched score, factor, error, and
    # final reconstruction matrix multiplies.
    assert operator_counts["aten::bmm"] == 8
    assert operator_counts["aten::sort"] == 2
    assert operator_counts["aten::gather"] == 2


def test_batched_factors_are_direct_views_into_the_packed_collective_buffer():
    first = torch.nn.Parameter(torch.zeros(3, 2))
    different = torch.nn.Parameter(torch.zeros(2, 4))
    second = torch.nn.Parameter(torch.zeros(3, 2))
    parameters = [first, different, second]
    bucket = FakeGradBucket(
        parameters,
        [
            torch.arange(6.0).view_as(first) + 1,
            torch.arange(8.0).view_as(different) + 11,
            torch.arange(6.0).view_as(second) + 21,
        ],
    )
    state = _state(
        [
            (first, "first", "matrix"),
            (different, "different", "matrix"),
            (second, "second", "matrix"),
        ],
        start_compress_step=0,
        update_interval=100,
    )
    state.committed_step = 1
    state.begin_step()
    context = state.note_bucket(bucket)
    score_buffer, matrix_batches, _dense_ranges = (
        hook_module._prepare_score_plus_aux_buffer(state, context)
    )

    factor_buffer = hook_module._prepare_factor_buffer(
        state, score_buffer, matrix_batches
    )

    factor_storage = factor_buffer.untyped_storage().data_ptr()
    works = [work for batch in matrix_batches for work in batch.works]
    assert [work.factor_offset for work in works] == [0, 3, 6]
    assert all(
        work.local_factor is not None
        and work.local_factor.untyped_storage().data_ptr() == factor_storage
        and work.factor_offset == work.local_factor.storage_offset()
        and work.local_factor.is_contiguous()
        for work in works
    )


def test_compressed_dense_only_bucket_metadata_has_no_matrix_payload():
    dense = torch.nn.Parameter(torch.zeros(3))
    bucket = FakeGradBucket([dense], [torch.ones_like(dense)])
    state = _state([(dense, "dense", "dense_aux")], start_compress_step=0)
    state.committed_step = 1

    state.begin_step()
    context = state.note_bucket(bucket)
    metadata = hook_module._bucket_profile_metadata(state, context)

    assert metadata["matrix_bytes"] == 0
    assert metadata["dense_aux_bytes"] == 12
    assert metadata["score_bytes"] == 0
    assert metadata["factor_bytes"] == 0
    assert metadata["basis_bytes"] == 0


def test_compressed_mixed_bucket_metadata_includes_aux_and_signed_scores():
    matrix = torch.nn.Parameter(torch.zeros(2, 3))
    dense = torch.nn.Parameter(torch.zeros(2))
    bucket = FakeGradBucket(
        [matrix, dense], [torch.ones_like(matrix), torch.ones_like(dense)]
    )
    state = _state(
        [(matrix, "matrix", "matrix"), (dense, "dense", "dense_aux")],
        start_compress_step=0,
    )
    state.committed_step = 1

    state.begin_step()
    context = state.note_bucket(bucket)
    metadata = hook_module._bucket_profile_metadata(state, context)

    assert metadata["matrix_bytes"] == 24
    assert metadata["dense_aux_bytes"] == 8
    # Two dense auxiliary values plus one signed score per matrix row.
    assert metadata["score_bytes"] == 16
    assert metadata["factor_bytes"] == 12


def test_dense_only_compressed_bucket_launches_one_dense_reduction_and_stops(monkeypatch):
    dense = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    gradient = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    bucket = FakeGradBucket([dense], [gradient])
    state = _state([(dense, "dense", "dense_aux")], start_compress_step=0)
    state.committed_step = 1
    observer = CollectiveObserver()

    def fake_all_reduce(current_state, tensor, category):
        with hook_module._collective_profile_range(category, "all_reduce", tensor):
            pass
        future = torch.futures.Future()
        future.set_result(tensor)
        return future

    monkeypatch.setattr(hook_module, "_all_reduce_future", fake_all_reduce)

    set_active_observer(observer)
    try:
        state.begin_step()
        result = greedy_lore_ddp_hook(state, bucket).wait()
        state.finish_step()
    finally:
        set_active_observer(None)

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, gradient)
    assert observer.signature() == [
        ("greedylore_hook/dense", "all_reduce", 3, "bfloat16", 6)
    ]


def test_registered_hook_callback_path_avoids_forbidden_synchronization():
    callback_path = (
        hook_module.greedy_lore_ddp_hook,
        hook_module.GreedyLoreDDPState.note_bucket,
        hook_module.GreedyLoreDDPState.execution_stream,
        hook_module._future_devices,
        hook_module._profile_range,
        hook_module._tensor_bytes,
        hook_module._collective_profile_range,
        hook_module._bucket_profile_metadata,
        hook_module.bridge_future,
        hook_module._future_tensor,
        hook_module.enqueue_bucket_chain,
        hook_module._all_reduce_future,
        hook_module._broadcast_future,
        hook_module._on_bucket_execution_stream,
        hook_module._on_bucket_execution_stream_result,
        hook_module._matrix_entries,
        hook_module._stable_matrix_entries,
        hook_module._prepare_refresh_buffer,
        hook_module._divide_completed_buffer,
        hook_module._refresh_local_svd,
        hook_module._launch_dense_bucket,
        hook_module._launch_refresh_bucket,
        hook_module._parameter_spec_for_dense,
        hook_module._ordered_compressed_entries,
        hook_module._prepare_score_plus_aux_buffer,
        hook_module._select_projector_batch_profiled,
        hook_module._compress_local_batch_profiled,
        hook_module._prepare_factor_buffer,
        hook_module._copy_averaged_dense_aux,
        hook_module._reconstruct_compressed_matrices,
        hook_module._mark_future_complete,
        hook_module._launch_compressed_bucket,
    )
    source = "\n".join(inspect.getsource(function) for function in callback_path)

    assert "Work.wait" not in source
    assert "torch.cuda.synchronize" not in source
    assert ".item(" not in source
    assert ".cpu(" not in source
    assert ".to(\"cpu\"" not in source
    assert ".to('cpu'" not in source
