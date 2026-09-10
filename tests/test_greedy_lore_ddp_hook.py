"""Single-rank reconstruction tests for the GreedyLore DDP communication hook."""

import torch
import pytest

from dion.greedy_lore import GreedyLoreConfig, canonicalize_svd_basis, orient_matrix
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


def test_non_refresh_compressed_step_is_not_silently_densified():
    matrix = torch.nn.Parameter(torch.zeros(2, 3))
    bucket = FakeGradBucket([matrix], [torch.ones_like(matrix)])
    state = _state([(matrix, "matrix", "matrix")], start_compress_step=0)

    state.begin_step()
    greedy_lore_ddp_hook(state, bucket).wait()
    state.finish_step()
    state.commit_step()

    state.begin_step()
    future = greedy_lore_ddp_hook(
        state,
        FakeGradBucket([matrix], [torch.ones_like(matrix)]),
    )

    with pytest.raises(NotImplementedError, match="compressed non-refresh"):
        future.wait()
    assert future.done()
