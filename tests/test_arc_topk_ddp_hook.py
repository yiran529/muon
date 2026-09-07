"""Single-rank reconstruction tests for the ARC DDP communication hook."""

import pytest
import torch

from dion.arc_topk_ddp_hook import (
    ArcTopKDDPParameterSpec,
    ArcTopKDDPState,
    arc_topk_ddp_hook,
)
from dion.arc_topk_sync import ArcTopKSyncConfig


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


def _state(parameters_and_roles, *, ratio=0.5):
    parameters = [item[0] for item in parameters_and_roles]
    return ArcTopKDDPState(
        process_group=None,
        fingerprint="e" * 64,
        parameter_specs=[
            ArcTopKDDPParameterSpec(parameter, name, index, role)
            for index, (parameter, name, role) in enumerate(parameters_and_roles)
        ],
        optimizer_parameters=parameters,
        config=ArcTopKSyncConfig(
            ratio=ratio,
            projection_rank=2,
            eta=0.25,
            seed=17,
            start_compress_step=2,
        ),
    )


def test_full_support_reconstructs_mixed_shape_and_role_offsets_exactly():
    first = torch.nn.Parameter(torch.zeros(2, 2))
    dense = torch.nn.Parameter(torch.zeros(3))
    second = torch.nn.Parameter(torch.zeros(3, 1))
    gradients = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([5.0, 6.0, 7.0]),
        torch.tensor([[8.0], [9.0], [10.0]]),
    ]
    bucket = FakeGradBucket([first, dense, second], gradients)
    state = _state(
        [
            (first, "first", "arc_matrix"),
            (dense, "dense", "dense_aux"),
            (second, "second", "arc_matrix"),
        ]
    )

    state.begin_step()
    result = arc_topk_ddp_hook(state, bucket).wait()
    state.finish_step()

    expected = torch.cat([gradient.flatten() for gradient in gradients])
    assert result is bucket.buffer()
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert result.device == expected.device
    torch.testing.assert_close(result, expected)
    for parameter, expected_gradient in ((first, gradients[0]), (second, gradients[2])):
        parameter_state = state.parameter_state(parameter)
        torch.testing.assert_close(parameter_state.h_local, expected_gradient)
        torch.testing.assert_close(parameter_state.g_local, expected_gradient)
        torch.testing.assert_close(parameter_state.g_global, expected_gradient)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_support_preserves_each_realistic_bucket_dtype(dtype):
    parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    bucket = FakeGradBucket([parameter], [gradient])
    state = _state([(parameter, "matrix", "arc_matrix")], ratio=1.0)

    state.begin_step()
    result = arc_topk_ddp_hook(state, bucket).wait()
    state.finish_step()

    assert result.dtype == dtype
    torch.testing.assert_close(result.view_as(parameter), gradient)
