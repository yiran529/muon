"""Single-rank reconstruction tests for the ARC DDP communication hook."""

import math

import pytest
import torch

from dion.arc_topk_ddp_hook import (
    ArcTopKDDPParameterSpec,
    ArcTopKDDPState,
    arc_topk_ddp_hook,
)
from dion.arc_topk_sync import ArcTopKSyncConfig
from dion.arc_topk import derive_arc_seed


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


def _state(parameters_and_roles, *, ratio=0.5, start_compress_step=2):
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
            start_compress_step=start_compress_step,
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


def _manual_sparse_step(gradient, h_local, g_local, *, step, stable_id):
    h_local = gradient if step == 1 else h_local.lerp(gradient, 0.25)
    if step == 1:
        return h_local, h_local, torch.arange(gradient.shape[0])
    delta = h_local - g_local
    generator = torch.Generator().manual_seed(
        derive_arc_seed(base_seed=17, step=step, stable_task_id=stable_id)
    )
    projection = torch.randn(1, gradient.shape[1], 2, generator=generator)
    sketch = torch.bmm(delta.unsqueeze(0), projection).squeeze(0) / math.sqrt(2.0)
    k = math.ceil(0.5 * gradient.shape[0])
    support = sketch.square().sum(-1).topk(k, sorted=True).indices
    compressed = torch.zeros_like(delta)
    compressed[support] = delta[support]
    return h_local, g_local + compressed, support


def test_sparse_hook_preserves_per_parameter_state_across_mixed_shape_bucket_steps():
    first = torch.nn.Parameter(torch.zeros(3, 2))
    dense = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2, 3))
    state = _state(
        [
            (first, "first", "arc_matrix"),
            (dense, "dense", "dense_aux"),
            (second, "second", "arc_matrix"),
        ],
        start_compress_step=0,
    )
    expected = {
        first: (torch.zeros_like(first), torch.zeros_like(first)),
        second: (torch.zeros_like(second), torch.zeros_like(second)),
    }

    for step in range(1, 4):
        gradients = [
            torch.arange(6.0).view_as(first) + step,
            torch.tensor([10.0 + step, 20.0 + step]),
            torch.arange(6.0).view_as(second) + 2 * step,
        ]
        bucket = FakeGradBucket([first, dense, second], gradients)
        state.begin_step()
        result = arc_topk_ddp_hook(state, bucket).wait()
        state.finish_step()

        for parameter, gradient, stable_id in (
            (first, gradients[0], 0),
            (second, gradients[2], 2),
        ):
            h_local, g_local = expected[parameter]
            h_local, g_local, support = _manual_sparse_step(
                gradient,
                h_local,
                g_local,
                step=step,
                stable_id=stable_id,
            )
            expected[parameter] = (h_local, g_local)
            parameter_state = state.parameter_state(parameter)
            torch.testing.assert_close(parameter_state.h_local, h_local)
            torch.testing.assert_close(parameter_state.g_local, g_local)
            torch.testing.assert_close(parameter_state.g_global, g_local)
            torch.testing.assert_close(parameter_state.last_support, support)
            offset = 0 if parameter is first else first.numel() + dense.numel()
            torch.testing.assert_close(
                result[offset : offset + parameter.numel()].view_as(parameter),
                g_local,
            )
        torch.testing.assert_close(
            result[first.numel() : first.numel() + dense.numel()],
            gradients[1],
        )
        state.commit_step()


def test_sparse_hook_batches_compatible_shapes_without_changing_stable_seed_state():
    first = torch.nn.Parameter(torch.zeros(3, 2))
    different = torch.nn.Parameter(torch.zeros(2, 3))
    second = torch.nn.Parameter(torch.zeros(3, 2))
    parameters = [first, different, second]
    state = _state(
        [
            (first, "first", "arc_matrix"),
            (different, "different", "arc_matrix"),
            (second, "second", "arc_matrix"),
        ],
        start_compress_step=0,
    )
    first_step = [
        torch.arange(6.0).view_as(first) + 1,
        torch.arange(6.0).view_as(different) + 21,
        torch.arange(6.0).view_as(second) + 11,
    ]
    state.begin_step()
    arc_topk_ddp_hook(state, FakeGradBucket(parameters, first_step)).wait()
    state.finish_step()
    state.commit_step()

    second_step = [
        torch.tensor([[8.0, 1.0], [3.0, 7.0], [2.0, 9.0]]),
        torch.tensor([[9.0, 2.0, 8.0], [1.0, 7.0, 3.0]]),
        torch.tensor([[4.0, 12.0], [10.0, 5.0], [13.0, 6.0]]),
    ]
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        state.begin_step()
        result = arc_topk_ddp_hook(
            state,
            FakeGradBucket(parameters, second_step),
        ).wait()
        state.finish_step()

    operator_counts = {event.key: event.count for event in profiler.key_averages()}
    assert operator_counts["aten::bmm"] == 2
    assert operator_counts["aten::topk"] == 2
    assert operator_counts["aten::gather"] == 2

    offset = 0
    for stable_id, (parameter, previous, gradient) in enumerate(
        zip(parameters, first_step, second_step)
    ):
        expected_h, expected_g, expected_support = _manual_sparse_step(
            gradient,
            previous,
            previous,
            step=2,
            stable_id=stable_id,
        )
        parameter_state = state.parameter_state(parameter)
        torch.testing.assert_close(parameter_state.h_local, expected_h)
        torch.testing.assert_close(parameter_state.g_local, expected_g)
        torch.testing.assert_close(parameter_state.g_global, expected_g)
        torch.testing.assert_close(parameter_state.last_support, expected_support)
        torch.testing.assert_close(
            result[offset : offset + parameter.numel()].view_as(parameter),
            expected_g,
        )
        offset += parameter.numel()
