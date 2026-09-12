"""Single-rank behavior and lifecycle tests for the Sparse-K DDP hook."""

import copy

import pytest
import torch

from dion.sparse_k import SparseKConfig
from dion.sparse_k_ddp_hook import (
    SparseKDDPParameterSpec,
    SparseKDDPState,
    sparse_k_ddp_hook,
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


def _state(
    parameters_and_roles, *, method="topk", ratio=0.5, error_feedback="ef14", start=0
):
    parameters = [item[0] for item in parameters_and_roles]
    return SparseKDDPState(
        process_group=None,
        fingerprint="c" * 64,
        parameter_specs=[
            SparseKDDPParameterSpec(parameter, name, index, role)
            for index, (parameter, name, role) in enumerate(parameters_and_roles)
        ],
        optimizer_parameters=parameters,
        config=SparseKConfig(
            method=method,
            ratio=ratio,
            seed=17,
            start_compress_step=start,
            error_feedback=error_feedback,
        ),
    )


def _run(state, bucket):
    state.begin_step()
    result = sparse_k_ddp_hook(state, bucket).wait()
    state.finish_step()
    state.commit_step()
    return result


def test_topk_ef14_uses_local_support_and_keeps_the_local_residual():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    auxiliary = torch.nn.Parameter(torch.zeros(2))
    bucket = FakeGradBucket(
        [matrix, auxiliary],
        [torch.tensor([[1.0, -4.0], [3.0, 2.0]]), torch.tensor([5.0, 7.0])],
    )
    state = _state(
        [(matrix, "matrix", "sparse_matrix"), (auxiliary, "aux", "dense_aux")]
    )
    state.committed_step = 1

    result = _run(state, bucket)

    assert result is bucket.buffer()
    torch.testing.assert_close(
        bucket.gradients()[0], torch.tensor([[0.0, -4.0], [3.0, 0.0]])
    )
    torch.testing.assert_close(bucket.gradients()[1], torch.tensor([5.0, 7.0]))
    parameter_state = state.parameter_state(matrix)
    torch.testing.assert_close(
        parameter_state.residual, torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    )
    assert set(parameter_state.last_support.tolist()) == {1, 2}


def test_noef_does_not_allocate_or_checkpoint_a_residual():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    state = _state([(matrix, "matrix", "sparse_matrix")], error_feedback="noef")

    assert state.parameter_state(matrix).residual is None
    assert state.state_dict()["rank_0"] == {}


def test_dense_only_bucket_is_averaged_during_a_sparse_step():
    auxiliary = torch.nn.Parameter(torch.zeros(3))
    state = _state([(auxiliary, "aux", "dense_aux")])
    state.committed_step = 1
    bucket = FakeGradBucket([auxiliary], [torch.tensor([2.0, 4.0, 6.0])])

    result = _run(state, bucket)

    assert result is bucket.buffer()
    torch.testing.assert_close(result, torch.tensor([2.0, 4.0, 6.0]))


def test_full_support_consumes_existing_ef14_residual_and_clears_it():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    state = _state([(matrix, "matrix", "sparse_matrix")], ratio=1.0)
    state.parameter_state(matrix).residual.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    bucket = FakeGradBucket([matrix], [torch.tensor([[10.0, 20.0], [30.0, 40.0]])])

    _run(state, bucket)

    torch.testing.assert_close(bucket.buffer(), torch.tensor([11.0, 22.0, 33.0, 44.0]))
    torch.testing.assert_close(
        state.parameter_state(matrix).residual, torch.zeros_like(matrix)
    )


def test_randk_support_depends_on_step_and_stable_parameter_identity_not_bucket_order():
    first = torch.nn.Parameter(torch.zeros(2, 3))
    second = torch.nn.Parameter(torch.zeros(2, 3))
    state = _state(
        [(first, "first", "sparse_matrix"), (second, "second", "sparse_matrix")],
        method="randk",
    )
    state.committed_step = 1
    first_bucket = FakeGradBucket([second], [torch.arange(6.0).view(2, 3)])
    second_bucket = FakeGradBucket([first], [torch.arange(6.0).view(2, 3)])

    state.begin_step()
    sparse_k_ddp_hook(state, first_bucket).wait()
    sparse_k_ddp_hook(state, second_bucket).wait()
    state.finish_step()
    state.commit_step()

    first_support = state.parameter_state(first).last_support
    second_support = state.parameter_state(second).last_support
    assert first_support.numel() == second_support.numel() == 3
    assert not torch.equal(first_support, second_support)


def test_lifecycle_requires_complete_unique_parameter_coverage():
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2))
    state = _state([(first, "first", "sparse_matrix"), (second, "second", "dense_aux")])
    state.begin_step()
    context = state.note_bucket(FakeGradBucket([first], [torch.ones_like(first)]))
    context.completion_future.set_result(context.buffer)

    with pytest.raises(RuntimeError, match="missing.*second"):
        state.finish_step()
    with pytest.raises(RuntimeError, match="more than once"):
        state.note_bucket(FakeGradBucket([first], [torch.ones_like(first)]))


def test_checkpoint_round_trip_restores_residual_and_committed_step():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    source = _state([(matrix, "matrix", "sparse_matrix")])
    source.parameter_state(matrix).residual.fill_(3.0)
    source.committed_step = 8
    payload = copy.deepcopy(source.state_dict())

    other_matrix = torch.nn.Parameter(torch.zeros(2, 2))
    destination = _state([(other_matrix, "matrix", "sparse_matrix")])
    destination.load_state_dict(payload)

    assert destination.committed_step == 8
    torch.testing.assert_close(
        destination.parameter_state(other_matrix).residual,
        torch.full_like(other_matrix, 3.0),
    )


def test_checkpoint_rejects_method_or_layout_mismatch_without_overwriting_state():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    source = _state([(matrix, "matrix", "sparse_matrix")])
    payload = copy.deepcopy(source.state_dict())
    payload["shared"]["config"]["method"] = "randk"
    destination_matrix = torch.nn.Parameter(torch.zeros(2, 2))
    destination = _state([(destination_matrix, "matrix", "sparse_matrix")])
    destination.parameter_state(destination_matrix).residual.fill_(9.0)

    with pytest.raises(ValueError, match="config"):
        destination.load_state_dict(payload)

    torch.testing.assert_close(
        destination.parameter_state(destination_matrix).residual,
        torch.full_like(destination_matrix, 9.0),
    )


def test_checkpoint_tensor_validation_is_atomic_across_parameters():
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    source = _state(
        [(first, "first", "sparse_matrix"), (second, "second", "sparse_matrix")]
    )
    source.parameter_state(first).residual.fill_(1.0)
    source.parameter_state(second).residual.fill_(2.0)
    payload = copy.deepcopy(source.state_dict())
    payload["rank_0"]["second"]["residual"] = torch.zeros(3)

    destination_first = torch.nn.Parameter(torch.zeros(2, 2))
    destination_second = torch.nn.Parameter(torch.zeros(2, 2))
    destination = _state(
        [
            (destination_first, "first", "sparse_matrix"),
            (destination_second, "second", "sparse_matrix"),
        ]
    )
    destination.parameter_state(destination_first).residual.fill_(9.0)
    destination.parameter_state(destination_second).residual.fill_(9.0)

    with pytest.raises(ValueError, match="tensor schema"):
        destination.load_state_dict(payload)

    torch.testing.assert_close(
        destination.parameter_state(destination_first).residual,
        torch.full_like(destination_first, 9.0),
    )
    torch.testing.assert_close(
        destination.parameter_state(destination_second).residual,
        torch.full_like(destination_second, 9.0),
    )


def test_checkpoint_is_rejected_during_an_active_step():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    state = _state([(matrix, "matrix", "sparse_matrix")])
    state.begin_step()

    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()


def test_state_rejects_non_matrix_sparse_role_and_unused_parameter_mode():
    vector = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="two-dimensional"):
        _state([(vector, "vector", "sparse_matrix")])

    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="find_unused_parameters=False"):
        SparseKDDPState(
            process_group=None,
            fingerprint="c" * 64,
            parameter_specs=[
                SparseKDDPParameterSpec(matrix, "matrix", 0, "sparse_matrix")
            ],
            optimizer_parameters=[matrix],
            config=SparseKConfig(),
            find_unused_parameters=True,
        )
