"""Tests for stable ARC DDP-hook metadata and optimizer-step lifecycle."""

import pytest
import torch

from dion.arc_topk_ddp_hook import (
    ArcTopKDDPParameterSpec,
    ArcTopKDDPState,
)
from dion.arc_topk_sync import ArcTopKSyncConfig


class FakeGradBucket:
    def __init__(self, parameters):
        self._parameters = tuple(parameters)
        self._gradients = tuple(torch.zeros_like(parameter) for parameter in parameters)
        self._buffer = torch.cat([gradient.flatten() for gradient in self._gradients])

    def parameters(self):
        return list(self._parameters)

    def gradients(self):
        return list(self._gradients)

    def buffer(self):
        return self._buffer


def _config(error_feedback="ef21m"):
    return ArcTopKSyncConfig(
        ratio=0.5,
        projection_rank=2,
        eta=0.25,
        seed=17,
        start_compress_step=1,
        error_feedback=error_feedback,
    )


def _make_state(parameters, specs=None, config=None, **kwargs):
    if specs is None:
        specs = [
            ArcTopKDDPParameterSpec(
                parameter=parameter,
                stable_name=name,
                stable_id=index,
                role=role,
            )
            for index, (parameter, name, role) in enumerate(parameters)
        ]
    return ArcTopKDDPState(
        process_group=None,
        fingerprint="a" * 64,
        parameter_specs=specs,
        optimizer_parameters=[parameter for parameter, _, _ in parameters],
        config=_config() if config is None else config,
        **kwargs,
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_name", "stable names"),
        ("duplicate_id", "stable IDs"),
        ("duplicate_parameter", "parameters"),
        ("invalid_role", "role"),
    ],
)
def test_parameter_metadata_rejects_duplicate_or_missing_identity(mutation, match):
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    specs = [
        ArcTopKDDPParameterSpec(first, "first", 0, "arc_matrix"),
        ArcTopKDDPParameterSpec(second, "second", 1, "arc_matrix"),
    ]
    if mutation == "duplicate_name":
        specs[1] = ArcTopKDDPParameterSpec(second, "first", 1, "arc_matrix")
    elif mutation == "duplicate_id":
        specs[1] = ArcTopKDDPParameterSpec(second, "second", 0, "arc_matrix")
    elif mutation == "duplicate_parameter":
        specs[1] = ArcTopKDDPParameterSpec(first, "second", 1, "arc_matrix")
    else:
        specs[1] = ArcTopKDDPParameterSpec(second, "second", 1, "unsupported")

    with pytest.raises(ValueError, match=match):
        _make_state(
            [(first, "first", "arc_matrix"), (second, "second", "arc_matrix")],
            specs,
        )


def test_arc_matrix_role_requires_a_two_dimensional_parameter():
    parameter = torch.nn.Parameter(torch.zeros(4))

    with pytest.raises(ValueError, match="two-dimensional"):
        _make_state([(parameter, "vector", "arc_matrix")])


def test_specs_and_optimizer_parameters_must_have_exact_identity_coverage():
    model_only = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer_only = torch.nn.Parameter(torch.zeros(2, 2))
    model_spec = ArcTopKDDPParameterSpec(model_only, "model", 0, "arc_matrix")

    with pytest.raises(ValueError, match="not owned by the optimizer"):
        _make_state(
            [(optimizer_only, "optimizer", "arc_matrix")],
            specs=[model_spec],
        )
    with pytest.raises(ValueError, match="absent from the model"):
        _make_state(
            [(model_only, "model", "arc_matrix")],
            specs=[],
        )


def test_parameter_state_survives_bucket_reordering_by_object_identity():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    scalar = torch.nn.Parameter(torch.zeros(2))
    state = _make_state(
        [(matrix, "matrix", "arc_matrix"), (scalar, "scalar", "dense_aux")]
    )
    matrix_state = state.parameter_state(matrix)
    matrix_state.h_local.fill_(7.0)

    state.begin_step()
    first_context = state.note_bucket(FakeGradBucket([scalar]))
    second_context = state.note_bucket(FakeGradBucket([matrix]))
    first_context.completion_future.set_result(first_context.buffer)
    second_context.completion_future.set_result(second_context.buffer)
    state.finish_step()
    state.commit_step()

    state.begin_step()
    reordered = state.note_bucket(FakeGradBucket([matrix, scalar]))

    assert state.parameter_state(matrix) is matrix_state
    torch.testing.assert_close(matrix_state.h_local, torch.full_like(matrix, 7.0))
    assert reordered.parameter_states[0] is matrix_state


def test_lifecycle_requires_one_active_step_and_one_capture_per_parameter():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    state = _make_state([(matrix, "matrix", "arc_matrix")])

    assert state.begin_step() == 1
    with pytest.raises(RuntimeError, match="already active"):
        state.begin_step()
    context = state.note_bucket(FakeGradBucket([matrix]))
    with pytest.raises(RuntimeError, match="more than once"):
        state.note_bucket(FakeGradBucket([matrix]))
    with pytest.raises(RuntimeError, match="in flight"):
        state.finish_step()

    context.completion_future.set_result(context.buffer)
    state.finish_step()
    state.commit_step()
    assert state.committed_step == 1

    with pytest.raises(RuntimeError, match="no active"):
        state.commit_step()


def test_finish_step_rejects_incomplete_parameter_coverage():
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    state = _make_state(
        [(first, "first", "arc_matrix"), (second, "second", "arc_matrix")]
    )

    state.begin_step()
    context = state.note_bucket(FakeGradBucket([first]))
    context.completion_future.set_result(context.buffer)

    with pytest.raises(RuntimeError, match="missing.*second"):
        state.finish_step()


def test_hook_state_rejects_find_unused_parameters_mode():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))

    with pytest.raises(ValueError, match="find_unused_parameters=False"):
        _make_state(
            [(matrix, "matrix", "arc_matrix")],
            find_unused_parameters=True,
        )


def test_fresh_state_dict_preallocates_complete_stable_name_tensor_schema():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    scalar = torch.nn.Parameter(torch.zeros(2))
    state = _make_state(
        [(matrix, "matrix", "arc_matrix"), (scalar, "scalar", "dense_aux")]
    )

    payload = state.state_dict()

    assert set(payload["rank_0"]) == {"matrix"}
    assert set(payload["rank_0"]["matrix"]) == {"h_local", "g_local"}
    assert set(payload["shared"]["g_global"]) == {"matrix"}
    torch.testing.assert_close(
        payload["rank_0"]["matrix"]["h_local"], torch.zeros_like(matrix)
    )
    torch.testing.assert_close(
        payload["shared"]["g_global"]["matrix"], torch.zeros_like(matrix)
    )


def test_ef14_state_dict_contains_only_rank_local_residual():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    state = _make_state(
        [(matrix, "matrix", "arc_matrix")],
        config=_config(error_feedback="ef14"),
    )

    parameter_state = state.parameter_state(matrix)
    assert parameter_state.h_local is None
    assert parameter_state.g_local is None
    assert parameter_state.g_global is None
    torch.testing.assert_close(parameter_state.residual, torch.zeros_like(matrix))
    payload = state.state_dict()
    assert "g_global" not in payload["shared"]
    assert set(payload["rank_0"]["matrix"]) == {"residual"}


def test_ef14_checkpoint_round_trip_restores_residual_and_step():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    config = _config(error_feedback="ef14")
    source = _make_state(
        [(matrix, "matrix", "arc_matrix")], config=config
    )
    source.parameter_state(matrix).residual.copy_(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    source.committed_step = 7
    payload = source.state_dict()

    restored_matrix = torch.nn.Parameter(torch.zeros(2, 2))
    restored = _make_state(
        [(restored_matrix, "matrix", "arc_matrix")], config=config
    )
    restored.load_state_dict(payload)

    assert restored.committed_step == 7
    torch.testing.assert_close(
        restored.parameter_state(restored_matrix).residual,
        source.parameter_state(matrix).residual,
    )


def test_checkpoint_rejects_cross_error_feedback_mode():
    matrix = torch.nn.Parameter(torch.zeros(2, 2))
    ef14 = _make_state(
        [(matrix, "matrix", "arc_matrix")],
        config=_config(error_feedback="ef14"),
    )
    payload = ef14.state_dict()
    ef21m = _make_state([(matrix, "matrix", "arc_matrix")])

    with pytest.raises(ValueError, match="config"):
        ef21m.load_state_dict(payload)
