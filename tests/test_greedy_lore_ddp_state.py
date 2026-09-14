"""GreedyLore DDP state allocation and lifecycle contracts."""

import gc
import weakref

import pytest
import torch

from dion.greedy_lore import GreedyLoreConfig
from dion.greedy_lore_ddp_hook import (
    GreedyLoreDDPParameterSpec,
    GreedyLoreDDPState,
)


class FakeGradBucket:
    def __init__(self, parameters):
        self._parameters = tuple(parameters)
        self._gradients = tuple(torch.zeros_like(item) for item in parameters)
        self._buffer = torch.cat([item.flatten() for item in self._gradients])

    def parameters(self):
        return list(self._parameters)

    def gradients(self):
        return list(self._gradients)

    def buffer(self):
        return self._buffer


def _make_state(parameters, *, specs=None, rank=2, **kwargs):
    if specs is None:
        specs = tuple(
            GreedyLoreDDPParameterSpec(parameter, name, index, role)
            for index, (parameter, name, role) in enumerate(parameters)
        )
    return GreedyLoreDDPState(
        process_group=None,
        fingerprint="a" * 64,
        parameter_specs=specs,
        optimizer_parameters=tuple(parameter for parameter, _, _ in parameters),
        config=GreedyLoreConfig(rank=rank, start_compress_step=1),
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_state_preallocates_matrix_state_in_parameter_dtype(dtype):
    parameter = torch.nn.Parameter(torch.zeros(5, 3, dtype=dtype))
    state = _make_state(((parameter, "matrix", "matrix"),), rank=2)
    item = state.parameter_state(parameter)

    assert item.orientation.original_shape == (5, 3)
    assert item.error.shape == (3, 5)
    assert item.error.dtype == dtype
    assert item.basis.shape == (3, 3)
    assert item.basis.dtype == dtype
    assert torch.equal(item.basis, torch.eye(3, dtype=dtype))
    assert item.last_support.tolist() == [0, 1]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_name", "stable names"),
        ("duplicate_id", "stable IDs"),
        ("duplicate_parameter", "unique parameters"),
        ("duplicate_optimizer", "optimizer parameters"),
        ("unsupported_role", "role"),
    ],
)
def test_state_rejects_ambiguous_frozen_identity_tables(mutation, match):
    first = torch.nn.Parameter(torch.zeros(3, 4))
    second = torch.nn.Parameter(torch.zeros(3, 4))
    parameters = [(first, "first", "matrix"), (second, "second", "matrix")]
    specs = [
        GreedyLoreDDPParameterSpec(first, "first", 0, "matrix"),
        GreedyLoreDDPParameterSpec(second, "second", 1, "matrix"),
    ]
    if mutation == "duplicate_name":
        specs[1] = GreedyLoreDDPParameterSpec(second, "first", 1, "matrix")
    elif mutation == "duplicate_id":
        specs[1] = GreedyLoreDDPParameterSpec(second, "second", 0, "matrix")
    elif mutation == "duplicate_parameter":
        specs[1] = GreedyLoreDDPParameterSpec(first, "second", 1, "matrix")
    elif mutation == "unsupported_role":
        specs[1] = GreedyLoreDDPParameterSpec(second, "second", 1, "other")
    if mutation == "duplicate_optimizer":
        parameters[1] = (first, "second", "matrix")

    with pytest.raises(ValueError, match=match):
        _make_state(parameters, specs=specs)


def test_specs_and_optimizer_parameters_require_exact_identity_coverage():
    model_only = torch.nn.Parameter(torch.zeros(3, 4))
    optimizer_only = torch.nn.Parameter(torch.zeros(3, 4))
    spec = GreedyLoreDDPParameterSpec(model_only, "model", 0, "matrix")

    with pytest.raises(ValueError, match="not owned by the optimizer"):
        _make_state(((optimizer_only, "optimizer", "matrix"),), specs=(spec,))
    with pytest.raises(ValueError, match="absent from the model"):
        _make_state(((model_only, "model", "matrix"),), specs=())


@pytest.mark.parametrize(
    "parameter,rank,match",
    [
        (torch.nn.Parameter(torch.zeros(4)), 1, "two-dimensional"),
        (
            torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64)),
            1,
            "FP32 or BF16",
        ),
        (torch.nn.Parameter(torch.zeros(2, 3)), 3, "rank"),
    ],
)
def test_matrix_state_rejects_unsupported_shape_dtype_or_rank(parameter, rank, match):
    with pytest.raises(ValueError, match=match):
        _make_state(((parameter, "matrix", "matrix"),), rank=rank)


def test_state_rejects_find_unused_parameters_mode():
    parameter = torch.nn.Parameter(torch.zeros(3, 4))
    with pytest.raises(ValueError, match="find_unused_parameters=False"):
        _make_state(((parameter, "matrix", "matrix"),), find_unused_parameters=True)


def test_parameter_state_and_bucket_mapping_survive_bucket_reordering():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    auxiliary = torch.nn.Parameter(torch.zeros(4))
    state = _make_state(((matrix, "matrix", "matrix"), (auxiliary, "aux", "dense_aux")))
    matrix_state = state.parameter_state(matrix)
    matrix_state.error.fill_(7)

    state.begin_step()
    first = state.note_bucket(FakeGradBucket((auxiliary,)))
    second = state.note_bucket(FakeGradBucket((matrix,)))
    first.completion_future.set_result(first.buffer)
    second.completion_future.set_result(second.buffer)
    state.finish_step()
    state.commit_step()

    state.begin_step()
    reordered = state.note_bucket(FakeGradBucket((matrix, auxiliary)))
    assert reordered.parameter_states == (matrix_state, None)
    assert state.parameter_state(matrix) is matrix_state
    assert torch.equal(matrix_state.error, torch.full_like(matrix_state.error, 7))


def test_lifecycle_requires_exactly_one_active_step_and_parameter_capture():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    state = _make_state(((matrix, "matrix", "matrix"),))

    assert state.begin_step() == 1
    with pytest.raises(RuntimeError, match="already active"):
        state.begin_step()
    context = state.note_bucket(FakeGradBucket((matrix,)))
    assert context.step == 1
    assert context.phase is None
    with pytest.raises(RuntimeError, match="more than once"):
        state.note_bucket(FakeGradBucket((matrix,)))
    with pytest.raises(RuntimeError, match="in flight"):
        state.finish_step()
    with pytest.raises(RuntimeError, match="before finish"):
        state.commit_step()

    context.completion_future.set_result(context.buffer)
    state.finish_step()
    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.state_dict()
    state.commit_step()
    assert state.committed_step == 1
    assert state.state_dict()["shared"]["committed_step"] == 1
    with pytest.raises(RuntimeError, match="no active"):
        state.commit_step()


def test_finish_rejects_missing_coverage_and_bucket_rejects_unknown_parameter():
    first = torch.nn.Parameter(torch.zeros(3, 4))
    second = torch.nn.Parameter(torch.zeros(3, 4))
    unknown = torch.nn.Parameter(torch.zeros(3, 4))
    state = _make_state(((first, "first", "matrix"), (second, "second", "matrix")))
    state.begin_step()
    with pytest.raises(RuntimeError, match="outside the GreedyLore layout"):
        state.note_bucket(FakeGradBucket((unknown,)))
    context = state.note_bucket(FakeGradBucket((first,)))
    context.completion_future.set_result(context.buffer)
    with pytest.raises(RuntimeError, match="missing.*second"):
        state.finish_step()


def test_next_step_waits_for_tail_even_if_previous_step_is_not_active():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    state = _make_state(((matrix, "matrix", "matrix"),))
    pending = torch.futures.Future()
    state.tail_future = pending

    with pytest.raises(RuntimeError, match="previous tail is in flight"):
        state.begin_step()


def test_finish_step_waits_for_every_bucket_completion_not_only_latest():
    first = torch.nn.Parameter(torch.zeros(3, 4))
    second = torch.nn.Parameter(torch.zeros(3, 4))
    state = _make_state(
        ((first, "first", "matrix"), (second, "second", "matrix"))
    )
    state.begin_step()
    first_context = state.note_bucket(FakeGradBucket((first,)))
    second_context = state.note_bucket(FakeGradBucket((second,)))

    second_context.completion_future.set_result(second_context.buffer)

    with pytest.raises(RuntimeError, match="bucket tail is still in flight"):
        state.finish_step()

    first_context.completion_future.set_result(first_context.buffer)
    state.finish_step()


def test_basis_validation_requires_a_committed_step_boundary():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    state = _make_state(((matrix, "matrix", "matrix"),))
    state.begin_step()

    with pytest.raises(RuntimeError, match="committed step boundary"):
        state.validate_replicated_basis_across_ranks()


def test_bucket_context_is_retained_until_completion_future_finishes():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))
    state = _make_state(((matrix, "matrix", "matrix"),))
    state.begin_step()
    bucket = FakeGradBucket((matrix,))
    bucket_ref = weakref.ref(bucket)
    context = state.note_bucket(bucket)
    completion = context.completion_future

    del context, bucket
    gc.collect()
    assert bucket_ref() is not None

    completion.set_result(None)
    gc.collect()
    assert bucket_ref() is None


def test_dense_parameter_has_no_matrix_state():
    parameter = torch.nn.Parameter(torch.zeros(4))
    state = _make_state(((parameter, "aux", "dense_aux"),))
    with pytest.raises(ValueError, match="dense auxiliary"):
        state.parameter_state(parameter)


def test_state_does_not_retain_or_inspect_optimizer_step_fields():
    matrix = torch.nn.Parameter(torch.zeros(3, 4))

    class StepHostileOptimizer:
        param_groups = ({"params": (matrix,)},)

        @property
        def state(self):
            raise AssertionError("optimizer state must not be inspected")

    optimizer = StepHostileOptimizer()
    state = _make_state(
        tuple(
            (parameter, "matrix", "matrix")
            for parameter in optimizer.param_groups[0]["params"]
        )
    )

    assert not hasattr(state, "optimizer")
    assert state.state_dict()["shared"]["committed_step"] == 0
