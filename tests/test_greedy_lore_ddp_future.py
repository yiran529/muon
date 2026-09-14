"""Tests for the GreedyLore hook's explicit non-nesting Future bridge."""

import pytest
import torch
from types import SimpleNamespace

from dion.greedy_lore import GreedyLoreConfig
import dion.greedy_lore_ddp_hook as hook_module
from dion.greedy_lore_ddp_hook import GreedyLoreDDPParameterSpec, GreedyLoreDDPState


class FakeGradBucket:
    def __init__(self, parameters):
        self._parameters = tuple(parameters)
        self._gradients = tuple(torch.ones_like(parameter) for parameter in parameters)
        self._buffer = torch.cat([gradient.flatten() for gradient in self._gradients])

    def parameters(self):
        return list(self._parameters)

    def gradients(self):
        return list(self._gradients)

    def buffer(self):
        return self._buffer


def _state(parameters):
    specs = [
        GreedyLoreDDPParameterSpec(parameter, f"parameter_{index}", index, "matrix")
        for index, parameter in enumerate(parameters)
    ]
    return GreedyLoreDDPState(
        process_group=None,
        fingerprint="b" * 64,
        parameter_specs=specs,
        optimizer_parameters=parameters,
        config=GreedyLoreConfig(rank=1),
    )


def _completed(value):
    future = torch.futures.Future()
    future.set_result(value)
    return future


def test_bridge_future_resolves_to_a_tensor_not_a_nested_future():
    source = torch.futures.Future()
    destination = torch.futures.Future()
    hook_module.bridge_future(source, destination, lambda value: value + 2)

    source.set_result(torch.tensor([3.0]))

    value = destination.wait()
    assert isinstance(value, torch.Tensor)
    torch.testing.assert_close(value, torch.tensor([5.0]))


def test_bridge_future_propagates_transform_and_source_exceptions():
    transform_source = _completed(torch.tensor([1.0]))
    transform_destination = torch.futures.Future()

    def fail_transform(_value):
        raise RuntimeError("transform failed")

    hook_module.bridge_future(transform_source, transform_destination, fail_transform)
    with pytest.raises(RuntimeError, match="transform failed"):
        transform_destination.wait()

    source = torch.futures.Future()
    destination = torch.futures.Future()
    hook_module.bridge_future(source, destination, lambda value: value)
    source.set_exception(RuntimeError("collective failed"))
    with pytest.raises(RuntimeError, match="collective failed"):
        destination.wait()


def test_bucket_context_remains_live_until_async_launch_completion():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    state = _state([parameter])
    state.begin_step()
    context = state.note_bucket(FakeGradBucket([parameter]))
    pending = torch.futures.Future()

    result = hook_module.enqueue_bucket_chain(
        state, context, lambda _context: pending
    )

    assert result is context.completion_future
    assert context.context_id in state._active_contexts
    assert not result.done()
    pending.set_result(context.buffer)
    assert result.done()
    assert context.context_id not in state._active_contexts


def test_aggregate_tail_is_installed_before_completed_source_callback_runs_inline():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    state = _state([parameter])
    state.begin_step()
    context = state.note_bucket(FakeGradBucket([parameter]))
    observed = []

    def launch(current):
        observed.append(
            state.tail_future is not current.completion_future
            and not state.tail_future.done()
        )
        return _completed(current.buffer)

    result = hook_module.enqueue_bucket_chain(state, context, launch)

    assert observed == [True]
    assert result.done()
    torch.testing.assert_close(result.wait(), context.buffer)


def test_failed_bucket_poison_propagates_without_launching_later_bucket():
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    state = _state([first, second])
    state.begin_step()
    first_context = state.note_bucket(FakeGradBucket([first]))
    first_source = torch.futures.Future()
    first_result = hook_module.enqueue_bucket_chain(
        state,
        first_context,
        lambda _context: first_source,
    )
    second_context = state.note_bucket(FakeGradBucket([second]))
    later_launches = []
    second_result = hook_module.enqueue_bucket_chain(
        state,
        second_context,
        lambda _context: later_launches.append(True)
        or _completed(second_context.buffer),
    )

    first_source.set_exception(RuntimeError("first collective failed"))

    with pytest.raises(RuntimeError, match="first collective failed"):
        first_result.wait()
    with pytest.raises(RuntimeError, match="first collective failed"):
        second_result.wait()
    assert later_launches == []


def test_later_compressed_bucket_prepares_before_previous_bucket_completes(
    monkeypatch,
):
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    state = GreedyLoreDDPState(
        process_group=None,
        fingerprint="e" * 64,
        parameter_specs=(
            GreedyLoreDDPParameterSpec(first, "first", 0, "matrix"),
            GreedyLoreDDPParameterSpec(second, "second", 1, "matrix"),
        ),
        optimizer_parameters=(first, second),
        config=GreedyLoreConfig(rank=1, start_compress_step=0),
    )
    state.committed_step = 1
    state.begin_step()
    prepared_contexts = []
    collective_categories = []
    first_score = torch.futures.Future()
    real_prepare = hook_module._prepare_score_plus_aux_buffer

    def observed_prepare(current_state, context):
        prepared_contexts.append(context.context_id)
        return real_prepare(current_state, context)

    def delayed_first_score(_state, tensor, category):
        collective_categories.append(category)
        if len(collective_categories) == 1:
            return first_score
        return _completed(tensor)

    monkeypatch.setattr(
        hook_module,
        "_prepare_score_plus_aux_buffer",
        observed_prepare,
    )
    monkeypatch.setattr(hook_module, "_all_reduce_future", delayed_first_score)

    first_bucket = FakeGradBucket([first])
    first_result = hook_module.greedy_lore_ddp_hook(state, first_bucket)
    second_result = hook_module.greedy_lore_ddp_hook(state, FakeGradBucket([second]))

    assert prepared_contexts == [0, 1]
    assert collective_categories == ["greedylore_hook/score_plus_aux_allreduce"]
    assert not first_result.done()
    assert not second_result.done()


def test_next_score_launches_while_previous_reconstruction_is_pending(monkeypatch):
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    state = GreedyLoreDDPState(
        process_group=None,
        fingerprint="f" * 64,
        parameter_specs=(
            GreedyLoreDDPParameterSpec(first, "first", 0, "matrix"),
            GreedyLoreDDPParameterSpec(second, "second", 1, "matrix"),
        ),
        optimizer_parameters=(first, second),
        config=GreedyLoreConfig(rank=1, start_compress_step=0),
    )
    state.committed_step = 1
    state.begin_step()
    collective_categories = []
    second_score = torch.futures.Future()
    first_reconstruction = torch.futures.Future()

    def controlled_all_reduce(_state, tensor, category):
        collective_categories.append(category)
        if collective_categories == [
            "greedylore_hook/score_plus_aux_allreduce",
            "greedylore_hook/factor_allreduce",
            "greedylore_hook/score_plus_aux_allreduce",
        ]:
            return second_score
        return _completed(tensor)

    def controlled_reconstruction(_state, context, factor_buffer, matrix_work):
        if context.context_id == 0:
            return first_reconstruction
        hook_module._reconstruct_compressed_matrices(
            _state,
            factor_buffer,
            matrix_work,
        )
        return _completed(context.buffer)

    monkeypatch.setattr(hook_module, "_all_reduce_future", controlled_all_reduce)
    monkeypatch.setattr(
        hook_module,
        "_launch_compressed_reconstruction",
        controlled_reconstruction,
        raising=False,
    )

    first_bucket = FakeGradBucket([first])
    first_result = hook_module.greedy_lore_ddp_hook(state, first_bucket)
    second_result = hook_module.greedy_lore_ddp_hook(state, FakeGradBucket([second]))

    assert collective_categories == [
        "greedylore_hook/score_plus_aux_allreduce",
        "greedylore_hook/factor_allreduce",
        "greedylore_hook/score_plus_aux_allreduce",
    ]
    assert not first_result.done()
    assert not second_result.done()

    first_reconstruction.set_exception(RuntimeError("reconstruction failed"))
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        first_result.wait()


def test_compressed_collective_failure_poison_propagates_to_later_bucket(
    monkeypatch,
):
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    state = GreedyLoreDDPState(
        process_group=None,
        fingerprint="1" * 64,
        parameter_specs=(
            GreedyLoreDDPParameterSpec(first, "first", 0, "matrix"),
            GreedyLoreDDPParameterSpec(second, "second", 1, "matrix"),
        ),
        optimizer_parameters=(first, second),
        config=GreedyLoreConfig(rank=1, start_compress_step=0),
    )
    state.committed_step = 1
    state.begin_step()
    failed_score = torch.futures.Future()
    collective_calls = []

    def fail_first_score(_state, tensor, category):
        collective_calls.append(category)
        if len(collective_calls) == 1:
            return failed_score
        return _completed(tensor)

    monkeypatch.setattr(hook_module, "_all_reduce_future", fail_first_score)

    first_result = hook_module.greedy_lore_ddp_hook(state, FakeGradBucket([first]))
    second_result = hook_module.greedy_lore_ddp_hook(state, FakeGradBucket([second]))
    first_context = state._active_contexts[0]

    failed_score.set_exception(RuntimeError("score collective failed"))

    assert first_context.collective_completion_future.done()
    with pytest.raises(RuntimeError, match="score collective failed"):
        first_result.wait()
    with pytest.raises(RuntimeError, match="score collective failed"):
        second_result.wait()
    assert collective_calls == ["greedylore_hook/score_plus_aux_allreduce"]


def test_reconstruction_launch_failure_does_not_suppress_later_collectives(
    monkeypatch,
):
    first = torch.nn.Parameter(torch.zeros(2, 2))
    second = torch.nn.Parameter(torch.zeros(2, 2))
    state = GreedyLoreDDPState(
        process_group=None,
        fingerprint="2" * 64,
        parameter_specs=(
            GreedyLoreDDPParameterSpec(first, "first", 0, "matrix"),
            GreedyLoreDDPParameterSpec(second, "second", 1, "matrix"),
        ),
        optimizer_parameters=(first, second),
        config=GreedyLoreConfig(rank=1, start_compress_step=0),
    )
    state.committed_step = 1
    state.begin_step()
    collective_categories = []
    real_reconstruction = hook_module._launch_compressed_reconstruction

    def completed_all_reduce(_state, tensor, category):
        collective_categories.append(category)
        return _completed(tensor)

    def fail_first_reconstruction(_state, context, factor_buffer, matrix_work):
        if context.context_id == 0:
            raise RuntimeError("reconstruction launch failed")
        return real_reconstruction(_state, context, factor_buffer, matrix_work)

    monkeypatch.setattr(hook_module, "_all_reduce_future", completed_all_reduce)
    monkeypatch.setattr(
        hook_module,
        "_launch_compressed_reconstruction",
        fail_first_reconstruction,
    )

    first_result = hook_module.greedy_lore_ddp_hook(state, FakeGradBucket([first]))
    second_result = hook_module.greedy_lore_ddp_hook(state, FakeGradBucket([second]))

    with pytest.raises(RuntimeError, match="reconstruction launch failed"):
        first_result.wait()
    assert second_result.done()
    assert collective_categories == [
        "greedylore_hook/score_plus_aux_allreduce",
        "greedylore_hook/factor_allreduce",
        "greedylore_hook/score_plus_aux_allreduce",
        "greedylore_hook/factor_allreduce",
    ]


def test_cross_stream_intermediates_are_recorded_on_their_consumer_streams():
    class RecordingTensor:
        def __init__(self):
            self.streams = []

        def record_stream(self, stream):
            self.streams.append(stream)

    execution_stream = object()
    reconstruction_stream = object()
    score = RecordingTensor()
    corrected = RecordingTensor()
    signed_lambda = RecordingTensor()
    factor = RecordingTensor()
    projector = RecordingTensor()
    bucket_buffer = RecordingTensor()
    work = SimpleNamespace(
        corrected=corrected,
        signed_lambda=signed_lambda,
        projector=projector,
    )
    prepared = SimpleNamespace(score_plus_aux=score, matrix_work=[work])
    context = SimpleNamespace(buffer=bucket_buffer)

    hook_module._record_prepared_tensors_on_stream(prepared, execution_stream)
    hook_module._record_reconstruction_tensors_on_stream(
        context,
        factor,
        [work],
        reconstruction_stream,
    )

    assert score.streams == [execution_stream]
    assert corrected.streams == [execution_stream]
    assert signed_lambda.streams == [execution_stream]
    assert factor.streams == [reconstruction_stream]
    assert projector.streams == [reconstruction_stream]
    assert bucket_buffer.streams == [reconstruction_stream]
