"""Tests for the GreedyLore hook's explicit non-nesting Future bridge."""

import pytest
import torch

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


def test_new_tail_is_installed_before_completed_source_callback_runs_inline():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    state = _state([parameter])
    state.begin_step()
    context = state.note_bucket(FakeGradBucket([parameter]))
    observed = []

    def launch(current):
        observed.append(state.tail_future is current.completion_future)
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
