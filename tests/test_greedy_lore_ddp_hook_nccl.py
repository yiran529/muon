"""NCCL stream-completion test for the GreedyLore Future sequencer."""

import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.greedy_lore import GreedyLoreConfig
import dion.greedy_lore_ddp_hook as hook_module
from dion.greedy_lore_ddp_hook import GreedyLoreDDPParameterSpec, GreedyLoreDDPState


class _CudaBucket:
    def __init__(self, parameter, buffer):
        self._parameter = parameter
        self._buffer = buffer

    def parameters(self):
        return [self._parameter]

    def gradients(self):
        return [self._buffer.view_as(self._parameter)]

    def buffer(self):
        return self._buffer


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reduced_tensor(future):
    value = future.value()
    return value[0] if isinstance(value, (tuple, list)) else value


def _join_with_timeout(process_context, timeout):
    deadline = time.monotonic() + timeout
    while not process_context.join(timeout=max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            return False
    return True


def _worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        device = torch.device("cuda", rank)
        parameter = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="d" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(parameter, "parameter", 0, "matrix")
            ],
            optimizer_parameters=[parameter],
            config=GreedyLoreConfig(rank=1),
        )
        consumer = torch.cuda.Stream(device=device)
        completion_stream = torch.cuda.Stream(device=device)
        state.begin_step()
        buffer = torch.full((parameter.numel(),), -1000.0, device=device)
        context = state.note_bucket(_CudaBucket(parameter, buffer))

        def launch(current):
            work = dist.all_reduce(
                current.buffer,
                group=dist.group.WORLD,
                async_op=True,
            )

            def delayed_final_write(completed):
                reduced = _reduced_tensor(completed)
                execution_stream = state.execution_stream(device)
                execution_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(execution_stream):
                    torch.cuda._sleep(2_000_000)
                    reduced.fill_(7.0)
                return reduced

            with torch.cuda.stream(completion_stream):
                return work.get_future().then(delayed_final_write)

        result = hook_module.enqueue_bucket_chain(state, context, launch)
        with torch.cuda.stream(consumer):
            consumed = result.wait().clone()
        consumer.synchronize()
        torch.testing.assert_close(consumed, torch.full_like(consumed, 7.0))
        state.finish_step()
        state.commit_step()
        assert not state._active_contexts
    finally:
        dist.destroy_process_group()


def _broadcast_refresh_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    original_broadcast_future = hook_module._broadcast_future
    original_stream_result = hook_module._on_bucket_execution_stream_result
    try:
        device = torch.device("cuda", rank)
        parameter = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="e" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(parameter, "parameter", 0, "matrix")
            ],
            optimizer_parameters=[parameter],
            config=GreedyLoreConfig(
                rank=1,
                start_compress_step=0,
                basis_sync="broadcast",
            ),
        )
        consumer = torch.cuda.Stream(device=device)
        buffer = (
            torch.tensor(
                [[1.0 + rank, 2.0], [3.0, 4.0 - rank]],
                device=device,
            )
            .reshape(-1)
            .clone()
        )

        def delayed_broadcast(current_state, tensor, category):
            source = original_broadcast_future(current_state, tensor, category)

            def delayed_marker(completed):
                completed.value()
                execution_stream = current_state.execution_stream(device)
                execution_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(execution_stream):
                    torch.cuda._sleep(2_000_000)
                    buffer.fill_(11.0)
                return tensor

            return source.then(delayed_marker)

        export_calls = 0

        def counted_stream_result(current_state, current_context, callback):
            wrapped = original_stream_result(current_state, current_context, callback)

            def run(completed):
                nonlocal export_calls
                export_calls += 1
                return wrapped(completed)

            return run

        hook_module._broadcast_future = delayed_broadcast
        hook_module._on_bucket_execution_stream_result = counted_stream_result

        state.begin_step()
        result = hook_module.greedy_lore_ddp_hook(state, _CudaBucket(parameter, buffer))
        with torch.cuda.stream(consumer):
            consumed = result.wait().clone()
        consumer.synchronize()
        torch.testing.assert_close(consumed, torch.full_like(consumed, 11.0))
        assert export_calls == 1
        state.finish_step()
        state.commit_step()
        assert not state._active_contexts
    finally:
        hook_module._broadcast_future = original_broadcast_future
        hook_module._on_bucket_execution_stream_result = original_stream_result
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_returned_future_exports_final_compressor_stream_write():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    process_context = mp.spawn(
        _worker,
        args=(2, _free_port()),
        nprocs=2,
        join=False,
    )
    try:
        assert _join_with_timeout(process_context, timeout=60)
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.multi_gpu
def test_broadcast_refresh_future_exports_final_compressor_stream_write():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    process_context = mp.spawn(
        _broadcast_refresh_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=False,
    )
    try:
        assert _join_with_timeout(process_context, timeout=60)
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
