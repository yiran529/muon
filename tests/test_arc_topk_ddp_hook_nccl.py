"""NCCL stream-completion stress tests for the ARC DDP hook."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.arc_topk_ddp_hook import (
    ArcTopKDDPParameterSpec,
    ArcTopKDDPState,
    enqueue_bucket_chain,
)
from dion.arc_topk_sync import ArcTopKSyncConfig


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


def _reduced_tensor(future):
    value = future.value()
    return value[0] if isinstance(value, (tuple, list)) else value


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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
        first = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        second = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        state = ArcTopKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="d" * 64,
            parameter_specs=[
                ArcTopKDDPParameterSpec(first, "first", 0, "arc_matrix"),
                ArcTopKDDPParameterSpec(second, "second", 1, "arc_matrix"),
            ],
            optimizer_parameters=[first, second],
            config=ArcTopKSyncConfig(),
        )
        producer = torch.cuda.Stream(device=device)
        consumer = torch.cuda.Stream(device=device)

        def launch(context):
            work = dist.all_reduce(
                context.buffer,
                group=dist.group.WORLD,
                async_op=True,
            )
            return work.get_future().then(
                lambda future: _reduced_tensor(future).div_(world_size)
            )

        for iteration in range(8):
            state.begin_step()
            first_buffer = torch.full(
                (first.numel(),), float(rank + 1), device=device
            )
            first_context = state.note_bucket(_CudaBucket(first, first_buffer))
            enqueue_bucket_chain(state, first_context, launch)

            delayed_buffer = torch.full(
                (second.numel(),), -1000.0, device=device
            )
            expected = float(iteration * 10) + 1.5
            with torch.cuda.stream(producer):
                torch.cuda._sleep(2_000_000)
                delayed_buffer.fill_(float(iteration * 10 + rank + 1))
                second_context = state.note_bucket(
                    _CudaBucket(second, delayed_buffer)
                )
            result = enqueue_bucket_chain(state, second_context, launch)

            with torch.cuda.stream(consumer):
                consumed = result.wait().clone()
            consumer.synchronize()
            torch.testing.assert_close(
                consumed,
                torch.full_like(consumed, expected),
            )
            state.finish_step()
            state.commit_step()
            assert not state._active_contexts

            churn = [torch.empty(128 * 1024, device=device) for _ in range(8)]
            del churn
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_nccl_future_completion_waits_for_bucket_ready_stream_and_releases_contexts():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
