"""NCCL stream-completion stress tests for the ARC DDP hook."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import dion.arc_topk_ddp_hook as hook_module
from dion.arc_topk_ddp_hook import (
    ArcTopKDDPParameterSpec,
    ArcTopKDDPState,
    arc_topk_ddp_hook,
    enqueue_bucket_chain,
)
from dion.arc_topk_sync import ArcTopKSyncConfig
from dion.collective_observer import CollectiveObserver, set_active_observer


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


class _BucketedCudaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(16, 16, bias=False) for _ in range(8)]
        )

    def forward(self, value):
        for layer in self.layers:
            value = torch.tanh(layer(value))
        return value.sum()


def _sparse_hook_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    observer = CollectiveObserver()
    set_active_observer(observer)
    original_all_reduce = hook_module._all_reduce_future
    try:
        device = torch.device("cuda", rank)
        torch.manual_seed(1234)
        model = _BucketedCudaModel().to(device)
        ddp = DDP(
            model,
            device_ids=[rank],
            gradient_as_bucket_view=True,
            bucket_cap_mb=0.0005,
        )
        parameters = list(model.parameters())
        state = ArcTopKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="2" * 64,
            parameter_specs=[
                ArcTopKDDPParameterSpec(parameter, name, index, "arc_matrix")
                for index, (name, parameter) in enumerate(model.named_parameters())
            ],
            optimizer_parameters=parameters,
            config=ArcTopKSyncConfig(
                ratio=0.5,
                projection_rank=2,
                eta=0.5,
                seed=17,
                start_compress_step=0,
            ),
        )
        callback_count = 0

        def delayed_all_reduce(current_state, tensor, category):
            nonlocal callback_count
            callback_count += 1
            if category == "arc_hook/sketch" and callback_count % world_size == rank:
                torch.cuda._sleep(500_000)
            return original_all_reduce(current_state, tensor, category)

        hook_module._all_reduce_future = delayed_all_reduce
        hook_counts = []

        def counted_hook(hook_state, bucket):
            hook_counts[-1] += 1
            return arc_topk_ddp_hook(hook_state, bucket)

        ddp.register_comm_hook(state, counted_hook)
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        consumer = torch.cuda.Stream(device=device)
        for iteration in range(6):
            hook_counts.append(0)
            state.begin_step()
            inputs = torch.full(
                (8, 16),
                float(rank + iteration + 1) / 10,
                device=device,
            )
            ddp(inputs).backward()
            state.finish_step()
            consumer.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(consumer):
                consumed = [parameter.grad.clone() for parameter in parameters]
            consumer.synchronize()
            local_snapshot = {
                "gradients": [gradient.cpu() for gradient in consumed],
                "globals": [
                    state.parameter_state(parameter).g_global.cpu()
                    for parameter in parameters
                ],
            }
            gathered_snapshots = [None] * world_size
            dist.all_gather_object(gathered_snapshots, local_snapshot)
            for parameter_index in range(len(parameters)):
                rank_zero_gradient = gathered_snapshots[0]["gradients"][parameter_index]
                rank_one_gradient = gathered_snapshots[1]["gradients"][parameter_index]
                rank_zero_global = gathered_snapshots[0]["globals"][parameter_index]
                rank_one_global = gathered_snapshots[1]["globals"][parameter_index]
                torch.testing.assert_close(
                    rank_zero_global,
                    rank_one_global,
                    msg=lambda message: (
                        f"g_global iteration={iteration}, "
                        f"parameter={parameter_index}: {message}"
                    ),
                )
                torch.testing.assert_close(
                    rank_zero_gradient,
                    rank_zero_global,
                    msg=lambda message: (
                        f"rank-0 scatter iteration={iteration}, "
                        f"parameter={parameter_index}: {message}"
                    ),
                )
                torch.testing.assert_close(
                    rank_one_gradient,
                    rank_one_global,
                    msg=lambda message: (
                        f"rank-1 scatter iteration={iteration}, "
                        f"parameter={parameter_index}: {message}"
                    ),
                )
                torch.testing.assert_close(
                    rank_zero_gradient,
                    rank_one_gradient,
                    msg=lambda message: (
                        f"gradient iteration={iteration}, "
                        f"parameter={parameter_index}: {message}"
                    ),
                )
            optimizer.step()
            state.commit_step()
            optimizer.zero_grad(set_to_none=True)
            assert not state._active_contexts
            churn = [torch.empty(256 * 1024, device=device) for _ in range(16)]
            del churn

        assert hook_counts[-1] >= 2
        signature = observer.signature()
        gathered_signatures = [None] * world_size
        dist.all_gather_object(gathered_signatures, signature)
        assert gathered_signatures == [gathered_signatures[0]] * world_size
        assert all(event.category != "arc/seed" for event in observer.events)
    finally:
        hook_module._all_reduce_future = original_all_reduce
        set_active_observer(None)
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_sparse_nccl_hook_survives_rebuild_callback_delay_and_allocator_churn():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(_sparse_hook_worker, args=(2, _free_port()), nprocs=2, join=True)
