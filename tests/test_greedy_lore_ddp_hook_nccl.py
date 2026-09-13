"""NCCL stream-completion test for the GreedyLore Future sequencer."""

import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.collective_observer import CollectiveObserver, set_active_observer
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


class _BucketedCudaModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(16, 16, bias=False, dtype=dtype)
                for _ in range(8)
            ]
        )
        self.dense = torch.nn.Parameter(torch.zeros(16, dtype=dtype))

    def forward(self, value):
        for layer in self.layers:
            value = torch.tanh(layer(value))
        return value.sum() + (self.dense * value.sum(dim=0)).sum()


def _compressed_stress_worker(rank, world_size, port, dtype_name):
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
        dtype = getattr(torch, dtype_name)
        torch.manual_seed(1234)
        model = _BucketedCudaModel(dtype=dtype).to(device)
        ddp = DDP(
            model,
            device_ids=[rank],
            gradient_as_bucket_view=True,
            bucket_cap_mb=0.0005,
        )
        parameters = list(model.parameters())
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="7" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(
                    parameter,
                    name,
                    index,
                    "matrix" if parameter.ndim == 2 else "dense_aux",
                )
                for index, (name, parameter) in enumerate(model.named_parameters())
            ],
            optimizer_parameters=parameters,
            config=GreedyLoreConfig(
                rank=1,
                start_compress_step=0,
                update_interval=3,
                seed=29,
            ),
        )
        delayed_calls = 0

        def delayed_all_reduce(current_state, tensor, category):
            nonlocal delayed_calls
            delayed_calls += 1
            if (
                category == "greedylore_hook/factor_allreduce"
                and delayed_calls % world_size == rank
            ):
                torch.cuda._sleep(500_000)
            return original_all_reduce(current_state, tensor, category)

        hook_module._all_reduce_future = delayed_all_reduce
        ddp.register_comm_hook(state, hook_module.greedy_lore_ddp_hook)
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        consumer = torch.cuda.Stream(device=device)
        for iteration in range(6):
            if iteration == 2:
                with ddp.no_sync():
                    ddp(
                        torch.full(
                            (8, 16),
                            0.125 * (rank + 1),
                            device=device,
                            dtype=dtype,
                        )
                    ).backward()
            state.begin_step()
            inputs = torch.full(
                (8, 16),
                float(rank + iteration + 1) / 10,
                device=device,
                dtype=dtype,
            )
            if iteration == 3:
                inputs.zero_()
            ddp(inputs).backward()
            state.finish_step()
            consumer.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(consumer):
                consumed = [parameter.grad.detach().clone() for parameter in parameters]
            consumer.synchronize()
            gathered = [None] * world_size
            dist.all_gather_object(
                gathered,
                [gradient.cpu() for gradient in consumed],
            )
            for parameter_index in range(len(parameters)):
                torch.testing.assert_close(
                    gathered[0][parameter_index],
                    gathered[1][parameter_index],
                    msg=lambda message: (
                        f"iteration={iteration}, parameter={parameter_index}: {message}"
                    ),
                )
            optimizer.step()
            state.commit_step()
            optimizer.zero_grad(set_to_none=True)
            assert not state._active_contexts
            churn = [torch.empty(256 * 1024, device=device) for _ in range(16)]
            del churn

        signature = observer.signature()
        signatures = [None] * world_size
        dist.all_gather_object(signatures, signature)
        assert signatures == [signatures[0]] * world_size
        assert any(
            event.category == "greedylore_hook/score_plus_aux_allreduce"
            for event in observer.events
        )
        assert any(
            event.category == "greedylore_hook/factor_allreduce"
            for event in observer.events
        )
        assert all(event.dtype == dtype_name for event in observer.events)
        assert all(event.category != "greedylore_hook/seed" for event in observer.events)
    finally:
        hook_module._all_reduce_future = original_all_reduce
        set_active_observer(None)
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_compressed_nccl_hook_survives_rebuild_accumulation_delay_and_allocator_churn():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    process_context = mp.spawn(
        _compressed_stress_worker,
        args=(2, _free_port(), "float32"),
        nprocs=2,
        join=False,
    )
    try:
        assert _join_with_timeout(process_context, timeout=75)
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.multi_gpu
def test_bfloat16_compressed_nccl_hook_keeps_all_payloads_bucket_native():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    process_context = mp.spawn(
        _compressed_stress_worker,
        args=(2, _free_port(), "bfloat16"),
        nprocs=2,
        join=False,
    )
    try:
        assert _join_with_timeout(process_context, timeout=75)
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
