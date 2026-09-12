"""Two-rank Gloo tests for Sparse-K communication semantics."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.sparse_k import SparseKConfig
from dion.sparse_k_ddp_hook import (
    SparseKDDPParameterSpec,
    SparseKDDPState,
    sparse_k_ddp_hook,
)


class FakeGradBucket:
    def __init__(self, parameters, gradients):
        self._parameters = parameters
        self._buffer = torch.cat([gradient.flatten() for gradient in gradients])
        self._gradients = []
        offset = 0
        for parameter in parameters:
            self._gradients.append(
                self._buffer[offset : offset + parameter.numel()].view_as(parameter)
            )
            offset += parameter.numel()

    def parameters(self):
        return self._parameters

    def gradients(self):
        return self._gradients

    def buffer(self):
        return self._buffer


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, method, error_feedback, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        matrix = torch.nn.Parameter(torch.zeros(2, 2))
        auxiliary = torch.nn.Parameter(torch.zeros(2))
        state = SparseKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="d" * 64,
            parameter_specs=[
                SparseKDDPParameterSpec(matrix, "matrix", 0, "sparse_matrix"),
                SparseKDDPParameterSpec(auxiliary, "aux", 1, "dense_aux"),
            ],
            optimizer_parameters=[matrix, auxiliary],
            config=SparseKConfig(
                method=method,
                ratio=0.5,
                seed=71,
                start_compress_step=0,
                error_feedback=error_feedback,
            ),
        )
        state.committed_step = 1
        matrix_gradients = (
            torch.tensor([[10.0, 8.0], [1.0, 0.0]])
            if rank == 0
            else torch.tensor([[0.0, 9.0], [7.0, 1.0]])
        )
        bucket = FakeGradBucket(
            [matrix, auxiliary],
            [matrix_gradients, torch.tensor([rank + 1.0, rank + 3.0])],
        )
        state.begin_step()
        sparse_k_ddp_hook(state, bucket).wait()
        state.finish_step()
        state.commit_step()
        parameter_state = state.parameter_state(matrix)
        queue.put(
            (
                rank,
                bucket.buffer().tolist(),
                parameter_state.last_support.tolist(),
                (
                    None
                    if parameter_state.residual is None
                    else parameter_state.residual.flatten().tolist()
                ),
            )
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run(method, error_feedback):
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(
        _worker,
        args=(2, _free_port(), method, error_feedback, queue),
        nprocs=2,
        join=True,
    )
    return sorted(queue.get() for _ in range(2))


@pytest.mark.parametrize("error_feedback", ["ef14", "noef"])
def test_topk_allgathers_rank_local_supports_and_accumulates_overlap(error_feedback):
    results = _run("topk", error_feedback)

    assert results[0][1] == results[1][1] == [5.0, 8.5, 3.5, 0.0, 1.5, 3.5]
    assert set(results[0][2]) == {0, 1}
    assert set(results[1][2]) == {1, 2}
    if error_feedback == "ef14":
        assert results[0][3] == [0.0, 0.0, 1.0, 0.0]
        assert results[1][3] == [0.0, 0.0, 0.0, 1.0]
    else:
        assert results[0][3] is results[1][3] is None


def test_randk_uses_identical_support_and_allreduces_only_selected_values():
    results = _run("randk", "noef")

    assert results[0][1] == results[1][1]
    assert results[0][2] == results[1][2]
    assert len(results[0][2]) == 2
    assert results[0][1][-2:] == [1.5, 3.5]


def _real_ddp_worker(rank, world_size, port, method, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(3)
        ddp = DDP(torch.nn.Linear(4, 4, bias=True))
        parameters = list(ddp.module.parameters())
        state = SparseKDDPState(
            process_group=ddp.process_group,
            fingerprint="f" * 64,
            parameter_specs=[
                SparseKDDPParameterSpec(parameters[0], "weight", 0, "sparse_matrix"),
                SparseKDDPParameterSpec(parameters[1], "bias", 1, "dense_aux"),
            ],
            optimizer_parameters=parameters,
            config=SparseKConfig(
                method=method,
                ratio=0.5,
                seed=19,
                start_compress_step=0,
                error_feedback="ef14",
            ),
        )
        ddp.register_comm_hook(state, sparse_k_ddp_hook)
        for step in range(2):
            state.begin_step()
            for micro_step in range(2):
                context = ddp.no_sync() if micro_step == 0 else torch.enable_grad()
                with context:
                    inputs = torch.arange(8.0).view(2, 4) * (rank + 1) * (step + 1)
                    ddp(inputs).sum().div_(2).backward()
            state.finish_step()
            state.commit_step()
            if step == 1:
                queue.put(
                    (
                        rank,
                        [
                            gradient.tolist()
                            for gradient in (parameters[0].grad, parameters[1].grad)
                        ],
                    )
                )
            ddp.zero_grad(set_to_none=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method", ["randk", "topk"])
def test_real_ddp_hook_supports_two_microstep_gradient_accumulation(method):
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(
        _real_ddp_worker,
        args=(2, _free_port(), method, queue),
        nprocs=2,
        join=True,
    )
    results = sorted(queue.get() for _ in range(2))
    assert results[0][1] == results[1][1]
