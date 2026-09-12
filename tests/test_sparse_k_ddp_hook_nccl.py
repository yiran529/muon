"""Two-GPU NCCL smoke tests for Sparse-K hook stream visibility."""

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
    def __init__(self, parameter, gradient):
        self._parameter = parameter
        self._buffer = gradient.flatten().clone()

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


def _worker(rank, world_size, port, method, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        device = torch.device("cuda", rank)
        parameter = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        state = SparseKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="e" * 64,
            parameter_specs=[
                SparseKDDPParameterSpec(parameter, "matrix", 0, "sparse_matrix")
            ],
            optimizer_parameters=[parameter],
            config=SparseKConfig(
                method=method,
                ratio=0.5,
                seed=41,
                start_compress_step=0,
                error_feedback="ef14",
            ),
        )
        state.committed_step = 1
        gradient = torch.tensor(
            [[10.0, 8.0], [1.0, 0.0]] if rank == 0 else [[0.0, 9.0], [7.0, 1.0]],
            device=device,
        )
        bucket = FakeGradBucket(parameter, gradient)
        state.begin_step()
        sparse_k_ddp_hook(state, bucket).wait()
        state.finish_step()
        torch.cuda.synchronize(device)
        with open(os.path.join(output_dir, f"{method}-{rank}.txt"), "w") as handle:
            handle.write(
                ",".join(str(value) for value in bucket.buffer().cpu().tolist())
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("method", ["randk", "topk"])
def test_two_rank_nccl_sparse_k_smoke(method, tmp_path):
    mp.spawn(
        _worker,
        args=(2, _free_port(), method, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    first = (tmp_path / f"{method}-0.txt").read_text()
    second = (tmp_path / f"{method}-1.txt").read_text()
    assert first == second


class MatrixLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2, device=device))

    def forward(self, multiplier):
        return (self.weight * multiplier).sum()


def _real_ddp_worker(rank, world_size, port, method, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        device = torch.device("cuda", rank)
        ddp = DDP(MatrixLoss(device), device_ids=[rank])
        parameter = ddp.module.weight
        state = SparseKDDPState(
            process_group=ddp.process_group,
            fingerprint="a" * 64,
            parameter_specs=[
                SparseKDDPParameterSpec(parameter, "weight", 0, "sparse_matrix")
            ],
            optimizer_parameters=[parameter],
            config=SparseKConfig(
                method=method,
                ratio=0.5,
                seed=41,
                start_compress_step=0,
                error_feedback="noef",
            ),
        )
        state.committed_step = 1
        ddp.register_comm_hook(state, sparse_k_ddp_hook)
        multiplier = torch.tensor(
            [[10.0, 8.0], [1.0, 0.0]] if rank == 0 else [[0.0, 9.0], [7.0, 1.0]],
            device=device,
        )
        state.begin_step()
        ddp(multiplier).backward()
        state.finish_step()
        consumed_on_current_stream = parameter.grad.mul(2).add(1)
        with open(os.path.join(output_dir, f"real-{method}-{rank}.txt"), "w") as handle:
            handle.write(
                ",".join(
                    str(value)
                    for value in consumed_on_current_stream.flatten().cpu().tolist()
                )
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("method", ["randk", "topk"])
def test_real_nccl_ddp_consumes_hook_future_on_current_stream(method, tmp_path):
    mp.spawn(
        _real_ddp_worker,
        args=(2, _free_port(), method, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    first = (tmp_path / f"real-{method}-0.txt").read_text()
    second = (tmp_path / f"real-{method}-1.txt").read_text()
    assert first == second
    if method == "topk":
        assert [float(value) for value in first.split(",")] == [11.0, 18.0, 8.0, 1.0]
