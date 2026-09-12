"""Distributed-checkpoint continuation tests for Sparse-K residual state."""

import os
import socket
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[1]


class Loader:
    def __init__(self):
        self.position = 0

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict):
        self.position = state_dict["position"]


class TwoMatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.arange(16.0).view(4, 4) / 100)
        self.second = torch.nn.Parameter(torch.arange(16.0).view(4, 4) / 200)

    def forward(self, first_multiplier, second_multiplier):
        return (self.first * first_multiplier).sum() + (
            self.second * second_multiplier
        ).sum()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build(method, bucket_cap_mb):
    from dion.sparse_k import SparseKConfig
    from dion.sparse_k_ddp_hook import (
        SparseKDDPParameterSpec,
        SparseKDDPState,
        sparse_k_ddp_hook,
    )

    ddp = DDP(TwoMatrixModel(), bucket_cap_mb=bucket_cap_mb)
    parameters = list(ddp.module.parameters())
    state = SparseKDDPState(
        process_group=ddp.process_group,
        fingerprint="9" * 64,
        parameter_specs=[
            SparseKDDPParameterSpec(parameters[0], "first", 0, "sparse_matrix"),
            SparseKDDPParameterSpec(parameters[1], "second", 1, "sparse_matrix"),
        ],
        optimizer_parameters=parameters,
        config=SparseKConfig(
            method=method,
            ratio=0.25,
            seed=37,
            start_compress_step=0,
            error_feedback="ef14",
        ),
    )
    ddp.register_comm_hook(state, sparse_k_ddp_hook)
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    return ddp, optimizer, state


def _run_step(ddp, optimizer, state, rank, step):
    state.begin_step()
    base = torch.arange(1.0, 17.0).view(4, 4)
    ddp(base * (rank + 1) * step, base.flip(0) * (rank + 2) * step).backward()
    state.finish_step()
    optimizer.step()
    state.commit_step()
    ddp.zero_grad(set_to_none=True)


def _snapshot(ddp, state):
    return {
        "parameters": [
            parameter.detach().clone() for parameter in ddp.module.parameters()
        ],
        "residuals": [
            state.parameter_state(parameter).residual.detach().clone()
            for parameter in ddp.module.parameters()
        ],
        "step": state.committed_step,
    }


def _assert_snapshot_equal(actual, expected):
    assert actual["step"] == expected["step"]
    for actual_tensor, expected_tensor in zip(
        actual["parameters"] + actual["residuals"],
        expected["parameters"] + expected["residuals"],
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def _worker(rank, world_size, port, checkpoint_dir, method):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import train

        ddp, optimizer, state = _build(method, bucket_cap_mb=25)
        manager = train.CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=ddp,
            optimizer=optimizer,
            train_loader=Loader(),
            val_loader=Loader(),
            extra_stateful={"sparse_k_compressor": state},
        )
        for step in range(1, 4):
            _run_step(ddp, optimizer, state, rank, step)
        saved = _snapshot(ddp, state)
        manager.save(step=3)
        for step in range(4, 6):
            _run_step(ddp, optimizer, state, rank, step)
        uninterrupted = _snapshot(ddp, state)

        rebuilt_ddp, rebuilt_optimizer, rebuilt_state = _build(
            method, bucket_cap_mb=0.0001
        )
        rebuilt_manager = train.CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=rebuilt_ddp,
            optimizer=rebuilt_optimizer,
            train_loader=Loader(),
            val_loader=Loader(),
            extra_stateful={"sparse_k_compressor": rebuilt_state},
        )
        rebuilt_manager.load()
        _assert_snapshot_equal(_snapshot(rebuilt_ddp, rebuilt_state), saved)
        for step in range(4, 6):
            _run_step(rebuilt_ddp, rebuilt_optimizer, rebuilt_state, rank, step)
        _assert_snapshot_equal(_snapshot(rebuilt_ddp, rebuilt_state), uninterrupted)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method", ["randk", "topk"])
def test_two_rank_dcp_round_trip_preserves_residual_and_continuation(method):
    with tempfile.TemporaryDirectory(prefix="sparse-k-dcp-") as root:
        checkpoint_dir = str(Path(root, "checkpoint"))
        mp.spawn(
            _worker,
            args=(2, _free_port(), checkpoint_dir, method),
            nprocs=2,
            join=True,
        )
