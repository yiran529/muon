"""Two-rank numerical tests for ARC DDP bucket synchronization."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion import Muon
from dion.arc_topk_ddp_hook import (
    ArcTopKDDPParameterSpec,
    ArcTopKDDPState,
    arc_topk_ddp_hook,
)
from dion.arc_topk_sync import ArcTopKSyncConfig
from dion.collective_observer import CollectiveObserver, set_active_observer


class _ControlledGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.zeros(2, 2))
        self.matrix_two = torch.nn.Parameter(torch.zeros(2, 2))
        self.dense = torch.nn.Parameter(torch.zeros(2))

    def forward(self, matrix_source, matrix_two_source, dense_source):
        return (
            (self.matrix * matrix_source).sum()
            + (self.matrix_two * matrix_two_source).sum()
            + (self.dense * dense_source).sum()
        )


def _identity(value, epsilon=None):
    return value


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    observer = CollectiveObserver()
    set_active_observer(observer)
    try:
        model = _ControlledGradientModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        optimizer = Muon(
            [
                {"params": [model.matrix, model.matrix_two]},
                {
                    "params": [model.dense],
                    "algorithm": "adamw",
                    "lr": 0.05,
                    "betas": (0.0, 0.0),
                    "weight_decay": 0.0,
                },
            ],
            distributed_mesh=dist.group.WORLD,
            lr=0.05,
            mu=0.0,
            weight_decay=0.0,
            nesterov=False,
            adjust_lr=None,
            newton_schulz_func=_identity,
        )
        state = ArcTopKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="f" * 64,
            parameter_specs=[
                ArcTopKDDPParameterSpec(model.matrix, "matrix", 0, "arc_matrix"),
                ArcTopKDDPParameterSpec(
                    model.matrix_two, "matrix_two", 1, "arc_matrix"
                ),
                ArcTopKDDPParameterSpec(model.dense, "dense", 2, "dense_aux"),
            ],
            optimizer_parameters=[model.matrix, model.matrix_two, model.dense],
            config=ArcTopKSyncConfig(
                ratio=1.0,
                projection_rank=2,
                eta=0.25,
                seed=17,
                start_compress_step=2,
            ),
        )
        ddp.register_comm_hook(state, arc_topk_ddp_hook)
        expected_tracker = torch.zeros_like(model.matrix)
        expected_tracker_two = torch.zeros_like(model.matrix_two)

        for step in range(1, 4):
            matrix_gradient = torch.tensor(
                [[rank + step, 2.0 * step], [3.0 + rank, 4.0 - step]]
            )
            matrix_two_gradient = matrix_gradient + 5.0
            dense_gradient = torch.tensor([rank + step, rank - step], dtype=torch.float32)
            state.begin_step()
            ddp(matrix_gradient, matrix_two_gradient, dense_gradient).backward()
            state.finish_step()

            expected_tracker = (
                matrix_gradient
                if step == 1
                else expected_tracker.lerp(matrix_gradient, 0.25)
            )
            expected_tracker_two = (
                matrix_two_gradient
                if step == 1
                else expected_tracker_two.lerp(matrix_two_gradient, 0.25)
            )
            expected_dense = torch.tensor(
                [step + 0.5, 0.5 - step], dtype=torch.float32
            )

            torch.testing.assert_close(
                state.parameter_state(model.matrix).h_local,
                expected_tracker,
            )
            torch.testing.assert_close(
                state.parameter_state(model.matrix_two).h_local,
                expected_tracker_two,
            )
            for parameter in (model.matrix, model.matrix_two):
                averaged_tracker = state.parameter_state(parameter).h_local.clone()
                dist.all_reduce(averaged_tracker)
                averaged_tracker.div_(world_size)
                torch.testing.assert_close(parameter.grad, averaged_tracker)
            torch.testing.assert_close(model.dense.grad, expected_dense)
            optimizer.step()
            state.commit_step()
            for parameter in (model.matrix, model.matrix_two):
                gathered = [torch.empty_like(parameter) for _ in range(world_size)]
                dist.all_gather(gathered, parameter)
                torch.testing.assert_close(gathered[0], gathered[1])
            optimizer.zero_grad(set_to_none=True)

        categories = [event.category for event in observer.events]
        assert categories.count("arc_hook/dense") == 3
        assert "arc/dense_uncompressed" not in categories
        assert "muon/result_collective" in categories
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_two_rank_full_support_hook_matches_tracker_and_dense_oracles():
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
