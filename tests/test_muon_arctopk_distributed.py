"""Two-rank consistency tests for ARC-TopK-EF21M-Muon."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion import ArcTopKMuon
from dion.arc_topk_layout import ArcTopKLayoutMismatch
from dion.collective_observer import CollectiveObserver, set_active_observer


def _identity_orthogonalizer(x, epsilon=None):
    return x


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _assert_same_across_ranks(tensor: torch.Tensor, world_size: int) -> None:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    for other in gathered[1:]:
        torch.testing.assert_close(other, gathered[0])


def _matrix_gradient(rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.tensor(
            [[2.0, 0.0, 1.0], [0.0, 4.0, 2.0], [6.0, 2.0, 0.0], [1.0, 3.0, 5.0]]
        )
    return torch.tensor(
        [[0.0, 2.0, 3.0], [4.0, 0.0, 2.0], [2.0, 6.0, 4.0], [5.0, 1.0, 3.0]]
    )


def _worker(
    rank: int,
    world_size: int,
    port: int,
    scalar_algorithm: str,
    missing_matrix_gradient: bool,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        matrix_initial = torch.arange(12.0).reshape(4, 3)
        matrix = torch.nn.Parameter(matrix_initial.clone())
        param_groups = [{"params": [matrix]}]

        scalar = None
        if scalar_algorithm != "none":
            scalar = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            param_groups.append(
                {
                    "params": [scalar],
                    "algorithm": scalar_algorithm,
                    "lr": 0.125,
                    "betas": (0.5, 0.75),
                    "weight_decay": 0.0,
                }
            )

        optimizer = ArcTopKMuon(
            param_groups,
            distributed_mesh=dist.group.WORLD,
            arc_parameter_names={
                matrix: "matrix",
                **({scalar: "scalar"} if scalar is not None else {}),
            },
            lr=0.125,
            mu=0.0,
            weight_decay=0.0,
            nesterov=False,
            adjust_lr=None,
            newton_schulz_func=_identity_orthogonalizer,
            arc_topk_ratio=1.0,
            arc_projection_rank=2,
            arc_eta=1.0,
            arc_seed=23,
        )
        if not (missing_matrix_gradient and rank == 1):
            matrix.grad = _matrix_gradient(rank)
        if scalar is not None:
            scalar.grad = torch.tensor(
                [2.0, 4.0, 6.0, 8.0]
                if rank == 0
                else [-8.0, -6.0, -4.0, -2.0]
            )

        optimizer.step()

        _assert_same_across_ranks(matrix.detach(), world_size)
        matrix_state = optimizer.state[matrix]
        _assert_same_across_ranks(matrix_state["momentum"], world_size)
        _assert_same_across_ranks(matrix_state["arc_g_global"], world_size)

        expected_gradient = _matrix_gradient(0)
        if not missing_matrix_gradient:
            expected_gradient = (expected_gradient + _matrix_gradient(1)) / 2
        else:
            expected_gradient = expected_gradient / 2
        expected_matrix = matrix_initial - 0.125 * expected_gradient.to(torch.bfloat16).float()
        torch.testing.assert_close(matrix, expected_matrix)

        if scalar is not None:
            _assert_same_across_ranks(scalar.detach(), world_size)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scalar_algorithm,missing_matrix_gradient",
    [
        ("none", False),
        ("none", True),
        ("lion", False),
        ("adamw", False),
    ],
)
def test_arc_topk_muon_two_rank_consistency(
    scalar_algorithm,
    missing_matrix_gradient,
):
    mp.spawn(
        _worker,
        args=(2, _free_port(), scalar_algorithm, missing_matrix_gradient),
        nprocs=2,
        join=True,
    )


def _missing_parameter_names_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        parameter = torch.nn.Parameter(torch.zeros(4, 3))
        with pytest.raises(ValueError, match="arc_parameter_names"):
            ArcTopKMuon(
                [parameter],
                distributed_mesh=dist.group.WORLD,
                newton_schulz_func=_identity_orthogonalizer,
            )
    finally:
        dist.destroy_process_group()


def test_distributed_arc_topk_muon_requires_stable_parameter_names():
    mp.spawn(
        _missing_parameter_names_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def _layout_validation_count_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
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
        parameter = torch.nn.Parameter(torch.zeros(4, 3))
        optimizer = ArcTopKMuon(
            [parameter],
            distributed_mesh=dist.group.WORLD,
            arc_parameter_names={parameter: "matrix"},
            lr=0.125,
            mu=0.0,
            weight_decay=0.0,
            nesterov=False,
            adjust_lr=None,
            newton_schulz_func=_identity_orthogonalizer,
            arc_topk_ratio=0.5,
            arc_projection_rank=2,
            arc_eta=1.0,
            arc_seed=23,
            arc_start_compress_step=0,
        )
        for step in range(3):
            parameter.grad = _matrix_gradient(rank) + step
            optimizer.step()

        categories = [event.category for event in observer.events]
        assert categories.count("arc/layout_validation") == 1
        assert "arc/seed" not in categories
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_layout_validation_runs_once_across_three_optimizer_steps():
    mp.spawn(
        _layout_validation_count_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


def _optimizer_layout_mismatch_worker(
    rank: int,
    world_size: int,
    port: int,
    mismatch: str,
) -> None:
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
        first = torch.nn.Parameter(torch.zeros(4, 3))
        second = torch.nn.Parameter(torch.ones(4, 3))
        groups = (
            [{"params": [first, second]}]
            if rank == 0 or mismatch == "seed"
            else [{"params": [first]}, {"params": [second]}]
        )
        seed = 24 if mismatch == "seed" and rank == 1 else 23
        message = None
        try:
            ArcTopKMuon(
                groups,
                distributed_mesh=dist.group.WORLD,
                arc_parameter_names={first: "first", second: "second"},
                newton_schulz_func=_identity_orthogonalizer,
                arc_seed=seed,
            )
        except ArcTopKLayoutMismatch as exc:
            message = str(exc)
        assert message is not None
        messages = [None] * world_size
        dist.all_gather_object(messages, message)
        assert messages == [messages[0]] * world_size
        categories = [event.category for event in observer.events]
        assert categories == ["arc/layout_validation"]
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", ["seed", "group_boundary"])
def test_optimizer_seed_and_real_task_layout_mismatch_fail_before_training(
    mismatch,
):
    mp.spawn(
        _optimizer_layout_mismatch_worker,
        args=(2, _free_port(), mismatch),
        nprocs=2,
        join=True,
    )
