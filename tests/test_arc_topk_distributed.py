"""Two-rank CPU tests for the complete distributed ARC-TopK algorithm."""

import math
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.arc_topk import arc_topk_ef21m_async
from dion.opt_utils import AsyncRuntime, AsyncTask


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gradient_for_case(rank: int, case: str) -> torch.Tensor:
    if case == "different":
        if rank == 0:
            return torch.tensor(
                [[1.0, 3.0, 2.0], [4.0, 2.0, 1.0], [8.0, 1.0, 2.0], [2.0, 6.0, 4.0]]
            )
        return torch.tensor(
            [[3.0, 1.0, 4.0], [2.0, 5.0, 3.0], [6.0, 2.0, 1.0], [1.0, 4.0, 8.0]]
        )
    if case == "rank1_zero":
        if rank == 0:
            return torch.tensor(
                [[2.0, 4.0, 1.0], [7.0, 1.0, 3.0], [3.0, 5.0, 9.0], [6.0, 2.0, 8.0]]
            )
        return torch.zeros(4, 3)
    raise AssertionError(f"unknown case {case}")


def _manual_expected(case: str, ratio: float, eta: float, synchronized_seed: int):
    gradient0 = _gradient_for_case(0, case).unsqueeze(0)
    gradient1 = _gradient_for_case(1, case).unsqueeze(0)
    delta0 = eta * gradient0
    delta1 = eta * gradient1

    generator = torch.Generator(device="cpu").manual_seed(synchronized_seed)
    projection = torch.randn(1, 3, 2, generator=generator)
    sketch0 = torch.bmm(delta0, projection) / math.sqrt(2.0)
    sketch1 = torch.bmm(delta1, projection) / math.sqrt(2.0)
    scores = ((sketch0 + sketch1) / 2).square().sum(dim=-1)
    k = math.ceil(ratio * 4)
    indices = scores.topk(k=k, dim=-1, sorted=True).indices
    expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
    selected0 = torch.gather(delta0, 1, expanded)
    selected1 = torch.gather(delta1, 1, expanded)
    averaged_selected = (selected0 + selected1) / 2
    expected_global = torch.zeros_like(delta0).scatter(1, expanded, averaged_selected)
    expected_local = [
        torch.zeros_like(delta0).scatter(1, expanded, selected0),
        torch.zeros_like(delta1).scatter(1, expanded, selected1),
    ]
    return expected_global, expected_local


def _worker(rank: int, world_size: int, port: int, case: str, ratio: float) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        gradient = _gradient_for_case(rank, case)
        tracker = torch.zeros_like(gradient)
        local_estimate = torch.zeros_like(gradient)
        global_estimate = torch.zeros_like(gradient)
        result = {}

        def task():
            result["gradients"] = yield from arc_topk_ef21m_async(
                gradients=[gradient],
                trackers=[tracker],
                local_estimates=[local_estimate],
                global_estimates=[global_estimate],
                process_group=dist.group.WORLD,
                ratio=ratio,
                projection_rank=2,
                eta=0.5,
                base_seed=19 + 100 * rank,
                step=2,
                task_index=3,
            )

        runtime = AsyncRuntime(iter([AsyncTask(task())]), max_concurrent_tasks=1)
        runtime.run()

        synchronized_seed = 19 + 2 * 1_000_003 + 3
        expected_global, expected_local = _manual_expected(
            case, ratio, eta=0.5, synchronized_seed=synchronized_seed
        )
        torch.testing.assert_close(tracker, 0.5 * gradient)
        torch.testing.assert_close(local_estimate, expected_local[rank].squeeze(0))
        torch.testing.assert_close(global_estimate, expected_global.squeeze(0))
        torch.testing.assert_close(result["gradients"][0], global_estimate)

        gathered = [torch.empty_like(global_estimate) for _ in range(world_size)]
        dist.all_gather(gathered, global_estimate)
        torch.testing.assert_close(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "case,ratio",
    [
        ("different", 0.5),
        ("different", 1.0),
        ("rank1_zero", 0.5),
    ],
)
def test_arc_topk_two_rank_collectives(case, ratio):
    mp.spawn(
        _worker,
        args=(2, _free_port(), case, ratio),
        nprocs=2,
        join=True,
    )
