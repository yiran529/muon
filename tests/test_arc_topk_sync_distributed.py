"""Two-rank CPU tests for the shared ARC-TopK synchronization adapter."""

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.arc_topk_sync import ArcTopKSyncConfig, synchronize_arc_batch_async
from dion.opt_utils import AsyncRuntime, AsyncTask


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_generator(generator):
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        param = torch.nn.Parameter(torch.zeros(4, 3))
        param.grad = torch.full_like(param, float(rank + 1))
        state = {}
        config = ArcTopKSyncConfig(
            ratio=1.0, projection_rank=2, eta=1.0, seed=19, start_compress_step=1
        )
        result = {}

        def task():
            result["gradients"] = yield from synchronize_arc_batch_async(
                params=[param],
                states=[state],
                process_group=dist.group.WORLD,
                config=config,
                step=1,
                task_index=0,
            )

        AsyncRuntime(iter([AsyncTask(task())]), max_concurrent_tasks=1).run()

        expected_first = torch.full_like(param, 1.5)
        torch.testing.assert_close(result["gradients"][0], expected_first)
        torch.testing.assert_close(state["arc_h_local"], param.grad)
        torch.testing.assert_close(state["arc_g_local"], param.grad)
        torch.testing.assert_close(state["arc_g_global"], expected_first)

        param.grad = torch.full_like(param, float(3 * (rank + 1)))

        def second_task():
            result["gradients"] = yield from synchronize_arc_batch_async(
                params=[param],
                states=[state],
                process_group=dist.group.WORLD,
                config=config,
                step=2,
                task_index=0,
            )

        AsyncRuntime(iter([AsyncTask(second_task())]), max_concurrent_tasks=1).run()
        torch.testing.assert_close(
            result["gradients"][0], torch.full_like(param, 4.5)
        )

        # A missing local gradient participates in exactly the same collectives.
        param.grad = None if rank == 1 else torch.full_like(param, 5.0)
        missing_state = {}
        result.clear()

        def third_task():
            result["gradients"] = yield from synchronize_arc_batch_async(
                params=[param],
                states=[missing_state],
                process_group=dist.group.WORLD,
                config=config,
                step=1,
                task_index=1,
            )

        AsyncRuntime(iter([AsyncTask(third_task())]), max_concurrent_tasks=1).run()
        torch.testing.assert_close(
            result["gradients"][0], torch.full_like(param, 2.5)
        )
        assert missing_state["arc_h_local"].shape == param.shape

        # Both shape groups can be synchronized in stable order with distinct indices.
        p0 = torch.zeros(4, 3)
        p1 = torch.zeros(2, 3)
        p0.grad = torch.full_like(p0, float(rank + 1))
        p1.grad = torch.full_like(p1, float(2 * (rank + 1)))
        states = [{}, {}]
        outputs = []
        for task_index, (group_param, group_state) in enumerate(
            zip(([p0], [p1]), states)
        ):
            outputs.append(
                _run_generator(
                    synchronize_arc_batch_async(
                        params=group_param,
                        states=[group_state],
                        process_group=dist.group.WORLD,
                        config=config,
                        step=1,
                        task_index=task_index,
                    )
                )
            )
        torch.testing.assert_close(outputs[0][0], torch.full_like(p0, 1.5))
        torch.testing.assert_close(outputs[1][0], torch.full_like(p1, 3.0))
    finally:
        dist.destroy_process_group()


def test_shared_arc_synchronization_two_rank_collectives():
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
