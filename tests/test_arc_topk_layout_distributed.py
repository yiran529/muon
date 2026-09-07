"""Two-rank tests for ARC layout fingerprint validation."""

import os
import socket
from dataclasses import replace
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

import dion.arc_topk_layout as layout
from dion.arc_topk_sync import ArcTopKSyncConfig


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, mismatch: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        config = ArcTopKSyncConfig(
            ratio=0.5,
            projection_rank=2,
            eta=0.25,
            seed=17,
            start_compress_step=0,
        )
        parameters = (
            layout.ArcParameterDescriptor(
                stable_name="matrix",
                stable_id=0,
                shape=(4, 3),
                dtype="float32",
                role="arc_matrix",
            ),
        )
        task = layout.ArcOptimizerTaskDescriptor(
            group_id=0,
            task_id=0,
            ordered_parameter_names=("matrix",),
            shape=(4, 3),
            dtype="float32",
            config=config,
        )
        base_seed = 18 if mismatch == "seed" and rank == 1 else 17
        if mismatch == "role" and rank == 1:
            parameters = (replace(parameters[0], role="dense_aux"),)
        if mismatch == "task" and rank == 1:
            task = replace(task, ordered_parameter_names=("other",))
        fingerprint = layout.canonical_arc_fingerprint(
            base_seed=base_seed,
            config=config,
            group_ranks=(0, 1),
            parameters=parameters,
            optimizer_tasks=(task,),
        )

        if mismatch == "none":
            layout.validate_arc_fingerprint_across_ranks(
                fingerprint,
                dist.group.WORLD,
            )
            return

        message = None
        try:
            layout.validate_arc_fingerprint_across_ranks(
                fingerprint,
                dist.group.WORLD,
            )
        except layout.ArcTopKLayoutMismatch as exc:
            message = str(exc)
        assert message is not None
        messages = [None] * world_size
        dist.all_gather_object(messages, message)
        assert messages == [messages[0]] * world_size
        assert "rank 0" in message and "rank 1" in message
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", ["none", "seed", "role", "task"])
def test_two_rank_layout_validation_is_symmetric_and_does_not_hang(mismatch):
    mp.spawn(
        _worker,
        args=(2, _free_port(), mismatch),
        nprocs=2,
        join=True,
    )
