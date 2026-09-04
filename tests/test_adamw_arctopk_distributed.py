"""Two-rank Gloo coverage for ARC-TopK AdamW."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion import ArcTopKAdamW


def _distributed_worker(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        matrix = torch.nn.Parameter(torch.arange(12.0).reshape(4, 3))
        dense = torch.nn.Parameter(torch.arange(3.0))
        optimizer = ArcTopKAdamW(
            [
                {"params": [matrix], "arc_compress": True},
                {"params": [dense], "arc_compress": False},
            ],
            process_group=dist.group.WORLD,
            lr=0.01,
            weight_decay=0.0,
            arc_topk_ratio=0.5,
            arc_projection_rank=2,
            arc_eta=0.5,
            arc_seed=11,
        )

        # A missing matrix gradient on one rank is represented as zeros while
        # every rank still enters the same ARC collective sequence.
        matrix.grad = torch.full_like(matrix, float(rank + 1)) if rank == 0 else None
        dense.grad = torch.full_like(dense, float(rank + 1))
        optimizer.step()
        for parameter in (matrix, dense):
            maximum = parameter.detach().clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            minimum = parameter.detach().clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            if not torch.equal(maximum, minimum):
                raise AssertionError("parameter diverged across Gloo ranks")

        # Step 2 exercises the seeded ARC compression path with rank-specific
        # gradients and confirms the synchronized AdamW update stays identical.
        matrix.grad = torch.full_like(matrix, float(3 - rank))
        dense.grad = None if rank == 1 else torch.full_like(dense, 2.0)
        optimizer.step()
        for parameter in (matrix, dense):
            maximum = parameter.detach().clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            minimum = parameter.detach().clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            if not torch.equal(maximum, minimum):
                raise AssertionError("parameter diverged across Gloo ranks")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.name == "nt", reason="file init method is POSIX-only")
def test_arc_topk_adamw_two_rank_gloo_sync():
    fd, init_file = tempfile.mkstemp()
    os.close(fd)
    os.unlink(init_file)
    try:
        mp.spawn(
            _distributed_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
