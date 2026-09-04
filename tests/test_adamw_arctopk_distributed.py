"""Two-rank Gloo coverage for ARC-TopK AdamW."""

import os
import copy
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
        matrix_grad_ref = matrix.grad
        dense_grad_ref = dense.grad
        matrix_grad_value = None if matrix.grad is None else matrix.grad.clone()
        dense_grad_value = dense.grad.clone()
        optimizer.step()
        assert matrix.grad is matrix_grad_ref
        assert dense.grad is dense_grad_ref
        if matrix_grad_value is not None:
            torch.testing.assert_close(matrix.grad, matrix_grad_value)
        torch.testing.assert_close(dense.grad, dense_grad_value)
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
        matrix_grad_ref = matrix.grad
        dense_grad_ref = dense.grad
        matrix_grad_value = matrix.grad.clone()
        dense_grad_value = None if dense.grad is None else dense.grad.clone()
        optimizer.step()
        assert matrix.grad is matrix_grad_ref
        assert dense.grad is dense_grad_ref
        torch.testing.assert_close(matrix.grad, matrix_grad_value)
        if dense_grad_value is not None:
            torch.testing.assert_close(dense.grad, dense_grad_value)
        for parameter in (matrix, dense):
            maximum = parameter.detach().clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            minimum = parameter.detach().clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            if not torch.equal(maximum, minimum):
                raise AssertionError("parameter diverged across Gloo ranks")

        # Ratio=1/eta=1 must be exactly the dense-average AdamW update, including
        # the first and second moments.
        initial = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        ratio_one_param = torch.nn.Parameter(initial.clone())
        ratio_one_optimizer = ArcTopKAdamW(
            [{"params": [ratio_one_param], "arc_compress": True}],
            process_group=dist.group.WORLD,
            lr=0.01,
            betas=(0.8, 0.95),
            eps=1e-7,
            weight_decay=0.03,
            arc_topk_ratio=1.0,
            arc_projection_rank=2,
            arc_eta=1.0,
        )
        local_gradient = torch.full_like(ratio_one_param, float(rank + 1))
        ratio_one_param.grad = local_gradient
        ratio_one_optimizer.step()
        averaged = torch.full_like(initial, 1.5)
        momentum = (1 - 0.8) * averaged
        variance = (1 - 0.95) * averaged.square()
        bias_corrected_momentum = momentum / (1 - 0.8)
        bias_corrected_variance = variance / (1 - 0.95)
        expected = initial * (1 - 0.01 * 0.03)
        expected -= 0.01 * bias_corrected_momentum / (
            bias_corrected_variance.sqrt() + 1e-7
        )
        torch.testing.assert_close(ratio_one_param, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            ratio_one_optimizer.state[ratio_one_param]["momentum"], momentum
        )
        torch.testing.assert_close(
            ratio_one_optimizer.state[ratio_one_param]["variance"], variance
        )

        # A checkpoint resumed on each rank must produce the same next update.
        resumed_param = torch.nn.Parameter(ratio_one_param.detach().clone())
        resumed_optimizer = ArcTopKAdamW(
            [{"params": [resumed_param], "arc_compress": True}],
            process_group=dist.group.WORLD,
            lr=0.01,
            betas=(0.8, 0.95),
            eps=1e-7,
            weight_decay=0.03,
            arc_topk_ratio=1.0,
            arc_projection_rank=2,
            arc_eta=1.0,
        )
        resumed_optimizer.load_state_dict(copy.deepcopy(ratio_one_optimizer.state_dict()))
        next_gradient = torch.full_like(ratio_one_param, float(2 * rank + 1))
        ratio_one_param.grad = next_gradient.clone()
        resumed_param.grad = next_gradient.clone()
        ratio_one_optimizer.step()
        resumed_optimizer.step()
        torch.testing.assert_close(ratio_one_param, resumed_param)
        for key in ("momentum", "variance", "step_dev"):
            torch.testing.assert_close(
                ratio_one_optimizer.state[ratio_one_param][key],
                resumed_optimizer.state[resumed_param][key],
            )
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
