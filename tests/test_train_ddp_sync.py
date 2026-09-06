"""Distributed regression tests for the shared training micro-step."""

import json
import os
import socket
import tempfile

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from train import ddp_gradient_sync_context, forward_backward_micro_step


class _LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 2))

    def forward(self, x, _y):
        return (x @ self.weight.t()).sum()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reduced_tensor(future):
    value = future.value()
    return value[0] if isinstance(value, (list, tuple)) else value


def _worker(rank, world_size, port, optimizer_owns_gradient_sync, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        ddp = DDP(_LossModel())
        state = {"calls": 0}

        def hook(hook_state, bucket):
            hook_state["calls"] += 1
            work = dist.all_reduce(bucket.buffer(), async_op=True)
            return work.get_future().then(
                lambda future: _reduced_tensor(future).div_(world_size)
            )

        ddp.register_comm_hook(state, hook)
        inputs = (
            (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 2.0]]))
            if rank == 0
            else (torch.tensor([[3.0, 0.0]]), torch.tensor([[0.0, 4.0]]))
        )
        for micro_step, input_tensor in enumerate(inputs, start=1):
            forward_backward_micro_step(
                ddp,
                input_tensor,
                None,
                autocast_ctx=nullcontext(),
                micro_step=micro_step,
                grad_accum_steps=2,
                optimizer_owns_gradient_sync=optimizer_owns_gradient_sync,
            )

        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "hook_calls": state["calls"],
                    "gradient": ddp.module.weight.grad.detach().flatten().tolist(),
                }
            )
            + "\n"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_policy(optimizer_owns_gradient_sync: bool):
    with tempfile.TemporaryDirectory(prefix="train-ddp-sync-") as output_dir:
        mp.spawn(
            _worker,
            args=(2, _free_port(), optimizer_owns_gradient_sync, output_dir),
            nprocs=2,
            join=True,
        )
        return [
            json.loads(Path(output_dir, f"rank-{rank}.json").read_text())
            for rank in range(2)
        ]


def test_dense_policy_syncs_only_the_final_accumulation_micro_step():
    results = _run_policy(optimizer_owns_gradient_sync=False)

    assert [result["hook_calls"] for result in results] == [1, 1]
    assert [result["gradient"] for result in results] == [[1.0, 1.5], [1.0, 1.5]]


def test_optimizer_owned_policy_keeps_accumulated_gradients_rank_local():
    results = _run_policy(optimizer_owns_gradient_sync=True)

    assert [result["hook_calls"] for result in results] == [0, 0]
    assert [result["gradient"] for result in results] == [[0.5, 1.0], [1.5, 2.0]]


def test_ddp_sync_context_rejects_an_invalid_micro_step():
    with pytest.raises(ValueError, match="micro_step"):
        ddp_gradient_sync_context(
            _LossModel(),
            micro_step=0,
            grad_accum_steps=2,
            optimizer_owns_gradient_sync=False,
        )


def test_forward_backward_micro_step_runs_the_pre_backward_callback():
    model = _LossModel()
    calls = []

    train_loss, callback_result = forward_backward_micro_step(
        model,
        torch.tensor([[2.0, 4.0]]),
        None,
        autocast_ctx=nullcontext(),
        micro_step=1,
        grad_accum_steps=2,
        optimizer_owns_gradient_sync=False,
        before_backward=lambda: calls.append("called") or "next-batch",
    )

    assert train_loss.item() == 6.0
    assert callback_result == "next-batch"
    assert calls == ["called"]
    assert model.weight.grad.tolist() == [[1.0, 2.0]]
