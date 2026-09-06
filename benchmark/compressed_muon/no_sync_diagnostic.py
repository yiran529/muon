"""Characterize DDP reducer behavior for three ``no_sync`` placements."""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


PATTERNS = ("backward_only", "dense_correct", "optimizer_correct")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reduced_tensor(future):
    value = future.value()
    return value[0] if isinstance(value, (list, tuple)) else value


def _worker(rank: int, world_size: int, port: int, pattern: str, output_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ddp = DDP(model)
        hook_state = {"calls": 0, "inputs": []}

        def hook(state, bucket):
            buffer = bucket.buffer()
            state["calls"] += 1
            state["inputs"].append(buffer.detach().clone().tolist())
            work = dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=True)
            return work.get_future().then(
                lambda future: _reduced_tensor(future).div_(world_size)
            )

        ddp.register_comm_hook(hook_state, hook)
        inputs = (
            (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 2.0]]))
            if rank == 0
            else (torch.tensor([[3.0, 0.0]]), torch.tensor([[0.0, 4.0]]))
        )

        for micro_step, input_tensor in enumerate(inputs):
            if pattern == "backward_only":
                loss = ddp(input_tensor).sum()
                with ddp.no_sync():
                    loss.backward()
            elif pattern == "dense_correct":
                context = ddp.no_sync() if micro_step == 0 else nullcontext()
                with context:
                    ddp(input_tensor).sum().backward()
            elif pattern == "optimizer_correct":
                with ddp.no_sync():
                    ddp(input_tensor).sum().backward()
            else:  # Guarded by the parent, retained for spawned-process safety.
                raise ValueError(f"unknown no_sync pattern: {pattern}")

        payload = {
            "rank": rank,
            "hook_calls": hook_state["calls"],
            "hook_inputs": hook_state["inputs"],
            "gradient": model.weight.grad.detach().flatten().tolist(),
        }
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_no_sync_case(pattern: str, world_size: int = 2) -> dict:
    """Run one real-DDP case and return rank-level reducer observations."""

    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, got {pattern!r}")
    if world_size != 2:
        raise ValueError(f"diagnostic requires world_size=2, got {world_size}")

    with tempfile.TemporaryDirectory(prefix=f"no-sync-{pattern}-") as output_dir:
        mp.spawn(
            _worker,
            args=(world_size, _free_port(), pattern, output_dir),
            nprocs=world_size,
            join=True,
        )
        rank_results = [
            json.loads(Path(output_dir, f"rank-{rank}.json").read_text())
            for rank in range(world_size)
        ]

    gradients = [result["gradient"] for result in rank_results]
    max_abs_range = max(
        max(values) - min(values) for values in zip(*gradients)
    )
    hook_calls = [result["hook_calls"] for result in rank_results]
    expected_calls = {
        "backward_only": [2, 2],
        "dense_correct": [1, 1],
        "optimizer_correct": [0, 0],
    }[pattern]
    expect_equal_gradients = pattern != "optimizer_correct"
    passed = hook_calls == expected_calls and (
        (max_abs_range == 0.0) if expect_equal_gradients else (max_abs_range > 0.0)
    )
    return {
        "pattern": pattern,
        "world_size": world_size,
        "hook_calls_per_rank": hook_calls,
        "pre_optimizer_gradients": gradients,
        "gradient_ranges": {"max_abs": max_abs_range},
        "rank_results": rank_results,
        "passed": passed,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = [run_no_sync_case(pattern) for pattern in PATTERNS]
    payload = {"schema_version": 1, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
