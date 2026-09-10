"""Two-rank real-DDP ordering tests for the GreedyLore Future sequencer."""

import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.greedy_lore import GreedyLoreConfig
import dion.greedy_lore_ddp_hook as hook_module
from dion.greedy_lore_ddp_hook import GreedyLoreDDPParameterSpec, GreedyLoreDDPState


class _ManyParameterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(16, 16, bias=False) for _ in range(8)]
        )

    def forward(self, inputs):
        for layer in self.layers:
            inputs = torch.tanh(layer(inputs))
        return inputs.sum()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reduced_tensor(future):
    value = future.value()
    return value[0] if isinstance(value, (tuple, list)) else value


def _join_with_timeout(process_context, timeout):
    deadline = time.monotonic() + timeout
    while not process_context.join(timeout=max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            return False
    return True


def _worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        model = _ManyParameterModel()
        ddp = DDP(model, bucket_cap_mb=0.0005)
        parameters = list(model.parameters())
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="c" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(parameter, name, index, "matrix")
                for index, (name, parameter) in enumerate(model.named_parameters())
            ],
            optimizer_parameters=parameters,
            config=GreedyLoreConfig(rank=1),
        )
        per_iteration_signatures = []

        def hook(hook_state, bucket):
            context = hook_state.note_bucket(bucket)

            def launch(current):
                if rank == (current.context_id % world_size):
                    time.sleep(0.002)
                signature.append(
                    tuple(
                        hook_state._specs_by_parameter[id(parameter)].stable_name
                        for parameter in current.parameters
                    )
                )
                work = dist.all_reduce(
                    current.buffer,
                    group=dist.group.WORLD,
                    async_op=True,
                )
                return work.get_future().then(
                    lambda future: _reduced_tensor(future).div_(world_size)
                )

            return hook_module.enqueue_bucket_chain(hook_state, context, launch)

        ddp.register_comm_hook(state, hook)
        for iteration in range(4):
            signature = []
            state.begin_step()
            ddp(torch.full((4, 16), float(rank + iteration + 1))).backward()
            state.finish_step()
            state.commit_step()
            per_iteration_signatures.append(signature)
            ddp.zero_grad(set_to_none=True)

        assert len(per_iteration_signatures[-1]) >= 2
        gathered = [None] * world_size
        dist.all_gather_object(gathered, per_iteration_signatures[-1])
        assert gathered == [gathered[0]] * world_size
    finally:
        dist.destroy_process_group()


def test_two_rank_rebuilt_ddp_buckets_share_one_global_launch_order():
    process_context = mp.spawn(
        _worker,
        args=(2, _free_port()),
        nprocs=2,
        join=False,
    )
    try:
        assert _join_with_timeout(process_context, timeout=45)
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
