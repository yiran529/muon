"""Two-rank GreedyLore layout validation contracts."""

import os
import socket
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.greedy_lore import GreedyLoreConfig
from dion.greedy_lore_ddp_hook import (
    GreedyLoreDDPParameterSpec,
    GreedyLoreDDPState,
    GreedyLoreReplicatedStateMismatch,
)
import dion.greedy_lore_layout as layout


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, mismatch):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        config = GreedyLoreConfig(rank=2)
        parameters = (
            layout.GreedyLoreParameterDescriptor(
                "first", 0, (4, 3), "float32", "matrix"
            ),
            layout.GreedyLoreParameterDescriptor(
                "second", 1, (4,), "float32", "dense_aux"
            ),
        )
        if rank == 1 and mismatch == "basis_sync":
            config = replace(config, basis_sync="broadcast")
        elif rank == 1 and mismatch == "rank":
            config = replace(config, rank=1)
        elif rank == 1 and mismatch == "parameter_order":
            parameters = tuple(reversed(parameters))
        fingerprint = layout.canonical_greedy_lore_fingerprint(
            config=config,
            group_ranks=(0, 1),
            parameters=parameters,
        )

        if mismatch == "none":
            layout.validate_greedy_lore_fingerprint_across_ranks(
                fingerprint, dist.group.WORLD
            )
            return

        message = None
        try:
            layout.validate_greedy_lore_fingerprint_across_ranks(
                fingerprint, dist.group.WORLD
            )
        except layout.GreedyLoreLayoutMismatch as exc:
            message = str(exc)
        assert message is not None
        messages = [None] * world_size
        dist.all_gather_object(messages, message)
        assert messages == [messages[0]] * world_size
        assert "rank 0" in message and "rank 1" in message
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", ["none", "basis_sync", "rank", "parameter_order"])
def test_two_rank_layout_validation_is_symmetric_and_bounded(mismatch):
    mp.spawn(_worker, args=(2, _free_port(), mismatch), nprocs=2, join=True)


def _basis_worker(rank, world_size, port, mismatch):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        first = torch.nn.Parameter(torch.zeros(3, 4))
        second = torch.nn.Parameter(torch.zeros(2, 4))
        specs = (
            GreedyLoreDDPParameterSpec(first, "z-last", 0, "matrix"),
            GreedyLoreDDPParameterSpec(second, "a-first", 1, "matrix"),
        )
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="a" * 64,
            parameter_specs=specs,
            optimizer_parameters=(first, second),
            config=GreedyLoreConfig(rank=1),
        )
        target = state.parameter_state(first)
        if rank == 1 and mismatch == "basis":
            target.basis[0, 0] += 0.01
        elif rank == 1 and mismatch == "support":
            target.last_support[0] = 1

        if mismatch == "none":
            state.validate_replicated_basis_across_ranks()
            return
        with pytest.raises(GreedyLoreReplicatedStateMismatch, match=mismatch):
            state.validate_replicated_basis_across_ranks()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", ["none", "basis", "support"])
def test_two_rank_replicated_state_validation_is_symmetric(mismatch):
    mp.spawn(_basis_worker, args=(2, _free_port(), mismatch), nprocs=2, join=True)
