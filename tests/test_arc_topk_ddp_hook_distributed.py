"""Two-rank numerical tests for ARC DDP bucket synchronization."""

import os
import socket
import math
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion import Muon
from dion.arc_topk_ddp_hook import (
    ArcTopKDDPParameterSpec,
    ArcTopKDDPState,
    arc_topk_ddp_hook,
)
from dion.arc_topk_sync import ArcTopKSyncConfig
from dion.arc_topk import derive_arc_seed
from dion.collective_observer import CollectiveObserver, set_active_observer


class _ControlledGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.zeros(2, 2))
        self.matrix_two = torch.nn.Parameter(torch.zeros(2, 2))
        self.dense = torch.nn.Parameter(torch.zeros(2))

    def forward(self, matrix_source, matrix_two_source, dense_source):
        return (
            (self.matrix * matrix_source).sum()
            + (self.matrix_two * matrix_two_source).sum()
            + (self.dense * dense_source).sum()
        )


def _identity(value, epsilon=None):
    return value


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port):
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
        model = _ControlledGradientModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        optimizer = Muon(
            [
                {"params": [model.matrix, model.matrix_two]},
                {
                    "params": [model.dense],
                    "algorithm": "adamw",
                    "lr": 0.05,
                    "betas": (0.0, 0.0),
                    "weight_decay": 0.0,
                },
            ],
            distributed_mesh=dist.group.WORLD,
            lr=0.05,
            mu=0.0,
            weight_decay=0.0,
            nesterov=False,
            adjust_lr=None,
            newton_schulz_func=_identity,
        )
        state = ArcTopKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="f" * 64,
            parameter_specs=[
                ArcTopKDDPParameterSpec(model.matrix, "matrix", 0, "arc_matrix"),
                ArcTopKDDPParameterSpec(
                    model.matrix_two, "matrix_two", 1, "arc_matrix"
                ),
                ArcTopKDDPParameterSpec(model.dense, "dense", 2, "dense_aux"),
            ],
            optimizer_parameters=[model.matrix, model.matrix_two, model.dense],
            config=ArcTopKSyncConfig(
                ratio=1.0,
                projection_rank=2,
                eta=0.25,
                seed=17,
                start_compress_step=2,
            ),
        )
        ddp.register_comm_hook(state, arc_topk_ddp_hook)
        expected_tracker = torch.zeros_like(model.matrix)
        expected_tracker_two = torch.zeros_like(model.matrix_two)

        for step in range(1, 4):
            matrix_gradient = torch.tensor(
                [[rank + step, 2.0 * step], [3.0 + rank, 4.0 - step]]
            )
            matrix_two_gradient = matrix_gradient + 5.0
            dense_gradient = torch.tensor([rank + step, rank - step], dtype=torch.float32)
            state.begin_step()
            ddp(matrix_gradient, matrix_two_gradient, dense_gradient).backward()
            state.finish_step()

            expected_tracker = (
                matrix_gradient
                if step == 1
                else expected_tracker.lerp(matrix_gradient, 0.25)
            )
            expected_tracker_two = (
                matrix_two_gradient
                if step == 1
                else expected_tracker_two.lerp(matrix_two_gradient, 0.25)
            )
            expected_dense = torch.tensor(
                [step + 0.5, 0.5 - step], dtype=torch.float32
            )

            torch.testing.assert_close(
                state.parameter_state(model.matrix).h_local,
                expected_tracker,
            )
            torch.testing.assert_close(
                state.parameter_state(model.matrix_two).h_local,
                expected_tracker_two,
            )
            for parameter in (model.matrix, model.matrix_two):
                averaged_tracker = state.parameter_state(parameter).h_local.clone()
                dist.all_reduce(averaged_tracker)
                averaged_tracker.div_(world_size)
                torch.testing.assert_close(parameter.grad, averaged_tracker)
            torch.testing.assert_close(model.dense.grad, expected_dense)
            optimizer.step()
            state.commit_step()
            for parameter in (model.matrix, model.matrix_two):
                gathered = [torch.empty_like(parameter) for _ in range(world_size)]
                dist.all_gather(gathered, parameter)
                torch.testing.assert_close(gathered[0], gathered[1])
            optimizer.zero_grad(set_to_none=True)

        categories = [event.category for event in observer.events]
        assert categories.count("arc_hook/dense") == 3
        assert "arc/dense_uncompressed" not in categories
        assert "muon/result_collective" in categories
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_two_rank_full_support_hook_matches_tracker_and_dense_oracles():
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)


class _SparseControlledGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(3, 2))
        self.second = torch.nn.Parameter(torch.zeros(3, 2))
        self.dense = torch.nn.Parameter(torch.zeros(2))

    def forward(self, first_source, second_source, dense_source):
        return (
            (self.first * first_source).sum()
            + (self.second * second_source).sum()
            + (self.dense * dense_source).sum()
        )


def _sparse_gradients(rank, step):
    return (
        torch.arange(6.0).reshape(3, 2) + step + 2 * rank,
        torch.arange(6.0).reshape(3, 2) + 2 * step + 3 * rank,
        torch.tensor([step + rank, 2 * step - rank], dtype=torch.float32),
    )


def _manual_sparse_oracle(previous, gradients, *, step, stable_id, rows, columns):
    h_local, g_local, g_global = previous
    next_h = [
        gradients[rank]
        if step == 1
        else h_local[rank].lerp(gradients[rank], 0.5)
        for rank in range(2)
    ]
    if step == 1:
        return next_h, [value.clone() for value in next_h], sum(next_h) / 2, torch.arange(rows)

    deltas = [next_h[rank] - g_local[rank] for rank in range(2)]
    generator = torch.Generator().manual_seed(
        derive_arc_seed(base_seed=17, step=step, stable_task_id=stable_id)
    )
    projection = torch.randn(1, columns, 2, generator=generator)
    sketches = [
        torch.bmm(delta.unsqueeze(0), projection).squeeze(0) / math.sqrt(2.0)
        for delta in deltas
    ]
    k = math.ceil(rows * 0.5)
    support = ((sketches[0] + sketches[1]) / 2).square().sum(-1).topk(
        k, sorted=True
    ).indices
    compressed = []
    for delta in deltas:
        value = torch.zeros_like(delta)
        value[support] = delta[support]
        compressed.append(value)
    next_local = [g_local[rank] + compressed[rank] for rank in range(2)]
    next_global = g_global + (compressed[0] + compressed[1]) / 2
    return next_h, next_local, next_global, support


def _sparse_worker(rank, world_size, port):
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
        model = _SparseControlledGradientModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        optimizer = Muon(
            [
                {"params": [model.first, model.second]},
                {
                    "params": [model.dense],
                    "algorithm": "adamw",
                    "lr": 0.05,
                    "betas": (0.0, 0.0),
                    "weight_decay": 0.0,
                },
            ],
            distributed_mesh=dist.group.WORLD,
            lr=0.05,
            mu=0.0,
            weight_decay=0.0,
            nesterov=False,
            adjust_lr=None,
            newton_schulz_func=_identity,
        )
        state = ArcTopKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="1" * 64,
            parameter_specs=[
                ArcTopKDDPParameterSpec(model.first, "first", 0, "arc_matrix"),
                ArcTopKDDPParameterSpec(model.second, "second", 1, "arc_matrix"),
                ArcTopKDDPParameterSpec(model.dense, "dense", 2, "dense_aux"),
            ],
            optimizer_parameters=[model.first, model.second, model.dense],
            config=ArcTopKSyncConfig(
                ratio=0.5,
                projection_rank=2,
                eta=0.5,
                seed=17,
                start_compress_step=0,
            ),
        )
        ddp.register_comm_hook(state, arc_topk_ddp_hook)
        expected = {
            "first": ([torch.zeros_like(model.first) for _ in range(2)],
                      [torch.zeros_like(model.first) for _ in range(2)],
                      torch.zeros_like(model.first)),
            "second": ([torch.zeros_like(model.second) for _ in range(2)],
                       [torch.zeros_like(model.second) for _ in range(2)],
                       torch.zeros_like(model.second)),
        }

        for step in range(1, 4):
            all_gradients = [_sparse_gradients(source_rank, step) for source_rank in range(2)]
            local_gradients = all_gradients[rank]
            state.begin_step()
            ddp(*local_gradients).backward()
            state.finish_step()

            for index, (name, parameter, stable_id) in enumerate(
                (("first", model.first, 0), ("second", model.second, 1))
            ):
                next_state = _manual_sparse_oracle(
                    expected[name],
                    [all_gradients[0][index], all_gradients[1][index]],
                    step=step,
                    stable_id=stable_id,
                    rows=parameter.shape[0],
                    columns=parameter.shape[1],
                )
                expected[name] = next_state[:3]
                parameter_state = state.parameter_state(parameter)
                torch.testing.assert_close(parameter_state.h_local, next_state[0][rank])
                torch.testing.assert_close(parameter_state.g_local, next_state[1][rank])
                torch.testing.assert_close(parameter_state.g_global, next_state[2])
                torch.testing.assert_close(parameter_state.last_support, next_state[3])
                torch.testing.assert_close(parameter.grad, next_state[2])
            torch.testing.assert_close(
                model.dense.grad,
                (all_gradients[0][2] + all_gradients[1][2]) / 2,
            )

            optimizer.step()
            state.commit_step()
            for parameter in (model.first, model.second, model.dense):
                gathered = [torch.empty_like(parameter) for _ in range(world_size)]
                dist.all_gather(gathered, parameter)
                torch.testing.assert_close(gathered[0], gathered[1])
            optimizer.zero_grad(set_to_none=True)

        signature = [
            (event.category, event.bytes)
            for event in observer.events
            if event.category.startswith("arc_hook/")
        ]
        expected_signature = [
            ("arc_hook/dense", 56),
            ("arc_hook/dense", 8),
            ("arc_hook/sketch", 48),
            ("arc_hook/selected_values", 32),
            ("arc_hook/dense", 8),
            ("arc_hook/sketch", 48),
            ("arc_hook/selected_values", 32),
        ]
        assert signature == expected_signature, signature
        assert all(event.category != "arc/seed" for event in observer.events)
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_two_rank_sparse_hook_matches_three_step_parameter_oracle_and_signature():
    mp.spawn(_sparse_worker, args=(2, _free_port()), nprocs=2, join=True)


def _manual_ef14_oracle(residuals, gradients, *, step, stable_id, rows, columns):
    if step == 1:
        return [torch.zeros_like(value) for value in residuals], sum(gradients) / 2, torch.arange(rows)
    compensated = [
        gradient + residual for gradient, residual in zip(gradients, residuals)
    ]
    generator = torch.Generator().manual_seed(
        derive_arc_seed(base_seed=17, step=step, stable_task_id=stable_id)
    )
    projection = torch.randn(1, columns, 2, generator=generator)
    sketches = [
        torch.bmm(value.unsqueeze(0), projection).squeeze(0) / math.sqrt(2.0)
        for value in compensated
    ]
    support = ((sketches[0] + sketches[1]) / 2).square().sum(-1).topk(
        math.ceil(rows * 0.5), sorted=True
    ).indices
    compressed = []
    next_residuals = []
    for value in compensated:
        selected = torch.zeros_like(value)
        selected[support] = value[support]
        compressed.append(selected)
        next_residuals.append(value - selected)
    return next_residuals, sum(compressed) / 2, support


def _ef14_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        model = _SparseControlledGradientModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999))
        state = ArcTopKDDPState(
            process_group=dist.group.WORLD,
            fingerprint="2" * 64,
            parameter_specs=[
                ArcTopKDDPParameterSpec(model.first, "first", 0, "arc_matrix"),
                ArcTopKDDPParameterSpec(model.second, "second", 1, "arc_matrix"),
                ArcTopKDDPParameterSpec(model.dense, "dense", 2, "dense_aux"),
            ],
            optimizer_parameters=list(model.parameters()),
            config=ArcTopKSyncConfig(
                ratio=0.5,
                projection_rank=2,
                eta=1.0,
                seed=17,
                start_compress_step=0,
                error_feedback="ef14",
            ),
        )
        ddp.register_comm_hook(state, arc_topk_ddp_hook)
        residuals = {
            "first": [torch.zeros_like(model.first) for _ in range(2)],
            "second": [torch.zeros_like(model.second) for _ in range(2)],
        }

        for step in range(1, 4):
            all_gradients = [_sparse_gradients(source_rank, step) for source_rank in range(2)]
            state.begin_step()
            ddp(*all_gradients[rank]).backward()
            state.finish_step()
            for index, (name, parameter, stable_id) in enumerate(
                (("first", model.first, 0), ("second", model.second, 1))
            ):
                next_residuals, expected_gradient, support = _manual_ef14_oracle(
                    residuals[name],
                    [all_gradients[0][index], all_gradients[1][index]],
                    step=step,
                    stable_id=stable_id,
                    rows=parameter.shape[0],
                    columns=parameter.shape[1],
                )
                residuals[name] = next_residuals
                parameter_state = state.parameter_state(parameter)
                torch.testing.assert_close(parameter_state.residual, next_residuals[rank])
                torch.testing.assert_close(parameter_state.last_support, support)
                torch.testing.assert_close(parameter.grad, expected_gradient)
            torch.testing.assert_close(
                model.dense.grad, (all_gradients[0][2] + all_gradients[1][2]) / 2
            )
            optimizer.step()
            state.commit_step()
            for parameter in model.parameters():
                for tensor in (
                    parameter,
                    optimizer.state[parameter]["exp_avg"],
                    optimizer.state[parameter]["exp_avg_sq"],
                ):
                    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
                    dist.all_gather(gathered, tensor)
                    torch.testing.assert_close(gathered[0], gathered[1])
            optimizer.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


def test_two_rank_ef14_hook_matches_reference_and_adamw_moments_agree():
    mp.spawn(_ef14_worker, args=(2, _free_port()), nprocs=2, join=True)
