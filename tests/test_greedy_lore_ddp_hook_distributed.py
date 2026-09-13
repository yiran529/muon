"""Two-rank numerical tests for GreedyLore dense and refresh hook paths."""

import os
import socket
from datetime import timedelta
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.collective_observer import CollectiveObserver, set_active_observer
from dion.greedy_lore import (
    GreedyLoreConfig,
    approximate_signed_lambda,
    canonicalize_svd_basis,
    compress_local,
    corrected_gradient,
    compressed_phase,
    derive_greedy_lore_seed,
    make_random_vectors,
    matrix_orientation,
    orient_matrix,
    reconstruct_global,
    select_projector,
    unorient_matrix,
)
from dion.greedy_lore_ddp_hook import (
    GreedyLoreDDPParameterSpec,
    GreedyLoreDDPState,
    GreedyLoreReplicatedStateMismatch,
    greedy_lore_ddp_hook,
)


class _ControlledGradientModel(torch.nn.Module):
    def __init__(self, *, second_shape=(3, 2), dtype=torch.float32):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype))
        self.dense = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
        self.second = torch.nn.Parameter(torch.zeros(second_shape, dtype=dtype))

    def forward(self, first_source, dense_source, second_source):
        return (
            (self.first * first_source).sum()
            + (self.dense * dense_source).sum()
            + (self.second * second_source).sum()
        )


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _warmup_gradients(rank):
    first = torch.arange(6.0).reshape(2, 3) + rank
    dense = torch.tensor([10.0 + rank, 20.0 - rank])
    second = torch.arange(6.0).reshape(3, 2) + 2 * rank
    return first, dense, second


def _warmup_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    observer = CollectiveObserver()
    set_active_observer(observer)
    try:
        model = _ControlledGradientModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="f" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(parameter, name, index, role)
                for index, (name, parameter, role) in enumerate(
                    (
                        ("first", model.first, "matrix"),
                        ("dense", model.dense, "dense_aux"),
                        ("second", model.second, "matrix"),
                    )
                )
            ],
            optimizer_parameters=list(model.parameters()),
            config=GreedyLoreConfig(rank=1, start_compress_step=3),
        )
        ddp.register_comm_hook(state, greedy_lore_ddp_hook)

        state.begin_step()
        ddp(*_warmup_gradients(rank)).backward()
        state.finish_step()

        all_gradients = [_warmup_gradients(source_rank) for source_rank in range(2)]
        expected_first = (all_gradients[0][0] + all_gradients[1][0]) / 2
        expected_dense = (all_gradients[0][1] + all_gradients[1][1]) / 2
        expected_second = (all_gradients[0][2] + all_gradients[1][2]) / 2
        torch.testing.assert_close(model.first.grad, expected_first)
        torch.testing.assert_close(model.dense.grad, expected_dense)
        torch.testing.assert_close(model.second.grad, expected_second)
        assert [event.category for event in observer.events] == [
            "greedylore_hook/dense"
        ]
        assert observer.events[0].operation == "all_reduce"
        assert (
            observer.events[0].bytes
            == (model.first.numel() + model.dense.numel() + model.second.numel()) * 4
        )
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_two_rank_dense_warmup_averages_real_ddp_bucket_once():
    mp.spawn(_warmup_worker, args=(2, _free_port()), nprocs=2, join=True)


def _refresh_case(case, rank):
    if case == "zero":
        corrected_average = torch.zeros(2, 3)
        gradient = torch.full((2, 3), float(rank + 1))
        error = corrected_average - gradient
    elif case == "repeated":
        corrected_average = torch.eye(2, 3)
        gradient = corrected_average + rank
        error = corrected_average - gradient
    else:
        corrected_average = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.999, 0.0]])
        gradient = corrected_average + 2 * rank
        error = corrected_average - gradient
    return gradient, error, corrected_average


def _refresh_worker(rank, world_size, port, case):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    observer = CollectiveObserver()
    set_active_observer(observer)
    try:
        model = _ControlledGradientModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="a" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(model.first, "first", 0, "matrix"),
                GreedyLoreDDPParameterSpec(model.dense, "dense", 1, "dense_aux"),
                GreedyLoreDDPParameterSpec(model.second, "second", 2, "matrix"),
            ],
            optimizer_parameters=list(model.parameters()),
            config=GreedyLoreConfig(
                rank=1,
                start_compress_step=0,
                update_interval=4,
                basis_sync="local_svd",
            ),
        )
        first_gradient, first_error, first_average = _refresh_case(case, rank)
        second_gradient = torch.arange(6.0).reshape(3, 2) + rank
        second_error = torch.tensor(
            [
                [1.0 - rank, -1.0 - rank, 0.5 - rank],
                [2.0 - rank, -2.0 - rank, 1.5 - rank],
            ]
        )
        second_average = (
            sum(
                (torch.arange(6.0).reshape(3, 2) + source_rank).mT
                + torch.tensor(
                    [
                        [
                            1.0 - source_rank,
                            -1.0 - source_rank,
                            0.5 - source_rank,
                        ],
                        [
                            2.0 - source_rank,
                            -2.0 - source_rank,
                            1.5 - source_rank,
                        ],
                    ]
                )
                for source_rank in range(2)
            )
            / 2
        )
        dense_gradient = torch.tensor([5.0 + rank, 7.0 - rank])
        state.parameter_state(model.first).error.copy_(first_error)
        state.parameter_state(model.second).error.copy_(second_error)
        ddp.register_comm_hook(state, greedy_lore_ddp_hook)

        raw_first_average = (
            sum(_refresh_case(case, source_rank)[0] for source_rank in range(2)) / 2
        )

        state.begin_step()
        ddp(first_gradient, dense_gradient, second_gradient).backward()
        state.finish_step()

        torch.testing.assert_close(model.first.grad, first_average)
        torch.testing.assert_close(model.second.grad, second_average.mT)
        assert not torch.allclose(model.first.grad, raw_first_average)
        first_state = state.parameter_state(model.first)
        second_state = state.parameter_state(model.second)
        torch.testing.assert_close(
            first_state.error,
            torch.zeros_like(first_state.error),
        )
        torch.testing.assert_close(
            second_state.error,
            torch.zeros_like(second_state.error),
        )
        torch.testing.assert_close(first_state.last_support, torch.tensor([0]))
        expected_basis, _, _ = torch.linalg.svd(first_average, full_matrices=False)
        expected_basis = canonicalize_svd_basis(expected_basis)
        torch.testing.assert_close(first_state.basis, expected_basis)
        state.commit_step()
        state.validate_replicated_basis_across_ranks()
        if case == "repeated" and rank == 1:
            first_state.basis.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        if case == "repeated":
            with pytest.raises(GreedyLoreReplicatedStateMismatch, match="basis"):
                state.validate_replicated_basis_across_ranks()
        categories = [event.category for event in observer.events]
        assert categories == ["greedylore_hook/dense"]
        assert "greedylore_hook/factor_allreduce" not in categories
        assert "greedylore_hook/basis_broadcast" not in categories
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["zero", "repeated", "near_repeated"])
def test_two_rank_local_svd_refresh_averages_corrected_gradient_and_replicates_basis(
    case,
):
    mp.spawn(_refresh_worker, args=(2, _free_port(), case), nprocs=2, join=True)


def _broadcast_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    observer = CollectiveObserver()
    set_active_observer(observer)
    original_svd = torch.linalg.svd
    svd_calls = 0

    def counted_svd(*args, **kwargs):
        nonlocal svd_calls
        svd_calls += 1
        return original_svd(*args, **kwargs)

    try:
        model = _ControlledGradientModel(second_shape=(3, 5))
        ddp = DDP(model, gradient_as_bucket_view=True)
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="b" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(model.first, "z-last", 0, "matrix"),
                GreedyLoreDDPParameterSpec(model.dense, "middle", 1, "dense_aux"),
                GreedyLoreDDPParameterSpec(model.second, "a-first", 2, "matrix"),
            ],
            optimizer_parameters=list(model.parameters()),
            config=GreedyLoreConfig(
                rank=1,
                start_compress_step=0,
                update_interval=2,
                basis_sync="broadcast",
            ),
        )
        first_gradient = torch.arange(6.0).reshape(2, 3) + rank
        second_gradient = torch.arange(15.0).reshape(3, 5) - rank
        dense_gradient = torch.tensor([2.0 + rank, 3.0 - rank])
        ddp.register_comm_hook(state, greedy_lore_ddp_hook)

        with mock.patch("torch.linalg.svd", counted_svd):
            state.begin_step()
            ddp(first_gradient, dense_gradient, second_gradient).backward()
            state.finish_step()

        assert svd_calls == (2 if rank == 0 else 0)
        state.commit_step()
        state.validate_replicated_basis_across_ranks()
        basis_events = [
            event
            for event in observer.events
            if event.category == "greedylore_hook/basis_broadcast"
        ]
        assert [event.bytes for event in basis_events] == [36, 16]
        assert [event.operation for event in basis_events] == ["broadcast", "broadcast"]
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_two_rank_broadcast_refresh_runs_rank_zero_svd_and_broadcasts_stable_order():
    mp.spawn(_broadcast_worker, args=(2, _free_port()), nprocs=2, join=True)


def _compressed_refresh_gradients(rank):
    first = torch.tensor([[5.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    dense = torch.tensor([30.0 + rank, 40.0 - rank])
    second_oriented = torch.tensor([[6.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    return first, dense, second_oriented.mT


def _sign_cancel_matrix(shape, *, phase, stable_id, scale, rank):
    orientation = matrix_orientation(shape)
    rows, columns = orientation.compressed_shape
    seed = derive_greedy_lore_seed(
        base_seed=23,
        phase=phase,
        stable_parameter_id=stable_id,
    )
    vectors = make_random_vectors(
        rows=rows,
        columns=columns,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    oriented = torch.stack((vectors[0] * (4.0 * (-1.0 if rank else 1.0)), vectors[1] * scale))
    return unorient_matrix(oriented, orientation).clone()


def _compressed_gradients(rank, step):
    phase = compressed_phase(step, 0)
    assert phase is not None
    if step == 2:
        first = _sign_cancel_matrix((2, 3), phase=phase, stable_id=0, scale=0.75, rank=rank)
        second = _sign_cancel_matrix((3, 2), phase=phase, stable_id=2, scale=1.25, rank=rank)
    else:
        first = torch.tensor(
            [
                [0.5 * step + rank, -1.0 + rank, 2.0 - 0.25 * step],
                [1.5 - rank, 0.25 * step, -0.5 - rank],
            ]
        )
        second = torch.tensor(
            [
                [1.0 + rank, -0.5 * step],
                [2.0 - rank, 0.75 * step],
                [-1.0 + 0.5 * step, 1.25 + rank],
            ]
        )
    dense = torch.tensor([10.0 * step + rank, 20.0 - step - rank])
    return first, dense, second


def _compressed_oracle(config, expected_bases, expected_errors, all_gradients, step):
    phase = compressed_phase(step, config.start_compress_step)
    assert phase is not None
    matrices = (
        ("first", 0, 0, matrix_orientation((2, 3))),
        ("second", 2, 2, matrix_orientation((3, 2))),
    )
    result = {}
    next_errors = [{}, {}]
    for name, stable_id, gradient_index, orientation in matrices:
        corrected = [
            corrected_gradient(
                all_gradients[source_rank][gradient_index],
                expected_errors[source_rank][name],
                orientation,
            )
            for source_rank in range(2)
        ]
        rows, columns = orientation.compressed_shape
        random_vectors = make_random_vectors(
            rows=rows,
            columns=columns,
            seed=derive_greedy_lore_seed(
                base_seed=config.seed,
                phase=phase,
                stable_parameter_id=stable_id,
            ),
            device=corrected[0].device,
            dtype=corrected[0].dtype,
        )
        signed_lambdas = [
            approximate_signed_lambda(value, expected_bases[name], random_vectors)
            for value in corrected
        ]
        averaged_lambda = sum(signed_lambdas) / 2
        projector, support = select_projector(
            expected_bases[name],
            averaged_lambda,
            config.rank,
        )
        local_factors = []
        for source_rank, value in enumerate(corrected):
            local_factor, next_error = compress_local(value, projector)
            local_factors.append(local_factor)
            next_errors[source_rank][name] = next_error
        averaged_factor = sum(local_factors) / 2
        reconstructed = reconstruct_global(projector, averaged_factor)
        result[name] = {
            "averaged_lambda": averaged_lambda,
            "scores": averaged_lambda.square(),
            "support": support,
            "local_factors": local_factors,
            "next_errors": [next_errors[0][name], next_errors[1][name]],
            "averaged_factor": averaged_factor,
            "reconstructed": unorient_matrix(reconstructed, orientation),
        }
    result["dense"] = (all_gradients[0][1] + all_gradients[1][1]) / 2
    return result, next_errors


def _compressed_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=45)
    )
    observer = CollectiveObserver()
    set_active_observer(observer)
    try:
        model = _ControlledGradientModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        config = GreedyLoreConfig(
            rank=1,
            start_compress_step=0,
            update_interval=100,
            seed=23,
            basis_sync="local_svd",
        )
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="6" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(model.first, "first", 0, "matrix"),
                GreedyLoreDDPParameterSpec(model.dense, "dense", 1, "dense_aux"),
                GreedyLoreDDPParameterSpec(model.second, "second", 2, "matrix"),
            ],
            optimizer_parameters=list(model.parameters()),
            config=config,
        )
        ddp.register_comm_hook(state, greedy_lore_ddp_hook)
        state.begin_step()
        ddp(*_compressed_refresh_gradients(rank)).backward()
        state.finish_step()
        state.commit_step()
        ddp.zero_grad(set_to_none=True)

        expected_bases = {}
        expected_errors = [{"first": torch.zeros(2, 3), "second": torch.zeros(2, 3)} for _ in range(2)]
        for name, parameter, average in (
            ("first", model.first, _compressed_refresh_gradients(0)[0]),
            ("second", model.second, _compressed_refresh_gradients(0)[2].mT),
        ):
            basis, _, _ = torch.linalg.svd(average, full_matrices=False)
            expected_bases[name] = canonicalize_svd_basis(basis)
            parameter_state = state.parameter_state(parameter)
            torch.testing.assert_close(parameter_state.basis, expected_bases[name])
            torch.testing.assert_close(parameter_state.error, torch.zeros_like(parameter_state.error))

        for step in range(2, 5):
            all_gradients = [_compressed_gradients(source_rank, step) for source_rank in range(2)]
            expected, expected_errors = _compressed_oracle(
                config,
                expected_bases,
                expected_errors,
                all_gradients,
                step,
            )
            state.begin_step()
            ddp(*all_gradients[rank]).backward()
            state.finish_step()

            torch.testing.assert_close(model.first.grad, expected["first"]["reconstructed"])
            torch.testing.assert_close(model.second.grad, expected["second"]["reconstructed"])
            torch.testing.assert_close(model.dense.grad, expected["dense"])
            for name, parameter in (("first", model.first), ("second", model.second)):
                parameter_state = state.parameter_state(parameter)
                torch.testing.assert_close(parameter_state.basis, expected_bases[name])
                torch.testing.assert_close(parameter_state.last_support, expected[name]["support"])
                torch.testing.assert_close(parameter_state.error, expected[name]["next_errors"][rank])
                torch.testing.assert_close(
                    expected[name]["averaged_factor"],
                    sum(expected[name]["local_factors"]) / 2,
                )
                torch.testing.assert_close(
                    expected[name]["scores"],
                    expected[name]["averaged_lambda"].square(),
                )
            if step == 2:
                assert torch.equal(expected["first"]["support"], torch.tensor([1]))
                assert torch.equal(expected["second"]["support"], torch.tensor([1]))
            state.commit_step()
            ddp.zero_grad(set_to_none=True)

        signature = [(event.category, event.operation, event.bytes) for event in observer.events]
        assert signature == [
            ("greedylore_hook/dense", "all_reduce", 56),
            ("greedylore_hook/score_plus_aux_allreduce", "all_reduce", 24),
            ("greedylore_hook/factor_allreduce", "all_reduce", 24),
            ("greedylore_hook/score_plus_aux_allreduce", "all_reduce", 24),
            ("greedylore_hook/factor_allreduce", "all_reduce", 24),
            ("greedylore_hook/score_plus_aux_allreduce", "all_reduce", 24),
            ("greedylore_hook/factor_allreduce", "all_reduce", 24),
        ]
        assert all(event.category != "greedylore_hook/seed" for event in observer.events)
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_two_rank_compressed_hook_matches_three_step_oracle_and_signature():
    mp.spawn(_compressed_worker, args=(2, _free_port()), nprocs=2, join=True)


def _bfloat16_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=45)
    )
    observer = CollectiveObserver()
    set_active_observer(observer)
    try:
        model = _ControlledGradientModel(dtype=torch.bfloat16)
        ddp = DDP(model, gradient_as_bucket_view=True)
        state = GreedyLoreDDPState(
            process_group=dist.group.WORLD,
            fingerprint="7" * 64,
            parameter_specs=[
                GreedyLoreDDPParameterSpec(model.first, "first", 0, "matrix"),
                GreedyLoreDDPParameterSpec(model.dense, "dense", 1, "dense_aux"),
                GreedyLoreDDPParameterSpec(model.second, "second", 2, "matrix"),
            ],
            optimizer_parameters=list(model.parameters()),
            config=GreedyLoreConfig(
                rank=1,
                start_compress_step=0,
                update_interval=100,
                seed=23,
                basis_sync="local_svd",
            ),
        )
        ddp.register_comm_hook(state, greedy_lore_ddp_hook)

        for step in (1, 2):
            first = (
                torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) + rank + step
            )
            dense = torch.tensor(
                [10 * step + rank, 20 - step - rank], dtype=torch.bfloat16
            )
            second = (
                torch.arange(6, dtype=torch.bfloat16).reshape(3, 2) - rank + step
            )
            state.begin_step()
            ddp(first, dense, second).backward()
            state.finish_step()

            for parameter in model.parameters():
                assert parameter.grad is not None
                assert parameter.grad.dtype == torch.bfloat16
                gathered = [torch.empty_like(parameter.grad) for _ in range(world_size)]
                dist.all_gather(gathered, parameter.grad)
                for other in gathered[1:]:
                    torch.testing.assert_close(gathered[0], other)
            for parameter in (model.first, model.second):
                parameter_state = state.parameter_state(parameter)
                assert parameter_state.error.dtype == torch.bfloat16
                assert parameter_state.basis.dtype == torch.bfloat16

            state.commit_step()
            ddp.zero_grad(set_to_none=True)

        assert [
            (event.category, event.dtype, event.bytes) for event in observer.events
        ] == [
            ("greedylore_hook/dense", "bfloat16", 28),
            ("greedylore_hook/score_plus_aux_allreduce", "bfloat16", 12),
            ("greedylore_hook/factor_allreduce", "bfloat16", 12),
        ]
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def test_two_rank_bfloat16_hook_keeps_state_collectives_and_gradients_bucket_native():
    mp.spawn(_bfloat16_worker, args=(2, _free_port()), nprocs=2, join=True)
