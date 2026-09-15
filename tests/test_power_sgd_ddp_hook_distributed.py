"""Two-rank Gloo checks against hand-derived PowerSGD projections."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.collective_observer import CollectiveObserver, set_active_observer
from dion.power_sgd import PowerSGDConfig
import dion.power_sgd_ddp_hook as hook_module
from test_power_sgd_ddp_hook import FakeGradBucket, make_state


class ControlledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.zeros(4, 4))
        self.auxiliary = torch.nn.Parameter(torch.zeros(3))
        self.fallback = torch.nn.Parameter(torch.zeros(2, 2))

    def forward(self, matrix, auxiliary, fallback):
        return sum(
            (p * g).sum()
            for p, g in zip(self.parameters(), (matrix, auxiliary, fallback))
        )


def _init(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )


def _ef14_worker(rank, rendezvous):
    _init(rank, rendezvous)
    observer = CollectiveObserver()
    set_active_observer(observer)
    try:
        model = ControlledModel()
        ddp = DDP(model, gradient_as_bucket_view=True)
        state = make_state(
            [
                (model.matrix, "matrix"),
                (model.auxiliary, "dense_aux"),
                (model.fallback, "matrix"),
            ],
            process_group=dist.group.WORLD,
            config=PowerSGDConfig(
                start_compress_step=1,
                min_compression_rate=1,
                orthogonalization_epsilon=0,
            ),
        )
        ddp.register_comm_hook(state, hook_module.power_sgd_ddp_hook)
        gradient = torch.zeros(4, 4)
        gradient[rank, rank] = 2 * (rank + 1)
        auxiliary = torch.tensor([2.0, 4.0, 8.0]) + 2 * rank
        fallback = torch.full((2, 2), 4.0 + 2 * rank)
        item = state.parameter_state(model.matrix)

        # Warmup averages everything exactly and does not initialize compression.
        state.begin_step()
        ddp(gradient, auxiliary, fallback).backward()
        state.finish_step()
        state.commit_step()
        torch.testing.assert_close(
            model.matrix.grad, torch.diag(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        )
        assert not bool(item.q_initialized)
        assert observer.signature() == [
            ("powersgd_hook/dense", "all_reduce", 23, "float32", 92)
        ]

        # A known initial factor gives an exact hand-derived two-step oracle.
        item.q_memory.fill_(1)
        item.q_initialized.fill_(True)
        expected_first = torch.zeros(4, 4)
        expected_first[:2, :2] = torch.tensor([[1.0, 4.0], [2.0, 8.0]]) / 5
        expected_second = torch.zeros(4, 4)
        expected_second[:2, :2] = (
            torch.tensor([[217.0, -812.0], [-1426.0, 5336.0]]) / 2165
        )
        previous_error = torch.zeros(4, 4)
        for expected, expected_q in (
            (expected_first, torch.tensor([1.0, 4.0, 0.0, 0.0]) / (5**0.5)),
            (expected_second, torch.tensor([-31.0, 116.0, 0.0, 0.0]) / (2165**0.5)),
        ):
            ddp.zero_grad(set_to_none=True)
            observer.events.clear()
            state.begin_step()
            ddp(gradient, auxiliary, fallback).backward()
            state.finish_step()
            state.commit_step()
            torch.testing.assert_close(
                model.matrix.grad, expected, atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(
                item.error, gradient + previous_error - expected, atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(
                item.q_memory[:, 0], expected_q, atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(
                model.auxiliary.grad, torch.tensor([3.0, 5.0, 9.0])
            )
            torch.testing.assert_close(model.fallback.grad, torch.full((2, 2), 5.0))
            assert observer.signature() == [
                ("powersgd_hook/p_plus_aux", "all_reduce", 11, "float32", 44),
                ("powersgd_hook/q", "all_reduce", 4, "float32", 16),
            ]
            gathered = [torch.empty_like(model.matrix.grad) for _ in range(2)]
            dist.all_gather(gathered, model.matrix.grad)
            torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
            gathered_q = [torch.empty_like(item.q_memory) for _ in range(2)]
            dist.all_gather(gathered_q, item.q_memory)
            torch.testing.assert_close(gathered_q[0], gathered_q[1], atol=0, rtol=0)
            previous_error = item.error.clone()
    finally:
        set_active_observer(None)
        dist.destroy_process_group()


def _epsilon_worker(rank, rendezvous):
    _init(rank, rendezvous)
    try:
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        state = make_state(
            [(parameter, "matrix")],
            process_group=dist.group.WORLD,
            config=PowerSGDConfig(
                start_compress_step=0,
                min_compression_rate=1,
                orthogonalization_epsilon=0.5,
            ),
        )
        item = state.parameter_state(parameter)
        item.q_memory.fill_(1)
        item.q_initialized.fill_(True)
        gradient = torch.full((4, 4), 1.0 + 2 * rank)
        state.begin_step()
        bucket = FakeGradBucket([parameter], [gradient])
        hook_module.power_sgd_ddp_hook(state, bucket).wait()
        # Qinit=1/2.5; Psum=6.4; normalized P=64/133; Qavg=8P.
        expected = torch.full((4, 4), 32768 / 17689)
        torch.testing.assert_close(bucket.gradients()[0], expected)
        torch.testing.assert_close(item.error, gradient - expected)
        torch.testing.assert_close(item.q_memory, torch.full((4, 1), 512 / 133))
        state.finish_step()
    finally:
        dist.destroy_process_group()


def _random_worker(rank, rendezvous):
    _init(rank, rendezvous)
    try:
        diagonal = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for warm_start in (True, False):
            parameter = torch.nn.Parameter(torch.zeros(4, 4))
            state = make_state(
                [(parameter, "matrix")],
                process_group=dist.group.WORLD,
                config=PowerSGDConfig(
                    start_compress_step=0,
                    min_compression_rate=1,
                    orthogonalization_epsilon=0,
                    error_feedback="none",
                    warm_start=warm_start,
                ),
            )
            item = state.parameter_state(parameter)
            expected_q = None
            for phase in (0, 1, 2):
                if not warm_start or phase == 0:
                    initial = torch.randn(
                        4,
                        generator=torch.Generator().manual_seed(52 + phase * 1_000_003),
                    )
                else:
                    initial = expected_q
                direction = diagonal * (initial / torch.linalg.vector_norm(initial))
                p = direction / torch.linalg.vector_norm(direction)
                expected_q = diagonal * p
                state.begin_step()
                bucket = FakeGradBucket(
                    [parameter], [torch.diag(diagonal * (0.5 + rank))], index=phase + 7
                )
                hook_module.power_sgd_ddp_hook(state, bucket).wait()
                torch.testing.assert_close(
                    bucket.gradients()[0], torch.outer(p, expected_q)
                )
                torch.testing.assert_close(item.q_memory[:, 0], expected_q)
                torch.testing.assert_close(item.error, torch.zeros_like(parameter))
                gathered = [torch.empty_like(item.q_memory) for _ in range(2)]
                dist.all_gather(gathered, item.q_memory)
                torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
                state.finish_step()
                state.commit_step()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_real_ddp_warmup_and_two_step_ef14(tmp_path):
    mp.spawn(_ef14_worker, args=(str(tmp_path / "ef14"),), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_p_sum_is_not_averaged_before_orthogonalization(tmp_path):
    mp.spawn(_epsilon_worker, args=(str(tmp_path / "epsilon"),), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_seeded_initialization_and_warm_start_are_replicated(tmp_path):
    mp.spawn(_random_worker, args=(str(tmp_path / "random"),), nprocs=2, join=True)
