"""Training-entry contracts for GreedyLore gradient sync with ordinary Muon."""

import argparse
import importlib
import json
import os
import socket
import sys
import tempfile

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    pytest.importorskip("train", reason="requires the dion[train] extra")
    return importlib.import_module("train_greedylore")


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=False),
            torch.nn.Linear(8, 8, bias=False),
        )
        self.transformer.wte = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


class _DDPStub:
    process_group = None
    find_unused_parameters = False

    def __init__(self):
        self.registrations = []

    def register_comm_hook(self, state, hook):
        self.registrations.append((state, hook))


class _FormalLoopModel(_StubModel):
    def forward(self, x, _target):
        matrix = self.transformer.h[0](x).sum() + self.transformer.h[1](x).sum()
        auxiliary = self.transformer.wte.weight.sum() + self.lm_head.weight.sum()
        return matrix + auxiliary * 0.0


def _cli(**overrides):
    values = dict(
        use_gram_newton_schulz=False,
        no_triton=True,
        use_polar_express=False,
        _explicit_replicate_mesh_grad_sync=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_factory_builds_ordinary_muon_with_one_greedylore_hook_and_muon_group_roles():
    module = _module()
    from dion import GreedyLoreDDPState, Muon

    model = _StubModel()
    ddp = _DDPStub()
    hp = module.GreedyLoreHyperparameters(
        greedy_lore_rank=8,
        scalar_opt="adamw",
        scalar_lr=0.003,
        scalar_adam_beta1=0.8,
        scalar_adam_beta2=0.9,
        scalar_adam_eps=1e-7,
        scalar_weight_decay=0.04,
    )
    optimizer, runtime = module.init_greedy_lore_optimizer(
        model=model,
        device_mesh=None,
        ddp_model=ddp,
        hp=hp,
        cli_args=_cli(),
    )

    assert type(optimizer) is Muon
    assert runtime.optimizer_owns_gradient_sync is False
    assert len(ddp.registrations) == 1
    state, hook = ddp.registrations[0]
    assert isinstance(state, GreedyLoreDDPState)
    assert hook is module.greedy_lore_ddp_hook
    assert runtime.begin_step == state.begin_step
    assert runtime.finish_step == state.finish_step
    assert runtime.commit_step == state.commit_step
    assert runtime.checkpoint_state is state
    assert runtime.checkpoint_state_name == "greedy_lore_compressor"

    assert [spec.stable_name for spec in state.parameter_specs] == [
        name for name, _parameter in model.named_parameters()
    ]
    roles = {spec.stable_name: spec.role for spec in state.parameter_specs}
    assert roles["transformer.h.0.weight"] == "matrix"
    assert roles["transformer.h.1.weight"] == "matrix"
    assert roles["transformer.wte.weight"] == "dense_aux"
    assert roles["lm_head.weight"] == "dense_aux"

    assert [id(spec.parameter) for spec in state.parameter_specs] == [
        id(parameter) for _name, parameter in model.named_parameters()
    ]
    assert [spec.stable_id for spec in state.parameter_specs] == list(
        range(len(state.parameter_specs))
    )
    assert [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ] == [id(parameter) for _name, parameter in model.named_parameters()]
    assert optimizer.param_groups[1]["algorithm"] == "adamw"
    assert optimizer.param_groups[1]["lr"].item() == pytest.approx(0.003)
    assert optimizer.param_groups[1]["beta1"] == pytest.approx(0.8)
    assert optimizer.param_groups[1]["beta2"] == pytest.approx(0.9)
    assert optimizer.param_groups[1]["epsilon"] == pytest.approx(1e-7)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.04)
    assert optimizer.param_groups[2]["algorithm"] == "adamw"
    assert optimizer.param_groups[2]["beta1"] == pytest.approx(0.8)
    assert optimizer.param_groups[2]["beta2"] == pytest.approx(0.9)
    assert optimizer.param_groups[2]["epsilon"] == pytest.approx(1e-7)
    assert optimizer.param_groups[2]["weight_decay"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"device_mesh": object()}, "DDP only"),
        ({"ddp_model": None}, "requires a DDP model"),
    ],
)
def test_factory_rejects_non_ddp_construction(kwargs, match):
    module = _module()
    args = dict(
        model=_StubModel(),
        device_mesh=None,
        ddp_model=_DDPStub(),
        hp=module.GreedyLoreHyperparameters(),
        cli_args=_cli(),
    )
    args.update(kwargs)

    with pytest.raises(ValueError, match=match):
        module.init_greedy_lore_optimizer(**args)


@pytest.mark.parametrize(
    "hp, cli_args, ddp, match",
    [
        (
            lambda module: module.GreedyLoreHyperparameters(optimizer="muon"),
            _cli(),
            _DDPStub(),
            "Unsupported GreedyLore optimizer",
        ),
        (
            lambda module: module.GreedyLoreHyperparameters(scalar_opt="sgd"),
            _cli(),
            _DDPStub(),
            "Unrecognized scalar optimizer",
        ),
        (
            lambda module: module.GreedyLoreHyperparameters(),
            _cli(_explicit_replicate_mesh_grad_sync=True),
            _DDPStub(),
            "replicate_mesh_grad_sync",
        ),
        (
            lambda module: module.GreedyLoreHyperparameters(),
            _cli(),
            type(
                "_FindUnusedDDPStub",
                (_DDPStub,),
                {"find_unused_parameters": True},
            )(),
            "find_unused_parameters=False",
        ),
    ],
)
def test_factory_rejects_unsupported_runtime_modes(hp, cli_args, ddp, match):
    module = _module()

    with pytest.raises(ValueError, match=match):
        module.init_greedy_lore_optimizer(
            model=_StubModel(),
            device_mesh=None,
            ddp_model=ddp,
            hp=hp(module),
            cli_args=cli_args,
        )


def test_checkpoint_runtime_names_arc_fallback_greedylore_and_optimizer_only():
    import train

    state = object()
    unnamed_arc = train.GradientSyncRuntime(
        optimizer_owns_gradient_sync=False,
        checkpoint_state=state,
    )
    named_greedylore = train.GradientSyncRuntime(
        optimizer_owns_gradient_sync=False,
        checkpoint_state=state,
        checkpoint_state_name="greedy_lore_compressor",
    )
    optimizer_only = train.GradientSyncRuntime(optimizer_owns_gradient_sync=True)

    assert train.extra_stateful_from_gradient_sync_runtime(unnamed_arc) == {
        "arc_compressor": state
    }
    assert train.extra_stateful_from_gradient_sync_runtime(named_greedylore) == {
        "greedy_lore_compressor": state
    }
    assert train.extra_stateful_from_gradient_sync_runtime(optimizer_only) is None


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_update(ddp, optimizer, runtime, rank, step, grad_accum_steps=2):
    runtime.begin_step()
    inputs = (
        (torch.tensor([[1.0] + [0.0] * 7]), torch.tensor([[0.0, 2.0] + [0.0] * 6]))
        if rank == 0
        else (torch.tensor([[3.0] + [0.0] * 7]), torch.tensor([[0.0, 4.0] + [0.0] * 6]))
    )
    for micro_step, input_tensor in enumerate(inputs[:grad_accum_steps], start=1):
        train_loss, _ = __import__("train").forward_backward_micro_step(
            ddp,
            input_tensor,
            None,
            autocast_ctx=nullcontext(),
            micro_step=micro_step,
            grad_accum_steps=grad_accum_steps,
            optimizer_owns_gradient_sync=runtime.optimizer_owns_gradient_sync,
        )
        assert torch.isfinite(train_loss)
    state = runtime.checkpoint_state
    hook_calls_before_finish = state._next_context_id
    gradient_before_finish = ddp.module.transformer.h[0].weight.grad.detach().clone()
    first_error_before_clip = state.parameter_state(
        ddp.module.transformer.h[0].weight
    ).error.detach().clone()
    runtime.finish_step()
    grad_norm_before_clip = torch.sqrt(
        sum(
            parameter.grad.detach().to(torch.float32).square().sum()
            for parameter in ddp.module.parameters()
            if parameter.grad is not None
        )
    )
    grad_norm = __import__("train").compute_and_clip_grad_norm_(
        ddp.module.parameters(), 0.5 if step == 2 else None
    )
    first_error_after_clip = state.parameter_state(
        ddp.module.transformer.h[0].weight
    ).error.detach().clone()
    optimizer.step()
    runtime.commit_step()
    ddp.zero_grad(set_to_none=True)
    return {
        "hook_calls": hook_calls_before_finish,
        "committed_step": state.committed_step,
        "gradient": gradient_before_finish[0].tolist(),
        "grad_norm": float(grad_norm),
        "grad_norm_before_clip": float(grad_norm_before_clip),
        "error_before_clip": first_error_before_clip[0].tolist(),
        "error_after_clip": first_error_after_clip[0].tolist(),
    }


def _integration_worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        from dion.collective_observer import CollectiveObserver, set_active_observer

        train = __import__("train")
        module = _module()
        torch.manual_seed(0)
        ddp = DDP(_FormalLoopModel())
        observer = CollectiveObserver()
        set_active_observer(observer)
        optimizer, runtime = module.init_greedy_lore_optimizer(
            model=ddp.module,
            device_mesh=None,
            ddp_model=ddp,
            hp=module.GreedyLoreHyperparameters(
                greedy_lore_rank=1,
                greedy_lore_update_interval=4,
                greedy_lore_start_compress_step=0,
                scalar_opt="adamw",
                lr=0.01,
                weight_decay=0.1,
                adjust_lr=None,
            ),
            cli_args=_cli(),
        )
        state = runtime.checkpoint_state
        try:
            step1 = _run_update(ddp, optimizer, runtime, rank, step=1)
            step2 = _run_update(ddp, optimizer, runtime, rank, step=2)
            matrix_state = state.parameter_state(ddp.module.transformer.h[0].weight)
            state.validate_replicated_basis_across_ranks()
            gathered_weight = [
                torch.empty_like(ddp.module.transformer.h[0].weight)
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_weight, ddp.module.transformer.h[0].weight)
            assert all(torch.equal(gathered_weight[0], value) for value in gathered_weight[1:])
            Path(output_dir, f"integration-rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "step1": step1,
                        "step2": step2,
                        "support": matrix_state.last_support.tolist(),
                        "signature": observer.signature(),
                        "parameters": {
                            name: parameter.detach().tolist()
                            for name, parameter in ddp.module.named_parameters()
                        },
                        "momentum": {
                            name: optimizer.state[parameter]["momentum"].detach().tolist()
                            for name, parameter in ddp.module.named_parameters()
                            if "momentum" in optimizer.state[parameter]
                        },
                    }
                )
            )
            dist.barrier()
        finally:
            set_active_observer(None)
    finally:
        dist.destroy_process_group()


def _run_integration_case():
    with tempfile.TemporaryDirectory(prefix="greedylore-train-integration-") as output_dir:
        mp.spawn(
            _integration_worker,
            args=(2, _free_port(), output_dir),
            nprocs=2,
            join=True,
        )
        return [
            json.loads(Path(output_dir, f"integration-rank-{rank}.json").read_text())
            for rank in range(2)
        ]


def test_real_two_rank_factory_loop_refresh_then_compressed_step_and_ordering():
    results = _run_integration_case()

    assert [result["step1"]["hook_calls"] for result in results] == [1, 1]
    assert [result["step1"]["committed_step"] for result in results] == [1, 1]
    assert [result["step2"]["hook_calls"] for result in results] == [2, 2]
    assert [result["step2"]["committed_step"] for result in results] == [2, 2]
    assert [result["step1"]["gradient"] for result in results] == [
        pytest.approx([1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        pytest.approx([1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    assert [result["step2"]["error_before_clip"] for result in results] == [
        result["step2"]["error_after_clip"] for result in results
    ]
    assert [result["step2"]["grad_norm"] for result in results] == pytest.approx(
        [result["step2"]["grad_norm_before_clip"] for result in results]
    )
    assert results[0]["support"] == results[1]["support"]
    assert results[0]["signature"] == results[1]["signature"]
    categories = [event[0] for event in results[0]["signature"]]
    assert categories == [
        "greedylore/layout_validation",
        "greedylore_hook/dense",
        "muon/result_collective",
        "greedylore_hook/score_plus_aux_allreduce",
        "greedylore_hook/factor_allreduce",
        "muon/result_collective",
    ]
    for name in results[0]["parameters"]:
        torch.testing.assert_close(
            torch.tensor(results[0]["parameters"][name]),
            torch.tensor(results[1]["parameters"][name]),
            rtol=0,
            atol=0,
        )
    for name in results[0]["momentum"]:
        torch.testing.assert_close(
            torch.tensor(results[0]["momentum"][name]),
            torch.tensor(results[1]["momentum"][name]),
            rtol=1e-6,
            atol=1e-6,
        )


def _full_rank_worker(rank, world_size, port, case_name, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        from dion.collective_observer import CollectiveObserver, set_active_observer

        train = __import__("train")
        module = _module()
        torch.manual_seed(123)
        ddp = DDP(_FormalLoopModel())
        observer = CollectiveObserver()
        set_active_observer(observer)
        try:
            if case_name == "dense":
                optimizer = train.init_optimizer(
                    model=ddp.module,
                    device_mesh=None,
                    ddp_model=ddp,
                    hp=train.Hyperparameters(
                        optimizer="muon",
                        scalar_opt="adamw",
                        lr=0.01,
                        weight_decay=0.1,
                        adjust_lr=None,
                    ),
                    cli_args=_cli(),
                )
                runtime = train.GradientSyncRuntime(optimizer_owns_gradient_sync=False)
            else:
                optimizer, runtime = module.init_greedy_lore_optimizer(
                    model=ddp.module,
                    device_mesh=None,
                    ddp_model=ddp,
                    hp=module.GreedyLoreHyperparameters(
                        greedy_lore_rank=8,
                        greedy_lore_update_interval=2,
                        greedy_lore_start_compress_step=0,
                        scalar_opt="adamw",
                        lr=0.01,
                        weight_decay=0.1,
                        adjust_lr=None,
                    ),
                    cli_args=_cli(),
                )
            gradients = {}
            for step in range(2):
                if runtime.begin_step is not None:
                    runtime.begin_step()
                train.forward_backward_micro_step(
                    ddp,
                    torch.tensor([[float(rank + step + 1)] + [0.0] * 7]),
                    None,
                    autocast_ctx=nullcontext(),
                    micro_step=1,
                    grad_accum_steps=1,
                    optimizer_owns_gradient_sync=runtime.optimizer_owns_gradient_sync,
                )
                if runtime.finish_step is not None:
                    runtime.finish_step()
                gradients = {
                    name: parameter.grad.detach().tolist()
                    for name, parameter in ddp.module.named_parameters()
                    if parameter.grad is not None
                }
                optimizer.step()
                if runtime.commit_step is not None:
                    runtime.commit_step()
                ddp.zero_grad(set_to_none=True)
            Path(output_dir, f"{case_name}-rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "signature": observer.signature(),
                        "parameters": {
                            name: parameter.detach().tolist()
                            for name, parameter in ddp.module.named_parameters()
                        },
                        "gradients": gradients,
                        "momentum": {
                            name: optimizer.state[parameter]["momentum"].detach().tolist()
                            for name, parameter in ddp.module.named_parameters()
                            if "momentum" in optimizer.state[parameter]
                        },
                    }
                )
            )
            dist.barrier()
        finally:
            set_active_observer(None)
    finally:
        dist.destroy_process_group()


def _run_full_rank_case(case_name):
    with tempfile.TemporaryDirectory(prefix="greedylore-full-rank-") as output_dir:
        mp.spawn(
            _full_rank_worker,
            args=(2, _free_port(), case_name, output_dir),
            nprocs=2,
            join=True,
        )
        return [
            json.loads(Path(output_dir, f"{case_name}-rank-{rank}.json").read_text())
            for rank in range(2)
        ]


def test_full_rank_greedylore_matches_dense_muon_through_refresh_and_compressed_steps():
    dense_results = _run_full_rank_case("dense")
    greedy_results = _run_full_rank_case("greedy")

    for dense_rank, greedy_rank in zip(dense_results, greedy_results):
        for name in dense_rank["parameters"]:
            torch.testing.assert_close(
                torch.tensor(greedy_rank["parameters"][name]),
                torch.tensor(dense_rank["parameters"][name]),
                rtol=1e-5,
                atol=1e-6,
            )
        for name in dense_rank["gradients"]:
            torch.testing.assert_close(
                torch.tensor(greedy_rank["gradients"][name]),
                torch.tensor(dense_rank["gradients"][name]),
                rtol=1e-5,
                atol=1e-6,
            )
        for name in dense_rank["momentum"]:
            torch.testing.assert_close(
                torch.tensor(greedy_rank["momentum"][name]),
                torch.tensor(dense_rank["momentum"][name]),
                rtol=1e-5,
                atol=1e-6,
            )

    greedy_categories = [event[0] for event in greedy_results[0]["signature"]]
    assert greedy_categories == [
        "greedylore/layout_validation",
        "greedylore_hook/dense",
        "muon/result_collective",
        "greedylore_hook/score_plus_aux_allreduce",
        "greedylore_hook/factor_allreduce",
        "muon/result_collective",
    ]
