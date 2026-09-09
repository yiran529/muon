"""Training-entry contracts for optimizer-owned and DDP-hook ARC sync."""

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
    return importlib.import_module("train_arctopk")


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Linear(8, 8, bias=False)
        self.transformer.wte = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


class _DDPStub:
    process_group = None

    def __init__(self):
        self.registrations = []

    def register_comm_hook(self, state, hook):
        self.registrations.append((state, hook))


class _FormalLoopModel(_StubModel):
    def forward(self, x, _target):
        matrix_loss = (x @ self.transformer.h.weight.t()).sum()
        auxiliary_coverage = self.transformer.wte.weight.sum() + self.lm_head.weight.sum()
        return matrix_loss + auxiliary_coverage * 0.0


def _cli(**overrides):
    values = dict(
        use_gram_newton_schulz=False,
        no_triton=True,
        use_polar_express=False,
        _explicit_replicate_mesh_grad_sync=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_optimizer_mode_builds_only_arctopk_muon_and_owns_gradient_sync():
    module = _module()
    from dion import ArcTopKMuon

    ddp = _DDPStub()
    optimizer, runtime = module.init_arc_topk_optimizer(
        model=_StubModel(),
        device_mesh=None,
        ddp_model=ddp,
        hp=module.ArcTopKHyperparameters(arc_sync_mode="optimizer"),
        cli_args=_cli(),
    )

    assert type(optimizer) is ArcTopKMuon
    assert runtime.optimizer_owns_gradient_sync is True
    assert runtime.begin_step is None
    assert runtime.finish_step is None
    assert runtime.commit_step is None
    assert runtime.checkpoint_state is None
    assert ddp.registrations == []


def test_adamw_optimizer_side_mode_is_rejected():
    module = _module()
    with pytest.raises(ValueError, match="arc_topk_adamw.*ddp_hook"):
        module.init_arc_topk_optimizer(
            model=_StubModel(),
            device_mesh=None,
            ddp_model=_DDPStub(),
            hp=module.ArcTopKHyperparameters(
                optimizer="arc_topk_adamw",
                arc_sync_mode="optimizer",
            ),
            cli_args=_cli(),
        )


def test_hook_mode_builds_ordinary_muon_and_registers_exactly_one_hook():
    module = _module()
    from dion import ArcTopKDDPState, Muon

    model = _StubModel()
    ddp = _DDPStub()
    optimizer, runtime = module.init_arc_topk_optimizer(
        model=model,
        device_mesh=None,
        ddp_model=ddp,
        hp=module.ArcTopKHyperparameters(arc_sync_mode="ddp_hook"),
        cli_args=_cli(),
    )

    assert type(optimizer) is Muon
    assert runtime.optimizer_owns_gradient_sync is False
    assert len(ddp.registrations) == 1
    state, hook = ddp.registrations[0]
    assert isinstance(state, ArcTopKDDPState)
    assert hook is module.arc_topk_ddp_hook
    assert runtime.begin_step == state.begin_step
    assert runtime.finish_step == state.finish_step
    assert runtime.commit_step == state.commit_step
    assert runtime.checkpoint_state is state
    assert [spec.stable_name for spec in state.parameter_specs] == [
        name for name, _parameter in model.named_parameters()
    ]
    assert [spec.role for spec in state.parameter_specs] == [
        "arc_matrix",
        "arc_matrix",
        "arc_matrix",
    ]


def test_hook_mode_keeps_non_matrix_parameters_dense():
    module = _module()

    model = _StubModel()
    model.transformer.wte.scale = torch.nn.Parameter(torch.ones(8))
    ddp = _DDPStub()
    module.init_arc_topk_optimizer(
        model=model,
        device_mesh=None,
        ddp_model=ddp,
        hp=module.ArcTopKHyperparameters(arc_sync_mode="ddp_hook"),
        cli_args=_cli(),
    )

    state, _hook = ddp.registrations[0]
    roles = {spec.stable_name: spec.role for spec in state.parameter_specs}
    assert roles["transformer.h.weight"] == "arc_matrix"
    assert roles["transformer.wte.weight"] == "arc_matrix"
    assert roles["lm_head.weight"] == "arc_matrix"
    assert roles["transformer.wte.scale"] == "dense_aux"


def test_adamw_hook_mode_builds_standard_adamw_and_compresses_all_2d():
    module = _module()
    model = _StubModel()
    model.transformer.wte.scale = torch.nn.Parameter(torch.ones(8))
    ddp = _DDPStub()
    optimizer, runtime = module.init_arc_topk_optimizer(
        model=model,
        device_mesh=None,
        ddp_model=ddp,
        hp=module.ArcTopKHyperparameters(
            optimizer="arc_topk_adamw",
            arc_sync_mode="ddp_hook",
            lr=3e-4,
            weight_decay=0.1,
        ),
        cli_args=_cli(),
    )

    assert type(optimizer) is torch.optim.AdamW
    assert runtime.optimizer_owns_gradient_sync is False
    assert len(ddp.registrations) == 1
    state, hook = ddp.registrations[0]
    assert hook is module.arc_topk_ddp_hook
    roles = {spec.stable_name: spec.role for spec in state.parameter_specs}
    assert roles["transformer.h.weight"] == "arc_matrix"
    assert roles["transformer.wte.weight"] == "arc_matrix"
    assert roles["lm_head.weight"] == "arc_matrix"
    assert roles["transformer.wte.scale"] == "dense_aux"


@pytest.mark.parametrize("mode, owns", [("optimizer", True), ("ddp_hook", False)])
def test_arc_sync_mode_is_the_only_ownership_policy(mode, owns):
    module = _module()

    assert module.arc_optimizer_owns_gradient_sync(mode) is owns


def test_explicit_legacy_ownership_flag_is_rejected_with_migration_message():
    module = _module()

    with pytest.raises(ValueError, match="arc_sync_mode"):
        module.init_arc_topk_optimizer(
            model=_StubModel(),
            device_mesh=None,
            ddp_model=_DDPStub(),
            hp=module.ArcTopKHyperparameters(),
            cli_args=_cli(_explicit_replicate_mesh_grad_sync=True),
        )


def test_hook_mode_rejects_find_unused_parameters():
    module = _module()
    ddp = _DDPStub()
    ddp.find_unused_parameters = True

    with pytest.raises(ValueError, match="find_unused_parameters=False"):
        module.init_arc_topk_optimizer(
            model=_StubModel(),
            device_mesh=None,
            ddp_model=ddp,
            hp=module.ArcTopKHyperparameters(arc_sync_mode="ddp_hook"),
            cli_args=_cli(),
        )


def test_shared_runtime_normalizes_legacy_optimizer_factories():
    import train

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)
    normalized_optimizer, runtime = train.normalize_gradient_sync_runtime(
        optimizer,
        optimizer_owns_gradient_sync=True,
    )

    assert normalized_optimizer is optimizer
    assert runtime == train.GradientSyncRuntime(optimizer_owns_gradient_sync=True)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _formal_loop_worker(rank, world_size, port, mode, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        import train

        module = _module()
        torch.manual_seed(0)
        ddp = DDP(_FormalLoopModel())
        optimizer, runtime = module.init_arc_topk_optimizer(
            model=ddp.module,
            device_mesh=None,
            ddp_model=ddp,
            hp=module.ArcTopKHyperparameters(
                arc_sync_mode=mode,
                arc_topk_ratio=1.0,
                arc_eta=1.0,
                scalar_opt="adamw",
            ),
            cli_args=_cli(),
        )
        inputs = (
            (torch.tensor([[1.0] + [0.0] * 7]), torch.tensor([[0.0, 2.0] + [0.0] * 6]))
            if rank == 0
            else (torch.tensor([[3.0] + [0.0] * 7]), torch.tensor([[0.0, 4.0] + [0.0] * 6]))
        )
        if runtime.begin_step is not None:
            runtime.begin_step()
        for micro_step, input_tensor in enumerate(inputs, start=1):
            train.forward_backward_micro_step(
                ddp,
                input_tensor,
                None,
                autocast_ctx=nullcontext(),
                micro_step=micro_step,
                grad_accum_steps=2,
                optimizer_owns_gradient_sync=runtime.optimizer_owns_gradient_sync,
            )
        gradient = ddp.module.transformer.h.weight.grad.detach().clone()
        hook_calls = 0
        tracker = None
        if runtime.checkpoint_state is not None:
            state = runtime.checkpoint_state
            hook_calls = state._next_context_id
            tracker = state.parameter_state(ddp.module.transformer.h.weight).h_local.clone()
            runtime.finish_step()
        optimizer.step()
        if runtime.commit_step is not None:
            runtime.commit_step()
        Path(output_dir, f"{mode}-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "gradient": gradient[0].tolist(),
                    "tracker": None if tracker is None else tracker[0].tolist(),
                    "hook_calls": hook_calls,
                }
            )
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["optimizer", "ddp_hook"])
def test_real_two_rank_formal_loop_uses_the_selected_sync_owner(mode):
    with tempfile.TemporaryDirectory(prefix="arc-formal-loop-") as output_dir:
        mp.spawn(
            _formal_loop_worker,
            args=(2, _free_port(), mode, output_dir),
            nprocs=2,
            join=True,
        )
        results = [
            json.loads(Path(output_dir, f"{mode}-rank-{rank}.json").read_text())
            for rank in range(2)
        ]

    if mode == "optimizer":
        assert [result["hook_calls"] for result in results] == [0, 0]
        assert [result["gradient"][:2] for result in results] == [
            [0.5, 1.0],
            [1.5, 2.0],
        ]
    else:
        assert all(result["hook_calls"] >= 1 for result in results)
        assert [result["gradient"][:2] for result in results] == [
            [1.0, 1.5],
            [1.0, 1.5],
        ]
        assert [result["tracker"][:2] for result in results] == [
            [0.5, 1.0],
            [1.5, 2.0],
        ]


def _large_accumulation_worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        import train

        module = _module()
        ddp = DDP(_FormalLoopModel())
        optimizer, runtime = module.init_arc_topk_optimizer(
            model=ddp.module,
            device_mesh=None,
            ddp_model=ddp,
            hp=module.ArcTopKHyperparameters(
                arc_sync_mode="ddp_hook",
                arc_topk_ratio=1.0,
                arc_eta=1.0,
                scalar_opt="adamw",
            ),
            cli_args=_cli(),
        )
        runtime.begin_step()
        for micro_step in range(1, 257):
            train.forward_backward_micro_step(
                ddp,
                torch.ones(1, 8),
                None,
                autocast_ctx=nullcontext(),
                micro_step=micro_step,
                grad_accum_steps=256,
                optimizer_owns_gradient_sync=False,
            )
        state = runtime.checkpoint_state
        runtime.finish_step()
        optimizer.step()
        runtime.commit_step()
        tracker = state.parameter_state(ddp.module.transformer.h.weight).h_local
        Path(output_dir, "large-accumulation.json").write_text(
            json.dumps(
                {
                    "hook_calls": state._next_context_id,
                    "committed_step": state.committed_step,
                    "tracker": tracker[0].tolist(),
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_ga256_advances_each_tracker_once_per_optimizer_step():
    with tempfile.TemporaryDirectory(prefix="arc-ga256-") as output_dir:
        mp.spawn(
            _large_accumulation_worker,
            args=(1, _free_port(), output_dir),
            nprocs=1,
            join=True,
        )
        result = json.loads(Path(output_dir, "large-accumulation.json").read_text())

    assert result["hook_calls"] == 1
    assert result["committed_step"] == 1
    assert result["tracker"] == pytest.approx([1.0] * 8)
