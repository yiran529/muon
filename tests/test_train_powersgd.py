"""Training-entry contracts for PowerSGD-Muon DDP synchronization."""

import argparse
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/compressed_muon/m005_power_sgd_muon_ddp.yaml"


def _module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    pytest.importorskip("train", reason="requires the dion[train] extra")
    return importlib.import_module("train_powersgd")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Linear(16, 16, bias=True)
        self.transformer.wte = torch.nn.Embedding(32, 16)
        self.lm_head = torch.nn.Linear(16, 32, bias=False)


class _DDP:
    process_group = None
    find_unused_parameters = False

    def __init__(self):
        self.registrations = []

    def register_comm_hook(self, state, hook):
        self.registrations.append((state, hook))


def _cli(**overrides):
    values = dict(
        use_gram_newton_schulz=False,
        no_triton=True,
        use_polar_express=False,
        _explicit_replicate_mesh_grad_sync=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "mesh,ddp,message",
    [
        (object(), None, "DDP only"),
        (None, None, "requires a DDP model"),
    ],
)
def test_factory_requires_ddp(mesh, ddp, message):
    module = _module()
    with pytest.raises(ValueError, match=message):
        module.init_power_sgd_optimizer(
            _Model(), mesh, ddp, module.PowerSGDHyperparameters(), _cli()
        )


def test_factory_rejects_explicit_optimizer_owned_sync():
    module = _module()
    with pytest.raises(ValueError, match="replicate_mesh_grad_sync"):
        module.init_power_sgd_optimizer(
            _Model(),
            None,
            _DDP(),
            module.PowerSGDHyperparameters(),
            _cli(_explicit_replicate_mesh_grad_sync=True),
        )


def test_factory_uses_ordinary_muon_and_only_muon_group_is_compressed():
    module = _module()
    from dion import Muon, PowerSGDDDPState

    model, ddp = _Model(), _DDP()
    optimizer, runtime = module.init_power_sgd_optimizer(
        model, None, ddp, module.PowerSGDHyperparameters(scalar_opt="adamw"), _cli()
    )

    assert type(optimizer) is Muon
    assert [group["algorithm"] for group in optimizer.param_groups] == [
        "muon",
        "adamw",
        "adamw",
    ]
    assert len(ddp.registrations) == 1
    state, hook = ddp.registrations[0]
    assert isinstance(state, PowerSGDDDPState)
    assert hook is module.power_sgd_ddp_hook
    assert {spec.stable_name: spec.role for spec in state.parameter_specs} == {
        "transformer.h.weight": "matrix",
        "transformer.h.bias": "dense_aux",
        "transformer.wte.weight": "dense_aux",
        "lm_head.weight": "dense_aux",
    }
    assert runtime.optimizer_owns_gradient_sync is False
    assert runtime.begin_step == state.begin_step
    assert runtime.finish_step == state.finish_step
    assert runtime.commit_step == state.commit_step
    assert runtime.checkpoint_state is state
    assert runtime.checkpoint_state_name == "power_sgd_compressor"


def test_invalid_configuration_is_rejected_before_hook_registration():
    module = _module()
    ddp = _DDP()
    with pytest.raises(ValueError, match="rank must be positive"):
        module.init_power_sgd_optimizer(
            _Model(),
            None,
            ddp,
            module.PowerSGDHyperparameters(power_sgd_rank=0),
            _cli(),
        )
    assert ddp.registrations == []


def test_cross_rank_fingerprint_failure_prevents_hook_registration(monkeypatch):
    module = _module()
    from dion.power_sgd_layout import PowerSGDLayoutMismatch

    ddp = _DDP()
    ddp.process_group = object()
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda _group: [0, 1])

    def mismatch(fingerprint, process_group):
        assert len(fingerprint) == 64
        assert process_group is ddp.process_group
        raise PowerSGDLayoutMismatch("layout mismatch across ranks")

    monkeypatch.setattr(module, "validate_power_sgd_fingerprint_across_ranks", mismatch)
    with pytest.raises(PowerSGDLayoutMismatch, match="layout mismatch across ranks"):
        module.init_power_sgd_optimizer(
            _Model(), None, ddp, module.PowerSGDHyperparameters(), _cli()
        )
    assert ddp.registrations == []


def test_parser_exposes_power_sgd_options_and_validation():
    module = _module()
    parser = argparse.ArgumentParser()
    module.configure_power_sgd_parser(parser)
    args = parser.parse_args(
        [
            "--power_sgd_rank",
            "4",
            "--power_sgd_start_compress_step",
            "1000",
            "--power_sgd_min_compression_rate",
            "2",
            "--power_sgd_error_feedback",
            "ef14",
            "--power_sgd_warm_start",
            "--power_sgd_seed",
            "7",
            "--power_sgd_orthogonalization_epsilon",
            "1e-8",
        ]
    )
    assert args.power_sgd_rank == 4
    assert args.power_sgd_start_compress_step == 1000
    assert args.power_sgd_min_compression_rate == 2
    assert args.power_sgd_error_feedback == "ef14"
    assert args.power_sgd_warm_start is True
    assert args.power_sgd_seed == 7
    assert args.power_sgd_orthogonalization_epsilon == 1e-8
    with pytest.raises(ValueError, match="Unsupported PowerSGD optimizer"):
        module.validate_power_sgd_hyperparameters(
            module.PowerSGDHyperparameters(optimizer="muon")
        )


def test_ddp_yaml_loads_as_explicit_bf16_power_sgd_recipe():
    module = _module()
    import train

    values = yaml.safe_load(CONFIG.read_text())
    with patch.object(sys, "argv", ["train_powersgd.py", "--config", str(CONFIG)]):
        args = train.parse_cli_args(configure_parser=module.configure_power_sgd_parser)
    hp = train.override_args_from_cli(module.PowerSGDHyperparameters(), args)
    module.validate_power_sgd_hyperparameters(hp)
    assert values["model_dtype"] == "bfloat16"
    assert (values["dp_size"], values["fs_size"], values["tp_size"]) == (
        None,
        None,
        None,
    )
    assert values["power_sgd_rank"] == 4
    assert values["power_sgd_start_compress_step"] == 1000
    assert values["power_sgd_error_feedback"] == "ef14"
    assert values["power_sgd_warm_start"] is True
