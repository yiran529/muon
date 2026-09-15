"""Regression tests for the injectable seams in the shared training entry point."""

import inspect
import sys

from pathlib import Path
from unittest.mock import patch

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_train():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return pytest.importorskip(
        "train", reason="train.py and its deps need the dion[train] extra"
    )


def test_parse_cli_args_accepts_extension():
    train = _import_train()

    def configure(parser):
        parser.add_argument("--arc_eta", type=float, default=None)

    with patch.object(sys, "argv", ["train.py", "--arc_eta", "0.25"]):
        args = train.parse_cli_args(configure_parser=configure)

    assert args.arc_eta == 0.25


def test_main_exposes_defaulted_factories():
    train = _import_train()
    signature = inspect.signature(train.main)

    assert (
        signature.parameters["hyperparameters_factory"].default
        is train.Hyperparameters
    )
    assert signature.parameters["optimizer_factory"].default is train.init_optimizer
    assert signature.parameters["configure_parser"].default is None
    assert signature.parameters["ddp_kwargs_factory"].default is None


def test_build_ddp_kwargs_combines_standard_and_entry_specific_settings():
    train = _import_train()
    model = _StubModel()
    hp = train.Hyperparameters()
    cli_args = type("Args", (), {"bucket_cap_mb": 64.0})()
    calls = []

    def extension(current_model, current_hp, current_cli_args):
        calls.append((current_model, current_hp, current_cli_args))
        return {"bucket_cap_mb_list": [1.0, 2.0, 3.0]}

    kwargs = train.build_ddp_kwargs(
        model,
        hp,
        cli_args,
        local_rank=3,
        ddp_kwargs_factory=extension,
    )

    assert calls == [(model, hp, cli_args)]
    assert kwargs == {
        "device_ids": [3],
        "output_device": 3,
        "bucket_cap_mb": 64.0,
        "bucket_cap_mb_list": [1.0, 2.0, 3.0],
    }


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Linear(8, 8, bias=False)
        self.transformer.wte = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


def test_muon_parameter_groups_use_independent_scalar_adamw_settings():
    train = _import_train()
    hp = train.Hyperparameters(
        scalar_opt="adamw",
        lr=0.02,
        scalar_lr=0.001,
        scalar_adam_beta1=0.9,
        scalar_adam_beta2=0.999,
        scalar_adam_eps=1e-8,
        scalar_weight_decay=0.0,
    )

    groups = train.build_muon_param_groups(_StubModel(), hp)

    assert len(groups) == 3
    assert "lr" not in groups[0]
    for group in groups[1:]:
        assert group["algorithm"] == "adamw"
        assert group["lr"] == pytest.approx(0.001)
        assert group["beta1"] == pytest.approx(0.9)
        assert group["beta2"] == pytest.approx(0.999)
        assert group["epsilon"] == pytest.approx(1e-8)
        assert group["weight_decay"] == pytest.approx(0.0)
        assert "betas" not in group


def test_scalar_optimizer_cli_arguments_round_trip():
    train = _import_train()
    argv = [
        "train.py",
        "--scalar_lr", "0.001",
        "--scalar_adam_beta1", "0.9",
        "--scalar_adam_beta2", "0.999",
        "--scalar_adam_eps", "1e-8",
        "--scalar_weight_decay", "0",
    ]

    with patch.object(sys, "argv", argv):
        args = train.parse_cli_args()

    assert args.scalar_lr == pytest.approx(0.001)
    assert args.scalar_adam_beta1 == pytest.approx(0.9)
    assert args.scalar_adam_beta2 == pytest.approx(0.999)
    assert args.scalar_adam_eps == pytest.approx(1e-8)
    assert args.scalar_weight_decay == pytest.approx(0.0)
