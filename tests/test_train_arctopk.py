"""Tests for the dedicated ARC-TopK Muon training entry point."""

import argparse
import importlib
import sys

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/compressed_muon/m001_arc_topk_muon_ddp.yaml"


def _import_train_arctopk():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    pytest.importorskip(
        "train", reason="training dependencies need the dion[train] extra"
    )
    return importlib.import_module("train_arctopk")


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.Linear(8, 8, bias=False)
        self.transformer.wte = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


def test_arctopk_hyperparameter_defaults_select_ddp_arc_topk():
    module = _import_train_arctopk()

    hp = module.ArcTopKHyperparameters()

    assert hp.optimizer == "arc_topk_muon"
    assert hp.replicate_mesh_grad_sync is True
    assert hp.arc_topk_ratio == 0.2
    assert hp.arc_projection_rank == 4
    assert hp.arc_eta == 0.1
    assert hp.arc_seed == 42
    assert hp.arc_start_compress_step == 1000


def test_arctopk_parser_accepts_method_specific_arguments():
    module = _import_train_arctopk()
    parser = argparse.ArgumentParser()
    module.configure_arc_topk_parser(parser)

    args = parser.parse_args(
        [
            "--arc_topk_ratio",
            "0.125",
            "--arc_projection_rank",
            "8",
            "--arc_eta",
            "0.3",
            "--arc_seed",
            "7",
            "--arc_start_compress_step",
            "250",
        ]
    )

    assert args.arc_topk_ratio == 0.125
    assert args.arc_projection_rank == 8
    assert args.arc_eta == 0.3
    assert args.arc_seed == 7
    assert args.arc_start_compress_step == 250


def test_arctopk_optimizer_rejects_fsdp_device_mesh():
    module = _import_train_arctopk()
    hp = module.ArcTopKHyperparameters()
    cli_args = argparse.Namespace(
        use_gram_newton_schulz=False,
        no_triton=True,
        use_polar_express=False,
    )

    with pytest.raises(ValueError, match="DDP only"):
        module.init_arc_topk_optimizer(
            model=_StubModel(),
            device_mesh=object(),
            ddp_model=None,
            hp=hp,
            cli_args=cli_args,
        )


def test_arctopk_optimizer_builds_matrix_and_scalar_groups():
    module = _import_train_arctopk()
    from dion import ArcTopKMuon

    hp = module.ArcTopKHyperparameters(scalar_opt="adamw")
    cli_args = argparse.Namespace(
        use_gram_newton_schulz=False,
        no_triton=True,
        use_polar_express=False,
    )
    ddp_model = argparse.Namespace(process_group=None)

    opt = module.init_arc_topk_optimizer(
        model=_StubModel(),
        device_mesh=None,
        ddp_model=ddp_model,
        hp=hp,
        cli_args=cli_args,
    )

    assert type(opt) is ArcTopKMuon
    assert [group["algorithm"] for group in opt.param_groups] == [
        "muon",
        "adamw",
        "adamw",
    ]
    assert opt.param_groups[0]["arc_topk_ratio"] == hp.arc_topk_ratio
    assert opt.param_groups[0]["arc_projection_rank"] == hp.arc_projection_rank
    assert opt.param_groups[0]["arc_eta"] == hp.arc_eta
    assert opt.param_groups[0]["arc_seed"] == hp.arc_seed
    assert (
        opt.param_groups[0]["arc_start_compress_step"]
        == hp.arc_start_compress_step
    )


def test_m001_yaml_loads_through_shared_parser():
    module = _import_train_arctopk()
    import train

    with CONFIG_PATH.open() as config_file:
        raw_config = yaml.safe_load(config_file)

    assert raw_config["dp_size"] is None
    assert raw_config["fs_size"] is None
    assert raw_config["tp_size"] is None
    assert raw_config["checkpoint_freq"] == 0

    with patch.object(sys, "argv", ["train_arctopk.py", "--config", str(CONFIG_PATH)]):
        args = train.parse_cli_args(
            configure_parser=module.configure_arc_topk_parser
        )

    hp = module.ArcTopKHyperparameters(
        **{
            key: value
            for key, value in vars(args).items()
            if key in module.ArcTopKHyperparameters.__dataclass_fields__
        }
    )
    assert hp.optimizer == "arc_topk_muon"
    assert hp.replicate_mesh_grad_sync is True
    assert hp.arc_topk_ratio == 0.2
    assert hp.arc_projection_rank == 4
    assert hp.arc_eta == 0.1
    assert hp.arc_seed == 42
    assert hp.arc_start_compress_step == 1000
