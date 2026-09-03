"""Regression tests for the injectable seams in the shared training entry point."""

import inspect
import sys

from pathlib import Path
from unittest.mock import patch

import pytest


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
