"""Contracts for selecting the trainable model parameter dtype."""

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


def test_default_materialization_keeps_all_trainable_parameters_float32():
    train = _import_train()
    from models.gpt_model import GPT, GPTConfig

    assert train.Hyperparameters().model_dtype == "float32"
    with torch.device("meta"):
        model = GPT(
            GPTConfig(
                sequence_len=8,
                vocab_size=32,
                n_layer=1,
                n_head=1,
                n_embd=8,
            )
        )

    train.materialize_and_initialize_model(
        model,
        device="cpu",
        dtype=train.resolve_model_dtype("float32"),
    )

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_bfloat16_materialization_covers_transformer_embedding_and_lm_head():
    train = _import_train()
    from models.gpt_model import GPT, GPTConfig

    with torch.device("meta"):
        model = GPT(
            GPTConfig(
                sequence_len=8,
                vocab_size=32,
                n_layer=1,
                n_head=1,
                n_embd=8,
            )
        )

    train.materialize_and_initialize_model(
        model,
        device="cpu",
        dtype=train.resolve_model_dtype("bfloat16"),
    )

    assert model.transformer.h[0].attn.c_q.weight.dtype == torch.bfloat16
    assert model.transformer.wte.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}


def test_model_dtype_cli_round_trips_to_hyperparameters():
    train = _import_train()

    with patch.object(sys, "argv", ["train.py", "--model_dtype", "bfloat16"]):
        args = train.parse_cli_args()
    hp = train.override_args_from_cli(train.Hyperparameters(), args)

    assert hp.model_dtype == "bfloat16"


def test_resolve_model_dtype_rejects_unvalidated_callers():
    train = _import_train()

    with pytest.raises(ValueError, match="Unsupported model dtype"):
        train.resolve_model_dtype("float16")


def test_bfloat16_parameter_storage_is_rejected_for_device_mesh_training():
    train = _import_train()

    with pytest.raises(ValueError, match="DDP only"):
        train.validate_model_dtype_parallelism("bfloat16", object())


def test_float32_parameter_storage_remains_valid_for_device_mesh_training():
    train = _import_train()

    train.validate_model_dtype_parallelism("float32", object())
