"""Tests for the shared dense AdamW optimizer builder."""

import pytest
import torch


train = pytest.importorskip(
    "train", reason="train.py and its deps need the dion[train] extra"
)


def test_build_adamw_optimizer_preserves_dense_training_semantics():
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    embedding = torch.nn.Parameter(torch.ones(3, 2))
    head = torch.nn.Parameter(torch.ones(2, 3))
    groups = [
        {"params": [matrix]},
        {"params": [embedding], "algorithm": "adamw", "weight_decay": 0.0},
        {"params": [head], "algorithm": "adamw", "weight_decay": 0.0},
    ]
    hp = train.Hyperparameters(lr=3e-4, weight_decay=0.1)
    optimizer = train.build_adamw_optimizer(groups, hp)
    assert type(optimizer) is torch.optim.AdamW
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4] * 3
    assert [group["betas"] for group in optimizer.param_groups] == [(0.9, 0.95)] * 3
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0, 0.0]
    assert [group["params"][0] for group in optimizer.param_groups] == [matrix, embedding, head]


def test_build_adamw_optimizer_uses_configured_paper_betas_and_epsilon():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    hp = train.Hyperparameters(
        lr=1e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
    )

    optimizer = train.build_adamw_optimizer([{"params": [parameter]}], hp)

    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
