"""Tests for configurable training schedules and gradient clipping."""

import math

import pytest
import torch


train = pytest.importorskip(
    "train", reason="train.py and its deps need the dion[train] extra"
)


def test_default_schedule_preserves_ratio_based_linear_warmdown():
    hp = train.Hyperparameters(num_iterations=100, warmup_ratio=0.1, warmdown_ratio=0.2)

    assert train.learning_rate_multiplier(0, hp) == pytest.approx(0.1)
    assert train.learning_rate_multiplier(9, hp) == pytest.approx(1.0)
    assert train.learning_rate_multiplier(80, hp) == pytest.approx(1.0)
    assert train.learning_rate_multiplier(90, hp) == pytest.approx(0.5)
    assert train.learning_rate_multiplier(100, hp) == pytest.approx(0.0)


def test_cosine_schedule_uses_exact_warmup_steps():
    hp = train.Hyperparameters(
        num_iterations=8393,
        warmup_steps=1000,
        lr_schedule="cosine",
    )

    assert train.learning_rate_multiplier(0, hp) == pytest.approx(0.001)
    assert train.learning_rate_multiplier(999, hp) == pytest.approx(1.0)
    assert train.learning_rate_multiplier(1000, hp) == pytest.approx(1.0)
    midpoint = 1000 + (8393 - 1000) / 2
    assert train.learning_rate_multiplier(midpoint, hp) == pytest.approx(0.5)
    assert train.learning_rate_multiplier(8393, hp) == pytest.approx(0.0)


def test_gradient_norm_helper_clips_only_when_configured():
    unclipped = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    unclipped.grad = torch.tensor([3.0, 4.0])
    norm = train.compute_and_clip_grad_norm_([unclipped], None)
    assert norm.item() == pytest.approx(5.0)
    assert unclipped.grad.tolist() == [3.0, 4.0]

    clipped = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    clipped.grad = torch.tensor([3.0, 4.0])
    norm = train.compute_and_clip_grad_norm_([clipped], 1.0)
    assert norm.item() == pytest.approx(5.0)
    assert torch.linalg.vector_norm(clipped.grad).item() == pytest.approx(1.0)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_gradient_norm_helper_rejects_invalid_clip_norm(value):
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.ones(1)
    with pytest.raises(ValueError, match="grad_clip_norm"):
        train.compute_and_clip_grad_norm_([parameter], value)
