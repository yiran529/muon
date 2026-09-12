"""Unit tests for the standalone Rand-K and Top-K primitives."""

import pytest
import torch

from dion.sparse_k import (
    SparseKConfig,
    derive_sparse_k_seed,
    select_randk_indices,
    select_topk_indices,
    sparse_k_count,
    update_ef14_residual_,
)


@pytest.mark.parametrize("ratio,expected", [(0.01, 1), (0.5, 5), (1.0, 10)])
def test_sparse_k_count_uses_a_clamped_floor(ratio, expected):
    assert sparse_k_count(10, ratio) == expected


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"method": "other"}, "method"),
        ({"ratio": 0.0}, "ratio"),
        ({"ratio": 1.1}, "ratio"),
        ({"error_feedback": "ef21"}, "error_feedback"),
        ({"start_compress_step": -1}, "start_compress_step"),
    ],
)
def test_sparse_k_config_rejects_unsupported_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SparseKConfig(**kwargs)


def test_topk_selects_largest_absolute_values_with_local_indices():
    values = torch.tensor([1.0, -7.0, 3.0, -5.0])

    indices = select_topk_indices(values, 2)

    assert set(indices.tolist()) == {1, 3}
    assert indices.dtype == torch.int64


def test_randk_is_reproducible_unique_and_does_not_change_global_rng():
    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)

    first = select_randk_indices(20, 7, seed=919, device=torch.device("cpu"))
    actual_next = torch.rand(4)
    second = select_randk_indices(20, 7, seed=919, device=torch.device("cpu"))

    assert torch.equal(first, second)
    assert first.unique().numel() == 7
    assert torch.equal(actual_next, expected_next)


def test_seed_changes_with_step_and_stable_parameter_id():
    baseline = derive_sparse_k_seed(base_seed=17, step=3, stable_parameter_id=5)

    assert derive_sparse_k_seed(base_seed=17, step=4, stable_parameter_id=5) != baseline
    assert derive_sparse_k_seed(base_seed=17, step=3, stable_parameter_id=6) != baseline
    assert 0 <= baseline < 2**63


def test_ef14_residual_uses_the_local_compressor_output():
    residual = torch.tensor([0.5, -0.5, 1.0, -1.0])
    compensated = torch.tensor([2.0, 4.0, 6.0, 8.0])
    indices = torch.tensor([1, 3])
    local_values = torch.tensor([4.0, 8.0])

    update_ef14_residual_(residual, compensated, indices, local_values)

    torch.testing.assert_close(residual, torch.tensor([2.0, 0.0, 6.0, 0.0]))
