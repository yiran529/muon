import pytest
import torch

from dion.power_sgd import (
    PowerSGDConfig,
    compressed_phase,
    compute_left_factor,
    compute_right_factor,
    corrected_gradient,
    derive_power_sgd_seed,
    make_random_factor,
    orthogonalize,
    reconstruct,
    should_compress,
)


def test_should_compress_counts_both_factors():
    assert should_compress(8, 16, rank=2, min_compression_rate=2.0)
    assert not should_compress(4, 4, rank=2, min_compression_rate=2.0)


def test_config_defaults_and_rejects_invalid_values():
    assert PowerSGDConfig() == PowerSGDConfig(
        rank=1,
        start_compress_step=1000,
        min_compression_rate=2.0,
        error_feedback="ef14",
        warm_start=True,
        seed=42,
        orthogonalization_epsilon=1e-8,
        seed_scheme_version=1,
    )
    with pytest.raises(ValueError):
        PowerSGDConfig(rank=0)
    with pytest.raises(ValueError):
        PowerSGDConfig(error_feedback="bad")


def test_compressed_phase_is_zero_based_after_warmup():
    assert [compressed_phase(step, 7) for step in range(7, 12)] == [None, 0, 1, 2, 3]


def test_seed_and_random_factor_are_deterministic_without_global_rng_changes():
    seed = derive_power_sgd_seed(base_seed=42, phase=3, stable_parameter_id=17)
    assert seed == derive_power_sgd_seed(base_seed=42, phase=3, stable_parameter_id=17)
    assert seed != derive_power_sgd_seed(base_seed=42, phase=4, stable_parameter_id=17)
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    factor_a = make_random_factor(5, 2, seed, torch.device("cpu"), torch.bfloat16)
    observed = torch.rand(1)
    torch.manual_seed(123)
    _ = torch.rand(1)
    assert torch.equal(observed, expected)
    factor_b = make_random_factor(5, 2, seed, torch.device("cpu"), torch.bfloat16)
    assert torch.equal(factor_a, factor_b)
    assert factor_a.dtype == torch.bfloat16


def test_orthogonalize_accumulates_in_fp32_and_returns_input_dtype():
    matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.bfloat16)
    result = orthogonalize(matrix, epsilon=1e-8)
    assert result.dtype == matrix.dtype
    assert torch.allclose(result.float().T @ result.float(), torch.eye(2), atol=2e-2)


def test_power_sgd_factors_reconstruct_and_full_rank_is_exact():
    corrected = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16
    )
    right = orthogonalize(make_random_factor(3, 2, 9, corrected.device, corrected.dtype))
    left = compute_left_factor(corrected, right)
    right_factor = compute_right_factor(corrected, left)
    reconstructed = reconstruct(left, right_factor)
    assert left.shape == (2, 2)
    assert right_factor.shape == (3, 2)
    assert reconstructed.shape == corrected.shape
    assert reconstructed.dtype == corrected.dtype

    full_left = torch.eye(2, dtype=corrected.dtype)
    full_right = corrected.mT
    exact = reconstruct(full_left, full_right)
    assert torch.allclose(exact.float(), corrected.float(), atol=1e-5)


def test_corrected_gradient_and_error_feedback_recurrence():
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    error = torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=torch.float32)
    corrected = corrected_gradient(gradient, error)
    reconstructed = torch.tensor([[1.0, 1.0], [3.0, 5.0]], dtype=torch.bfloat16)
    next_error = corrected - reconstructed
    assert corrected.dtype == torch.float32
    assert torch.equal(next_error, corrected - reconstructed)
