import pytest
import torch

import dion.power_sgd as power_sgd_module
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


def _reference_orthogonalize(matrix, epsilon=1e-8):
    work = matrix.float().clone()
    for index in range(work.shape[-1]):
        column = work[..., :, index]
        norm = torch.linalg.vector_norm(column, dim=-1, keepdim=True)
        normalized = column / (norm + epsilon)
        work[..., :, index] = normalized
        if index + 1 < work.shape[-1]:
            remaining = work[..., :, index + 1 :]
            coefficients = normalized.unsqueeze(-2) @ remaining
            remaining -= normalized.unsqueeze(-1) @ coefficients
    return work.to(dtype=matrix.dtype)


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
    with pytest.raises(ValueError):
        PowerSGDConfig(min_compression_rate=float("nan"))
    with pytest.raises(ValueError):
        PowerSGDConfig(min_compression_rate=float("inf"))
    with pytest.raises(ValueError):
        PowerSGDConfig(orthogonalization_epsilon=float("nan"))
    with pytest.raises(ValueError):
        PowerSGDConfig(orthogonalization_epsilon=float("inf"))


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


def test_orthogonalize_uses_scripted_gram_schmidt_core():
    assert isinstance(
        power_sgd_module._orthogonalize_gram_schmidt,
        torch.jit.ScriptFunction,
    )


def test_orthogonalize_does_not_clone_after_low_precision_conversion(monkeypatch):
    clone_dtypes = []
    original_clone = torch.Tensor.clone

    def observed_clone(tensor, *args, **kwargs):
        clone_dtypes.append(tensor.dtype)
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", observed_clone)

    orthogonalize(torch.randn(12, 4, dtype=torch.bfloat16))

    assert clone_dtypes == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(12, 4), (3, 12, 4)])
def test_scripted_orthogonalize_matches_reference_exactly(dtype, shape):
    generator = torch.Generator().manual_seed(17)
    matrix = torch.randn(shape, dtype=dtype, generator=generator)
    original = matrix.clone()

    result = orthogonalize(matrix, epsilon=1e-8)
    expected = _reference_orthogonalize(matrix, epsilon=1e-8)

    assert torch.equal(result, expected)
    assert torch.equal(matrix, original)


def test_orthogonalize_batches_matrices_without_changing_individual_results():
    matrices = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [1.0, 3.0], [0.0, 1.0]],
        ],
        dtype=torch.bfloat16,
    )

    batched = orthogonalize(matrices, epsilon=1e-8)
    individual = torch.stack(
        [orthogonalize(matrix, epsilon=1e-8) for matrix in matrices]
    )

    assert batched.dtype == matrices.dtype
    torch.testing.assert_close(batched, individual)
    torch.testing.assert_close(
        batched.float().mT @ batched.float(),
        torch.eye(2).expand(2, 2, 2),
        atol=2e-2,
        rtol=2e-2,
    )


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

    corrected_full = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=corrected.dtype)
    full_q = torch.eye(2, dtype=corrected.dtype)
    full_left = orthogonalize(compute_left_factor(corrected_full, full_q))
    full_right = compute_right_factor(corrected_full, full_left)
    exact = reconstruct(full_left, full_right)
    assert torch.allclose(exact.float(), corrected_full.float(), atol=1e-5)


def test_corrected_gradient_and_error_feedback_recurrence():
    gradient = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.bfloat16)
    error = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.bfloat16)
    corrected = corrected_gradient(gradient, error)
    right = torch.tensor([[1.0], [0.0]], dtype=torch.bfloat16)
    left = orthogonalize(compute_left_factor(corrected, right))
    reconstructed = reconstruct(left, compute_right_factor(corrected, left))
    next_error = corrected - reconstructed
    assert corrected.dtype == torch.bfloat16
    assert torch.equal(corrected, torch.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=torch.bfloat16))
    assert torch.equal(reconstructed, torch.tensor([[3.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16))
    assert torch.equal(next_error, torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=torch.bfloat16))
