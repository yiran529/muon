import math

import pytest
import torch

from dion.greedy_lore import (
    GreedyLoreConfig,
    approximate_signed_lambda,
    canonicalize_svd_basis,
    compress_local,
    compressed_phase,
    corrected_gradient,
    derive_greedy_lore_seed,
    is_refresh_step,
    make_random_vectors,
    matrix_orientation,
    orient_matrix,
    reconstruct_global,
    refresh_basis,
    select_projector,
    unorient_matrix,
)


def test_config_defaults_match_spec():
    config = GreedyLoreConfig()

    assert config.rank == 32
    assert config.update_interval == 200
    assert config.seed == 42
    assert config.start_compress_step == 1000
    assert config.basis_sync == "local_svd"
    assert config.seed_scheme_version == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"update_interval": 0},
        {"start_compress_step": -1},
        {"basis_sync": "unknown"},
        {"seed_scheme_version": 0},
        {"seed_scheme_version": 2},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GreedyLoreConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": True},
        {"update_interval": False},
        {"seed": True},
        {"start_compress_step": True},
        {"seed_scheme_version": True},
    ],
)
def test_config_rejects_boolean_integer_values(kwargs):
    with pytest.raises(TypeError):
        GreedyLoreConfig(**kwargs)


def test_config_is_frozen():
    config = GreedyLoreConfig()

    with pytest.raises((AttributeError, TypeError)):
        config.rank = 8


def test_first_compressed_step_is_refresh_independent_of_absolute_step():
    config = GreedyLoreConfig(start_compress_step=7, update_interval=3)

    assert [compressed_phase(step, 7) for step in range(7, 12)] == [None, 0, 1, 2, 3]
    assert [is_refresh_step(step, config) for step in range(7, 12)] == [
        False,
        True,
        False,
        False,
        True,
    ]


def test_update_interval_one_refreshes_every_compressed_step():
    config = GreedyLoreConfig(start_compress_step=2, update_interval=1)

    assert [is_refresh_step(step, config) for step in range(2, 7)] == [
        False,
        True,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize(
    "shape,expected,transposed",
    [
        ((3, 5), (3, 5), False),
        ((5, 3), (3, 5), True),
        ((4, 4), (4, 4), False),
    ],
)
def test_orientation_round_trip(shape, expected, transposed):
    tensor = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    orientation = matrix_orientation(shape)

    assert orientation.original_shape == shape
    assert orientation.compressed_shape == expected
    assert orientation.transposed is transposed
    assert torch.equal(
        unorient_matrix(orient_matrix(tensor, orientation), orientation), tensor
    )


def test_matrix_orientation_requires_two_dimensional_input():
    with pytest.raises(ValueError, match="two-dimensional input"):
        matrix_orientation((2, 3, 4))


def test_transposed_orientation_uses_storage_aliasing_views():
    tensor = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    orientation = matrix_orientation(tensor.shape)

    oriented = orient_matrix(tensor, orientation)
    restored = unorient_matrix(oriented, orientation)

    assert oriented.data_ptr() == tensor.data_ptr()
    assert restored.data_ptr() == tensor.data_ptr()


def test_corrected_gradient_handles_square_and_transposed_matrices():
    square_gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    square_error = torch.tensor([[0.5, -0.5], [1.5, -1.0]], dtype=torch.bfloat16)
    square_orientation = matrix_orientation(square_gradient.shape)

    square_corrected = corrected_gradient(
        square_gradient, square_error, square_orientation
    )

    assert square_orientation.transposed is False
    assert square_corrected.dtype == torch.bfloat16
    assert torch.allclose(
        square_corrected,
        torch.tensor([[1.5, 1.5], [4.5, 3.0]], dtype=torch.bfloat16),
    )

    tall_gradient = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
    )
    tall_error = torch.tensor([[0.25, -0.5, 0.75], [1.0, -1.25, 1.5]])
    tall_orientation = matrix_orientation(tall_gradient.shape)

    tall_corrected = corrected_gradient(tall_gradient, tall_error, tall_orientation)

    assert tall_orientation.transposed is True
    assert tall_corrected.shape == (2, 3)
    assert torch.allclose(tall_corrected, tall_gradient.mT + tall_error)


def test_rows_less_than_columns_matches_selecting_columns_from_u():
    tensor = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=torch.float32)
    orientation = matrix_orientation(tensor.shape)
    normalized = orient_matrix(tensor, orientation)
    original_u, _, _ = torch.linalg.svd(tensor, full_matrices=False)
    normalized_u, _, _ = torch.linalg.svd(normalized, full_matrices=False)
    original_u = canonicalize_svd_basis(original_u)
    normalized_u = canonicalize_svd_basis(normalized_u)
    rank = 1
    support = torch.arange(rank)

    expected = original_u.index_select(1, support).T @ tensor
    actual = normalized_u.index_select(1, support).T @ normalized

    assert torch.allclose(actual, expected)


def test_rows_greater_than_columns_matches_selecting_rows_from_vh_and_right_multiplying():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 5.0], [4.0, 7.0]], dtype=torch.float32)
    orientation = matrix_orientation(tensor.shape)
    normalized = orient_matrix(tensor, orientation)
    _, _, original_vh = torch.linalg.svd(tensor, full_matrices=False)
    normalized_u, _, _ = torch.linalg.svd(normalized, full_matrices=False)
    original_v = canonicalize_svd_basis(original_vh.T)
    normalized_u = canonicalize_svd_basis(normalized_u)
    rank = 1

    selected_rows = original_v[:, :rank].T
    expected = tensor @ selected_rows.T
    actual = (normalized_u[:, :rank].T @ normalized).T

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_derive_greedy_lore_seed_uses_version_one_formula():
    seed = derive_greedy_lore_seed(
        base_seed=42,
        phase=5,
        stable_parameter_id=9,
    )

    assert seed == (42 + 5 * 1_000_003 + 9) % (2**63 - 1)


def test_random_vectors_are_local_and_reproducible():
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    seed = derive_greedy_lore_seed(
        base_seed=42,
        phase=5,
        stable_parameter_id=9,
    )

    first = make_random_vectors(
        rows=3,
        columns=4,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = make_random_vectors(
        rows=3,
        columns=4,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(first, second)
    assert first.shape == (3, 4)
    assert first.dtype == torch.float32
    assert first.device == torch.device("cpu")
    assert torch.equal(torch.random.get_rng_state(), before)


def test_random_vectors_include_negative_standard_normal_values():
    vectors = make_random_vectors(
        rows=8,
        columns=8,
        seed=42,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert vectors.dtype == torch.bfloat16
    assert (vectors < 0).any()
    assert (vectors > 0).any()


def test_canonicalize_svd_basis_uses_smallest_maximum_index_as_positive_pivot():
    basis = torch.tensor([[-0.5, 0.0], [-0.5, -1.0]])

    result = canonicalize_svd_basis(basis)

    assert result[0, 0] > 0
    assert result[1, 1] > 0
    assert torch.allclose(result[:, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(result.T @ result, basis.T @ basis)


def test_zero_corrected_gradient_stays_zero_through_recurrence_primitives():
    corrected = torch.zeros(2, 3)
    basis = torch.eye(2)
    random_vectors = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])

    signed_lambda = approximate_signed_lambda(corrected, basis, random_vectors)
    projector, support = select_projector(basis, signed_lambda, rank=1)
    local_factor, next_error = compress_local(corrected, projector)
    reconstructed = reconstruct_global(projector, local_factor)

    assert torch.equal(signed_lambda, torch.zeros(2))
    assert torch.equal(support, torch.tensor([0]))
    assert torch.equal(local_factor, torch.zeros(1, 3))
    assert torch.equal(next_error, torch.zeros(2, 3))
    assert torch.equal(reconstructed, torch.zeros(2, 3))


def test_select_projector_uses_stable_score_ties():
    basis = torch.eye(3)
    averaged_lambda = torch.tensor([2.0, -2.0, 1.0])

    projector, support = select_projector(basis, averaged_lambda, rank=2)

    assert torch.equal(support, torch.tensor([0, 1]))
    assert torch.equal(projector, torch.eye(3)[:, :2])


def test_full_rank_recurrence_round_trip_uses_projector_path():
    global_corrected = torch.tensor([[2.0, 1.0, 0.5], [0.25, 3.0, 4.0]])
    basis, _, _ = refresh_basis(global_corrected, rank=2)
    averaged_lambda = torch.tensor([0.25, -0.75])
    projector, support = select_projector(basis, averaged_lambda, rank=2)
    corrected = torch.tensor([[1.0, -2.0, 0.5], [3.0, 1.5, -4.0]])

    local_factor, next_error = compress_local(corrected, projector)
    reconstructed = reconstruct_global(projector, local_factor)

    assert torch.equal(support, torch.tensor([1, 0]))
    assert local_factor.shape == (2, 3)
    assert torch.allclose(next_error, torch.zeros_like(corrected), atol=1e-5, rtol=1e-5)
    assert torch.allclose(reconstructed, corrected, atol=1e-5, rtol=1e-5)


def test_recurrence_primitives_keep_bucket_dtype_after_bfloat16_gradient(monkeypatch):
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    error = torch.tensor([[0.125, -0.25], [0.5, -1.0]], dtype=torch.bfloat16)
    orientation = matrix_orientation(gradient.shape)
    original_svd = torch.linalg.svd
    observed_svd_dtypes = []

    def recording_svd(value, *args, **kwargs):
        observed_svd_dtypes.append(value.dtype)
        return original_svd(value, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", recording_svd)

    corrected = corrected_gradient(gradient, error, orientation)
    basis, projector, _ = refresh_basis(corrected, rank=1)
    signed_lambda = approximate_signed_lambda(
        corrected, basis, torch.ones_like(corrected)
    )
    local_factor, next_error = compress_local(corrected, projector)
    reconstructed = reconstruct_global(projector, local_factor)

    assert observed_svd_dtypes == [torch.float32]
    assert corrected.dtype == torch.bfloat16
    assert basis.dtype == torch.bfloat16
    assert projector.dtype == torch.bfloat16
    assert signed_lambda.dtype == torch.bfloat16
    assert local_factor.dtype == torch.bfloat16
    assert next_error.dtype == torch.bfloat16
    assert reconstructed.dtype == torch.bfloat16


def test_refresh_basis_uses_left_gram_path_for_very_wide_matrix(monkeypatch):
    corrected = torch.tensor(
        [
            [4.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
        ]
    )

    def reject_svd(*_args, **_kwargs):
        raise AssertionError("very wide refresh must not materialize an SVD Vh")

    monkeypatch.setattr(torch.linalg, "svd", reject_svd)

    basis, projector, support = refresh_basis(corrected, rank=2)

    assert torch.equal(support, torch.tensor([0, 1]))
    assert torch.equal(projector, basis[:, :2])
    assert torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=1e-5)
    pivot_rows = basis.abs().argmax(dim=0)
    assert torch.all(basis[pivot_rows, torch.arange(3)] >= 0)
