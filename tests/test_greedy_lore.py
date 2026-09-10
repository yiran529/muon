import math

import pytest
import torch

from dion.greedy_lore import (
    GreedyLoreConfig,
    canonicalize_svd_basis,
    compressed_phase,
    derive_greedy_lore_seed,
    is_refresh_step,
    make_random_vectors,
    matrix_orientation,
    orient_matrix,
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
        rows=3, columns=4, seed=seed, device=torch.device("cpu")
    )
    second = make_random_vectors(
        rows=3, columns=4, seed=seed, device=torch.device("cpu")
    )

    assert torch.equal(first, second)
    assert first.shape == (3, 4)
    assert first.dtype == torch.float32
    assert first.device == torch.device("cpu")
    assert torch.equal(torch.random.get_rng_state(), before)


def test_canonicalize_svd_basis_uses_smallest_maximum_index_as_positive_pivot():
    basis = torch.tensor([[-0.5, 0.0], [-0.5, -1.0]])

    result = canonicalize_svd_basis(basis)

    assert result[0, 0] > 0
    assert result[1, 1] > 0
    assert torch.allclose(result[:, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(result.T @ result, basis.T @ basis)
