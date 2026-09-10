import torch

from dion.greedy_lore import (
    approximate_signed_lambda,
    compress_local,
    corrected_gradient,
    reconstruct_global,
    refresh_basis,
    select_projector,
    matrix_orientation,
)


def _assert_close(actual, expected):
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_two_rank_three_step_recurrence_matches_paper_oracle():
    rank = 1
    update_interval = 2
    orientation = matrix_orientation((2, 3))
    gradients_rank0 = [
        torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]),
        torch.tensor([[0.5, -1.0, 2.0], [2.0, 0.0, -0.5]]),
        torch.tensor([[1.5, 0.0, -1.0], [0.5, 2.0, 1.0]]),
    ]
    gradients_rank1 = [
        torch.tensor([[2.0, 0.0, 1.0], [1.0, -1.0, 2.0]]),
        torch.tensor([[-1.0, 1.5, 0.5], [0.0, 2.5, -1.5]]),
        torch.tensor([[0.0, 1.0, 2.0], [2.0, -0.5, 0.5]]),
    ]
    random_vectors = [
        torch.tensor([[0.25, -1.0, 0.5], [1.5, 0.75, -0.25]]),
        torch.tensor([[-0.5, 1.25, 0.75], [0.5, -1.5, 1.0]]),
        torch.tensor([[1.0, 0.5, -0.75], [-1.25, 0.25, 0.5]]),
    ]
    error_rank0 = torch.zeros(2, 3)
    error_rank1 = torch.zeros(2, 3)
    basis = None

    for step in range(3):
        corrected_rank0 = corrected_gradient(
            gradients_rank0[step], error_rank0, orientation
        )
        corrected_rank1 = corrected_gradient(
            gradients_rank1[step], error_rank1, orientation
        )
        expected_corrected_rank0 = gradients_rank0[step].float() + error_rank0
        expected_corrected_rank1 = gradients_rank1[step].float() + error_rank1
        _assert_close(corrected_rank0, expected_corrected_rank0)
        _assert_close(corrected_rank1, expected_corrected_rank1)

        is_refresh = step % update_interval == 0
        if is_refresh:
            global_corrected = (corrected_rank0 + corrected_rank1) / 2
            basis, projector, support = refresh_basis(global_corrected, rank)
            expected_basis, _, _ = torch.linalg.svd(
                global_corrected, full_matrices=False
            )
            expected_basis = expected_basis * torch.where(
                expected_basis[expected_basis.abs().argmax(dim=0), torch.arange(2)] < 0,
                -torch.ones(2),
                torch.ones(2),
            ).unsqueeze(0)
            expected_support = torch.arange(rank, dtype=torch.int64)
            expected_projector = expected_basis.index_select(1, expected_support)
        else:
            assert basis is not None
            lambda_rank0 = approximate_signed_lambda(
                corrected_rank0, basis, random_vectors[step]
            )
            lambda_rank1 = approximate_signed_lambda(
                corrected_rank1, basis, random_vectors[step]
            )
            averaged_lambda = (lambda_rank0 + lambda_rank1) / 2
            scores = averaged_lambda.square()
            expected_support = torch.argsort(scores, descending=True, stable=True)[
                :rank
            ]
            projector, support = select_projector(basis, averaged_lambda, rank)
            expected_projector = basis.index_select(1, expected_support)

            _assert_close(
                lambda_rank0,
                (basis.mT @ corrected_rank0 * random_vectors[step]).sum(dim=1),
            )
            _assert_close(
                lambda_rank1,
                (basis.mT @ corrected_rank1 * random_vectors[step]).sum(dim=1),
            )
            _assert_close(scores, ((lambda_rank0 + lambda_rank1) / 2).square())

        assert torch.equal(support, expected_support)
        _assert_close(projector, expected_projector)

        local_rank0, next_error_rank0 = compress_local(corrected_rank0, projector)
        local_rank1, next_error_rank1 = compress_local(corrected_rank1, projector)
        expected_local_rank0 = projector.T @ corrected_rank0
        expected_local_rank1 = projector.T @ corrected_rank1
        expected_error_rank0 = corrected_rank0 - projector @ expected_local_rank0
        expected_error_rank1 = corrected_rank1 - projector @ expected_local_rank1
        averaged_factor = (local_rank0 + local_rank1) / 2
        reconstructed = reconstruct_global(projector, averaged_factor)
        expected_reconstructed = projector @ averaged_factor

        _assert_close(local_rank0, expected_local_rank0)
        _assert_close(local_rank1, expected_local_rank1)
        _assert_close(next_error_rank0, expected_error_rank0)
        _assert_close(next_error_rank1, expected_error_rank1)
        _assert_close(
            averaged_factor, (expected_local_rank0 + expected_local_rank1) / 2
        )
        _assert_close(reconstructed, expected_reconstructed)

        error_rank0 = next_error_rank0
        error_rank1 = next_error_rank1


def test_signed_lambda_is_squared_only_after_rank_average():
    basis = torch.eye(2)
    random_vectors = torch.ones(2, 2)
    corrected_rank0 = torch.tensor([[3.0, 0.0], [0.5, 0.5]])
    corrected_rank1 = torch.tensor([[-3.0, 0.0], [1.0, 1.0]])

    lambda_rank0 = approximate_signed_lambda(corrected_rank0, basis, random_vectors)
    lambda_rank1 = approximate_signed_lambda(corrected_rank1, basis, random_vectors)
    averaged_lambda = (lambda_rank0 + lambda_rank1) / 2
    mean_square_scores = (lambda_rank0.square() + lambda_rank1.square()) / 2

    projector, support = select_projector(basis, averaged_lambda, rank=1)

    assert torch.equal(lambda_rank0, torch.tensor([3.0, 1.0]))
    assert torch.equal(lambda_rank1, torch.tensor([-3.0, 2.0]))
    assert torch.equal(
        torch.argsort(mean_square_scores, descending=True)[:1], torch.tensor([0])
    )
    assert torch.equal(support, torch.tensor([1]))
    _assert_close(projector, torch.tensor([[0.0], [1.0]]))


def test_sigma_type_one_cross_check_agrees_without_ties_but_stable_ties_are_kept():
    basis = torch.eye(3)
    averaged_lambda = torch.tensor([-0.5, 2.0, -1.25])

    _, support = select_projector(basis, averaged_lambda, rank=3)
    abs_mean_support = torch.argsort(averaged_lambda.abs(), descending=True)[:3]

    assert torch.equal(support, abs_mean_support)

    tied_lambda = torch.tensor([2.0, -2.0, 1.0])
    _, tied_support = select_projector(basis, tied_lambda, rank=2)

    assert torch.equal(tied_support, torch.tensor([0, 1]))


def test_refresh_reduces_gradient_plus_old_error_before_error_reset():
    orientation = matrix_orientation((2, 3))
    gradient = torch.tensor([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    old_error = torch.tensor([[-4.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

    corrected = corrected_gradient(gradient, old_error, orientation)
    basis, projector, support = refresh_basis(corrected, rank=1)
    raw_basis, raw_projector, _ = refresh_basis(gradient, rank=1)

    _assert_close(corrected, torch.tensor([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]]))
    assert torch.equal(support, torch.tensor([0]))
    _assert_close(projector, torch.tensor([[0.0], [1.0]]))
    assert not torch.allclose(projector, raw_projector)
    assert not torch.allclose(basis, raw_basis)
