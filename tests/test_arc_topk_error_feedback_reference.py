"""Small independent recurrence oracles for ARC error-feedback modes."""

import torch

from dion.arc_topk import (
    finalize_arc_sparse_,
    gather_rows,
    prepare_arc_batch,
)
from dion.arc_topk_sync import ArcTopKSyncConfig


GRADIENTS = [
    torch.tensor([[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]]),
    torch.tensor([[[0.0, 0.0], [3.0, 3.0], [4.0, 4.0]]]),
    torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]),
    torch.tensor([[[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]]]),
]
SUPPORTS = [
    torch.tensor([[0]]),
    torch.tensor([[1]]),
    torch.tensor([[2]]),
    torch.tensor([[1]]),
]


def _scatter_selected(source, support):
    result = torch.zeros_like(source)
    result.scatter_(1, support.unsqueeze(-1).expand(-1, -1, source.shape[-1]), gather_rows(source, support))
    return result


def test_ef21m_production_primitives_match_independent_multistep_reference():
    config = ArcTopKSyncConfig(ratio=1 / 3, projection_rank=1, eta=1.0)
    tracker = torch.zeros_like(GRADIENTS[0])
    local_estimate = torch.zeros_like(tracker)
    global_estimate = torch.zeros_like(tracker)
    reference_estimate = torch.zeros_like(tracker)

    for step, (gradient, support) in enumerate(zip(GRADIENTS, SUPPORTS), start=2):
        reference_delta = gradient - reference_estimate
        reference_compressed = _scatter_selected(reference_delta, support)
        reference_estimate = reference_estimate + reference_compressed

        prepared = prepare_arc_batch(
            gradient,
            tracker,
            local_estimate,
            global_estimate,
            config=config,
            step=step,
            projection_batch=torch.ones(1, 2, 1),
        )
        selected = gather_rows(prepared.delta_batch, support)
        output = finalize_arc_sparse_(prepared, support, selected, selected)

        torch.testing.assert_close(tracker, gradient)
        torch.testing.assert_close(local_estimate, reference_estimate)
        torch.testing.assert_close(global_estimate, reference_estimate)
        torch.testing.assert_close(output, reference_estimate)


def test_ef14_production_primitives_match_reference_and_conservation_identity():
    from dion.arc_topk import finalize_ef14_sparse_, prepare_ef14_batch

    residual = torch.zeros_like(GRADIENTS[0])
    reference_residual = torch.zeros_like(residual)
    cumulative_gradient = torch.zeros_like(residual)
    cumulative_output = torch.zeros_like(residual)

    for gradient, support in zip(GRADIENTS, SUPPORTS):
        reference_input = gradient + reference_residual
        reference_output = _scatter_selected(reference_input, support)
        reference_residual = reference_input - reference_output

        prepared = prepare_ef14_batch(
            gradient,
            residual,
            projection_batch=torch.ones(1, 2, 1),
        )
        selected = gather_rows(prepared.compensated_batch, support)
        output = finalize_ef14_sparse_(prepared, support, selected)

        cumulative_gradient += gradient
        cumulative_output += output
        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(residual, reference_residual)
        torch.testing.assert_close(cumulative_output + residual, cumulative_gradient)


def test_error_feedback_mode_is_explicit_and_validated():
    assert ArcTopKSyncConfig(error_feedback="ef21m").error_feedback == "ef21m"
    assert ArcTopKSyncConfig(error_feedback="ef14").error_feedback == "ef14"
    try:
        ArcTopKSyncConfig(error_feedback="unknown")
    except ValueError as exc:
        assert "error_feedback" in str(exc)
    else:
        raise AssertionError("unknown error-feedback mode was accepted")
