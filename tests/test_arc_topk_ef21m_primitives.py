"""Oracle tests for collective-free ARC/EF21M prepare and finalize primitives."""

import math

import pytest
import torch

from dion.arc_topk import (
    finalize_arc_full_support_,
    finalize_arc_sparse_,
    prepare_arc_batch,
)
from dion.arc_topk_sync import ArcTopKSyncConfig


def _config(*, ratio=0.5, eta=0.5, start_compress_step=0):
    return ArcTopKSyncConfig(
        ratio=ratio,
        projection_rank=2,
        eta=eta,
        seed=17,
        start_compress_step=start_compress_step,
    )


@pytest.mark.parametrize("step", [1, 2])
def test_first_step_and_final_warmup_step_use_the_complete_tracker(step):
    gradient = torch.tensor([[[4.0, 8.0], [12.0, 16.0]]])
    tracker = torch.full_like(gradient, 2.0)
    local_estimate = torch.full_like(gradient, 3.0)
    global_estimate = torch.full_like(gradient, 5.0)

    prepared = prepare_arc_batch(
        gradient,
        tracker,
        local_estimate,
        global_estimate,
        config=_config(eta=0.25, start_compress_step=2),
        step=step,
        projection_batch=None,
    )
    averaged_tracker = torch.tensor([[[3.0, 5.0], [7.0, 9.0]]])
    result = finalize_arc_full_support_(prepared, averaged_tracker)

    expected_tracker = gradient if step == 1 else torch.tensor(
        [[[2.5, 3.5], [4.5, 5.5]]]
    )
    torch.testing.assert_close(tracker, expected_tracker)
    torch.testing.assert_close(local_estimate, expected_tracker)
    torch.testing.assert_close(global_estimate, averaged_tracker)
    assert result is global_estimate
    assert prepared.delta_batch is None
    assert prepared.local_sketch_batch is None


def test_ratio_one_uses_full_support_without_a_projection():
    gradient = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0]],
            [[8.0, 0.0], [0.0, 8.0]],
        ]
    )
    tracker = torch.zeros_like(gradient)
    local_estimate = torch.ones_like(gradient)
    global_estimate = torch.full_like(gradient, 2.0)

    prepared = prepare_arc_batch(
        gradient,
        tracker,
        local_estimate,
        global_estimate,
        config=_config(ratio=1.0, eta=0.25),
        step=3,
        projection_batch=None,
    )
    averaged_tracker = torch.tensor(
        [
            [[0.5, 0.0], [0.0, 0.5]],
            [[1.5, 0.0], [0.0, 1.5]],
        ]
    )
    finalize_arc_full_support_(prepared, averaged_tracker)

    torch.testing.assert_close(tracker, gradient * 0.25)
    torch.testing.assert_close(local_estimate, gradient * 0.25)
    torch.testing.assert_close(global_estimate, averaged_tracker)
    assert prepared.delta_batch is None


def test_first_and_subsequent_sparse_steps_follow_hand_calculated_ef21m_state():
    tracker = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    local_estimate = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    global_estimate = torch.tensor([[[0.5, 0.0], [0.0, 0.5]]])
    projection = torch.eye(2).unsqueeze(0)

    first = prepare_arc_batch(
        torch.tensor([[[4.0, 0.0], [0.0, 0.0]]]),
        tracker,
        local_estimate,
        global_estimate,
        config=_config(),
        step=2,
        projection_batch=projection,
    )

    torch.testing.assert_close(tracker, torch.tensor([[[3.0, 0.0], [0.0, 1.0]]]))
    torch.testing.assert_close(first.delta_batch, torch.tensor([[[2.0, 0.0], [0.0, 0.0]]]))
    torch.testing.assert_close(first.local_sketch_batch, first.delta_batch / math.sqrt(2.0))
    result = finalize_arc_sparse_(
        first,
        indices=torch.tensor([[0]]),
        local_selected=torch.tensor([[[2.0, 0.0]]]),
        averaged_selected=torch.tensor([[[1.0, 0.0]]]),
    )

    torch.testing.assert_close(local_estimate, torch.tensor([[[3.0, 0.0], [0.0, 1.0]]]))
    torch.testing.assert_close(global_estimate, torch.tensor([[[1.5, 0.0], [0.0, 0.5]]]))
    assert result is global_estimate

    second = prepare_arc_batch(
        torch.tensor([[[1.0, 0.0], [0.0, 5.0]]]),
        tracker,
        local_estimate,
        global_estimate,
        config=_config(),
        step=3,
        projection_batch=projection,
    )
    finalize_arc_sparse_(
        second,
        indices=torch.tensor([[1]]),
        local_selected=torch.tensor([[[0.0, 2.0]]]),
        averaged_selected=torch.tensor([[[0.0, 1.0]]]),
    )

    torch.testing.assert_close(tracker, torch.tensor([[[2.0, 0.0], [0.0, 3.0]]]))
    torch.testing.assert_close(local_estimate, torch.tensor([[[3.0, 0.0], [0.0, 3.0]]]))
    torch.testing.assert_close(global_estimate, torch.tensor([[[1.5, 0.0], [0.0, 1.5]]]))


def test_sparse_primitives_preserve_bfloat16_state_dtype():
    dtype = torch.bfloat16
    gradient = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]], dtype=dtype)
    tracker = torch.zeros_like(gradient)
    local_estimate = torch.zeros_like(gradient)
    global_estimate = torch.zeros_like(gradient)

    prepared = prepare_arc_batch(
        gradient,
        tracker,
        local_estimate,
        global_estimate,
        config=_config(eta=0.25),
        step=2,
        projection_batch=torch.eye(2, dtype=dtype).unsqueeze(0),
    )
    finalize_arc_sparse_(
        prepared,
        indices=torch.tensor([[1]]),
        local_selected=torch.tensor([[[1.5, 2.0]]], dtype=dtype),
        averaged_selected=torch.tensor([[[0.75, 1.0]]], dtype=dtype),
    )

    assert tracker.dtype == dtype
    assert local_estimate.dtype == dtype
    assert global_estimate.dtype == dtype
    torch.testing.assert_close(tracker, gradient * 0.25)
    torch.testing.assert_close(
        local_estimate,
        torch.tensor([[[0.0, 0.0], [1.5, 2.0]]], dtype=dtype),
    )
    torch.testing.assert_close(
        global_estimate,
        torch.tensor([[[0.0, 0.0], [0.75, 1.0]]], dtype=dtype),
    )


def test_sparse_prepare_requires_projection_with_matching_batch_shape():
    gradient = torch.zeros(1, 2, 2)

    with pytest.raises(ValueError, match="projection_batch"):
        prepare_arc_batch(
            gradient,
            torch.zeros_like(gradient),
            torch.zeros_like(gradient),
            torch.zeros_like(gradient),
            config=_config(),
            step=2,
            projection_batch=None,
        )
