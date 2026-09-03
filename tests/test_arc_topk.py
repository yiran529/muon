"""Unit tests for ARC-TopK tensor operations and EF21M state updates."""

import math

import pytest
import torch

from dion.arc_topk import (
    arc_topk_ef21m_async,
    arc_topk_local_sketch,
    arc_topk_support,
    ef21m_apply_delta_,
    ef21m_update_tracker_,
    gather_rows,
    make_gaussian_projection,
    scatter_rows,
    validate_arc_topk_config,
)


def _run_generator(generator):
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


@pytest.mark.parametrize(
    "ratio,projection_rank,eta",
    [
        (0.0, 2, 0.5),
        (1.1, 2, 0.5),
        (0.5, 0, 0.5),
        (0.5, True, 0.5),
        (0.5, 2, 0.0),
        (0.5, 2, 1.1),
    ],
)
def test_invalid_arc_config_is_rejected(ratio, projection_rank, eta):
    with pytest.raises(ValueError):
        validate_arc_topk_config(ratio, projection_rank, eta)


def test_valid_arc_config_is_accepted():
    validate_arc_topk_config(0.25, 4, 0.1)
    validate_arc_topk_config(1.0, 1, 1.0)


@pytest.mark.parametrize("start_compress_step", [-1, True, 1.5])
def test_invalid_start_compress_step_is_rejected(start_compress_step):
    with pytest.raises(ValueError):
        validate_arc_topk_config(0.25, 4, 0.1, start_compress_step)


def test_valid_start_compress_step_is_accepted():
    validate_arc_topk_config(0.25, 4, 0.1, 0)
    validate_arc_topk_config(0.25, 4, 0.1, 1000)


def test_gaussian_projection_is_reproducible_without_using_global_rng():
    torch.manual_seed(1234)
    expected_next_global_draw = torch.randn(1)
    torch.manual_seed(1234)

    first = make_gaussian_projection(
        2,
        5,
        3,
        seed=17,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = make_gaussian_projection(
        2,
        5,
        3,
        seed=17,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first.shape == (2, 5, 3)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.randn(1), expected_next_global_draw)


def test_local_sketch_applies_batched_projection_and_rank_normalization():
    delta = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    projection = torch.eye(2).unsqueeze(0)

    sketch = arc_topk_local_sketch(delta, projection)

    torch.testing.assert_close(sketch, delta / math.sqrt(2.0))


def test_local_sketch_preserves_bfloat16_communication_dtype():
    delta = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
    projection = torch.eye(2, dtype=torch.bfloat16).unsqueeze(0)

    sketch = arc_topk_local_sketch(delta, projection)

    assert sketch.dtype == torch.bfloat16


def test_support_selects_rows_with_largest_sketch_norms():
    sketch = torch.tensor([[[3.0], [1.0], [4.0], [2.0]]])

    indices = arc_topk_support(sketch, k=2)

    assert indices.tolist() == [[2, 0]]


def test_gather_and_scatter_preserve_only_selected_rows():
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
        ]
    )
    indices = torch.tensor([[3, 1], [0, 2]])

    selected = gather_rows(values, indices)
    rebuilt = scatter_rows(selected, indices, rows=4)

    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                [[7.0, 8.0], [3.0, 4.0]],
                [[9.0, 10.0], [13.0, 14.0]],
            ]
        ),
    )
    torch.testing.assert_close(
        rebuilt,
        torch.tensor(
            [
                [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0], [7.0, 8.0]],
                [[9.0, 10.0], [0.0, 0.0], [13.0, 14.0], [0.0, 0.0]],
            ]
        ),
    )


def test_ef21m_tracker_matches_equation_11a_over_multiple_steps():
    tracker = torch.zeros(1, 2, 2)
    first_gradient = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    second_gradient = torch.tensor([[[10.0, 8.0], [6.0, 4.0]]])

    ef21m_update_tracker_(tracker, first_gradient, eta=0.25)
    torch.testing.assert_close(tracker, 0.25 * first_gradient)

    ef21m_update_tracker_(tracker, second_gradient, eta=0.25)
    expected = 0.75 * (0.25 * first_gradient) + 0.25 * second_gradient
    torch.testing.assert_close(tracker, expected)


def test_ef21m_estimates_accumulate_local_and_global_compressed_deltas():
    local_estimate = torch.zeros(1, 2, 2)
    global_estimate = torch.zeros_like(local_estimate)
    local_delta = torch.tensor([[[0.5, 1.0], [0.0, 0.0]]])
    averaged_delta = torch.tensor([[[0.25, 0.75], [0.0, 0.0]]])

    ef21m_apply_delta_(
        local_estimate,
        global_estimate,
        local_delta,
        averaged_delta,
    )
    ef21m_apply_delta_(
        local_estimate,
        global_estimate,
        local_delta,
        averaged_delta,
    )

    torch.testing.assert_close(local_estimate, 2 * local_delta)
    torch.testing.assert_close(global_estimate, 2 * averaged_delta)


def test_first_step_dense_initializes_h_and_g_before_compression():
    gradient = torch.arange(12.0).reshape(4, 3)
    tracker = torch.zeros_like(gradient)
    local_estimate = torch.zeros_like(gradient)
    global_estimate = torch.zeros_like(gradient)

    result = _run_generator(
        arc_topk_ef21m_async(
            gradients=[gradient],
            trackers=[tracker],
            local_estimates=[local_estimate],
            global_estimates=[global_estimate],
            process_group=None,
            ratio=0.25,
            projection_rank=2,
            eta=0.1,
            base_seed=17,
            step=1,
            task_index=0,
            start_compress_step=0,
        )
    )

    torch.testing.assert_close(tracker, gradient)
    torch.testing.assert_close(local_estimate, gradient)
    torch.testing.assert_close(global_estimate, gradient)
    torch.testing.assert_close(result[0], gradient)


def test_warmup_uses_dense_tracker_update_through_configured_step():
    previous = torch.full((4, 3), 2.0)
    gradient = torch.arange(12.0).reshape(4, 3)
    tracker = previous.clone()
    local_estimate = previous.clone()
    global_estimate = previous.clone()

    result = _run_generator(
        arc_topk_ef21m_async(
            gradients=[gradient],
            trackers=[tracker],
            local_estimates=[local_estimate],
            global_estimates=[global_estimate],
            process_group=None,
            ratio=0.25,
            projection_rank=2,
            eta=0.25,
            base_seed=17,
            step=1000,
            task_index=0,
            start_compress_step=1000,
        )
    )

    expected_tracker = 0.75 * previous + 0.25 * gradient
    torch.testing.assert_close(tracker, expected_tracker)
    torch.testing.assert_close(local_estimate, expected_tracker)
    torch.testing.assert_close(global_estimate, expected_tracker)
    torch.testing.assert_close(result[0], expected_tracker)


def test_step_after_warmup_uses_bfloat16_arc_compression():
    gradient = torch.arange(12.0, dtype=torch.bfloat16).reshape(4, 3)
    tracker = torch.zeros_like(gradient)
    local_estimate = torch.zeros_like(gradient)
    global_estimate = torch.zeros_like(gradient)

    projection = make_gaussian_projection(
        1,
        3,
        2,
        seed=17 + 1001 * 1_000_003,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert projection.dtype == torch.bfloat16

    result = _run_generator(
        arc_topk_ef21m_async(
            gradients=[gradient],
            trackers=[tracker],
            local_estimates=[local_estimate],
            global_estimates=[global_estimate],
            process_group=None,
            ratio=0.25,
            projection_rank=2,
            eta=1.0,
            base_seed=17,
            step=1001,
            task_index=0,
            start_compress_step=1000,
        )
    )

    assert result[0].dtype == torch.bfloat16
    assert torch.count_nonzero(local_estimate).item() == 3
    torch.testing.assert_close(global_estimate, local_estimate)
