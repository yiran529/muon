"""Tests for the shared ARC-TopK synchronization adapter."""

import math

import pytest
import torch

from dion.arc_topk_sync import (
    ArcTopKLogicalBytes,
    ArcTopKSyncConfig,
    average_gradients_async,
    estimate_arc_logical_bytes,
    group_parameters_by_shape_dtype,
    initialize_arc_state_,
)


def _run_generator(generator):
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def test_grouping_is_shape_dtype_stable():
    p0 = torch.nn.Parameter(torch.zeros(4, 3))
    p1 = torch.nn.Parameter(torch.zeros(2, 3))
    p2 = torch.nn.Parameter(torch.ones(4, 3))

    groups = group_parameters_by_shape_dtype([p0, p1, p2])

    assert groups == [[p0, p2], [p1]]


def test_initialize_arc_state_creates_distinct_matching_zero_tensors():
    param = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16))
    state = {}

    initialize_arc_state_(state, param)

    values = [state[key] for key in ("arc_h_local", "arc_g_local", "arc_g_global")]
    assert all(value.shape == param.shape for value in values)
    assert all(value.dtype == param.dtype for value in values)
    assert all(value.device == param.device for value in values)
    assert all(value is not param for value in values)
    assert len({id(value) for value in values}) == 3
    assert all(torch.count_nonzero(value) == 0 for value in values)


def test_initialize_arc_state_preserves_existing_values():
    param = torch.nn.Parameter(torch.ones(2, 2))
    existing = {
        "arc_h_local": torch.full_like(param, 3),
        "arc_g_local": torch.full_like(param, 4),
    }
    original_h = existing["arc_h_local"]
    original_g = existing["arc_g_local"]

    initialize_arc_state_(existing, param)
    initialize_arc_state_(existing, param)

    assert existing["arc_h_local"] is original_h
    assert existing["arc_g_local"] is original_g
    assert torch.equal(existing["arc_h_local"], torch.full_like(param, 3))
    assert torch.equal(existing["arc_g_local"], torch.full_like(param, 4))
    assert torch.count_nonzero(existing["arc_g_global"]) == 0


def test_arc_logical_bytes_count_dense_and_compressed_payloads():
    config = ArcTopKSyncConfig(ratio=0.2, projection_rank=4)
    compressed = [[torch.zeros(10, 8, dtype=torch.bfloat16) for _ in range(2)]]
    uncompressed = [torch.zeros(7, 8, dtype=torch.bfloat16)]

    result = estimate_arc_logical_bytes(
        compressed_batches=compressed,
        uncompressed_params=uncompressed,
        config=config,
        step=2,
    )

    assert result == ArcTopKLogicalBytes(
        dense_gradient=2 * 10 * 8 * 2,
        arc_seed=8,
        arc_sketch=2 * 10 * 4 * 2,
        arc_selected_values=2 * math.ceil(10 * 0.2) * 8 * 2,
        uncompressed=7 * 8 * 2,
    )


def test_arc_logical_bytes_use_dense_payload_before_compression_steps():
    config = ArcTopKSyncConfig(ratio=0.2, projection_rank=4, start_compress_step=2)
    compressed = [[torch.zeros(10, 8, dtype=torch.bfloat16) for _ in range(2)]]
    uncompressed = [torch.zeros(7, 8, dtype=torch.bfloat16)]

    result = estimate_arc_logical_bytes(
        compressed_batches=compressed,
        uncompressed_params=uncompressed,
        config=config,
        step=1,
    )

    assert result == ArcTopKLogicalBytes(
        dense_gradient=320,
        arc_seed=0,
        arc_sketch=0,
        arc_selected_values=0,
        uncompressed=112,
    )


def test_average_gradients_without_process_group_returns_copies():
    gradients = [torch.ones(2, 3), torch.full((2, 3), 2)]

    averaged = _run_generator(average_gradients_async(gradients, None))

    assert len(averaged) == len(gradients)
    for original, result in zip(gradients, averaged):
        assert result is not original
        torch.testing.assert_close(result, original)


@pytest.mark.parametrize("step", [0, 1, 2])
def test_arc_config_defaults_are_valid(step):
    config = ArcTopKSyncConfig()
    assert config.start_compress_step == 0
    assert step >= 0
