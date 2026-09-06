"""Real-DDP characterization for where ``no_sync`` encloses the micro-step."""

import pytest

from benchmark.compressed_muon.no_sync_diagnostic import run_no_sync_case


def test_no_sync_placement_controls_reducer_calls_and_local_gradients():
    historical = run_no_sync_case("backward_only")
    dense = run_no_sync_case("dense_correct")
    optimizer = run_no_sync_case("optimizer_correct")

    assert historical["hook_calls_per_rank"] == [2, 2]
    assert dense["hook_calls_per_rank"] == [1, 1]
    assert optimizer["hook_calls_per_rank"] == [0, 0]

    assert historical["gradient_ranges"]["max_abs"] == pytest.approx(0.0)
    assert dense["gradient_ranges"]["max_abs"] == pytest.approx(0.0)
    assert optimizer["gradient_ranges"]["max_abs"] > 0.0

    assert historical["passed"] is True
    assert dense["passed"] is True
    assert optimizer["passed"] is True


def test_no_sync_diagnostic_rejects_an_unknown_pattern():
    with pytest.raises(ValueError, match="pattern"):
        run_no_sync_case("unknown")
