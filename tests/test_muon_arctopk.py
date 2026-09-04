"""Single-process tests for ARC-TopK-EF21M-Muon integration."""

import copy
from unittest.mock import patch

import pytest
import torch

from dion import ArcTopKMuon
import dion.muon_arctopk as muon_arctopk_module
from dion.arc_topk_sync import ArcTopKSyncConfig


def _identity_orthogonalizer(x, epsilon=None):
    return x


def _make_optimizer(param, **kwargs):
    options = dict(
        lr=0.125,
        mu=0.0,
        weight_decay=0.0,
        nesterov=False,
        adjust_lr=None,
        newton_schulz_func=_identity_orthogonalizer,
        arc_topk_ratio=1.0,
        arc_projection_rank=2,
        arc_eta=1.0,
        arc_seed=17,
        arc_start_compress_step=0,
    )
    options.update(kwargs)
    return ArcTopKMuon([param], **options)


def test_arc_topk_muon_prepopulates_full_state():
    parameter = torch.nn.Parameter(torch.arange(32.0).reshape(8, 4))

    optimizer = _make_optimizer(parameter)

    state = optimizer.state[parameter]
    assert state.keys() >= {
        "momentum",
        "arc_h_local",
        "arc_g_local",
        "arc_g_global",
    }
    for name in ("arc_h_local", "arc_g_local", "arc_g_global"):
        assert state[name].shape == parameter.shape
        assert torch.count_nonzero(state[name]).item() == 0


def test_arc_topk_muon_routes_sync_through_shared_adapter():
    first = torch.nn.Parameter(torch.zeros(4, 3))
    second = torch.nn.Parameter(torch.zeros(2, 3))
    first_sentinel = torch.full_like(first, 2.5)
    second_sentinel = torch.full_like(second, -3.5)
    calls = []

    def fake_synchronize(**kwargs):
        calls.append(kwargs)
        yield
        return [
            first_sentinel if kwargs["task_index"] == 0 else second_sentinel,
        ]

    optimizer = ArcTopKMuon(
        [{"params": [first]}, {"params": [second]}],
        arc_topk_ratio=0.2,
        arc_projection_rank=4,
        arc_eta=0.1,
        arc_seed=42,
        arc_start_compress_step=0,
        lr=0.125,
        mu=0.0,
        weight_decay=0.0,
        nesterov=False,
        adjust_lr=None,
        newton_schulz_func=_identity_orthogonalizer,
    )
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)

    with patch.object(
        muon_arctopk_module,
        "synchronize_arc_batch_async",
        fake_synchronize,
        create=True,
    ):
        optimizer.step()

    assert [call["task_index"] for call in calls] == [0, 1]
    assert all(
        call["config"] == ArcTopKSyncConfig(0.2, 4, 0.1, 42, 0)
        for call in calls
    )
    torch.testing.assert_close(optimizer.state[first]["momentum"], first_sentinel)
    torch.testing.assert_close(optimizer.state[second]["momentum"], second_sentinel)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arc_topk_ratio": 0.0},
        {"arc_topk_ratio": 1.1},
        {"arc_projection_rank": 0},
        {"arc_projection_rank": True},
        {"arc_eta": 0.0},
        {"arc_eta": 1.1},
        {"arc_start_compress_step": -1},
        {"arc_start_compress_step": True},
    ],
)
def test_arc_topk_muon_rejects_invalid_arc_configuration(kwargs):
    parameter = torch.nn.Parameter(torch.zeros(4, 3))

    with pytest.raises(ValueError):
        _make_optimizer(parameter, **kwargs)


def test_arc_topk_muon_rejects_unsupported_matrix_layout_options():
    parameter = torch.nn.Parameter(torch.zeros(4, 4))

    with pytest.raises(ValueError, match="flatten"):
        _make_optimizer(parameter, flatten=True)
    with pytest.raises(ValueError, match="num_heads"):
        ArcTopKMuon(
            [{"params": [parameter], "num_heads": 2}],
            newton_schulz_func=_identity_orthogonalizer,
        )
    with pytest.raises(ValueError, match="split_sizes"):
        ArcTopKMuon(
            [{"params": [parameter], "split_sizes": (2, 2)}],
            newton_schulz_func=_identity_orthogonalizer,
        )


def test_ratio_one_first_step_feeds_dense_gradient_into_existing_muon_path():
    initial = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )
    gradient = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 4.0, 2.0], [6.0, 2.0, 0.0], [1.0, 3.0, 5.0]]
    )
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = _make_optimizer(parameter)
    parameter.grad = gradient.clone()

    optimizer.step()

    state = optimizer.state[parameter]
    torch.testing.assert_close(state["arc_h_local"], gradient)
    torch.testing.assert_close(state["arc_g_local"], gradient)
    torch.testing.assert_close(state["arc_g_global"], gradient)
    torch.testing.assert_close(state["momentum"], gradient)
    expected = initial - 0.125 * gradient.to(torch.bfloat16).float()
    torch.testing.assert_close(parameter, expected)


def test_arc_topk_state_dict_round_trip_restores_all_trackers():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = _make_optimizer(parameter, arc_topk_ratio=0.5, arc_eta=0.25)
    parameter.grad = torch.arange(12.0).reshape(4, 3)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())

    restored_parameter = torch.nn.Parameter(torch.zeros(4, 3))
    restored = _make_optimizer(
        restored_parameter,
        arc_topk_ratio=0.5,
        arc_eta=0.25,
    )
    restored.load_state_dict(saved)

    original_state = optimizer.state[parameter]
    restored_state = restored.state[restored_parameter]
    for name in ("momentum", "arc_h_local", "arc_g_local", "arc_g_global"):
        torch.testing.assert_close(restored_state[name], original_state[name])
    assert restored.param_groups[0]["arc_topk_ratio"] == 0.5
    assert restored.param_groups[0]["arc_projection_rank"] == 2
    assert restored.param_groups[0]["arc_eta"] == 0.25
    assert restored.param_groups[0]["arc_seed"] == 17
    assert restored.param_groups[0]["arc_start_compress_step"] == 0


def test_legacy_state_dict_resumes_with_immediate_compression():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = _make_optimizer(parameter, arc_topk_ratio=0.5, arc_eta=0.25)
    parameter.grad = torch.arange(12.0).reshape(4, 3)
    optimizer.step()
    legacy = copy.deepcopy(optimizer.state_dict())
    del legacy["param_groups"][0]["arc_start_compress_step"]

    restored_parameter = torch.nn.Parameter(torch.zeros(4, 3))
    restored = _make_optimizer(
        restored_parameter,
        arc_topk_ratio=0.5,
        arc_eta=0.25,
        arc_start_compress_step=1000,
    )
    restored.load_state_dict(legacy)
    restored_parameter.grad = torch.full((4, 3), 3.0)
    restored.step()

    assert restored.param_groups[0]["arc_start_compress_step"] == 0
    assert restored.param_groups[0]["step"] == 2
