"""Single-process tests for ARC-TopK-EF21M-Muon integration."""

import copy
from unittest.mock import patch

import pytest
import torch

from dion import ArcTopKMuon
import dion.arc_topk as arc_topk_module
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
            first_sentinel if kwargs["stable_task_id"] == 0 else second_sentinel,
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

    assert [call["stable_task_id"] for call in calls] == [0, 1]
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


def test_lossy_same_seed_three_step_optimizer_trajectory_is_frozen(monkeypatch):
    parameters = [
        torch.nn.Parameter(
            torch.tensor(
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
            )
        ),
        torch.nn.Parameter(
            torch.tensor(
                [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]
            )
        ),
    ]
    optimizer = ArcTopKMuon(
        parameters,
        lr=0.125,
        mu=0.5,
        weight_decay=0.0,
        nesterov=False,
        adjust_lr=None,
        newton_schulz_func=_identity_orthogonalizer,
        arc_topk_ratio=0.5,
        arc_projection_rank=2,
        arc_eta=0.25,
        arc_seed=17,
        arc_start_compress_step=0,
    )
    gradients = [
        (
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[8.0, 7.0], [6.0, 5.0], [4.0, 3.0], [2.0, 1.0]],
        ),
        (
            [[2.0, 1.0], [0.0, 3.0], [4.0, -1.0], [5.0, 2.0]],
            [[-1.0, 2.0], [3.0, 0.0], [1.0, 4.0], [2.0, 5.0]],
        ),
        (
            [[0.0, 2.0], [6.0, 1.0], [-2.0, 4.0], [3.0, 7.0]],
            [[5.0, -1.0], [2.0, 6.0], [0.0, 3.0], [4.0, 2.0]],
        ),
    ]
    expected_projections = [
        [
            [[-1.6053898335, 0.5725221634], [1.5612484217, 0.2325389534]],
            [[-0.3034220338, -2.02993536], [-0.6423370838, 0.2279635221]],
        ],
        [
            [[-0.5499795079, 0.6124502420], [0.6397058964, 0.4384637773]],
            [[-0.4881804287, 0.8683331013], [-1.3636255264, -0.9923884273]],
        ],
    ]
    expected_supports = [[[2, 3], [0, 1]], [[2, 1], [0, 3]]]
    expected_steps = [
        [
            {
                "arc_h_local": gradients[0][0],
                "arc_g_local": gradients[0][0],
                "arc_g_global": gradients[0][0],
                "momentum": gradients[0][0],
                "parameter": [[-0.125, 0.75], [1.625, 2.5], [3.375, 4.25], [5.125, 6.0]],
            },
            {
                "arc_h_local": gradients[0][1],
                "arc_g_local": gradients[0][1],
                "arc_g_global": gradients[0][1],
                "momentum": gradients[0][1],
                "parameter": [[0.0, -1.875], [1.25, -2.625], [2.5, -3.375], [3.75, -4.125]],
            },
        ],
        [
            {
                "arc_h_local": [[1.25, 1.75], [2.25, 3.75], [4.75, 4.25], [6.5, 6.5]],
                "arc_g_local": [[1.0, 2.0], [3.0, 4.0], [4.75, 4.25], [6.5, 6.5]],
                "arc_g_global": [[1.0, 2.0], [3.0, 4.0], [4.75, 4.25], [6.5, 6.5]],
                "momentum": [[1.5, 3.0], [4.5, 6.0], [7.25, 7.25], [10.0, 10.5]],
                "parameter": [[-0.3125, 0.375], [1.0625, 1.75], [2.46875, 3.34375], [3.875, 4.6875]],
            },
            {
                "arc_h_local": [[5.75, 5.75], [5.25, 3.75], [3.25, 3.25], [2.0, 2.0]],
                "arc_g_local": [[5.75, 5.75], [5.25, 3.75], [4.0, 3.0], [2.0, 1.0]],
                "arc_g_global": [[5.75, 5.75], [5.25, 3.75], [4.0, 3.0], [2.0, 1.0]],
                "momentum": [[9.75, 9.25], [8.25, 6.25], [6.0, 4.5], [3.0, 1.5]],
                "parameter": [[-1.21875, -3.03125], [0.21875, -3.40625], [1.75, -3.9375], [3.375, -4.3125]],
            },
        ],
        [
            {
                "arc_h_local": [[0.9375, 1.8125], [3.1875, 3.0625], [3.0625, 4.1875], [5.625, 6.625]],
                "arc_g_local": [[1.0, 2.0], [3.1875, 3.0625], [3.0625, 4.1875], [6.5, 6.5]],
                "arc_g_global": [[1.0, 2.0], [3.1875, 3.0625], [3.0625, 4.1875], [6.5, 6.5]],
                "momentum": [[1.75, 3.5], [5.4375, 6.0625], [6.6875, 7.8125], [11.5, 11.75]],
                "parameter": [[-0.53125, -0.0625], [0.3828125, 0.9921875], [1.6328125, 2.3671875], [2.4375, 3.21875]],
            },
            {
                "arc_h_local": [[5.5625, 4.0625], [4.4375, 4.3125], [2.4375, 3.1875], [2.5, 2.0]],
                "arc_g_local": [[5.5625, 4.0625], [5.25, 3.75], [4.0, 3.0], [2.5, 2.0]],
                "arc_g_global": [[5.5625, 4.0625], [5.25, 3.75], [4.0, 3.0], [2.5, 2.0]],
                "momentum": [[10.4375, 8.6875], [9.375, 6.875], [7.0, 5.25], [4.0, 2.75]],
                "parameter": [[-2.5234375, -4.1171875], [-0.953125, -4.265625], [0.875, -4.59375], [2.875, -4.65625]],
            },
        ],
    ]

    captured_projections = []
    captured_supports = []
    real_projection = arc_topk_module.make_gaussian_projection
    real_support = arc_topk_module.arc_topk_support

    def capture_projection(*args, **kwargs):
        projection = real_projection(*args, **kwargs)
        captured_projections.append(projection.clone())
        return projection

    def capture_support(sketch, k):
        support = real_support(sketch, k)
        captured_supports.append(support.clone())
        return support

    monkeypatch.setattr(arc_topk_module, "make_gaussian_projection", capture_projection)
    monkeypatch.setattr(arc_topk_module, "arc_topk_support", capture_support)

    for step, step_gradients in enumerate(gradients):
        for parameter, gradient in zip(parameters, step_gradients):
            parameter.grad = torch.tensor(gradient)
        optimizer.step()

        for parameter, expected in zip(parameters, expected_steps[step]):
            state = optimizer.state[parameter]
            for name in ("arc_h_local", "arc_g_local", "arc_g_global", "momentum"):
                torch.testing.assert_close(state[name], torch.tensor(expected[name]))
            torch.testing.assert_close(parameter, torch.tensor(expected["parameter"]))

    for actual, expected in zip(captured_projections, expected_projections):
        torch.testing.assert_close(actual, torch.tensor(expected))
    assert [support.tolist() for support in captured_supports] == expected_supports


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
