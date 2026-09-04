"""Tests for the pure AdamW ARC-TopK optimizer."""

import copy

import pytest
import torch

from dion import ArcTopKAdamW
import dion.adamw_arctopk as adamw_arctopk_module


def _make_optimizer(*params, **kwargs):
    options = dict(
        lr=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
        arc_topk_ratio=0.5,
        arc_projection_rank=2,
        arc_eta=0.25,
        arc_seed=17,
        arc_start_compress_step=0,
    )
    options.update(kwargs)
    return ArcTopKAdamW(list(params), **options)


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
def test_rejects_invalid_arc_configuration(kwargs):
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    with pytest.raises(ValueError):
        _make_optimizer(parameter, **kwargs)


def test_compressed_groups_require_2d_parameters():
    parameter = torch.nn.Parameter(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="2D"):
        ArcTopKAdamW([{"params": [parameter], "arc_compress": True}])


def test_arc_compress_must_be_bool():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="arc_compress"):
        ArcTopKAdamW([{"params": [parameter], "arc_compress": 1}])


def test_prepopulates_adamw_state_and_arc_state_only_for_compressed_group():
    matrix = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    vector = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    optimizer = ArcTopKAdamW(
        [
            {"params": [matrix], "arc_compress": True},
            {"params": [vector], "arc_compress": False},
        ]
    )

    assert set(optimizer.state[matrix]) == {
        "momentum",
        "variance",
        "step_dev",
        "arc_h_local",
        "arc_g_local",
        "arc_g_global",
    }
    assert set(optimizer.state[vector]) == {"momentum", "variance", "step_dev"}
    assert optimizer.state[matrix]["step_dev"].dtype == torch.float32
    assert optimizer.state[matrix]["step_dev"].device == matrix.device


def test_state_dict_carries_one_optimizer_wide_arc_step_and_restores_it():
    matrix = torch.nn.Parameter(torch.zeros(4, 3))
    vector = torch.nn.Parameter(torch.zeros(3))
    optimizer = ArcTopKAdamW(
        [
            {"params": [matrix], "arc_compress": True},
            {"params": [vector], "arc_compress": False},
        ]
    )
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    assert optimizer._arc_step == 1
    assert [group["arc_step"] for group in saved["param_groups"]] == [1, 1]

    restored_matrix = torch.nn.Parameter(torch.zeros(4, 3))
    restored_vector = torch.nn.Parameter(torch.zeros(3))
    restored = ArcTopKAdamW(
        [
            {"params": [restored_matrix], "arc_compress": True},
            {"params": [restored_vector], "arc_compress": False},
        ]
    )
    restored.load_state_dict(saved)
    assert restored._arc_step == 1


def test_lr_is_persistent_device_tensor_and_tracks_scheduler_reassignment():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = ArcTopKAdamW([parameter], lr=0.05)

    lr_tensor = optimizer._hyperparam_tensors[(0, "lr")]
    assert isinstance(lr_tensor, torch.Tensor)
    assert lr_tensor.ndim == 0
    assert lr_tensor.dtype == torch.float32
    assert lr_tensor.device == parameter.device
    assert optimizer.param_groups[0]["lr"] is lr_tensor

    optimizer.param_groups[0]["lr"] = 0.025
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer.param_groups[0]["lr"] is lr_tensor
    assert lr_tensor.item() == pytest.approx(0.025)


def test_optimizer_step_does_not_replace_or_mutate_param_grad():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = ArcTopKAdamW(
        [{"params": [parameter], "arc_compress": True}],
        arc_topk_ratio=0.5,
        arc_projection_rank=2,
    )
    gradient = torch.arange(12.0).reshape(4, 3)
    parameter.grad = gradient.clone()
    grad_identity = parameter.grad
    optimizer.step()
    assert parameter.grad is grad_identity
    torch.testing.assert_close(parameter.grad, gradient)

    dense_parameter = torch.nn.Parameter(torch.zeros(3))
    dense_optimizer = ArcTopKAdamW([dense_parameter])
    dense_gradient = torch.arange(3.0)
    dense_parameter.grad = dense_gradient.clone()
    dense_grad_identity = dense_parameter.grad
    dense_optimizer.step()
    assert dense_parameter.grad is dense_grad_identity
    torch.testing.assert_close(dense_parameter.grad, dense_gradient)


def test_load_migrates_missing_or_low_precision_step_dev():
    parameter = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    optimizer = ArcTopKAdamW([parameter])
    saved = optimizer.state_dict()
    saved["state"][0]["step_dev"] = torch.tensor(3.0, dtype=torch.bfloat16)

    restored_parameter = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    restored = ArcTopKAdamW([restored_parameter])
    restored.load_state_dict(saved)
    step_dev = restored.state[restored_parameter]["step_dev"]
    assert step_dev.dtype == torch.float32
    assert step_dev.device == restored_parameter.device
    assert step_dev.item() == pytest.approx(3.0)

    del saved["state"][0]["step_dev"]
    restored.load_state_dict(saved)
    step_dev = restored.state[restored_parameter]["step_dev"]
    assert step_dev.dtype == torch.float32
    assert step_dev.item() == pytest.approx(0.0)


def test_step_uses_three_way_async_runtime(monkeypatch):
    first = torch.nn.Parameter(torch.zeros(4, 3))
    second = torch.nn.Parameter(torch.zeros(2, 3))
    repeated_shape = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = ArcTopKAdamW(
        [{"params": [first, second, repeated_shape], "arc_compress": True}],
        arc_topk_ratio=0.5,
        arc_projection_rank=2,
    )
    for parameter in (first, second, repeated_shape):
        parameter.grad = torch.ones_like(parameter)
    observed = {}
    calls = []

    class RecordingRuntime:
        def __init__(self, tasks, max_concurrent_tasks):
            observed["max_concurrent_tasks"] = max_concurrent_tasks
            self._runtime = torch_opt_utils.AsyncRuntime(
                tasks, max_concurrent_tasks=max_concurrent_tasks
            )

        def run(self):
            return self._runtime.run()

    def fake_synchronize(**kwargs):
        # Delay recording until AsyncRuntime actually advances the task.  This
        # ensures the test exercises task consumption, not task construction.
        yield
        calls.append(
            (kwargs["task_index"], [tuple(param.shape) for param in kwargs["params"]])
        )
        yield
        return [torch.zeros_like(param) for param in kwargs["params"]]

    import dion.opt_utils as torch_opt_utils

    monkeypatch.setattr(
        adamw_arctopk_module,
        "synchronize_arc_batch_async",
        fake_synchronize,
    )
    monkeypatch.setattr(adamw_arctopk_module, "AsyncRuntime", RecordingRuntime)
    optimizer.step()
    assert observed["max_concurrent_tasks"] == 3
    assert calls == [(0, [(4, 3), (4, 3)]), (1, [(2, 3)])]


def test_load_rejects_inconsistent_arc_step_across_groups():
    matrix = torch.nn.Parameter(torch.zeros(4, 3))
    vector = torch.nn.Parameter(torch.zeros(3))
    optimizer = ArcTopKAdamW(
        [
            {"params": [matrix], "arc_compress": True},
            {"params": [vector], "arc_compress": False},
        ]
    )
    saved = optimizer.state_dict()
    saved["param_groups"][1]["arc_step"] = 1
    with pytest.raises(ValueError, match="arc_step"):
        optimizer.load_state_dict(saved)


def test_legacy_state_dict_defaults_missing_arc_start_to_zero():
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = ArcTopKAdamW(
        [{"params": [parameter], "arc_compress": True}],
        arc_start_compress_step=20,
    )
    saved = optimizer.state_dict()
    del saved["param_groups"][0]["arc_start_compress_step"]

    restored_parameter = torch.nn.Parameter(torch.zeros(4, 3))
    restored = ArcTopKAdamW(
        [{"params": [restored_parameter], "arc_compress": True}],
        arc_start_compress_step=20,
    )
    restored.load_state_dict(saved)
    assert restored.param_groups[0]["arc_start_compress_step"] == 0


def test_ratio_one_eta_one_matches_torch_adamw_for_two_steps():
    initial = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    ours_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())
    kwargs = dict(
        lr=0.01,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.03,
        arc_topk_ratio=1.0,
        arc_projection_rank=2,
        arc_eta=1.0,
    )
    ours = ArcTopKAdamW([{"params": [ours_param], "arc_compress": True}], **kwargs)
    reference = torch.optim.AdamW(
        [reference_param],
        **{k: kwargs[k] for k in ("lr", "betas", "eps", "weight_decay")},
    )
    gradients = [
        torch.tensor([[0.5, -0.25, 0.75], [1.0, -1.5, 0.25]]),
        torch.tensor([[-0.5, 0.125, 0.25], [0.75, 0.5, -1.0]]),
    ]
    for gradient in gradients:
        ours_param.grad = gradient.clone()
        reference_param.grad = gradient.clone()
        ours.step()
        reference.step()
    torch.testing.assert_close(ours_param, reference_param, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        ours.state[ours_param]["momentum"],
        reference.state[reference_param]["exp_avg"],
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        ours.state[ours_param]["variance"],
        reference.state[reference_param]["exp_avg_sq"],
        rtol=2e-5,
        atol=2e-6,
    )
