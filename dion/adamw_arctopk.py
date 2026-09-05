"""Pure AdamW with shared ARC-TopK gradient synchronization."""

from itertools import chain
from typing import Generator, Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import Optimizer, ParamsT

from .arc_topk import validate_arc_topk_config
from .arc_topk_sync import (
    ArcTopKSyncConfig,
    average_gradients_async,
    group_parameters_by_shape_dtype,
    initialize_arc_state_,
    synchronize_arc_batch_async,
)
from .opt_utils import AsyncRuntime, AsyncTask, as_scalar_tensor, to_local
from .scalar_opts import adamw_update_foreach_async


class ArcTopKAdamW(Optimizer):
    """AdamW whose selected matrix groups use ARC-TopK + EF21M."""

    def __init__(
        self,
        params: ParamsT,
        process_group: Optional[ProcessGroup] = None,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        arc_topk_ratio: float = 0.2,
        arc_projection_rank: int = 4,
        arc_eta: float = 0.1,
        arc_seed: int = 42,
        arc_start_compress_step: int = 0,
    ):
        if isinstance(process_group, DeviceMesh):
            raise ValueError("ArcTopKAdamW accepts a ProcessGroup, not a DeviceMesh")
        if process_group is not None and not isinstance(process_group, ProcessGroup):
            raise TypeError(
                "process_group must be a torch.distributed.ProcessGroup or None"
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or not 0.0 <= betas[0] < 1.0
            or not 0.0 <= betas[1] < 1.0
        ):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        validate_arc_topk_config(
            arc_topk_ratio,
            arc_projection_rank,
            arc_eta,
            arc_start_compress_step,
        )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            arc_compress=False,
            arc_topk_ratio=arc_topk_ratio,
            arc_projection_rank=arc_projection_rank,
            arc_eta=arc_eta,
            arc_seed=arc_seed,
            arc_start_compress_step=arc_start_compress_step,
            arc_step=0,
        )
        super().__init__(params, defaults)

        self._process_group = process_group
        self._arc_step = 0
        self._hyperparam_tensors: dict[tuple[int, str], Tensor] = {}
        self._state_prepopulated = False
        for group in self.param_groups:
            self._validate_group(group)
            self._prepopulate_group_state(group)
        self._sync_hyperparam_tensors()
        self._state_prepopulated = True

    def _validate_group(self, group: dict) -> None:
        compressed = group.get("arc_compress", False)
        if not isinstance(compressed, bool):
            raise ValueError("arc_compress must be a bool")
        validate_arc_topk_config(
            group["arc_topk_ratio"],
            group["arc_projection_rank"],
            group["arc_eta"],
            group["arc_start_compress_step"],
        )
        if compressed and any(param.ndim != 2 for param in group["params"]):
            raise ValueError(
                "ArcTopKAdamW compressed groups require only 2D parameters"
            )

    def _prepopulate_group_state(self, group: dict) -> None:
        compressed = group.get("arc_compress", False)
        for param in group["params"]:
            state = self._get_or_initialize_state(param)
            if compressed:
                initialize_arc_state_(state, param)

    def _get_or_initialize_state(self, param: Tensor) -> dict:
        state = self.state[param]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(param)
        if "variance" not in state:
            state["variance"] = torch.zeros_like(param)
        device = to_local(param).device
        step_dev = state.get("step_dev")
        if step_dev is None:
            step_dev = torch.zeros((), dtype=torch.float32, device=device)
        elif not isinstance(step_dev, Tensor):
            step_dev = torch.tensor(step_dev, dtype=torch.float32, device=device)
        elif step_dev.dtype != torch.float32 or step_dev.device != device:
            step_dev = step_dev.to(device=device, dtype=torch.float32)
        state["step_dev"] = step_dev
        return state

    def _ensure_lr_tensor(self, index: int) -> Optional[Tensor]:
        group = self.param_groups[index]
        params = group["params"]
        if not params:
            return None
        value = group["lr"]
        tensor = self._hyperparam_tensors.get((index, "lr"))
        device = to_local(params[0]).device
        if value is tensor and tensor is not None and tensor.device == device:
            return tensor
        if tensor is None or tensor.device != device:
            tensor = torch.empty((), dtype=torch.float32, device=device)
            self._hyperparam_tensors[(index, "lr")] = tensor
        if isinstance(value, Tensor):
            tensor.copy_(value)
        else:
            tensor.fill_(value)
        group["lr"] = tensor
        return tensor

    def _sync_hyperparam_tensors(self) -> None:
        for index in range(len(self.param_groups)):
            self._ensure_lr_tensor(index)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if not getattr(self, "_state_prepopulated", False):
            return
        group = self.param_groups[-1]
        group["arc_step"] = self._arc_step
        self._validate_group(group)
        self._prepopulate_group_state(group)
        self._ensure_lr_tensor(len(self.param_groups) - 1)

    def state_dict(self):
        result = super().state_dict()
        for group in result["param_groups"]:
            if isinstance(group.get("lr"), Tensor):
                group["lr"] = float(group["lr"].item())
            group["arc_step"] = self._arc_step
        return result

    def load_state_dict(self, state_dict):
        saved_groups = state_dict.get("param_groups", [])
        arc_steps = [group.get("arc_step", 0) for group in saved_groups]
        if any(
            isinstance(step, bool) or not isinstance(step, int) or step < 0
            for step in arc_steps
        ):
            raise ValueError(
                "arc_step must be a non-negative integer in every parameter group"
            )
        if any(step != arc_steps[0] for step in arc_steps[1:]):
            raise ValueError(
                "all parameter groups must carry the same optimizer-wide arc_step"
            )
        arc_step = arc_steps[0] if arc_steps else 0
        if isinstance(arc_step, bool) or not isinstance(arc_step, int) or arc_step < 0:
            raise ValueError(f"arc_step must be a non-negative integer, got {arc_step!r}")

        migrated = dict(state_dict)
        migrated["param_groups"] = [dict(group) for group in saved_groups]
        for group in migrated["param_groups"]:
            group.setdefault("arc_step", arc_step)
            group.setdefault("arc_start_compress_step", 0)
        super().load_state_dict(migrated)
        self._arc_step = arc_step
        for group in self.param_groups:
            self._validate_group(group)
            self._prepopulate_group_state(group)
        self._sync_hyperparam_tensors()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._sync_hyperparam_tensors()
        self._arc_step += 1
        for group in self.param_groups:
            group["arc_step"] = self._arc_step

        compressed_groups = [
            group for group in self.param_groups if group.get("arc_compress", False)
        ]
        dense_groups = [
            group for group in self.param_groups if not group.get("arc_compress", False)
        ]
        tasks = chain(
            self._create_compressed_tasks(compressed_groups),
            self._create_dense_tasks(dense_groups),
        )
        AsyncRuntime(tasks, max_concurrent_tasks=3).run()
        return loss

    def _create_compressed_tasks(self, param_groups: list[dict]):
        task_index = 0
        for group in param_groups:
            self._validate_group(group)
            for params in group_parameters_by_shape_dtype(group["params"]):
                states = [self._get_or_initialize_state(param) for param in params]
                config = ArcTopKSyncConfig(
                    ratio=group["arc_topk_ratio"],
                    projection_rank=group["arc_projection_rank"],
                    eta=group["arc_eta"],
                    seed=group["arc_seed"],
                    start_compress_step=group["arc_start_compress_step"],
                )
                yield AsyncTask(
                    _arc_adamw_update_async(
                        params=params,
                        states=states,
                        group=group,
                        process_group=self._process_group,
                        config=config,
                        step=self._arc_step,
                        task_index=task_index,
                    )
                )
                task_index += 1

    def _create_dense_tasks(self, param_groups: list[dict]):
        for group in param_groups:
            params = list(group["params"])
            if params:
                yield AsyncTask(
                    _dense_adamw_update_async(
                        params=params,
                        states=[self._get_or_initialize_state(param) for param in params],
                        group=group,
                        process_group=self._process_group,
                    )
                )


def _adamw_update(
    params: list[Tensor],
    gradients: list[Tensor],
    states: list[dict],
    group: dict,
):
    yield from adamw_update_foreach_async(
        X=[to_local(param) for param in params],
        G=gradients,
        M=[to_local(state["momentum"]) for state in states],
        V=[to_local(state["variance"]) for state in states],
        lr=as_scalar_tensor(group["lr"]),
        beta1=as_scalar_tensor(group["betas"][0]),
        beta2=as_scalar_tensor(group["betas"][1]),
        weight_decay=as_scalar_tensor(group["weight_decay"]),
        state_steps=[state["step_dev"] for state in states],
        epsilon=group["eps"],
    )


def _arc_adamw_update_async(
    *,
    params: list[Tensor],
    states: list[dict],
    group: dict,
    process_group: Optional[ProcessGroup],
    config: ArcTopKSyncConfig,
    step: int,
    task_index: int,
) -> Generator[None, None, None]:
    gradients = yield from synchronize_arc_batch_async(
        params=params,
        states=states,
        process_group=process_group,
        config=config,
        step=step,
        task_index=task_index,
    )
    yield from _adamw_update(params, gradients, states, group)


def _dense_adamw_update_async(
    *,
    params: list[Tensor],
    states: list[dict],
    group: dict,
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, None]:
    gradients = [
        to_local(param.grad).clone()
        if param.grad is not None
        else torch.zeros_like(to_local(param))
        for param in params
    ]
    gradients = yield from average_gradients_async(gradients, process_group)
    yield from _adamw_update(params, gradients, states, group)
