"""DDP-only Muon with complete ARC-TopK and EF21M gradient synchronization."""

from dataclasses import asdict

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import ParamsT
from torch.nn import Parameter
from typing import Callable, Generator, List, Mapping, Optional, Union

from .arc_topk import validate_arc_topk_config
from .arc_topk_layout import (
    ArcOptimizerTaskDescriptor,
    ArcParameterDescriptor,
    ArcTopKLayoutMismatch,
    canonical_arc_fingerprint,
    validate_arc_fingerprint_across_ranks,
)
from .arc_topk_sync import (
    ArcTopKSyncConfig,
    average_gradients_async,
    group_parameters_by_shape_dtype,
    initialize_arc_state_,
    synchronize_arc_batch_async,
)
from .megabatch_base import (
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    megabatch_orthogonalize_async,
)
from .muon import (
    Muon,
    muon_update_post_orthogonalize,
    muon_update_pre_orthogonalize,
)
from .opt_utils import AsyncTask, as_scalar_tensor, to_local
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


def _dtype_name(tensor: Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _serialize_optimizer_tasks(
    tasks: tuple[ArcOptimizerTaskDescriptor, ...],
) -> list[dict]:
    return [asdict(task) for task in tasks]


def _checkpoint_groups_match_tasks(
    saved_groups: list[dict],
    current_groups: list[dict],
    tasks: tuple[ArcOptimizerTaskDescriptor, ...],
) -> bool:
    if len(saved_groups) != len(current_groups):
        return False
    for saved_group, current_group in zip(saved_groups, current_groups):
        if saved_group.get("algorithm") != current_group.get("algorithm"):
            return False
        if len(saved_group.get("params", ())) != len(current_group["params"]):
            return False
    for task in tasks:
        if task.group_id >= len(saved_groups):
            return False
        group = saved_groups[task.group_id]
        saved_config = ArcTopKSyncConfig(
            ratio=group["arc_topk_ratio"],
            projection_rank=group["arc_projection_rank"],
            eta=group["arc_eta"],
            seed=group["arc_seed"],
            start_compress_step=group["arc_start_compress_step"],
            seed_scheme_version=group.get("arc_seed_scheme_version", 1),
        )
        if saved_config != task.config:
            return False
    return True


class ArcTopKMuon(Muon):
    """Muon whose DDP matrix gradients are synchronized with ARC-TopK + EF21M."""

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        *,
        arc_topk_ratio: float = 0.2,
        arc_projection_rank: int = 4,
        arc_eta: float = 0.1,
        arc_seed: int = 42,
        arc_start_compress_step: int = 300,
        arc_parameter_names: Mapping[Parameter, str] | None = None,
        **kwargs,
    ):
        self._arc_layout_frozen = False
        validate_arc_topk_config(
            arc_topk_ratio,
            arc_projection_rank,
            arc_eta,
            arc_start_compress_step,
        )
        if isinstance(distributed_mesh, DeviceMesh):
            raise ValueError("ArcTopKMuon first version is DDP only")
        self._arc_topk_ratio = arc_topk_ratio
        self._arc_projection_rank = arc_projection_rank
        self._arc_eta = arc_eta
        self._arc_seed = arc_seed
        self._arc_start_compress_step = arc_start_compress_step
        super().__init__(params, distributed_mesh=distributed_mesh, **kwargs)

        if self._process_group is not None and arc_parameter_names is None:
            raise ValueError(
                "distributed ArcTopKMuon requires arc_parameter_names from "
                "model.named_parameters()"
            )
        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue
            if group.get("flatten"):
                raise ValueError("ArcTopKMuon does not support flatten=True")
            if self._resolve_num_heads(group) is not None:
                raise ValueError("ArcTopKMuon does not support num_heads > 1")
            if group.get("split_sizes") is not None:
                raise ValueError("ArcTopKMuon does not support split_sizes")
            group["arc_topk_ratio"] = arc_topk_ratio
            group["arc_projection_rank"] = arc_projection_rank
            group["arc_eta"] = arc_eta
            group["arc_seed"] = arc_seed
            group["arc_start_compress_step"] = arc_start_compress_step
            group["arc_seed_scheme_version"] = 1

        optimizer_parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]
        if arc_parameter_names is None:
            self._arc_parameter_names = {
                parameter: f"parameter_{stable_id}"
                for stable_id, parameter in enumerate(optimizer_parameters)
            }
            named_parameters = list(self._arc_parameter_names.items())
        else:
            self._arc_parameter_names = dict(arc_parameter_names)
            missing_positions = [
                position
                for position, parameter in enumerate(optimizer_parameters, start=1)
                if parameter not in self._arc_parameter_names
            ]
            if missing_positions:
                raise ValueError(
                    "arc_parameter_names is missing stable names for "
                    f"optimizer parameter at position {missing_positions[0]}"
                )
            optimizer_parameter_ids = {id(parameter) for parameter in optimizer_parameters}
            named_parameters = [
                (parameter, name)
                for parameter, name in self._arc_parameter_names.items()
                if id(parameter) in optimizer_parameter_ids
            ]

        roles = {
            id(parameter): (
                "arc_matrix" if group["algorithm"] == "muon" else "dense_aux"
            )
            for group in self.param_groups
            for parameter in group["params"]
        }
        self._arc_parameters = tuple(
            ArcParameterDescriptor(
                stable_name=name,
                stable_id=stable_id,
                shape=tuple(parameter.shape),
                dtype=_dtype_name(parameter),
                role=roles[id(parameter)],
            )
            for stable_id, (parameter, name) in enumerate(named_parameters)
        )

        self._arc_task_ids = {}
        optimizer_tasks = []
        next_task_id = 0
        for group_id, group in enumerate(self.param_groups):
            if group["algorithm"] != "muon":
                continue
            for task_params in group_parameters_by_shape_dtype(group["params"]):
                self._arc_task_ids[tuple(map(id, task_params))] = next_task_id
                optimizer_tasks.append(
                    ArcOptimizerTaskDescriptor(
                        group_id=group_id,
                        task_id=next_task_id,
                        ordered_parameter_names=tuple(
                            self._arc_parameter_names[parameter]
                            for parameter in task_params
                        ),
                        shape=tuple(task_params[0].shape),
                        dtype=_dtype_name(task_params[0]),
                        config=ArcTopKSyncConfig(
                            ratio=group["arc_topk_ratio"],
                            projection_rank=group["arc_projection_rank"],
                            eta=group["arc_eta"],
                            seed=group["arc_seed"],
                            start_compress_step=group["arc_start_compress_step"],
                        ),
                    )
                )
                next_task_id += 1
        self._arc_optimizer_tasks = tuple(optimizer_tasks)
        group_ranks = (
            tuple(dist.get_process_group_ranks(self._process_group))
            if self._process_group is not None
            else (0,)
        )
        default_config = ArcTopKSyncConfig(
            ratio=arc_topk_ratio,
            projection_rank=arc_projection_rank,
            eta=arc_eta,
            seed=arc_seed,
            start_compress_step=arc_start_compress_step,
        )
        self._arc_layout_fingerprint = canonical_arc_fingerprint(
            base_seed=arc_seed,
            config=default_config,
            group_ranks=group_ranks,
            parameters=self._arc_parameters,
            optimizer_tasks=self._arc_optimizer_tasks,
        )
        if self._process_group is not None:
            validate_arc_fingerprint_across_ranks(
                self._arc_layout_fingerprint,
                self._process_group,
            )
        self._arc_layout_frozen = True

    def add_param_group(self, param_group: dict) -> None:
        if getattr(self, "_arc_layout_frozen", False):
            raise RuntimeError("cannot add a parameter group to a frozen ARC layout")
        super().add_param_group(param_group)

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == "muon":
            initialize_arc_state_(state, param)
        return state

    def load_state_dict(self, state_dict):
        migrated = dict(state_dict)
        migrated["param_groups"] = [dict(group) for group in state_dict["param_groups"]]
        for group in migrated["param_groups"]:
            if group.get("algorithm") == "muon":
                group.setdefault("arc_start_compress_step", 0)
                group.setdefault("arc_seed_scheme_version", 1)

        saved_fingerprint = migrated.get("arc_layout_fingerprint")
        saved_tasks = migrated.get("arc_optimizer_tasks")
        if (
            saved_fingerprint != self._arc_layout_fingerprint
            or saved_tasks != _serialize_optimizer_tasks(self._arc_optimizer_tasks)
            or not _checkpoint_groups_match_tasks(
                migrated["param_groups"],
                self.param_groups,
                self._arc_optimizer_tasks,
            )
        ):
            raise ArcTopKLayoutMismatch(
                "checkpoint ARC layout does not match the frozen optimizer layout"
            )
        super().load_state_dict(migrated)

    def state_dict(self):
        result = super().state_dict()
        result["arc_layout_fingerprint"] = self._arc_layout_fingerprint
        result["arc_optimizer_tasks"] = _serialize_optimizer_tasks(
            self._arc_optimizer_tasks
        )
        return result

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator[AsyncTask, None, None]:
        for group in param_groups:
            assert group["algorithm"] == "muon"
            if not all(p.ndim == 2 for p in group["params"]):
                raise ValueError("ArcTopKMuon only supports 2D matrix parameters")

            for params in group_parameters_by_shape_dtype(group["params"]):
                stable_task_id = self._arc_task_ids[tuple(map(id, params))]
                states = [self._get_or_initialize_state(p, "muon") for p in params]
                sync_config = ArcTopKSyncConfig(
                    ratio=group["arc_topk_ratio"],
                    projection_rank=group["arc_projection_rank"],
                    eta=group["arc_eta"],
                    seed=group["arc_seed"],
                    start_compress_step=group["arc_start_compress_step"],
                )
                yield AsyncTask(
                    arc_topk_muon_update_megabatch_async(
                        X=params,
                        M=[state["momentum"] for state in states],
                        states=states,
                        lr=group["lr"],
                        momentum=torch.tensor(group["mu"]),
                        weight_decay=as_scalar_tensor(group["weight_decay"]),
                        epsilon=torch.tensor(group["epsilon"]),
                        nesterov=group["nesterov"],
                        adjust_lr=group["adjust_lr"],
                        device_rank=self._device_rank,
                        world_size=self._world_size,
                        process_group=self._process_group,
                        newton_schulz_func=self._newton_schulz_func,
                        cautious_wd=group["cautious_wd"],
                        sync_config=sync_config,
                        step=group["step"],
                        stable_task_id=stable_task_id,
                    )
                )

    def _create_lion_tasks(
        self, param_groups: List[dict]
    ) -> Generator[AsyncTask, None, None]:
        for group in param_groups:
            params = list(group["params"])
            if not params:
                continue
            gradients = [
                to_local(p.grad) if p.grad is not None else torch.zeros_like(to_local(p))
                for p in params
            ]
            states = [self._get_or_initialize_state(p, "lion") for p in params]
            yield AsyncTask(
                _lion_update_allreduce_async(
                    X=to_local(params),
                    G=gradients,
                    M=to_local([state["momentum"] for state in states]),
                    lr=group["lr"],
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=as_scalar_tensor(group["weight_decay"]),
                    cautious_wd=group.get("cautious_wd", False),
                    process_group=self._process_group,
                )
            )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator[AsyncTask, None, None]:
        for group in param_groups:
            params = list(group["params"])
            if not params:
                continue
            gradients = [
                to_local(p.grad) if p.grad is not None else torch.zeros_like(to_local(p))
                for p in params
            ]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            yield AsyncTask(
                _adamw_update_allreduce_async(
                    X=to_local(params),
                    G=gradients,
                    M=to_local([state["momentum"] for state in states]),
                    V=to_local([state["variance"] for state in states]),
                    lr=group["lr"],
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=as_scalar_tensor(group["weight_decay"]),
                    state_steps=[state["step_dev"] for state in states],
                    epsilon=torch.tensor(group["epsilon"]),
                    cautious_wd=group.get("cautious_wd", False),
                    process_group=self._process_group,
                )
            )


def arc_topk_muon_update_megabatch_async(
    X: List[Tensor],
    M: List[Tensor],
    states: List[dict],
    lr: Tensor,
    momentum: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
    newton_schulz_func: Callable,
    cautious_wd: bool,
    sync_config: ArcTopKSyncConfig,
    step: int,
    stable_task_id: int,
) -> Generator[None, None, None]:
    synchronized_gradients = yield from synchronize_arc_batch_async(
        params=X,
        states=states,
        process_group=process_group,
        config=sync_config,
        step=step,
        stable_task_id=stable_task_id,
    )

    updates = muon_update_pre_orthogonalize(
        G=synchronized_gradients,
        M=M,
        momentum=momentum,
        nesterov=nesterov,
    )
    updates = yield from megabatch_orthogonalize_async(
        updates,
        comm_dim=None,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=False,
        epsilon=epsilon,
        global_comm_dim_size=None,
    )

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=False)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=False)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    muon_update_post_orthogonalize(
        X=to_local(X),
        U=updates,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
    )


def _lion_update_allreduce_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool,
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, None]:
    averaged = yield from average_gradients_async(G, process_group)
    yield from lion_update_foreach_async(
        X,
        averaged,
        M,
        lr,
        beta1,
        beta2,
        weight_decay,
        cautious_wd,
    )


def _adamw_update_allreduce_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    state_steps: List[Tensor],
    epsilon: Tensor,
    cautious_wd: bool,
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, None]:
    averaged = yield from average_gradients_async(G, process_group)
    yield from adamw_update_foreach_async(
        X,
        averaged,
        M,
        V,
        lr,
        beta1,
        beta2,
        weight_decay,
        state_steps=state_steps,
        epsilon=epsilon,
        cautious_wd=cautious_wd,
    )
