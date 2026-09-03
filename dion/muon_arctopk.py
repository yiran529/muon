"""DDP-only Muon with complete ARC-TopK and EF21M gradient synchronization."""

import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .arc_topk import arc_topk_ef21m_async, validate_arc_topk_config
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
        arc_start_compress_step: int = 1000,
        **kwargs,
    ):
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

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == "muon":
            state.setdefault("arc_h_local", torch.zeros_like(param))
            state.setdefault("arc_g_local", torch.zeros_like(param))
            state.setdefault("arc_g_global", torch.zeros_like(param))
        return state

    def load_state_dict(self, state_dict):
        migrated = dict(state_dict)
        migrated["param_groups"] = [dict(group) for group in state_dict["param_groups"]]
        for group in migrated["param_groups"]:
            if group.get("algorithm") == "muon":
                group.setdefault("arc_start_compress_step", 0)
        super().load_state_dict(migrated)

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator[AsyncTask, None, None]:
        task_index = 0
        for group in param_groups:
            assert group["algorithm"] == "muon"
            if not all(p.ndim == 2 for p in group["params"]):
                raise ValueError("ArcTopKMuon only supports 2D matrix parameters")

            shape_groups: dict[tuple, list[Tensor]] = defaultdict(list)
            for param in group["params"]:
                shape_groups[(param.shape, param.dtype)].append(param)

            for params in shape_groups.values():
                gradients = [
                    p.grad if p.grad is not None else torch.zeros_like(p)
                    for p in params
                ]
                states = [self._get_or_initialize_state(p, "muon") for p in params]
                yield AsyncTask(
                    arc_topk_muon_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=[state["momentum"] for state in states],
                        H=[state["arc_h_local"] for state in states],
                        G_local=[state["arc_g_local"] for state in states],
                        G_global=[state["arc_g_global"] for state in states],
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
                        arc_topk_ratio=group["arc_topk_ratio"],
                        arc_projection_rank=group["arc_projection_rank"],
                        arc_eta=group["arc_eta"],
                        arc_seed=group["arc_seed"],
                        step=group["step"],
                        task_index=task_index,
                        start_compress_step=group["arc_start_compress_step"],
                    )
                )
                task_index += 1

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
    G: List[Tensor],
    M: List[Tensor],
    H: List[Tensor],
    G_local: List[Tensor],
    G_global: List[Tensor],
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
    arc_topk_ratio: float,
    arc_projection_rank: int,
    arc_eta: float,
    arc_seed: int,
    step: int,
    task_index: int,
    start_compress_step: int,
) -> Generator[None, None, None]:
    synchronized_gradients = yield from arc_topk_ef21m_async(
        gradients=G,
        trackers=H,
        local_estimates=G_local,
        global_estimates=G_global,
        process_group=process_group,
        ratio=arc_topk_ratio,
        projection_rank=arc_projection_rank,
        eta=arc_eta,
        base_seed=arc_seed,
        step=step,
        task_index=task_index,
        start_compress_step=start_compress_step,
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


def _average_gradients_async(
    gradients: List[Tensor],
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, List[Tensor]]:
    averaged = [gradient.clone() for gradient in gradients]
    if process_group is None:
        return averaged
    world_size = torch.distributed.get_world_size(process_group)
    if world_size == 1:
        return averaged
    for gradient in averaged:
        work = torch.distributed.all_reduce(
            gradient,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
            async_op=True,
        )
        yield
        work.wait()
        gradient.div_(world_size)
    return averaged


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
    averaged = yield from _average_gradients_async(G, process_group)
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
    averaged = yield from _average_gradients_async(G, process_group)
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
