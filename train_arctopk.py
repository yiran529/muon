"""Dedicated DDP training entry point for ARC-TopK-EF21M-Muon."""

import argparse

from dataclasses import dataclass
from typing import Literal, Optional

from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.distributed as dist

import train
from dion import ArcTopKDDPParameterSpec, ArcTopKDDPState, ArcTopKMuon, Muon
from dion.arc_topk_ddp_hook import arc_topk_ddp_hook
from dion.arc_topk_layout import (
    ArcParameterDescriptor,
    canonical_arc_fingerprint,
    validate_arc_fingerprint_across_ranks,
)
from dion.arc_topk_sync import ArcTopKSyncConfig
from dion.opt_utils import lm_head_lr_scale


ArcSyncMode = Literal["optimizer", "ddp_hook"]


@dataclass
class ArcTopKHyperparameters(train.Hyperparameters):
    """Shared training parameters plus the ARC-TopK method parameters."""

    optimizer: str = "arc_topk_muon"
    arc_sync_mode: ArcSyncMode = "optimizer"
    arc_topk_ratio: float = 0.2
    arc_projection_rank: int = 4
    arc_eta: float = 0.1
    arc_seed: int = 42
    arc_start_compress_step: int = 300


def configure_arc_topk_parser(parser: argparse.ArgumentParser) -> None:
    """Add ARC-TopK arguments to the shared training parser."""

    parser.add_argument("--arc_topk_ratio", type=float, default=None)
    parser.add_argument("--arc_projection_rank", type=int, default=None)
    parser.add_argument("--arc_eta", type=float, default=None)
    parser.add_argument("--arc_seed", type=int, default=None)
    parser.add_argument("--arc_start_compress_step", type=int, default=None)
    parser.add_argument(
        "--arc_sync_mode",
        choices=("optimizer", "ddp_hook"),
        default=None,
    )


def arc_optimizer_owns_gradient_sync(arc_sync_mode: ArcSyncMode) -> bool:
    if arc_sync_mode == "optimizer":
        return True
    if arc_sync_mode == "ddp_hook":
        return False
    raise ValueError(f"unsupported ARC sync mode: {arc_sync_mode!r}")


def install_arc_topk_ddp_hook(
    model,
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    hp: ArcTopKHyperparameters,
) -> train.GradientSyncRuntime:
    config = ArcTopKSyncConfig(
        ratio=hp.arc_topk_ratio,
        projection_rank=hp.arc_projection_rank,
        eta=hp.arc_eta,
        seed=hp.arc_seed,
        start_compress_step=hp.arc_start_compress_step,
    )
    named_parameters = list(model.named_parameters())
    specs = tuple(
        ArcTopKDDPParameterSpec(
            parameter=parameter,
            stable_name=name,
            stable_id=stable_id,
            role="arc_matrix" if parameter.ndim == 2 else "dense_aux",
        )
        for stable_id, (name, parameter) in enumerate(named_parameters)
    )
    group_ranks = (
        tuple(dist.get_process_group_ranks(ddp_model.process_group))
        if ddp_model.process_group is not None
        else (0,)
    )
    fingerprint = canonical_arc_fingerprint(
        base_seed=config.seed,
        config=config,
        group_ranks=group_ranks,
        parameters=tuple(
            ArcParameterDescriptor(
                stable_name=spec.stable_name,
                stable_id=spec.stable_id,
                shape=tuple(spec.parameter.shape),
                dtype=str(spec.parameter.dtype).removeprefix("torch."),
                role=spec.role,
            )
            for spec in specs
        ),
    )
    if ddp_model.process_group is not None:
        validate_arc_fingerprint_across_ranks(
            fingerprint,
            ddp_model.process_group,
        )
    state = ArcTopKDDPState(
        process_group=ddp_model.process_group,
        fingerprint=fingerprint,
        parameter_specs=specs,
        optimizer_parameters=[
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ],
        config=config,
        find_unused_parameters=getattr(
            ddp_model, "find_unused_parameters", False
        ),
    )
    ddp_model.register_comm_hook(state, arc_topk_ddp_hook)
    return train.GradientSyncRuntime(
        optimizer_owns_gradient_sync=False,
        begin_step=state.begin_step,
        finish_step=state.finish_step,
        commit_step=state.commit_step,
        checkpoint_state=state,
    )


def init_arc_topk_optimizer(
    model,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: ArcTopKHyperparameters,
    cli_args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, train.GradientSyncRuntime]:
    """Build ARC-TopK-EF21M-Muon while preserving the baseline grouping."""

    if device_mesh is not None:
        raise ValueError("ARC-TopK-EF21M-Muon first version is DDP only")
    if ddp_model is None:
        raise ValueError("ARC-TopK-EF21M-Muon requires a DDP model")
    if hp.optimizer not in ("arc_topk_muon", "arc_topk_adamw"):
        raise ValueError(f"Unsupported ARC optimizer: {hp.optimizer}")
    if hp.optimizer == "arc_topk_adamw" and hp.arc_sync_mode != "ddp_hook":
        raise ValueError("arc_topk_adamw requires arc_sync_mode=ddp_hook")
    if hp.scalar_opt not in ("adamw", "lion"):
        raise ValueError(f"Unrecognized scalar optimizer: {hp.scalar_opt}")
    if getattr(cli_args, "_explicit_replicate_mesh_grad_sync", False):
        raise ValueError(
            "replicate_mesh_grad_sync is no longer accepted by train_arctopk.py; "
            "select --arc_sync_mode optimizer or ddp_hook"
        )
    arc_optimizer_owns_gradient_sync(hp.arc_sync_mode)

    matrix_params = list(model.transformer.h.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    lm_head_lr = hp.lr * lm_head_lr_scale(hp.scalar_opt, hp.model_dim)

    param_groups = [
        dict(params=matrix_params),
        dict(
            params=embedding_params,
            algorithm=hp.scalar_opt,
            lr=hp.lr,
            betas=(0.95, 0.98),
            weight_decay=0,
        ),
        dict(
            params=lm_head_params,
            algorithm=hp.scalar_opt,
            lr=lm_head_lr,
            betas=(0.95, 0.98),
            weight_decay=0,
        ),
    ]

    train.print0(f"ARC-TopK ratio: {hp.arc_topk_ratio}")
    train.print0(f"ARC projection rank: {hp.arc_projection_rank}")
    train.print0(f"EF21M eta: {hp.arc_eta}")
    train.print0(f"ARC compression starts after step: {hp.arc_start_compress_step}")
    train.print0(f"Muon LR adjust method: {hp.adjust_lr}")
    train.print0(f"Triton Newton-Schulz kernels: {not cli_args.no_triton}")

    optimizer_kwargs = dict(
        distributed_mesh=ddp_model.process_group,
        lr=hp.lr,
        mu=hp.mu,
        weight_decay=hp.weight_decay,
        nesterov=True,
        adjust_lr=hp.adjust_lr,
        use_gram_newton_schulz=cli_args.use_gram_newton_schulz,
        use_triton=not cli_args.no_triton,
        use_polar_express=cli_args.use_polar_express,
    )
    if hp.arc_sync_mode == "optimizer":
        optimizer = ArcTopKMuon(
            param_groups,
            **optimizer_kwargs,
            arc_topk_ratio=hp.arc_topk_ratio,
            arc_projection_rank=hp.arc_projection_rank,
            arc_eta=hp.arc_eta,
            arc_seed=hp.arc_seed,
            arc_start_compress_step=hp.arc_start_compress_step,
            arc_parameter_names={
                parameter: name for name, parameter in model.named_parameters()
            },
        )
        return optimizer, train.GradientSyncRuntime(
            optimizer_owns_gradient_sync=True
        )

    if hp.optimizer == "arc_topk_adamw":
        optimizer = train.build_adamw_optimizer(param_groups, hp)
    else:
        optimizer = Muon(param_groups, **optimizer_kwargs)
    return optimizer, install_arc_topk_ddp_hook(model, ddp_model, optimizer, hp)


if __name__ == "__main__":
    train.main(
        hyperparameters_factory=ArcTopKHyperparameters,
        optimizer_factory=init_arc_topk_optimizer,
        configure_parser=configure_arc_topk_parser,
    )
