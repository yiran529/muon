"""Dedicated DDP training entry point for ARC-TopK-EF21M-Muon."""

import argparse

from dataclasses import dataclass
from typing import Optional

from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

import train
from dion import ArcTopKMuon
from dion.opt_utils import lm_head_lr_scale


@dataclass
class ArcTopKHyperparameters(train.Hyperparameters):
    """Shared training parameters plus the ARC-TopK method parameters."""

    optimizer: str = "arc_topk_muon"
    replicate_mesh_grad_sync: bool = True
    arc_topk_ratio: float = 0.2
    arc_projection_rank: int = 4
    arc_eta: float = 0.1
    arc_seed: int = 42


def configure_arc_topk_parser(parser: argparse.ArgumentParser) -> None:
    """Add ARC-TopK arguments to the shared training parser."""

    parser.add_argument("--arc_topk_ratio", type=float, default=None)
    parser.add_argument("--arc_projection_rank", type=int, default=None)
    parser.add_argument("--arc_eta", type=float, default=None)
    parser.add_argument("--arc_seed", type=int, default=None)


def init_arc_topk_optimizer(
    model,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: ArcTopKHyperparameters,
    cli_args: argparse.Namespace,
) -> ArcTopKMuon:
    """Build ARC-TopK-EF21M-Muon while preserving the baseline grouping."""

    if device_mesh is not None:
        raise ValueError("ARC-TopK-EF21M-Muon first version is DDP only")
    if ddp_model is None:
        raise ValueError("ARC-TopK-EF21M-Muon requires a DDP model")
    if hp.scalar_opt not in ("adamw", "lion"):
        raise ValueError(f"Unrecognized scalar optimizer: {hp.scalar_opt}")

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
    train.print0(f"Muon LR adjust method: {hp.adjust_lr}")
    train.print0(f"Triton Newton-Schulz kernels: {not cli_args.no_triton}")

    return ArcTopKMuon(
        param_groups,
        distributed_mesh=ddp_model.process_group,
        lr=hp.lr,
        mu=hp.mu,
        weight_decay=hp.weight_decay,
        nesterov=True,
        adjust_lr=hp.adjust_lr,
        use_gram_newton_schulz=cli_args.use_gram_newton_schulz,
        use_triton=not cli_args.no_triton,
        use_polar_express=cli_args.use_polar_express,
        arc_topk_ratio=hp.arc_topk_ratio,
        arc_projection_rank=hp.arc_projection_rank,
        arc_eta=hp.arc_eta,
        arc_seed=hp.arc_seed,
    )


if __name__ == "__main__":
    train.main(
        hyperparameters_factory=ArcTopKHyperparameters,
        optimizer_factory=init_arc_topk_optimizer,
        configure_parser=configure_arc_topk_parser,
    )
