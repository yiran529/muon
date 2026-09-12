"""Dedicated DDP training entry point for Rand-K and Top-K Muon."""

import argparse
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

import train
from dion.muon import Muon
from dion.sparse_k import SparseKConfig
from dion.sparse_k_ddp_hook import (
    SparseKDDPParameterSpec,
    SparseKDDPState,
    sparse_k_ddp_hook,
)
from dion.sparse_k_layout import (
    SparseKParameterDescriptor,
    canonical_sparse_k_fingerprint,
    validate_sparse_k_fingerprint_across_ranks,
)


@dataclass
class SparseKHyperparameters(train.Hyperparameters):
    optimizer: str = "sparse_k_muon"
    sparse_k_method: Literal["randk", "topk"] = "topk"
    sparse_k_ratio: float = 0.2
    sparse_k_error_feedback: Literal["ef14", "noef"] = "ef14"
    sparse_k_seed: int = 42
    sparse_k_start_compress_step: int = 1000


def configure_sparse_k_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sparse_k_method", choices=("randk", "topk"), default=None)
    parser.add_argument("--sparse_k_ratio", type=float, default=None)
    parser.add_argument(
        "--sparse_k_error_feedback", choices=("ef14", "noef"), default=None
    )
    parser.add_argument("--sparse_k_seed", type=int, default=None)
    parser.add_argument("--sparse_k_start_compress_step", type=int, default=None)


def validate_sparse_k_hyperparameters(hp: SparseKHyperparameters) -> None:
    if hp.optimizer != "sparse_k_muon":
        raise ValueError(f"Unsupported Sparse-K optimizer: {hp.optimizer}")
    SparseKConfig(
        method=hp.sparse_k_method,
        ratio=hp.sparse_k_ratio,
        error_feedback=hp.sparse_k_error_feedback,
        seed=hp.sparse_k_seed,
        start_compress_step=hp.sparse_k_start_compress_step,
    )


def _install_sparse_k_ddp_hook(
    model,
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    hp: SparseKHyperparameters,
) -> train.GradientSyncRuntime:
    config = SparseKConfig(
        method=hp.sparse_k_method,
        ratio=hp.sparse_k_ratio,
        error_feedback=hp.sparse_k_error_feedback,
        seed=hp.sparse_k_seed,
        start_compress_step=hp.sparse_k_start_compress_step,
    )
    specs = tuple(
        SparseKDDPParameterSpec(
            parameter=parameter,
            stable_name=name,
            stable_id=stable_id,
            role="sparse_matrix" if parameter.ndim == 2 else "dense_aux",
        )
        for stable_id, (name, parameter) in enumerate(model.named_parameters())
    )
    group_ranks = (
        tuple(dist.get_process_group_ranks(ddp_model.process_group))
        if ddp_model.process_group is not None
        else (0,)
    )
    fingerprint = canonical_sparse_k_fingerprint(
        config=config,
        group_ranks=group_ranks,
        parameters=tuple(
            SparseKParameterDescriptor(
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
        validate_sparse_k_fingerprint_across_ranks(fingerprint, ddp_model.process_group)
    state = SparseKDDPState(
        process_group=ddp_model.process_group,
        fingerprint=fingerprint,
        parameter_specs=specs,
        optimizer_parameters=[
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ],
        config=config,
        find_unused_parameters=getattr(ddp_model, "find_unused_parameters", False),
    )
    ddp_model.register_comm_hook(state, sparse_k_ddp_hook)
    return train.GradientSyncRuntime(
        optimizer_owns_gradient_sync=False,
        begin_step=state.begin_step,
        finish_step=state.finish_step,
        commit_step=state.commit_step,
        checkpoint_state=state,
        checkpoint_state_name="sparse_k_compressor",
    )


def init_sparse_k_optimizer(
    model,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: SparseKHyperparameters,
    cli_args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, train.GradientSyncRuntime]:
    if device_mesh is not None:
        raise ValueError("Sparse-K-Muon first version is DDP only")
    if ddp_model is None:
        raise ValueError("Sparse-K-Muon requires a DDP model")
    validate_sparse_k_hyperparameters(hp)
    if getattr(cli_args, "_explicit_replicate_mesh_grad_sync", False):
        raise ValueError(
            "replicate_mesh_grad_sync is not accepted by train_sparsek.py; "
            "Sparse-K sync is owned by the DDP hook"
        )

    param_groups = train.build_muon_param_groups(model, hp)
    train.print0(f"Sparse-K method: {hp.sparse_k_method}")
    train.print0(f"Sparse-K ratio: {hp.sparse_k_ratio}")
    train.print0(f"Sparse-K error feedback: {hp.sparse_k_error_feedback}")
    train.print0(
        f"Sparse-K compression starts after step: {hp.sparse_k_start_compress_step}"
    )
    train.print0(f"Muon LR adjust method: {hp.adjust_lr}")
    train.print0(f"Triton Newton-Schulz kernels: {not cli_args.no_triton}")
    optimizer = Muon(
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
    )
    return optimizer, _install_sparse_k_ddp_hook(model, ddp_model, optimizer, hp)


if __name__ == "__main__":
    train.main(
        hyperparameters_factory=SparseKHyperparameters,
        optimizer_factory=init_sparse_k_optimizer,
        configure_parser=configure_sparse_k_parser,
        validate_hyperparameters=validate_sparse_k_hyperparameters,
    )
