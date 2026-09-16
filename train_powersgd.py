"""Dedicated DDP training entry point for PowerSGD gradient sync with Muon."""

import argparse
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

import train
from dion import Muon, PowerSGDConfig, PowerSGDDDPParameterSpec, PowerSGDDDPState
from dion.power_sgd_ddp_hook import power_sgd_ddp_hook
from dion.power_sgd_layout import (
    PowerSGDParameterDescriptor,
    canonical_power_sgd_fingerprint,
    validate_power_sgd_fingerprint_across_ranks,
)


@dataclass
class PowerSGDHyperparameters(train.Hyperparameters):
    optimizer: str = "power_sgd_muon"
    power_sgd_rank: int = 4
    power_sgd_start_compress_step: int = 1000
    power_sgd_min_compression_rate: float = 2.0
    power_sgd_error_feedback: Literal["ef14", "none"] = "ef14"
    power_sgd_warm_start: bool = True
    power_sgd_seed: int = 42
    power_sgd_orthogonalization_epsilon: float = 1e-8
    power_sgd_seed_scheme_version: int = 1


def configure_power_sgd_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--power_sgd_rank", type=int, default=None)
    parser.add_argument("--power_sgd_start_compress_step", type=int, default=None)
    parser.add_argument("--power_sgd_min_compression_rate", type=float, default=None)
    parser.add_argument(
        "--power_sgd_error_feedback", choices=("ef14", "none"), default=None
    )
    parser.add_argument(
        "--power_sgd_warm_start", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--power_sgd_seed", type=int, default=None)
    parser.add_argument(
        "--power_sgd_orthogonalization_epsilon", type=float, default=None
    )
    parser.add_argument("--power_sgd_seed_scheme_version", type=int, default=None)


def _power_sgd_config(hp: PowerSGDHyperparameters) -> PowerSGDConfig:
    return PowerSGDConfig(
        rank=hp.power_sgd_rank,
        start_compress_step=hp.power_sgd_start_compress_step,
        min_compression_rate=hp.power_sgd_min_compression_rate,
        error_feedback=hp.power_sgd_error_feedback,
        warm_start=hp.power_sgd_warm_start,
        seed=hp.power_sgd_seed,
        orthogonalization_epsilon=hp.power_sgd_orthogonalization_epsilon,
        seed_scheme_version=hp.power_sgd_seed_scheme_version,
    )


def validate_power_sgd_hyperparameters(hp: PowerSGDHyperparameters) -> None:
    if hp.optimizer != "power_sgd_muon":
        raise ValueError(f"Unsupported PowerSGD optimizer: {hp.optimizer}")
    if hp.replicate_mesh_grad_sync:
        raise ValueError("replicate_mesh_grad_sync is not accepted by PowerSGD-Muon")
    _power_sgd_config(hp)


def init_power_sgd_optimizer(
    model,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: PowerSGDHyperparameters,
    cli_args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, train.GradientSyncRuntime]:
    if device_mesh is not None:
        raise ValueError("PowerSGD-Muon is DDP only")
    if ddp_model is None:
        raise ValueError("PowerSGD-Muon requires a DDP model")
    validate_power_sgd_hyperparameters(hp)
    if getattr(cli_args, "_explicit_replicate_mesh_grad_sync", False):
        raise ValueError(
            "replicate_mesh_grad_sync is not accepted by train_powersgd.py; "
            "gradient sync is owned by the DDP hook"
        )

    config = _power_sgd_config(hp)
    param_groups = train.build_muon_param_groups(model, hp)
    muon_parameter_ids = {id(parameter) for parameter in param_groups[0]["params"]}
    specs = tuple(
        PowerSGDDDPParameterSpec(
            parameter=parameter,
            stable_name=name,
            stable_id=stable_id,
            role=(
                "matrix"
                if id(parameter) in muon_parameter_ids and parameter.ndim == 2
                else "dense_aux"
            ),
        )
        for stable_id, (name, parameter) in enumerate(model.named_parameters())
    )
    group_ranks = (
        tuple(dist.get_process_group_ranks(ddp_model.process_group))
        if ddp_model.process_group is not None
        else (0,)
    )
    fingerprint = canonical_power_sgd_fingerprint(
        config=config,
        group_ranks=group_ranks,
        parameters=tuple(
            PowerSGDParameterDescriptor(
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
        validate_power_sgd_fingerprint_across_ranks(
            fingerprint, ddp_model.process_group
        )

    train.print0(f"PowerSGD rank: {config.rank}")
    train.print0(
        f"PowerSGD compression starts after step: {config.start_compress_step}"
    )
    train.print0(f"PowerSGD error feedback: {config.error_feedback}")
    train.print0(f"PowerSGD warm start: {config.warm_start}")

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
    state = PowerSGDDDPState(
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
    ddp_model.register_comm_hook(state, power_sgd_ddp_hook)
    return optimizer, train.GradientSyncRuntime(
        optimizer_owns_gradient_sync=False,
        begin_step=state.begin_step,
        finish_step=state.finish_step,
        commit_step=state.commit_step,
        checkpoint_state=state,
        checkpoint_state_name="power_sgd_compressor",
    )


if __name__ == "__main__":
    train.main(
        hyperparameters_factory=PowerSGDHyperparameters,
        optimizer_factory=init_power_sgd_optimizer,
        configure_parser=configure_power_sgd_parser,
        validate_hyperparameters=validate_power_sgd_hyperparameters,
    )
