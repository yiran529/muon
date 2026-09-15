"""Dedicated DDP training entry point for GreedyLore gradient sync with Muon."""

import argparse
import inspect
import math

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

import train
from dion import GreedyLoreConfig, GreedyLoreDDPParameterSpec, GreedyLoreDDPState, Muon
from dion.greedy_lore_ddp_hook import greedy_lore_ddp_hook
from dion.greedy_lore_layout import (
    GreedyLoreParameterDescriptor,
    canonical_greedy_lore_fingerprint,
    validate_greedy_lore_fingerprint_across_ranks,
)


@dataclass
class GreedyLoreHyperparameters(train.Hyperparameters):
    optimizer: str = "greedy_lore_muon"
    greedy_lore_rank: int = 32
    greedy_lore_update_interval: int = 200
    greedy_lore_seed: int = 42
    greedy_lore_start_compress_step: int = 1000
    greedy_lore_basis_sync: Literal["local_svd", "broadcast"] = "local_svd"
    greedy_lore_dense_aux_communication_dtype: Literal[
        "bucket", "float32", "bfloat16"
    ] = "bucket"
    greedy_lore_compress_embedding_lm_head: bool = False
    greedy_lore_isolate_dense_aux_buckets: bool = False


def configure_greedy_lore_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--greedy_lore_rank", type=int, default=None)
    parser.add_argument("--greedy_lore_update_interval", type=int, default=None)
    parser.add_argument("--greedy_lore_seed", type=int, default=None)
    parser.add_argument("--greedy_lore_start_compress_step", type=int, default=None)
    parser.add_argument(
        "--greedy_lore_basis_sync",
        choices=("local_svd", "broadcast"),
        default=None,
    )
    parser.add_argument(
        "--greedy_lore_dense_aux_communication_dtype",
        choices=("bucket", "float32", "bfloat16"),
        default=None,
    )
    parser.add_argument(
        "--greedy_lore_compress_embedding_lm_head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compress both embedding and LM-head gradients with GreedyLore",
    )
    parser.add_argument(
        "--greedy_lore_isolate_dense_aux_buckets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use exact DDP bucket caps to isolate the dense embedding and LM head "
            "from compressed Transformer blocks"
        ),
    )
    parser.add_argument(
        "--greedy-lore-bucket-cap-mb-list",
        type=_parse_bucket_cap_mb_list,
        default=None,
        help="Comma-separated calibrated DDP bucket caps in reducer-ready order",
    )
    parser.add_argument(
        "--greedy-lore-bucket-layout-output-dir",
        default=None,
        help="Write the observed per-rank reducer bucket layout as JSON",
    )
    parser.add_argument(
        "--greedy-lore-bucket-layout-capture-step",
        type=int,
        default=None,
        help="GreedyLore active step whose rebuilt bucket layout is written",
    )
    parser.add_argument(
        "--greedy-lore-require-role-aligned-buckets",
        action="store_true",
        help="Fail after DDP rebuild if any bucket mixes matrix and dense auxiliary roles",
    )


def validate_greedy_lore_hyperparameters(hp: GreedyLoreHyperparameters) -> None:
    if hp.optimizer != "greedy_lore_muon":
        raise ValueError(f"Unsupported GreedyLore optimizer: {hp.optimizer}")
    GreedyLoreConfig(
        rank=hp.greedy_lore_rank,
        update_interval=hp.greedy_lore_update_interval,
        seed=hp.greedy_lore_seed,
        start_compress_step=hp.greedy_lore_start_compress_step,
        basis_sync=hp.greedy_lore_basis_sync,
        dense_aux_communication_dtype=hp.greedy_lore_dense_aux_communication_dtype,
    )
    if (
        hp.greedy_lore_isolate_dense_aux_buckets
        and hp.greedy_lore_compress_embedding_lm_head
    ):
        raise ValueError(
            "greedy_lore_isolate_dense_aux_buckets and "
            "greedy_lore_compress_embedding_lm_head cannot be enabled together"
        )


_MIB = 1024 * 1024


def _parse_bucket_cap_mb_list(value: str) -> tuple[float, ...]:
    try:
        caps = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bucket cap list must contain comma-separated numbers"
        ) from exc
    if not caps or any(not math.isfinite(cap) or cap <= 0 for cap in caps):
        raise argparse.ArgumentTypeError(
            "bucket cap list values must be finite and positive"
        )
    return caps


def _parameter_bytes(parameters) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in parameters)


def build_isolated_dense_aux_bucket_cap_mb_list(
    model,
    *,
    target_cap_mb: float,
) -> list[float]:
    """Return exact caps for embedding, whole-block groups, and LM head."""

    if not math.isfinite(target_cap_mb) or target_cap_mb <= 0:
        raise ValueError("target_cap_mb must be finite and positive")

    embedding = tuple(model.transformer.wte.parameters())
    blocks = tuple(tuple(block.parameters()) for block in model.transformer.h)
    lm_head = tuple(model.lm_head.parameters())
    expected_parameters = (
        *embedding,
        *(p for block in blocks for p in block),
        *lm_head,
    )
    actual_parameters = tuple(model.parameters())
    if tuple(map(id, actual_parameters)) != tuple(map(id, expected_parameters)):
        raise ValueError(
            "dense auxiliary bucket isolation requires GPT parameter registration "
            "order: embedding, Transformer blocks, LM head"
        )

    embedding_bytes = _parameter_bytes(embedding)
    lm_head_bytes = _parameter_bytes(lm_head)
    if embedding_bytes == 0 or lm_head_bytes == 0:
        raise ValueError("embedding and LM head must each contain parameters")

    target_bytes = int(target_cap_mb * _MIB)
    block_bucket_bytes: list[int] = []
    current_bytes = 0
    for block in reversed(blocks):
        block_bytes = _parameter_bytes(block)
        if block_bytes == 0:
            continue
        if current_bytes and current_bytes + block_bytes > target_bytes:
            block_bucket_bytes.append(current_bytes)
            current_bytes = 0
        current_bytes += block_bytes
    if current_bytes:
        block_bucket_bytes.append(current_bytes)
    if not block_bucket_bytes:
        raise ValueError(
            "at least one trainable Transformer block parameter is required"
        )

    return [
        value / _MIB for value in (lm_head_bytes, *block_bucket_bytes, embedding_bytes)
    ]


def require_ddp_bucket_cap_mb_list_support(ddp_type=DDP) -> None:
    if "bucket_cap_mb_list" not in inspect.signature(ddp_type).parameters:
        raise RuntimeError(
            "greedy_lore_isolate_dense_aux_buckets requires PyTorch DDP support "
            "for bucket_cap_mb_list (available in PyTorch 2.11 or newer)"
        )


def greedy_lore_ddp_kwargs(model, hp, cli_args) -> dict:
    """Opt into role-aligned DDP buckets without changing the GPT module tree."""

    if not hp.greedy_lore_isolate_dense_aux_buckets:
        return {}
    require_ddp_bucket_cap_mb_list_support()
    calibrated_caps = getattr(cli_args, "greedy_lore_bucket_cap_mb_list", None)
    if calibrated_caps is None:
        target_cap_mb = (
            25.0
            if getattr(cli_args, "bucket_cap_mb", None) is None
            else cli_args.bucket_cap_mb
        )
        caps = build_isolated_dense_aux_bucket_cap_mb_list(
            model,
            target_cap_mb=target_cap_mb,
        )
    else:
        caps = list(calibrated_caps)
    train.print0(f"GreedyLore DDP bucket caps (MiB): {caps}")
    return {"bucket_cap_mb_list": caps}


def _install_greedy_lore_ddp_hook(
    model,
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    hp: GreedyLoreHyperparameters,
    compressed_parameters: set[int],
    cli_args: argparse.Namespace,
) -> train.GradientSyncRuntime:
    config = GreedyLoreConfig(
        rank=hp.greedy_lore_rank,
        update_interval=hp.greedy_lore_update_interval,
        seed=hp.greedy_lore_seed,
        start_compress_step=hp.greedy_lore_start_compress_step,
        basis_sync=hp.greedy_lore_basis_sync,
        dense_aux_communication_dtype=hp.greedy_lore_dense_aux_communication_dtype,
    )
    specs = tuple(
        GreedyLoreDDPParameterSpec(
            parameter=parameter,
            stable_name=name,
            stable_id=stable_id,
            role="matrix" if id(parameter) in compressed_parameters else "dense_aux",
        )
        for stable_id, (name, parameter) in enumerate(model.named_parameters())
    )
    group_ranks = (
        tuple(dist.get_process_group_ranks(ddp_model.process_group))
        if ddp_model.process_group is not None
        else (0,)
    )
    fingerprint = canonical_greedy_lore_fingerprint(
        config=config,
        group_ranks=group_ranks,
        parameters=tuple(
            GreedyLoreParameterDescriptor(
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
        validate_greedy_lore_fingerprint_across_ranks(
            fingerprint,
            ddp_model.process_group,
        )
    state = GreedyLoreDDPState(
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
            ddp_model,
            "find_unused_parameters",
            False,
        ),
        bucket_layout_output_dir=getattr(
            cli_args, "greedy_lore_bucket_layout_output_dir", None
        ),
        bucket_layout_capture_step=getattr(
            cli_args, "greedy_lore_bucket_layout_capture_step", None
        ),
        require_role_aligned_buckets=getattr(
            cli_args, "greedy_lore_require_role_aligned_buckets", False
        ),
        required_bucket_count=(
            len(cli_args.greedy_lore_bucket_cap_mb_list)
            if getattr(cli_args, "greedy_lore_bucket_cap_mb_list", None) is not None
            else None
        ),
    )
    ddp_model.register_comm_hook(state, greedy_lore_ddp_hook)
    return train.GradientSyncRuntime(
        optimizer_owns_gradient_sync=False,
        begin_step=state.begin_step,
        finish_step=state.finish_step,
        commit_step=state.commit_step,
        checkpoint_state=state,
        checkpoint_state_name="greedy_lore_compressor",
    )


def init_greedy_lore_optimizer(
    model,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: GreedyLoreHyperparameters,
    cli_args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, train.GradientSyncRuntime]:
    if device_mesh is not None:
        raise ValueError("GreedyLore-Muon first version is DDP only")
    if ddp_model is None:
        raise ValueError("GreedyLore-Muon requires a DDP model")
    validate_greedy_lore_hyperparameters(hp)
    if getattr(cli_args, "_explicit_replicate_mesh_grad_sync", False):
        raise ValueError(
            "replicate_mesh_grad_sync is not accepted by train_greedylore.py; "
            "GreedyLore sync is owned by the DDP hook"
        )

    param_groups = train.build_muon_param_groups(model, hp)
    compressed_parameters = {id(parameter) for parameter in param_groups[0]["params"]}
    if hp.greedy_lore_compress_embedding_lm_head:
        compressed_parameters.update(
            {
                id(model.transformer.wte.weight),
                id(model.lm_head.weight),
            }
        )

    train.print0(f"GreedyLore rank: {hp.greedy_lore_rank}")
    train.print0(f"GreedyLore update interval: {hp.greedy_lore_update_interval}")
    train.print0(
        f"GreedyLore compression starts after step: "
        f"{hp.greedy_lore_start_compress_step}"
    )
    train.print0(f"GreedyLore basis sync: {hp.greedy_lore_basis_sync}")
    train.print0(
        "GreedyLore dense auxiliary communication dtype: "
        f"{hp.greedy_lore_dense_aux_communication_dtype}"
    )
    train.print0(
        "GreedyLore compress embedding and LM head: "
        f"{hp.greedy_lore_compress_embedding_lm_head}"
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
    runtime = _install_greedy_lore_ddp_hook(
        model,
        ddp_model,
        optimizer,
        hp,
        compressed_parameters,
        cli_args,
    )
    return optimizer, runtime


if __name__ == "__main__":
    train.main(
        hyperparameters_factory=GreedyLoreHyperparameters,
        optimizer_factory=init_greedy_lore_optimizer,
        configure_parser=configure_greedy_lore_parser,
        validate_hyperparameters=validate_greedy_lore_hyperparameters,
        ddp_kwargs_factory=greedy_lore_ddp_kwargs,
    )
