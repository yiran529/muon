"""Shared deterministic ARC-TopK gradient synchronization helpers."""

import math
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch import Tensor
from torch.distributed import ProcessGroup

from .arc_topk import arc_topk_ef21m_async, validate_arc_topk_config


@dataclass(frozen=True)
class ArcTopKSyncConfig:
    """Configuration shared by ARC-TopK optimizer implementations."""

    ratio: float = 0.2
    projection_rank: int = 4
    eta: float = 0.1
    seed: int = 42
    start_compress_step: int = 0

    def __post_init__(self) -> None:
        validate_arc_topk_config(
            self.ratio,
            self.projection_rank,
            self.eta,
            self.start_compress_step,
        )


@dataclass(frozen=True)
class ArcTopKLogicalBytes:
    """Per-rank logical input payload sizes, before ring amplification."""

    dense_gradient: int
    arc_seed: int
    arc_sketch: int
    arc_selected_values: int
    uncompressed: int


def initialize_arc_state_(state: dict, param: Tensor) -> None:
    """Initialize ARC-TopK EF21M state entries without replacing existing values."""

    state.setdefault("arc_h_local", torch.zeros_like(param))
    state.setdefault("arc_g_local", torch.zeros_like(param))
    state.setdefault("arc_g_global", torch.zeros_like(param))


def group_parameters_by_shape_dtype(
    params: Iterable[Tensor],
) -> list[list[Tensor]]:
    """Group parameters by shape and dtype while preserving first-seen order."""

    groups: dict[tuple[tuple[int, ...], torch.dtype], list[Tensor]] = {}
    for param in params:
        key = (tuple(param.shape), param.dtype)
        groups.setdefault(key, []).append(param)
    return list(groups.values())


def synchronize_arc_batch_async(
    *,
    params: list[Tensor],
    states: list[dict],
    process_group: Optional[ProcessGroup],
    config: ArcTopKSyncConfig,
    step: int,
    task_index: int,
) -> Generator[None, None, list[Tensor]]:
    """Synchronize one same-shaped parameter batch using the ARC-TopK primitive."""

    if len(params) != len(states):
        raise ValueError("params and states must have equal lengths")
    if not params:
        return []

    for param, state in zip(params, states):
        initialize_arc_state_(state, param)

    gradients = [
        param.grad if param.grad is not None else torch.zeros_like(param)
        for param in params
    ]
    return (yield from arc_topk_ef21m_async(
        gradients=gradients,
        trackers=[state["arc_h_local"] for state in states],
        local_estimates=[state["arc_g_local"] for state in states],
        global_estimates=[state["arc_g_global"] for state in states],
        process_group=process_group,
        ratio=config.ratio,
        projection_rank=config.projection_rank,
        eta=config.eta,
        base_seed=config.seed,
        step=step,
        task_index=task_index,
        start_compress_step=config.start_compress_step,
    ))


def average_gradients_async(
    gradients: list[Tensor],
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, list[Tensor]]:
    """Return cloned gradients averaged across a process group."""

    averaged = [gradient.clone() for gradient in gradients]
    if process_group is None:
        return averaged
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return averaged
    for gradient in averaged:
        with record_function("arc/dense_uncompressed"):
            work = dist.all_reduce(
                gradient,
                op=dist.ReduceOp.SUM,
                group=process_group,
                async_op=True,
            )
        yield
        work.wait()
        gradient.div_(world_size)
    return averaged


def estimate_arc_logical_bytes(
    *,
    compressed_batches: list[list[Tensor]],
    uncompressed_params: list[Tensor],
    config: ArcTopKSyncConfig,
    step: int,
) -> ArcTopKLogicalBytes:
    """Estimate logical collective inputs for one ARC-TopK synchronization step."""

    dense_gradient = 0
    arc_seed = 0
    arc_sketch = 0
    arc_selected_values = 0

    for batch in compressed_batches:
        if not batch:
            continue
        first = batch[0]
        if first.ndim != 2 or any(
            tensor.shape != first.shape or tensor.dtype != first.dtype
            for tensor in batch
        ):
            raise ValueError("compressed batches require same-shaped, same-dtype 2D tensors")
        dense_gradient += sum(tensor.numel() * tensor.element_size() for tensor in batch)
        if step == 1 or step <= config.start_compress_step:
            continue
        rows, columns = first.shape
        k = max(1, math.ceil(rows * config.ratio))
        element_size = first.element_size()
        arc_seed += 8
        arc_sketch += len(batch) * rows * config.projection_rank * element_size
        arc_selected_values += len(batch) * k * columns * element_size

    uncompressed = sum(param.numel() * param.element_size() for param in uncompressed_params)
    return ArcTopKLogicalBytes(
        dense_gradient=dense_gradient,
        arc_seed=arc_seed,
        arc_sketch=arc_sketch,
        arc_selected_values=arc_selected_values,
        uncompressed=uncompressed,
    )
