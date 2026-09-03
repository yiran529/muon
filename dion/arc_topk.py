"""ARC-TopK compression primitives and EF21M state updates."""

import math

import torch
from torch import Tensor


def validate_arc_topk_config(
    ratio: float,
    projection_rank: int,
    eta: float,
) -> None:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if (
        isinstance(projection_rank, bool)
        or not isinstance(projection_rank, int)
        or projection_rank < 1
    ):
        raise ValueError(
            f"projection_rank must be a positive integer, got {projection_rank!r}"
        )
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")


def make_gaussian_projection(
    batch: int,
    columns: int,
    rank: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        batch,
        columns,
        rank,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(dtype=dtype)


def arc_topk_local_sketch(delta: Tensor, projection: Tensor) -> Tensor:
    if delta.ndim != 3 or projection.ndim != 3:
        raise ValueError("delta and projection must both be batched 3D tensors")
    if delta.shape[0] != projection.shape[0] or delta.shape[2] != projection.shape[1]:
        raise ValueError(
            f"incompatible delta/projection shapes: {tuple(delta.shape)} and "
            f"{tuple(projection.shape)}"
        )
    return torch.bmm(delta.float(), projection.float()) / math.sqrt(
        projection.shape[-1]
    )


def arc_topk_support(global_sketch: Tensor, k: int) -> Tensor:
    if global_sketch.ndim != 3:
        raise ValueError("global_sketch must be a batched 3D tensor")
    rows = global_sketch.shape[1]
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= rows:
        raise ValueError(f"k must be in [1, {rows}], got {k!r}")
    scores = global_sketch.square().sum(dim=-1)
    return scores.topk(k=k, dim=-1, sorted=True).indices


def gather_rows(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim != 3 or indices.ndim != 2:
        raise ValueError("values must be 3D and indices must be 2D")
    if values.shape[0] != indices.shape[0]:
        raise ValueError("values and indices must have the same batch size")
    expanded = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, dim=1, index=expanded)


def scatter_rows(selected: Tensor, indices: Tensor, rows: int) -> Tensor:
    if selected.ndim != 3 or indices.ndim != 2:
        raise ValueError("selected must be 3D and indices must be 2D")
    if selected.shape[:2] != indices.shape:
        raise ValueError("selected and indices must agree on batch and selected rows")
    result = selected.new_zeros(selected.shape[0], rows, selected.shape[-1])
    expanded = indices.unsqueeze(-1).expand_as(selected)
    result.scatter_(dim=1, index=expanded, src=selected)
    return result


def ef21m_update_tracker_(tracker: Tensor, gradient: Tensor, eta: float) -> Tensor:
    tracker.lerp_(gradient.to(dtype=tracker.dtype), eta)
    return tracker


def ef21m_apply_delta_(
    local_estimate: Tensor,
    global_estimate: Tensor,
    local_delta: Tensor,
    averaged_delta: Tensor,
) -> None:
    local_estimate.add_(local_delta.to(dtype=local_estimate.dtype))
    global_estimate.add_(averaged_delta.to(dtype=global_estimate.dtype))
