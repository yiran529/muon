"""ARC-TopK compression primitives and EF21M state updates."""

import math
from dataclasses import dataclass
from typing import Generator, List, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch import Tensor
from torch.distributed import ProcessGroup
from .collective_observer import observe_collective

if TYPE_CHECKING:
    from .arc_topk_sync import ArcTopKSyncConfig


_TORCH_SEED_MODULUS = 1 << 64


@dataclass
class ArcPreparedBatch:
    """Collective-free ARC state prepared for full or sparse finalization."""

    tracker_batch: Tensor
    local_estimate_batch: Tensor
    global_estimate_batch: Tensor
    delta_batch: Optional[Tensor]
    projection_batch: Optional[Tensor]
    local_sketch_batch: Optional[Tensor]


@dataclass
class EF14PreparedBatch:
    """Collective-free EF14 tensors for a same-shaped parameter batch."""

    residual_batch: Tensor
    compensated_batch: Tensor
    local_sketch_batch: Tensor


def derive_arc_seed(*, base_seed: int, step: int, stable_task_id: int) -> int:
    """Derive a task-local ARC seed accepted by ``Generator.manual_seed``."""

    seed = int(base_seed) + int(step) * 1_000_003 + int(stable_task_id)
    return seed % _TORCH_SEED_MODULUS


def validate_arc_topk_config(
    ratio: float,
    projection_rank: int,
    eta: float,
    start_compress_step: int = 0,
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
    if (
        isinstance(start_compress_step, bool)
        or not isinstance(start_compress_step, int)
        or start_compress_step < 0
    ):
        raise ValueError(
            "start_compress_step must be a non-negative integer, got "
            f"{start_compress_step!r}"
        )


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
    return torch.bmm(delta, projection) / math.sqrt(projection.shape[-1])


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


def prepare_arc_batch(
    gradient_batch: Tensor,
    tracker_batch: Tensor,
    local_estimate_batch: Tensor,
    global_estimate_batch: Tensor,
    *,
    config: "ArcTopKSyncConfig",
    step: int,
    projection_batch: Optional[Tensor],
) -> ArcPreparedBatch:
    """Update the EF21M tracker and prepare collective-free ARC tensors."""

    if gradient_batch.ndim != 3:
        raise ValueError("gradient_batch must be a batched 3D tensor")
    if not (
        tracker_batch.shape
        == local_estimate_batch.shape
        == global_estimate_batch.shape
        == gradient_batch.shape
    ):
        raise ValueError("gradient and EF21M state batches must have equal shapes")

    if step == 1:
        tracker_batch.copy_(gradient_batch.to(dtype=tracker_batch.dtype))
    else:
        ef21m_update_tracker_(tracker_batch, gradient_batch, config.eta)

    is_warmup = step == 1 or step <= config.start_compress_step
    if is_warmup or (config.ratio == 1.0 and projection_batch is None):
        return ArcPreparedBatch(
            tracker_batch=tracker_batch,
            local_estimate_batch=local_estimate_batch,
            global_estimate_batch=global_estimate_batch,
            delta_batch=None,
            projection_batch=None,
            local_sketch_batch=None,
        )
    if projection_batch is None:
        raise ValueError("projection_batch is required for sparse ARC preparation")

    delta_batch = tracker_batch - local_estimate_batch
    return ArcPreparedBatch(
        tracker_batch=tracker_batch,
        local_estimate_batch=local_estimate_batch,
        global_estimate_batch=global_estimate_batch,
        delta_batch=delta_batch,
        projection_batch=projection_batch,
        local_sketch_batch=arc_topk_local_sketch(delta_batch, projection_batch),
    )


def finalize_arc_full_support_(
    prepared: ArcPreparedBatch,
    averaged_tracker_batch: Tensor,
) -> Tensor:
    """Commit complete tracker support into local and global estimates."""

    prepared.local_estimate_batch.copy_(prepared.tracker_batch)
    prepared.global_estimate_batch.copy_(
        averaged_tracker_batch.to(dtype=prepared.global_estimate_batch.dtype)
    )
    return prepared.global_estimate_batch


def finalize_arc_sparse_(
    prepared: ArcPreparedBatch,
    indices: Tensor,
    local_selected: Tensor,
    averaged_selected: Tensor,
) -> Tensor:
    """Commit selected local and averaged deltas into EF21M estimates."""

    if prepared.delta_batch is None:
        raise ValueError("sparse finalization requires a prepared delta batch")
    rows = prepared.tracker_batch.shape[1]
    local_compressed = scatter_rows(local_selected, indices, rows)
    averaged_compressed = scatter_rows(averaged_selected, indices, rows)
    ef21m_apply_delta_(
        prepared.local_estimate_batch,
        prepared.global_estimate_batch,
        local_compressed,
        averaged_compressed,
    )
    return prepared.global_estimate_batch


def prepare_ef14_batch(
    gradient_batch: Tensor,
    residual_batch: Tensor,
    *,
    projection_batch: Tensor,
) -> EF14PreparedBatch:
    """Prepare ``gradient + residual`` for ARC compression under EF14."""

    if gradient_batch.ndim != 3 or residual_batch.shape != gradient_batch.shape:
        raise ValueError("EF14 gradient and residual batches must have equal 3D shapes")
    compensated_batch = gradient_batch + residual_batch.to(dtype=gradient_batch.dtype)
    return EF14PreparedBatch(
        residual_batch=residual_batch,
        compensated_batch=compensated_batch,
        local_sketch_batch=arc_topk_local_sketch(
            compensated_batch, projection_batch
        ),
    )


def finalize_ef14_sparse_(
    prepared: EF14PreparedBatch,
    indices: Tensor,
    averaged_selected: Tensor,
) -> Tensor:
    """Commit EF14 residual and return the averaged sparse compressor output."""

    rows = prepared.compensated_batch.shape[1]
    local_selected = gather_rows(prepared.compensated_batch, indices)
    local_compressed = scatter_rows(local_selected, indices, rows)
    prepared.residual_batch.copy_(
        (prepared.compensated_batch - local_compressed).to(
            dtype=prepared.residual_batch.dtype
        )
    )
    return scatter_rows(averaged_selected, indices, rows)


def arc_topk_ef21m_async(
    gradients: List[Tensor],
    trackers: List[Tensor],
    local_estimates: List[Tensor],
    global_estimates: List[Tensor],
    *,
    process_group: Optional[ProcessGroup],
    ratio: float,
    projection_rank: int,
    eta: float,
    base_seed: int,
    step: int,
    stable_task_id: int,
    start_compress_step: int = 0,
) -> Generator[None, None, List[Tensor]]:
    """Apply ARC-TopK Algorithm 1 to an EF21M shape group.

    The four tensor lists contain corresponding two-dimensional tensors. State
    tensors are updated in place; the returned list is ``global_estimates`` so
    it can feed directly into the caller's optimizer update.
    """
    validate_arc_topk_config(ratio, projection_rank, eta, start_compress_step)
    count = len(gradients)
    if count == 0:
        return []
    if not (
        count == len(trackers) == len(local_estimates) == len(global_estimates)
    ):
        raise ValueError("gradient and EF21M state lists must have equal lengths")
    shape = gradients[0].shape
    if len(shape) != 2 or any(t.shape != shape for t in gradients):
        raise ValueError("ARC-TopK shape groups require same-shaped 2D gradients")

    with record_function("arc/state_stack"):
        gradient_batch = torch.stack(
            [g.to(dtype=trackers[i].dtype) for i, g in enumerate(gradients)]
        )
        tracker_batch = torch.stack(trackers)
        local_estimate_batch = torch.stack(local_estimates)
        global_estimate_batch = torch.stack(global_estimates)

    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    if step == 1 or step <= start_compress_step:
        from .arc_topk_sync import ArcTopKSyncConfig

        with record_function("arc/ef21m"):
            prepared = prepare_arc_batch(
                gradient_batch,
                tracker_batch,
                local_estimate_batch,
                global_estimate_batch,
                config=ArcTopKSyncConfig(
                    ratio=ratio,
                    projection_rank=projection_rank,
                    eta=eta,
                    seed=base_seed,
                    start_compress_step=start_compress_step,
                ),
                step=step,
                projection_batch=None,
            )
        averaged_tracker_batch = tracker_batch.clone()
        if process_group is not None and world_size > 1:
            observe_collective(
                "arc/dense_uncompressed", "all_reduce", averaged_tracker_batch
            )
            with record_function("arc/dense_uncompressed"):
                work = dist.all_reduce(
                    averaged_tracker_batch,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                    async_op=True,
                )
            yield
            with record_function("arc/dense_uncompressed_wait"):
                work.wait()
            averaged_tracker_batch.div_(world_size)

        with record_function("arc/ef21m"):
            finalize_arc_full_support_(prepared, averaged_tracker_batch)

        torch._foreach_copy_(trackers, list(tracker_batch.unbind(0)))
        torch._foreach_copy_(local_estimates, list(local_estimate_batch.unbind(0)))
        torch._foreach_copy_(global_estimates, list(global_estimate_batch.unbind(0)))
        return global_estimates

    seed = derive_arc_seed(
        base_seed=base_seed,
        step=step,
        stable_task_id=stable_task_id,
    )

    rows, columns = shape
    with record_function("arc/projection"):
        projection = make_gaussian_projection(
            count,
            columns,
            projection_rank,
            seed=seed,
            device=gradient_batch.device,
            dtype=gradient_batch.dtype,
        )
    from .arc_topk_sync import ArcTopKSyncConfig

    with record_function("arc/ef21m"):
        prepared = prepare_arc_batch(
            gradient_batch,
            tracker_batch,
            local_estimate_batch,
            global_estimate_batch,
            config=ArcTopKSyncConfig(
                ratio=ratio,
                projection_rank=projection_rank,
                eta=eta,
                seed=base_seed,
                start_compress_step=start_compress_step,
            ),
            step=step,
            projection_batch=projection,
        )
    with record_function("arc/sketch_compute"):
        global_sketch = prepared.local_sketch_batch
        assert global_sketch is not None
    if process_group is not None and world_size > 1:
        observe_collective("arc/sketch", "all_reduce", global_sketch)
        with record_function("arc/sketch"):
            work = dist.all_reduce(
                global_sketch,
                op=dist.ReduceOp.SUM,
                group=process_group,
                async_op=True,
            )
        yield
        with record_function("arc/sketch_wait"):
            work.wait()
        global_sketch.div_(world_size)

    k = math.ceil(ratio * rows)
    with record_function("arc/topk"):
        indices = arc_topk_support(global_sketch, k)
    with record_function("arc/gather"):
        assert prepared.delta_batch is not None
        local_selected = gather_rows(prepared.delta_batch, indices)
    averaged_selected = local_selected.clone()
    if process_group is not None and world_size > 1:
        observe_collective("arc/selected_values", "all_reduce", averaged_selected)
        with record_function("arc/selected_values"):
            work = dist.all_reduce(
                averaged_selected,
                op=dist.ReduceOp.SUM,
                group=process_group,
                async_op=True,
            )
        yield
        with record_function("arc/selected_values_wait"):
            work.wait()
        averaged_selected.div_(world_size)

    with record_function("arc/ef21m"):
        finalize_arc_sparse_(
            prepared,
            indices,
            local_selected,
            averaged_selected,
        )

    with record_function("arc/state_copy"):
        torch._foreach_copy_(trackers, list(tracker_batch.unbind(0)))
        torch._foreach_copy_(local_estimates, list(local_estimate_batch.unbind(0)))
        torch._foreach_copy_(global_estimates, list(global_estimate_batch.unbind(0)))
    return global_estimates
