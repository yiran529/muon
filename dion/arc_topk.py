"""ARC-TopK compression primitives and EF21M state updates."""

import math
from typing import Generator, List, Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch import Tensor
from torch.distributed import ProcessGroup
from .collective_observer import observe_collective


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
    task_index: int,
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

    gradient_batch = torch.stack(
        [g.to(dtype=trackers[i].dtype) for i, g in enumerate(gradients)]
    )
    tracker_batch = torch.stack(trackers)
    local_estimate_batch = torch.stack(local_estimates)
    global_estimate_batch = torch.stack(global_estimates)

    with record_function("arc/ef21m"):
        if step == 1:
            tracker_batch.copy_(gradient_batch)
        else:
            ef21m_update_tracker_(tracker_batch, gradient_batch, eta)

    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    if step == 1 or step <= start_compress_step:
        local_estimate_batch.copy_(tracker_batch)
        global_estimate_batch.copy_(tracker_batch)
        if process_group is not None and world_size > 1:
            observe_collective("arc/dense_uncompressed", "all_reduce", global_estimate_batch)
            with record_function("arc/dense_uncompressed"):
                work = dist.all_reduce(
                    global_estimate_batch,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                    async_op=True,
                )
            yield
            work.wait()
            global_estimate_batch.div_(world_size)

        torch._foreach_copy_(trackers, list(tracker_batch.unbind(0)))
        torch._foreach_copy_(local_estimates, list(local_estimate_batch.unbind(0)))
        torch._foreach_copy_(global_estimates, list(global_estimate_batch.unbind(0)))
        return global_estimates

    group_rank = dist.get_rank(process_group) if process_group is not None else 0
    source_rank = (
        dist.get_process_group_ranks(process_group)[0]
        if process_group is not None
        else 0
    )
    seed_value = int(base_seed) + int(step) * 1_000_003 + int(task_index)
    seed_tensor = torch.zeros((), dtype=torch.int64, device=gradients[0].device)
    if group_rank == 0:
        seed_tensor.fill_(seed_value)
    if process_group is not None and world_size > 1:
        observe_collective("arc/seed", "broadcast", seed_tensor)
        with record_function("arc/seed"):
            work = dist.broadcast(
                seed_tensor,
                src=source_rank,
                group=process_group,
                async_op=True,
            )
        yield
        work.wait()
    synchronized_seed = int(seed_tensor.item())

    delta_batch = tracker_batch - local_estimate_batch
    rows, columns = shape
    with record_function("arc/projection"):
        projection = make_gaussian_projection(
            count,
            columns,
            projection_rank,
            seed=synchronized_seed,
            device=gradient_batch.device,
            dtype=gradient_batch.dtype,
        )
    global_sketch = arc_topk_local_sketch(delta_batch, projection)
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
        work.wait()
        global_sketch.div_(world_size)

    k = math.ceil(ratio * rows)
    with record_function("arc/topk"):
        indices = arc_topk_support(global_sketch, k)
    with record_function("arc/selected_values"):
        local_selected = gather_rows(delta_batch, indices)
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
        work.wait()
        averaged_selected.div_(world_size)

    local_compressed = scatter_rows(local_selected, indices, rows)
    averaged_compressed = scatter_rows(averaged_selected, indices, rows)
    with record_function("arc/ef21m"):
        ef21m_apply_delta_(
            local_estimate_batch,
            global_estimate_batch,
            local_compressed,
            averaged_compressed,
        )

    torch._foreach_copy_(trackers, list(tracker_batch.unbind(0)))
    torch._foreach_copy_(local_estimates, list(local_estimate_batch.unbind(0)))
    torch._foreach_copy_(global_estimates, list(global_estimate_batch.unbind(0)))
    return global_estimates
