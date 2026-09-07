"""Asynchronous DDP bucket communication state for ARC-TopK/EF21M."""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence
import math
import threading

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Parameter

from .arc_topk_sync import ArcTopKSyncConfig
from .arc_topk import (
    arc_topk_support,
    derive_arc_seed,
    finalize_arc_full_support_,
    finalize_arc_sparse_,
    gather_rows,
    make_gaussian_projection,
    prepare_arc_batch,
)
from .collective_observer import observe_collective


ARC_COMPRESSOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArcTopKDDPParameterSpec:
    parameter: Parameter
    stable_name: str
    stable_id: int
    role: Literal["arc_matrix", "dense_aux"]


@dataclass
class ArcParameterState:
    spec: ArcTopKDDPParameterSpec
    h_local: Tensor
    g_local: Tensor
    g_global: Tensor
    last_support: Optional[Tensor] = None


@dataclass
class BucketContext:
    context_id: int
    bucket: dist.GradBucket
    buffer: Tensor
    gradients: tuple[Tensor, ...]
    parameters: tuple[Parameter, ...]
    parameter_states: tuple[Optional[ArcParameterState], ...]
    step: int
    entry_stream: Optional[torch.cuda.Stream]
    bucket_ready_event: Optional[torch.cuda.Event]
    previous_tail: torch.futures.Future
    completion_future: torch.futures.Future
    retained: list[Any] = field(default_factory=list)


def _future_devices(device: torch.device) -> list[torch.device]:
    return [device] if device.type == "cuda" else []


def _completed_future(device: torch.device) -> torch.futures.Future:
    future = torch.futures.Future(devices=_future_devices(device))
    future.set_result(None)
    return future


class ArcTopKDDPState:
    """Stable per-parameter ARC state and explicit optimizer-step lifecycle."""

    def __init__(
        self,
        *,
        process_group: Optional[ProcessGroup],
        fingerprint: str,
        parameter_specs: Sequence[ArcTopKDDPParameterSpec],
        optimizer_parameters: Sequence[Parameter],
        config: ArcTopKSyncConfig,
        find_unused_parameters: bool = False,
    ) -> None:
        if find_unused_parameters:
            raise ValueError("ARC DDP hook requires find_unused_parameters=False")
        if not parameter_specs:
            if optimizer_parameters:
                raise ValueError(
                    "an optimizer parameter is absent from the model parameter table"
                )
            raise ValueError("ARC DDP hook requires at least one parameter spec")

        stable_names = [spec.stable_name for spec in parameter_specs]
        stable_ids = [spec.stable_id for spec in parameter_specs]
        parameter_ids = [id(spec.parameter) for spec in parameter_specs]
        if len(stable_names) != len(set(stable_names)):
            raise ValueError("ARC parameter stable names must be unique")
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("ARC parameter stable IDs must be unique")
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("ARC parameter specs must reference unique parameters")
        for spec in parameter_specs:
            if spec.role not in ("arc_matrix", "dense_aux"):
                raise ValueError(f"unsupported ARC parameter role {spec.role!r}")
            if spec.role == "arc_matrix" and spec.parameter.ndim != 2:
                raise ValueError("arc_matrix parameters must be two-dimensional")

        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        optimizer_id_set = set(optimizer_ids)
        spec_id_set = set(parameter_ids)
        unowned = [
            spec.stable_name
            for spec in parameter_specs
            if id(spec.parameter) not in optimizer_id_set
        ]
        if unowned:
            raise ValueError(
                f"model parameter {unowned[0]!r} is not owned by the optimizer"
            )
        if optimizer_id_set != spec_id_set or len(optimizer_ids) != len(parameter_ids):
            raise ValueError("an optimizer parameter is absent from the model parameter table")

        self.process_group = process_group
        self.fingerprint = fingerprint
        self.config = config
        self.parameter_specs = tuple(parameter_specs)
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.global_rank = dist.get_rank() if process_group is not None else 0
        self.group_ranks = (
            tuple(dist.get_process_group_ranks(process_group))
            if process_group is not None
            else (0,)
        )

        self._parameter_states = {
            id(spec.parameter): ArcParameterState(
                spec=spec,
                h_local=torch.zeros_like(spec.parameter),
                g_local=torch.zeros_like(spec.parameter),
                g_global=torch.zeros_like(spec.parameter),
            )
            for spec in self.parameter_specs
            if spec.role == "arc_matrix"
        }
        self._specs_by_parameter = {
            id(spec.parameter): spec for spec in self.parameter_specs
        }
        self._all_parameter_ids = set(self._specs_by_parameter)
        device = self.parameter_specs[0].parameter.device
        self.tail_future = _completed_future(device)
        self.committed_step = 0
        self._active_step: Optional[int] = None
        self._finished_step = False
        self._seen_parameter_ids: set[int] = set()
        self._active_contexts: dict[int, BucketContext] = {}
        self._next_context_id = 0
        self._context_lock = threading.Lock()
        self._execution_streams: dict[torch.device, torch.cuda.Stream] = {}

    def execution_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.type != "cuda":
            raise ValueError("ARC execution streams are only defined for CUDA devices")
        stream = self._execution_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._execution_streams[device] = stream
        return stream

    def begin_step(self) -> int:
        if self._active_step is not None:
            raise RuntimeError(f"ARC optimizer step {self._active_step} is already active")
        if not self.tail_future.done():
            raise RuntimeError("cannot begin an ARC step while the previous tail is in flight")
        self._active_step = self.committed_step + 1
        self._finished_step = False
        self._seen_parameter_ids.clear()
        return self._active_step

    def parameter_state(self, parameter: Parameter) -> ArcParameterState:
        try:
            return self._parameter_states[id(parameter)]
        except KeyError as exc:
            spec = self._specs_by_parameter.get(id(parameter))
            if spec is not None:
                raise ValueError(
                    f"dense auxiliary parameter {spec.stable_name!r} has no ARC state"
                ) from exc
            raise ValueError("parameter is not part of the frozen ARC layout") from exc

    def note_bucket(self, bucket: dist.GradBucket) -> BucketContext:
        if self._active_step is None or self._finished_step:
            raise RuntimeError("note_bucket requires an unfinished active ARC step")
        parameters = tuple(bucket.parameters())
        gradients = tuple(bucket.gradients())
        if len(parameters) != len(gradients):
            raise RuntimeError("DDP bucket parameter and gradient views disagree")
        for parameter in parameters:
            parameter_id = id(parameter)
            if parameter_id not in self._all_parameter_ids:
                raise RuntimeError("DDP bucket contains a parameter outside the ARC layout")
            if parameter_id in self._seen_parameter_ids:
                name = self._specs_by_parameter[parameter_id].stable_name
                raise RuntimeError(f"ARC parameter {name!r} appeared more than once")
            self._seen_parameter_ids.add(parameter_id)

        buffer = bucket.buffer()
        entry_stream = None
        bucket_ready_event = None
        if buffer.device.type == "cuda":
            entry_stream = torch.cuda.current_stream(buffer.device)
            bucket_ready_event = torch.cuda.Event()
            bucket_ready_event.record(entry_stream)

        previous_tail = self.tail_future
        completion_future = torch.futures.Future(
            devices=_future_devices(buffer.device)
        )
        context_id = self._next_context_id
        self._next_context_id += 1
        context = BucketContext(
            context_id=context_id,
            bucket=bucket,
            buffer=buffer,
            gradients=gradients,
            parameters=parameters,
            parameter_states=tuple(
                self._parameter_states.get(id(parameter)) for parameter in parameters
            ),
            step=self._active_step,
            entry_stream=entry_stream,
            bucket_ready_event=bucket_ready_event,
            previous_tail=previous_tail,
            completion_future=completion_future,
        )
        with self._context_lock:
            self._active_contexts[context_id] = context
            self.tail_future = completion_future

        def release(_future: torch.futures.Future) -> None:
            with self._context_lock:
                self._active_contexts.pop(context_id, None)

        completion_future.add_done_callback(release)
        return context

    def finish_step(self) -> None:
        if self._active_step is None or self._finished_step:
            raise RuntimeError("finish_step requires an unfinished active ARC step")
        missing_ids = self._all_parameter_ids - self._seen_parameter_ids
        if missing_ids:
            missing_names = sorted(
                self._specs_by_parameter[parameter_id].stable_name
                for parameter_id in missing_ids
            )
            raise RuntimeError(
                "ARC step has missing parameter coverage: " + ", ".join(missing_names)
            )
        if not self.tail_future.done():
            raise RuntimeError("ARC bucket tail is still in flight")
        self.tail_future.value()
        self._finished_step = True

    def commit_step(self) -> None:
        if self._active_step is None:
            raise RuntimeError("cannot commit with no active ARC step")
        if not self._finished_step:
            raise RuntimeError("cannot commit ARC step before finish_step")
        self.committed_step = self._active_step
        self._active_step = None
        self._finished_step = False

    def state_dict(self) -> dict:
        if self._active_step is not None or not self.tail_future.done():
            raise RuntimeError("ARC compressor checkpoint requires a committed step boundary")
        parameter_table = [
            {
                "stable_name": spec.stable_name,
                "stable_id": spec.stable_id,
                "shape": list(spec.parameter.shape),
                "dtype": str(spec.parameter.dtype).removeprefix("torch."),
                "role": spec.role,
            }
            for spec in self.parameter_specs
        ]
        arc_states = [
            state
            for spec in self.parameter_specs
            if (state := self._parameter_states.get(id(spec.parameter))) is not None
        ]
        return {
            "shared": {
                "schema_version": ARC_COMPRESSOR_SCHEMA_VERSION,
                "dp_world_size": self.world_size,
                "group_ranks": list(self.group_ranks),
                "config_fingerprint": self.fingerprint,
                "config": asdict(self.config),
                "seed_scheme_version": self.config.seed_scheme_version,
                "committed_step": self.committed_step,
                "ordered_parameter_table": parameter_table,
                "g_global": {
                    state.spec.stable_name: state.g_global for state in arc_states
                },
            },
            f"rank_{self.global_rank}": {
                state.spec.stable_name: {
                    "h_local": state.h_local,
                    "g_local": state.g_local,
                }
                for state in arc_states
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self._active_step is not None or not self.tail_future.done():
            raise RuntimeError("ARC compressor load requires a committed step boundary")
        shared = state_dict["shared"]
        if shared["schema_version"] != ARC_COMPRESSOR_SCHEMA_VERSION:
            raise ValueError("ARC compressor checkpoint schema version mismatch")
        if shared["config_fingerprint"] != self.fingerprint:
            raise ValueError("ARC compressor checkpoint fingerprint mismatch")
        rank_state = state_dict[f"rank_{self.global_rank}"]
        for parameter_state in self._parameter_states.values():
            name = parameter_state.spec.stable_name
            parameter_state.h_local.copy_(rank_state[name]["h_local"])
            parameter_state.g_local.copy_(rank_state[name]["g_local"])
            parameter_state.g_global.copy_(shared["g_global"][name])
        self.committed_step = int(shared["committed_step"])


def bridge_future(
    source: torch.futures.Future,
    destination: torch.futures.Future,
    transform: Callable[[Any], Tensor],
) -> None:
    """Complete ``destination`` with one transformed value or source exception."""

    def complete(completed: torch.futures.Future) -> None:
        try:
            destination.set_result(transform(completed.value()))
        except BaseException as exc:
            destination.set_exception(exc)

    source.add_done_callback(complete)


def enqueue_bucket_chain(
    state: ArcTopKDDPState,
    context: BucketContext,
    launch: Callable[[BucketContext], torch.futures.Future],
) -> torch.futures.Future:
    """Launch one bucket after the prior complete bucket chain without nesting."""

    destination = context.completion_future

    def after_previous(previous: torch.futures.Future) -> None:
        try:
            previous.value()
            if context.buffer.device.type == "cuda":
                device = context.buffer.device
                callback_stream = torch.cuda.current_stream(device)
                execution_stream = state.execution_stream(device)
                execution_stream.wait_stream(callback_stream)
                assert context.bucket_ready_event is not None
                execution_stream.wait_event(context.bucket_ready_event)
                with torch.cuda.stream(execution_stream):
                    launched = launch(context)
            else:
                launched = launch(context)
            if launched is not destination:
                bridge_future(launched, destination, lambda value: value)
        except BaseException as exc:
            destination.set_exception(exc)

    context.previous_tail.add_done_callback(after_previous)
    return destination


def _future_tensor(value: Any) -> Tensor:
    return value[0] if isinstance(value, (tuple, list)) else value


def _launch_full_support_bucket(
    state: ArcTopKDDPState,
    context: BucketContext,
) -> torch.futures.Future:
    prepared_views = []
    for gradient, parameter_state in zip(
        context.gradients,
        context.parameter_states,
    ):
        if parameter_state is None:
            continue
        prepared = prepare_arc_batch(
            gradient.unsqueeze(0),
            parameter_state.h_local.unsqueeze(0),
            parameter_state.g_local.unsqueeze(0),
            parameter_state.g_global.unsqueeze(0),
            config=state.config,
            step=context.step,
            projection_batch=None,
        )
        gradient.copy_(parameter_state.h_local)
        prepared_views.append((prepared, gradient, parameter_state))
    context.retained.extend(prepared_views)

    if state.process_group is not None and state.world_size > 1:
        observe_collective("arc_hook/dense", "all_reduce", context.buffer)
        source = dist.all_reduce(
            context.buffer,
            op=dist.ReduceOp.SUM,
            group=state.process_group,
            async_op=True,
        ).get_future()
    else:
        source = torch.futures.Future(
            devices=_future_devices(context.buffer.device)
        )
        source.set_result(context.buffer)

    finalized = context.completion_future

    def finalize(value: Any) -> Tensor:
        buffer = _future_tensor(value)
        if state.world_size > 1:
            buffer.div_(state.world_size)
        for prepared, averaged_gradient, parameter_state in prepared_views:
            finalize_arc_full_support_(
                prepared,
                averaged_gradient.unsqueeze(0),
            )
            parameter_state.last_support = torch.arange(
                parameter_state.h_local.shape[0],
                device=parameter_state.h_local.device,
            )
        return context.buffer

    bridge_future(source, finalized, finalize)
    return finalized


def _all_reduce_future(
    state: ArcTopKDDPState,
    tensor: Tensor,
    category: str,
) -> torch.futures.Future:
    if state.process_group is not None and state.world_size > 1:
        observe_collective(category, "all_reduce", tensor)
        return dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            group=state.process_group,
            async_op=True,
        ).get_future()
    future = torch.futures.Future(devices=_future_devices(tensor.device))
    future.set_result(tensor)
    return future


def _on_bucket_execution_stream(
    state: ArcTopKDDPState,
    context: BucketContext,
    callback: Callable[[torch.futures.Future], None],
) -> Callable[[torch.futures.Future], None]:
    def run(completed: torch.futures.Future) -> None:
        if context.buffer.device.type != "cuda":
            callback(completed)
            return
        device = context.buffer.device
        callback_stream = torch.cuda.current_stream(device)
        execution_stream = state.execution_stream(device)
        execution_stream.wait_stream(callback_stream)
        with torch.cuda.stream(execution_stream):
            callback(completed)

    return run


def _on_bucket_execution_stream_result(
    state: ArcTopKDDPState,
    context: BucketContext,
    callback: Callable[[torch.futures.Future], Tensor],
) -> Callable[[torch.futures.Future], Tensor]:
    """Run a result-producing callback on the bucket stream and export its event."""

    def run(completed: torch.futures.Future) -> Tensor:
        if context.buffer.device.type != "cuda":
            return callback(completed)
        device = context.buffer.device
        callback_stream = torch.cuda.current_stream(device)
        execution_stream = state.execution_stream(device)
        execution_stream.wait_stream(callback_stream)
        with torch.cuda.stream(execution_stream):
            result = callback(completed)
        callback_stream.wait_stream(execution_stream)
        return result

    return run


def _launch_sparse_bucket(
    state: ArcTopKDDPState,
    context: BucketContext,
) -> torch.futures.Future:
    dense_views = [
        gradient
        for gradient, parameter_state in zip(
            context.gradients,
            context.parameter_states,
        )
        if parameter_state is None
    ]
    dense_buffer = (
        torch.cat([gradient.reshape(-1) for gradient in dense_views])
        if dense_views
        else context.buffer.new_empty(0)
    )
    prepared_entries = []
    sketch_parts = []
    for gradient, parameter_state in zip(
        context.gradients,
        context.parameter_states,
    ):
        if parameter_state is None:
            continue
        rows, columns = parameter_state.h_local.shape
        seed = derive_arc_seed(
            base_seed=state.config.seed,
            step=context.step,
            stable_task_id=parameter_state.spec.stable_id,
        )
        projection = make_gaussian_projection(
            1,
            columns,
            state.config.projection_rank,
            seed=seed,
            device=gradient.device,
            dtype=gradient.dtype,
        )
        prepared = prepare_arc_batch(
            gradient.unsqueeze(0),
            parameter_state.h_local.unsqueeze(0),
            parameter_state.g_local.unsqueeze(0),
            parameter_state.g_global.unsqueeze(0),
            config=state.config,
            step=context.step,
            projection_batch=projection,
        )
        assert prepared.local_sketch_batch is not None
        prepared_entries.append((prepared, gradient, parameter_state, rows, columns))
        sketch_parts.append(prepared.local_sketch_batch.reshape(-1))

    sketch_buffer = torch.cat(sketch_parts)
    final = context.completion_future
    context.retained.extend(
        [dense_buffer, sketch_buffer, prepared_entries, sketch_parts]
    )

    def fail(exc: BaseException) -> None:
        if not final.done():
            final.set_exception(exc)

    def finish_selected(completed: torch.futures.Future) -> Tensor:
        averaged_buffer = _future_tensor(completed.value())
        if state.world_size > 1:
            averaged_buffer.div_(state.world_size)
        offset = 0
        for prepared, gradient, parameter_state, rows, columns in prepared_entries:
            assert prepared.delta_batch is not None
            support = parameter_state.last_support
            assert support is not None
            count = support.numel() * columns
            averaged_selected = averaged_buffer[offset : offset + count].view(
                1, support.numel(), columns
            )
            local_selected = gather_rows(
                prepared.delta_batch,
                support.unsqueeze(0),
            )
            finalize_arc_sparse_(
                prepared,
                support.unsqueeze(0),
                local_selected,
                averaged_selected,
            )
            gradient.copy_(parameter_state.g_global)
            offset += count
        return context.buffer

    def finish_sketch(completed: torch.futures.Future) -> None:
        try:
            averaged_sketch = _future_tensor(completed.value())
            if state.world_size > 1:
                averaged_sketch.div_(state.world_size)
            sketch_offset = 0
            local_selected_parts = []
            for prepared, _gradient, parameter_state, rows, _columns in prepared_entries:
                sketch_count = rows * state.config.projection_rank
                parameter_sketch = averaged_sketch[
                    sketch_offset : sketch_offset + sketch_count
                ].view(1, rows, state.config.projection_rank)
                support = arc_topk_support(
                    parameter_sketch,
                    math.ceil(state.config.ratio * rows),
                ).squeeze(0)
                parameter_state.last_support = support
                assert prepared.delta_batch is not None
                local_selected_parts.append(
                    gather_rows(prepared.delta_batch, support.unsqueeze(0)).reshape(-1)
                )
                sketch_offset += sketch_count
            local_selected_buffer = torch.cat(local_selected_parts)
            averaged_selected_buffer = local_selected_buffer.clone()
            context.retained.extend(
                [
                    local_selected_parts,
                    local_selected_buffer,
                    averaged_selected_buffer,
                ]
            )
            selected_future = _all_reduce_future(
                state,
                averaged_selected_buffer,
                "arc_hook/selected_values",
            )
            selected_completion = selected_future.then(
                _on_bucket_execution_stream_result(
                    state,
                    context,
                    finish_selected,
                )
            )
            bridge_future(selected_completion, final, lambda value: value)
        except BaseException as exc:
            fail(exc)

    def finish_dense(completed: torch.futures.Future) -> None:
        try:
            if dense_views:
                averaged_dense = _future_tensor(completed.value())
                if state.world_size > 1:
                    averaged_dense.div_(state.world_size)
                offset = 0
                for gradient in dense_views:
                    gradient.copy_(
                        averaged_dense[offset : offset + gradient.numel()].view_as(
                            gradient
                        )
                    )
                    offset += gradient.numel()
            sketch_future = _all_reduce_future(
                state,
                sketch_buffer,
                "arc_hook/sketch",
            )
            sketch_future.add_done_callback(
                _on_bucket_execution_stream(
                    state,
                    context,
                    finish_sketch,
                )
            )
        except BaseException as exc:
            fail(exc)

    if dense_views:
        dense_future = _all_reduce_future(
            state,
            dense_buffer,
            "arc_hook/dense",
        )
    else:
        dense_future = torch.futures.Future(
            devices=_future_devices(context.buffer.device)
        )
        dense_future.set_result(dense_buffer)
    dense_future.add_done_callback(
        _on_bucket_execution_stream(
            state,
            context,
            finish_dense,
        )
    )
    return final


def arc_topk_ddp_hook(
    state: ArcTopKDDPState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[Tensor]:
    """Synchronize one DDP bucket through the globally sequenced ARC chain."""

    context = state.note_bucket(bucket)
    is_sparse_step = (
        context.step > state.config.start_compress_step
        and context.step != 1
        and state.config.ratio < 1.0
    )
    if is_sparse_step:
        return enqueue_bucket_chain(
            state,
            context,
            lambda current: _launch_sparse_bucket(state, current),
        )
    return enqueue_bucket_chain(
        state,
        context,
        lambda current: _launch_full_support_bucket(state, current),
    )
