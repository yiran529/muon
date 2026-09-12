"""DDP communication hook for tensor-wise Rand-K and Top-K."""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence
import threading

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Parameter
from torch.profiler import record_function

from .collective_observer import observe_collective
from .sparse_k import (
    SparseKConfig,
    compensated_gradient,
    select_sparse_k_indices,
    update_ef14_residual_,
)

SPARSE_K_COMPRESSOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SparseKDDPParameterSpec:
    parameter: Parameter
    stable_name: str
    stable_id: int
    role: Literal["sparse_matrix", "dense_aux"]


@dataclass
class SparseKParameterState:
    spec: SparseKDDPParameterSpec
    residual: Optional[Tensor]
    last_support: Optional[Tensor] = None


@dataclass
class SparseKBucketContext:
    context_id: int
    bucket: dist.GradBucket
    buffer: Tensor
    gradients: tuple[Tensor, ...]
    parameters: tuple[Parameter, ...]
    parameter_states: tuple[Optional[SparseKParameterState], ...]
    step: int
    previous_tail: torch.futures.Future
    completion_future: torch.futures.Future
    entry_stream: Optional[torch.cuda.Stream]
    ready_event: Optional[torch.cuda.Event]
    retained: list[Any] = field(default_factory=list)


def _future_devices(device: torch.device) -> list[torch.device]:
    return [device] if device.type == "cuda" else []


def _completed_future(tensor: Tensor) -> torch.futures.Future:
    future = torch.futures.Future(devices=_future_devices(tensor.device))
    future.set_result(tensor)
    return future


def _future_tensor(value: Any) -> Tensor:
    return value[0] if isinstance(value, (tuple, list)) else value


class SparseKDDPState:
    """Stable per-parameter Sparse-K state and optimizer-step lifecycle."""

    def __init__(
        self,
        *,
        process_group: Optional[ProcessGroup],
        fingerprint: str,
        parameter_specs: Sequence[SparseKDDPParameterSpec],
        optimizer_parameters: Sequence[Parameter],
        config: SparseKConfig,
        find_unused_parameters: bool = False,
    ) -> None:
        if find_unused_parameters:
            raise ValueError("Sparse-K DDP hook requires find_unused_parameters=False")
        if not parameter_specs:
            raise ValueError("Sparse-K DDP hook requires at least one parameter spec")
        names = [spec.stable_name for spec in parameter_specs]
        stable_ids = [spec.stable_id for spec in parameter_specs]
        parameter_ids = [id(spec.parameter) for spec in parameter_specs]
        if len(names) != len(set(names)):
            raise ValueError("Sparse-K parameter stable names must be unique")
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("Sparse-K parameter stable IDs must be unique")
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("Sparse-K specs must reference unique parameters")
        for spec in parameter_specs:
            if spec.role not in ("sparse_matrix", "dense_aux"):
                raise ValueError(f"unsupported Sparse-K parameter role {spec.role!r}")
            if spec.role == "sparse_matrix" and spec.parameter.ndim != 2:
                raise ValueError("sparse_matrix parameters must be two-dimensional")
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if set(optimizer_ids) != set(parameter_ids) or len(optimizer_ids) != len(
            parameter_ids
        ):
            raise ValueError(
                "an optimizer parameter is absent from the model parameter table"
            )

        self.process_group = process_group
        self.fingerprint = fingerprint
        self.parameter_specs = tuple(parameter_specs)
        self.config = config
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.global_rank = dist.get_rank() if process_group is not None else 0
        self.group_ranks = (
            tuple(dist.get_process_group_ranks(process_group))
            if process_group is not None
            else (0,)
        )
        self._states = {
            id(spec.parameter): SparseKParameterState(
                spec,
                (
                    torch.zeros_like(spec.parameter)
                    if config.error_feedback == "ef14"
                    else None
                ),
            )
            for spec in self.parameter_specs
            if spec.role == "sparse_matrix"
        }
        self._specs_by_id = {id(spec.parameter): spec for spec in self.parameter_specs}
        self._all_parameter_ids = set(self._specs_by_id)
        device = self.parameter_specs[0].parameter.device
        sentinel = torch.empty(0, device=device)
        self.tail_future = _completed_future(sentinel)
        self.committed_step = 0
        self._active_step: Optional[int] = None
        self._finished_step = False
        self._seen_parameter_ids: set[int] = set()
        self._active_contexts: dict[int, SparseKBucketContext] = {}
        self._next_context_id = 0
        self._lock = threading.Lock()
        self._execution_streams: dict[torch.device, torch.cuda.Stream] = {}

    def execution_stream(self, device: torch.device) -> torch.cuda.Stream:
        stream = self._execution_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._execution_streams[device] = stream
        return stream

    def begin_step(self) -> int:
        if self._active_step is not None:
            raise RuntimeError(
                f"Sparse-K optimizer step {self._active_step} is already active"
            )
        if not self.tail_future.done():
            raise RuntimeError(
                "cannot begin a Sparse-K step while the previous tail is in flight"
            )
        self._active_step = self.committed_step + 1
        self._finished_step = False
        self._seen_parameter_ids.clear()
        return self._active_step

    def parameter_state(self, parameter: Parameter) -> SparseKParameterState:
        try:
            return self._states[id(parameter)]
        except KeyError as exc:
            if id(parameter) in self._specs_by_id:
                raise ValueError(
                    "dense auxiliary parameter has no Sparse-K state"
                ) from exc
            raise ValueError("parameter is outside the Sparse-K layout") from exc

    def note_bucket(self, bucket: dist.GradBucket) -> SparseKBucketContext:
        if self._active_step is None or self._finished_step:
            raise RuntimeError(
                "note_bucket requires an unfinished active Sparse-K step"
            )
        parameters = tuple(bucket.parameters())
        gradients = tuple(bucket.gradients())
        if len(parameters) != len(gradients):
            raise RuntimeError("DDP bucket parameter and gradient views disagree")
        for parameter in parameters:
            parameter_id = id(parameter)
            if parameter_id not in self._all_parameter_ids:
                raise RuntimeError(
                    "DDP bucket contains a parameter outside the Sparse-K layout"
                )
            if parameter_id in self._seen_parameter_ids:
                name = self._specs_by_id[parameter_id].stable_name
                raise RuntimeError(
                    f"Sparse-K parameter {name!r} appeared more than once"
                )
            self._seen_parameter_ids.add(parameter_id)
        buffer = bucket.buffer()
        entry_stream = None
        ready_event = None
        if buffer.device.type == "cuda":
            entry_stream = torch.cuda.current_stream(buffer.device)
            ready_event = torch.cuda.Event()
            ready_event.record(entry_stream)
        completion = torch.futures.Future(devices=_future_devices(buffer.device))
        context = SparseKBucketContext(
            self._next_context_id,
            bucket,
            buffer,
            gradients,
            parameters,
            tuple(self._states.get(id(parameter)) for parameter in parameters),
            self._active_step,
            self.tail_future,
            completion,
            entry_stream,
            ready_event,
        )
        self._next_context_id += 1
        with self._lock:
            self._active_contexts[context.context_id] = context
            self.tail_future = completion

        def release(_future: torch.futures.Future) -> None:
            with self._lock:
                self._active_contexts.pop(context.context_id, None)

        completion.add_done_callback(release)
        return context

    def finish_step(self) -> None:
        if self._active_step is None or self._finished_step:
            raise RuntimeError(
                "finish_step requires an unfinished active Sparse-K step"
            )
        if not self.tail_future.done():
            raise RuntimeError(
                "cannot finish Sparse-K step while communication is in flight"
            )
        self.tail_future.wait()
        missing = self._all_parameter_ids - self._seen_parameter_ids
        if missing:
            names = sorted(self._specs_by_id[item].stable_name for item in missing)
            raise RuntimeError(
                f"Sparse-K step is missing parameters: {', '.join(names)}"
            )
        self._finished_step = True

    def commit_step(self) -> None:
        if self._active_step is None:
            raise RuntimeError("cannot commit Sparse-K step with no active step")
        if not self._finished_step:
            raise RuntimeError("cannot commit Sparse-K step before finish_step")
        self.committed_step = self._active_step
        self._active_step = None
        self._finished_step = False

    def _ordered_parameter_table(self) -> list[dict[str, Any]]:
        return [
            {
                "stable_name": spec.stable_name,
                "stable_id": spec.stable_id,
                "shape": list(spec.parameter.shape),
                "dtype": str(spec.parameter.dtype).removeprefix("torch."),
                "role": spec.role,
            }
            for spec in self.parameter_specs
        ]

    def _require_committed_boundary(self) -> None:
        if self._active_step is not None or not self.tail_future.done():
            raise RuntimeError("Sparse-K checkpoint requires a committed step boundary")

    def checkpoint_metadata(self) -> dict[str, Any]:
        self._require_committed_boundary()
        return {
            "schema_version": SPARSE_K_COMPRESSOR_SCHEMA_VERSION,
            "dp_world_size": self.world_size,
            "group_ranks": list(self.group_ranks),
            "config_fingerprint": self.fingerprint,
            "config": asdict(self.config),
            "seed_scheme_version": self.config.seed_scheme_version,
            "committed_step": self.committed_step,
            "ordered_parameter_table": self._ordered_parameter_table(),
        }

    def validate_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        expected = self.checkpoint_metadata()
        for key in (
            "schema_version",
            "dp_world_size",
            "group_ranks",
            "config_fingerprint",
            "config",
            "seed_scheme_version",
            "ordered_parameter_table",
        ):
            if metadata.get(key) != expected[key]:
                raise ValueError(f"Sparse-K compressor checkpoint {key} mismatch")
        step = metadata.get("committed_step")
        if not isinstance(step, int) or step < 0:
            raise ValueError("Sparse-K compressor checkpoint committed step is invalid")

    def state_dict(self) -> dict[str, Any]:
        shared = self.checkpoint_metadata()
        local = {}
        if self.config.error_feedback == "ef14":
            local = {
                state.spec.stable_name: {"residual": state.residual}
                for state in self._states.values()
            }
        return {"shared": shared, f"rank_{self.global_rank}": local}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        try:
            shared = state_dict["shared"]
            local = state_dict[f"rank_{self.global_rank}"]
        except KeyError as exc:
            raise ValueError("Sparse-K compressor checkpoint is incomplete") from exc
        self.validate_checkpoint_metadata(shared)
        expected_names = (
            {state.spec.stable_name for state in self._states.values()}
            if self.config.error_feedback == "ef14"
            else set()
        )
        if set(local) != expected_names:
            raise ValueError("Sparse-K compressor checkpoint parameter state mismatch")
        pending_residuals: list[tuple[Tensor, Tensor]] = []
        for state in self._states.values():
            if state.residual is None:
                continue
            local_entry = local[state.spec.stable_name]
            if not isinstance(local_entry, dict) or set(local_entry) != {"residual"}:
                raise ValueError(
                    "Sparse-K compressor checkpoint residual entry schema mismatch"
                )
            loaded = local_entry["residual"]
            if (
                not isinstance(loaded, Tensor)
                or loaded.shape != state.residual.shape
                or loaded.dtype != state.residual.dtype
            ):
                raise ValueError(
                    "Sparse-K compressor checkpoint residual tensor schema mismatch"
                )
            pending_residuals.append((state.residual, loaded))
        for residual, loaded in pending_residuals:
            residual.copy_(loaded)
        self.committed_step = int(shared["committed_step"])


def _bridge_future(
    source: torch.futures.Future,
    destination: torch.futures.Future,
    transform: Callable[[Any], Tensor],
) -> None:
    def complete(completed: torch.futures.Future) -> None:
        try:
            destination.set_result(transform(completed.value()))
        except BaseException as exc:
            destination.set_exception(exc)

    source.add_done_callback(complete)


def _enqueue_bucket(
    state: SparseKDDPState,
    context: SparseKBucketContext,
    launch: Callable[[SparseKBucketContext], torch.futures.Future],
) -> torch.futures.Future:
    destination = context.completion_future

    def after_previous(previous: torch.futures.Future) -> None:
        try:
            previous.value()
            if context.buffer.device.type == "cuda":
                callback_stream = torch.cuda.current_stream(context.buffer.device)
                execution_stream = state.execution_stream(context.buffer.device)
                execution_stream.wait_stream(callback_stream)
                assert context.ready_event is not None
                execution_stream.wait_event(context.ready_event)
                with torch.cuda.stream(execution_stream):
                    launched = launch(context)
            else:
                launched = launch(context)
            if launched is not destination:
                _bridge_future(
                    launched, destination, lambda value: _future_tensor(value)
                )
        except BaseException as exc:
            destination.set_exception(exc)

    context.previous_tail.add_done_callback(after_previous)
    return destination


def _all_reduce(
    state: SparseKDDPState, tensor: Tensor, category: str
) -> torch.futures.Future:
    if state.process_group is None or state.world_size == 1:
        return _completed_future(tensor)
    observe_collective(category, "all_reduce", tensor)
    with record_function(category):
        return dist.all_reduce(
            tensor, op=dist.ReduceOp.SUM, group=state.process_group, async_op=True
        ).get_future()


def _all_gather_into_tensor(
    state: SparseKDDPState, output: Tensor, tensor: Tensor, category: str
) -> torch.futures.Future:
    if state.process_group is None or state.world_size == 1:
        output.copy_(tensor.reshape(-1))
        return _completed_future(output)
    observe_collective(category, "all_gather_into_tensor", tensor)
    with record_function(category):
        return dist.all_gather_into_tensor(
            output, tensor, group=state.process_group, async_op=True
        ).get_future()


def _run_on_execution_stream(
    state: SparseKDDPState,
    context: SparseKBucketContext,
    callback: Callable[[torch.futures.Future], None],
) -> Callable[[torch.futures.Future], None]:
    def run(completed: torch.futures.Future) -> None:
        if context.buffer.device.type != "cuda":
            callback(completed)
            return
        callback_stream = torch.cuda.current_stream(context.buffer.device)
        execution_stream = state.execution_stream(context.buffer.device)
        execution_stream.wait_stream(callback_stream)
        with torch.cuda.stream(execution_stream):
            callback(completed)

    return run


def _launch_full_support(
    state: SparseKDDPState, context: SparseKBucketContext
) -> torch.futures.Future:
    for gradient, parameter_state in zip(context.gradients, context.parameter_states):
        if parameter_state is None:
            continue
        if parameter_state.residual is not None:
            gradient.add_(parameter_state.residual.to(gradient.dtype))
            parameter_state.residual.zero_()
        parameter_state.last_support = torch.arange(
            gradient.numel(), device=gradient.device, dtype=torch.int64
        )
    source = _all_reduce(state, context.buffer, "sparse_k_hook/dense")
    final = context.completion_future

    def finish(value: Any) -> Tensor:
        buffer = _future_tensor(value)
        if state.world_size > 1:
            buffer.div_(state.world_size)
        if context.buffer.device.type == "cuda":
            torch.cuda.current_stream(context.buffer.device).wait_stream(
                state.execution_stream(context.buffer.device)
            )
        return context.buffer

    _bridge_future(source, final, finish)
    return final


def _launch_sparse(
    state: SparseKDDPState, context: SparseKBucketContext
) -> torch.futures.Future:
    dense_views: list[Tensor] = []
    entries: list[tuple[Tensor, SparseKParameterState, Tensor, Tensor, Tensor]] = []
    for gradient, parameter_state in zip(context.gradients, context.parameter_states):
        if parameter_state is None:
            dense_views.append(gradient)
            continue
        compensated = compensated_gradient(gradient, parameter_state.residual)
        indices = select_sparse_k_indices(
            compensated,
            config=state.config,
            step=context.step,
            stable_parameter_id=parameter_state.spec.stable_id,
        )
        local_values = compensated.reshape(-1)[indices].clone()
        if parameter_state.residual is not None:
            update_ef14_residual_(
                parameter_state.residual, compensated, indices, local_values
            )
        parameter_state.last_support = indices
        entries.append((gradient, parameter_state, compensated, indices, local_values))

    if not entries:
        return _launch_full_support(state, context)
    dense_buffer = (
        torch.cat([view.reshape(-1) for view in dense_views])
        if dense_views
        else context.buffer.new_empty(0)
    )
    values = torch.cat([entry[4] for entry in entries])
    indices = torch.cat([entry[3].to(torch.int32) for entry in entries])
    context.retained.extend([dense_buffer, entries, values, indices])
    final = context.completion_future

    def fail(exc: BaseException) -> None:
        if not final.done():
            final.set_exception(exc)

    def reconstruct_randk(value: Any) -> Tensor:
        averaged = _future_tensor(value)
        if state.world_size > 1:
            averaged.div_(state.world_size)
        for gradient, *_rest in entries:
            gradient.zero_()
        offset = 0
        for gradient, _parameter_state, _compensated, support, local_values in entries:
            count = local_values.numel()
            gradient.reshape(-1).scatter_(0, support, averaged[offset : offset + count])
            offset += count
        return context.buffer

    def reconstruct_topk(gathered_values: Tensor, gathered_indices: Tensor) -> Tensor:
        for gradient, *_rest in entries:
            gradient.zero_()
        counts = [entry[4].numel() for entry in entries]
        for rank in range(state.world_size):
            offset = 0
            for (
                gradient,
                _parameter_state,
                _compensated,
                _support,
                _local,
            ), count in zip(entries, counts):
                support = gathered_indices[rank, offset : offset + count].to(
                    torch.int64
                )
                rank_values = gathered_values[rank, offset : offset + count]
                gradient.reshape(-1).index_add_(0, support, rank_values)
                offset += count
        if state.world_size > 1:
            for gradient, *_rest in entries:
                gradient.div_(state.world_size)
        return context.buffer

    def launch_sparse_collective(_completed: torch.futures.Future) -> None:
        try:
            if state.config.method == "randk":
                values_future = _all_reduce(
                    state, values, "sparse_k_hook/selected_values"
                )
                _bridge_future(values_future, final, reconstruct_randk)
                return
            gathered_values = values.new_empty(state.world_size * values.numel())
            gathered_indices = indices.new_empty(state.world_size * indices.numel())
            context.retained.extend([gathered_values, gathered_indices])

            def after_values(completed: torch.futures.Future) -> None:
                try:
                    _future_tensor(completed.value())
                    indices_future = _all_gather_into_tensor(
                        state,
                        gathered_indices,
                        indices,
                        "sparse_k_hook/indices",
                    )

                    def after_indices(done: Any) -> Tensor:
                        _future_tensor(done)
                        return reconstruct_topk(
                            gathered_values.view(state.world_size, -1),
                            gathered_indices.view(state.world_size, -1),
                        )

                    _bridge_future(indices_future, final, after_indices)
                except BaseException as exc:
                    fail(exc)

            values_future = _all_gather_into_tensor(
                state,
                gathered_values,
                values,
                "sparse_k_hook/selected_values",
            )
            values_future.add_done_callback(
                _run_on_execution_stream(state, context, after_values)
            )
        except BaseException as exc:
            fail(exc)

    def finish_dense(completed: torch.futures.Future) -> None:
        try:
            averaged_dense = _future_tensor(completed.value())
            if state.world_size > 1:
                averaged_dense.div_(state.world_size)
            offset = 0
            for gradient in dense_views:
                gradient.copy_(
                    averaged_dense[offset : offset + gradient.numel()].view_as(gradient)
                )
                offset += gradient.numel()
            launch_sparse_collective(completed)
        except BaseException as exc:
            fail(exc)

    dense_future = _all_reduce(state, dense_buffer, "sparse_k_hook/dense")
    dense_future.add_done_callback(
        _run_on_execution_stream(state, context, finish_dense)
    )
    return final


def sparse_k_ddp_hook(
    state: SparseKDDPState, bucket: dist.GradBucket
) -> torch.futures.Future[Tensor]:
    context = state.note_bucket(bucket)
    sparse_step = (
        context.step > state.config.start_compress_step
        and context.step != 1
        and state.config.ratio < 1.0
    )
    return _enqueue_bucket(
        state,
        context,
        (
            (lambda current: _launch_sparse(state, current))
            if sparse_step
            else (lambda current: _launch_full_support(state, current))
        ),
    )
