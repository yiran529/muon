"""Stable state and lifecycle for the GreedyLore DDP communication hook."""

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Parameter

from .greedy_lore import (
    GreedyLoreConfig,
    MatrixOrientation,
    compressed_phase,
    matrix_orientation,
)


class GreedyLoreStateError(RuntimeError):
    """Raised when the GreedyLore compressor lifecycle is used incorrectly."""


class GreedyLoreReplicatedStateMismatch(RuntimeError):
    """Raised when replicated GreedyLore bases or supports disagree."""


@dataclass(frozen=True)
class GreedyLoreDDPParameterSpec:
    parameter: Parameter
    stable_name: str
    stable_id: int
    role: Literal["matrix", "dense_aux"]


@dataclass
class GreedyLoreParameterState:
    spec: GreedyLoreDDPParameterSpec
    orientation: MatrixOrientation
    error: Tensor
    basis: Tensor
    last_support: Tensor


@dataclass
class BucketContext:
    context_id: int
    bucket: dist.GradBucket
    buffer: Tensor
    gradients: tuple[Tensor, ...]
    parameters: tuple[Parameter, ...]
    parameter_states: tuple[GreedyLoreParameterState | None, ...]
    step: int
    phase: int | None
    entry_stream: torch.cuda.Stream | None
    bucket_ready_event: torch.cuda.Event | None
    previous_tail: torch.futures.Future
    completion_future: torch.futures.Future
    retained: list[Any] = field(default_factory=list)


def _future_devices(device: torch.device) -> list[torch.device]:
    return [device] if device.type == "cuda" else []


def _completed_future(device: torch.device) -> torch.futures.Future:
    future = torch.futures.Future(devices=_future_devices(device))
    future.set_result(None)
    return future


class GreedyLoreDDPState:
    """Frozen per-parameter GreedyLore state and explicit step lifecycle."""

    def __init__(
        self,
        *,
        process_group: ProcessGroup | None,
        fingerprint: str,
        parameter_specs: Sequence[GreedyLoreDDPParameterSpec],
        optimizer_parameters: Sequence[Parameter],
        config: GreedyLoreConfig,
        find_unused_parameters: bool = False,
    ) -> None:
        if find_unused_parameters:
            raise ValueError(
                "GreedyLore DDP hook requires find_unused_parameters=False"
            )
        if not parameter_specs:
            if optimizer_parameters:
                raise ValueError(
                    "an optimizer parameter is absent from the model parameter table"
                )
            raise ValueError("GreedyLore DDP hook requires at least one parameter spec")

        stable_names = [spec.stable_name for spec in parameter_specs]
        stable_ids = [spec.stable_id for spec in parameter_specs]
        parameter_ids = [id(spec.parameter) for spec in parameter_specs]
        if len(stable_names) != len(set(stable_names)):
            raise ValueError("GreedyLore parameter stable names must be unique")
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("GreedyLore parameter stable IDs must be unique")
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError(
                "GreedyLore parameter specs must reference unique parameters"
            )

        for spec in parameter_specs:
            if spec.role not in ("matrix", "dense_aux"):
                raise ValueError(f"unsupported GreedyLore parameter role {spec.role!r}")
            if spec.role != "matrix":
                continue
            if spec.parameter.ndim != 2:
                raise ValueError("GreedyLore matrix parameters must be two-dimensional")
            if spec.parameter.dtype != torch.float32:
                raise ValueError("GreedyLore matrix parameters must be FP32")
            if config.rank > min(spec.parameter.shape):
                raise ValueError(
                    "GreedyLore rank must not exceed the smaller matrix dimension"
                )

        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError("GreedyLore optimizer parameters must be unique")
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
            raise ValueError(
                "an optimizer parameter is absent from the model parameter table"
            )

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
        self._specs_by_parameter = {
            id(spec.parameter): spec for spec in self.parameter_specs
        }
        self._all_parameter_ids = frozenset(self._specs_by_parameter)
        self._parameter_states = {}
        for spec in self.parameter_specs:
            if spec.role != "matrix":
                continue
            orientation = matrix_orientation(spec.parameter.shape)
            rows, columns = orientation.compressed_shape
            self._parameter_states[id(spec.parameter)] = GreedyLoreParameterState(
                spec=spec,
                orientation=orientation,
                error=torch.zeros(
                    (rows, columns),
                    dtype=torch.float32,
                    device=spec.parameter.device,
                ),
                basis=torch.eye(
                    rows,
                    dtype=torch.float32,
                    device=spec.parameter.device,
                ),
                last_support=torch.arange(
                    config.rank,
                    dtype=torch.int64,
                    device=spec.parameter.device,
                ),
            )

        device = self.parameter_specs[0].parameter.device
        self.tail_future = _completed_future(device)
        self.committed_step = 0
        self._active_step: int | None = None
        self._finished_step = False
        self._seen_parameter_ids: set[int] = set()
        self._active_contexts: dict[int, BucketContext] = {}
        self._next_context_id = 0
        self._context_lock = threading.Lock()
        self._execution_streams: dict[torch.device, torch.cuda.Stream] = {}

    def _require_committed_boundary(self, operation: str) -> None:
        if self._active_step is not None or not self.tail_future.done():
            raise GreedyLoreStateError(
                f"GreedyLore compressor {operation} requires a committed step boundary"
            )

    def execution_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.type != "cuda":
            raise ValueError(
                "GreedyLore execution streams are only defined for CUDA devices"
            )
        stream = self._execution_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._execution_streams[device] = stream
        return stream

    def begin_step(self) -> int:
        if self._active_step is not None:
            raise GreedyLoreStateError(
                f"GreedyLore optimizer step {self._active_step} is already active"
            )
        if not self.tail_future.done():
            raise GreedyLoreStateError(
                "cannot begin a GreedyLore step while the previous tail is in flight"
            )
        self._active_step = self.committed_step + 1
        self._finished_step = False
        self._seen_parameter_ids.clear()
        return self._active_step

    def parameter_state(self, parameter: Parameter) -> GreedyLoreParameterState:
        try:
            return self._parameter_states[id(parameter)]
        except KeyError as exc:
            spec = self._specs_by_parameter.get(id(parameter))
            if spec is not None:
                raise ValueError(
                    f"dense auxiliary parameter {spec.stable_name!r} "
                    "has no GreedyLore matrix state"
                ) from exc
            raise ValueError(
                "parameter is not part of the frozen GreedyLore layout"
            ) from exc

    def note_bucket(self, bucket: dist.GradBucket) -> BucketContext:
        if self._active_step is None or self._finished_step:
            raise GreedyLoreStateError(
                "note_bucket requires an unfinished active GreedyLore step"
            )
        parameters = tuple(bucket.parameters())
        gradients = tuple(bucket.gradients())
        if len(parameters) != len(gradients):
            raise GreedyLoreStateError(
                "DDP bucket parameter and gradient views disagree"
            )

        bucket_ids = [id(parameter) for parameter in parameters]
        for parameter_id in bucket_ids:
            if parameter_id not in self._all_parameter_ids:
                raise GreedyLoreStateError(
                    "DDP bucket contains a parameter outside the GreedyLore layout"
                )
        bucket_id_set = set(bucket_ids)
        duplicates = bucket_id_set & self._seen_parameter_ids
        if len(bucket_id_set) != len(bucket_ids):
            locally_seen = set()
            for parameter_id in bucket_ids:
                if parameter_id in locally_seen:
                    duplicates.add(parameter_id)
                locally_seen.add(parameter_id)
        if duplicates:
            parameter_id = next(
                identifier for identifier in bucket_ids if identifier in duplicates
            )
            name = self._specs_by_parameter[parameter_id].stable_name
            raise GreedyLoreStateError(
                f"GreedyLore parameter {name!r} appeared more than once"
            )
        self._seen_parameter_ids.update(bucket_ids)

        buffer = bucket.buffer()
        entry_stream = None
        bucket_ready_event = None
        if buffer.device.type == "cuda":
            entry_stream = torch.cuda.current_stream(buffer.device)
            bucket_ready_event = torch.cuda.Event()
            bucket_ready_event.record(entry_stream)

        previous_tail = self.tail_future
        completion_future = torch.futures.Future(devices=_future_devices(buffer.device))
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
            phase=compressed_phase(self._active_step, self.config.start_compress_step),
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
            raise GreedyLoreStateError(
                "finish_step requires an unfinished active GreedyLore step"
            )
        missing_ids = self._all_parameter_ids - self._seen_parameter_ids
        if missing_ids:
            missing_names = sorted(
                self._specs_by_parameter[parameter_id].stable_name
                for parameter_id in missing_ids
            )
            raise GreedyLoreStateError(
                "GreedyLore step has missing parameter coverage: "
                + ", ".join(missing_names)
            )
        if not self.tail_future.done():
            raise GreedyLoreStateError("GreedyLore bucket tail is still in flight")
        self.tail_future.value()
        self._finished_step = True

    def commit_step(self) -> None:
        if self._active_step is None:
            raise GreedyLoreStateError("cannot commit with no active GreedyLore step")
        if not self._finished_step:
            raise GreedyLoreStateError(
                "cannot commit GreedyLore step before finish_step"
            )
        self.committed_step = self._active_step
        self._active_step = None
        self._finished_step = False

    def validate_replicated_basis_across_ranks(
        self, *, atol: float = 1e-6, rtol: float = 1e-5
    ) -> None:
        """Validate replicated basis/support tensors at a committed boundary."""

        self._require_committed_boundary("basis validation")
        if self.process_group is None or self.world_size <= 1:
            return
        ordered_states = sorted(
            self._parameter_states.values(),
            key=lambda parameter_state: parameter_state.spec.stable_name,
        )
        for parameter_state in ordered_states:
            gathered_supports = [
                torch.empty_like(parameter_state.last_support)
                for _ in range(self.world_size)
            ]
            dist.all_gather(
                gathered_supports,
                parameter_state.last_support,
                group=self.process_group,
            )
            gathered_bases = [
                torch.empty_like(parameter_state.basis) for _ in range(self.world_size)
            ]
            dist.all_gather(
                gathered_bases,
                parameter_state.basis,
                group=self.process_group,
            )
            if any(
                not torch.equal(gathered_supports[0], support)
                for support in gathered_supports[1:]
            ):
                raise GreedyLoreReplicatedStateMismatch(
                    "GreedyLore replicated support differs across ranks for "
                    f"{parameter_state.spec.stable_name!r}"
                )
            if any(
                not torch.allclose(gathered_bases[0], basis, atol=atol, rtol=rtol)
                for basis in gathered_bases[1:]
            ):
                raise GreedyLoreReplicatedStateMismatch(
                    "GreedyLore replicated basis differs across ranks for "
                    f"{parameter_state.spec.stable_name!r}"
                )

    def state_dict(self) -> dict[str, Any]:
        """Expose preallocated state at a safe compressor boundary."""

        self._require_committed_boundary("checkpoint")
        matrix_states = sorted(
            self._parameter_states.values(),
            key=lambda parameter_state: parameter_state.spec.stable_name,
        )
        return {
            "shared": {"committed_step": self.committed_step},
            f"rank_{self.global_rank}": {
                state.spec.stable_name: {
                    "error": state.error,
                    "basis": state.basis,
                    "last_support": state.last_support,
                }
                for state in matrix_states
            },
        }


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


def _future_tensor(value: Any) -> Tensor:
    return value[0] if isinstance(value, (tuple, list)) else value


def enqueue_bucket_chain(
    state: GreedyLoreDDPState,
    context: BucketContext,
    launch: Callable[[BucketContext], torch.futures.Future],
) -> torch.futures.Future:
    """Launch one bucket after the prior complete bucket chain without nesting."""

    destination = context.completion_future

    def export_completion(value: Any) -> Tensor:
        if context.buffer.device.type == "cuda":
            device = context.buffer.device
            callback_stream = torch.cuda.current_stream(device)
            execution_stream = state.execution_stream(device)
            callback_stream.wait_stream(execution_stream)
        return _future_tensor(value)

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
                bridge_future(launched, destination, export_completion)
        except BaseException as exc:
            destination.set_exception(exc)

    context.previous_tail.add_done_callback(after_previous)
    return destination
