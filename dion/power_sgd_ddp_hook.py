"""Stable state and lifecycle for the PowerSGD DDP communication hook."""

import json
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Sequence
from urllib.parse import quote

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Parameter
from torch.profiler import record_function

from .collective_observer import observe_collective
from .power_sgd import (
    _orthogonalize_owned,
    PowerSGDConfig,
    compressed_phase,
    compute_left_factor,
    compute_right_factor,
    corrected_gradient,
    derive_power_sgd_seed,
    make_random_factor,
    orthogonalize,
    reconstruct,
    should_compress,
)

POWER_SGD_COMPRESSOR_SCHEMA_VERSION = 1


class PowerSGDStateError(RuntimeError):
    """Raised when the PowerSGD compressor lifecycle is used incorrectly."""


@dataclass(frozen=True)
class PowerSGDDDPParameterSpec:
    parameter: Parameter
    stable_name: str
    stable_id: int
    role: Literal["matrix", "dense_aux"]


@dataclass
class PowerSGDParameterState:
    spec: PowerSGDDDPParameterSpec
    error: Tensor
    q_memory: Tensor
    q_initialized: Tensor


@dataclass
class BucketContext:
    context_id: int
    bucket_index: int
    bucket: dist.GradBucket
    buffer: Tensor
    gradients: tuple[Tensor, ...]
    parameters: tuple[Parameter, ...]
    parameter_states: tuple[PowerSGDParameterState | None, ...]
    step: int
    phase: int | None
    entry_stream: torch.cuda.Stream | None
    bucket_ready_event: torch.cuda.Event | None
    previous_tail: torch.futures.Future
    previous_collective_tail: torch.futures.Future
    collective_completion_future: torch.futures.Future
    completion_future: torch.futures.Future
    prepare_done: torch.cuda.Event | None = None
    retained: list[Any] = field(default_factory=list)


def _future_devices(device: torch.device) -> list[torch.device]:
    return [device] if device.type == "cuda" else []


def _completed_future(device: torch.device) -> torch.futures.Future:
    future = torch.futures.Future(devices=_future_devices(device))
    future.set_result(None)
    return future


def _join_completion_futures(
    previous: torch.futures.Future,
    current: torch.futures.Future,
    device: torch.device,
) -> torch.futures.Future:
    """Join host completion, CUDA visibility, and errors from both inputs."""
    joined = torch.futures.Future(devices=_future_devices(device))

    def finish(completed: torch.futures.Future) -> None:
        try:
            result = None
            for future in completed.value():
                # collect_all guarantees host completion. wait() is therefore
                # only a stream dependency export, not device synchronization.
                result = future.wait()
            # collect_all's child Future is not CUDA-aware. Explicitly export a
            # tensor through our device-aware Future to retain the joined event.
            joined.set_result(result)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                failure = RuntimeError(f"{type(exc).__name__}: {exc}")
                failure.__cause__ = exc
                exc = failure
            joined.set_exception(exc)

    torch.futures.collect_all([previous, current]).add_done_callback(finish)
    return joined


class PowerSGDDDPState:
    """Frozen per-parameter PowerSGD state and explicit step lifecycle."""

    def __init__(
        self,
        *,
        process_group: ProcessGroup | None,
        fingerprint: str,
        parameter_specs: Sequence[PowerSGDDDPParameterSpec],
        optimizer_parameters: Sequence[Parameter],
        config: PowerSGDConfig,
        find_unused_parameters: bool = False,
    ) -> None:
        if find_unused_parameters:
            raise ValueError("PowerSGD DDP hook requires find_unused_parameters=False")
        if not parameter_specs:
            if optimizer_parameters:
                raise ValueError(
                    "an optimizer parameter is absent from the model parameter table"
                )
            raise ValueError("PowerSGD DDP hook requires at least one parameter spec")

        stable_names = [spec.stable_name for spec in parameter_specs]
        stable_ids = [spec.stable_id for spec in parameter_specs]
        parameter_ids = [id(spec.parameter) for spec in parameter_specs]
        if len(stable_names) != len(set(stable_names)):
            raise ValueError("PowerSGD parameter stable names must be unique")
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("PowerSGD parameter stable IDs must be unique")
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError(
                "PowerSGD parameter specs must reference unique parameters"
            )

        for spec in parameter_specs:
            if spec.role not in ("matrix", "dense_aux"):
                raise ValueError(f"unsupported PowerSGD parameter role {spec.role!r}")
            if spec.parameter.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("PowerSGD parameters must be FP32 or BF16")
            if spec.role == "matrix":
                if spec.parameter.ndim != 2:
                    raise ValueError(
                        "PowerSGD matrix parameters must be two-dimensional"
                    )
                if min(spec.parameter.shape) <= 0:
                    raise ValueError(
                        "PowerSGD matrix parameters must have positive dimensions"
                    )

        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError("PowerSGD optimizer parameters must be unique")
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
        self._parameter_states: dict[int, PowerSGDParameterState] = {}
        for spec in self.parameter_specs:
            parameter = spec.parameter
            if spec.role != "matrix" or not should_compress(
                parameter.shape[0],
                parameter.shape[1],
                config.rank,
                config.min_compression_rate,
            ):
                continue
            effective_rank = min(config.rank, parameter.shape[0], parameter.shape[1])
            self._parameter_states[id(parameter)] = PowerSGDParameterState(
                spec=spec,
                error=torch.zeros_like(parameter),
                q_memory=torch.zeros(
                    (parameter.shape[1], effective_rank),
                    dtype=parameter.dtype,
                    device=parameter.device,
                ),
                q_initialized=torch.zeros(
                    (), dtype=torch.bool, device=parameter.device
                ),
            )

        device = self.parameter_specs[0].parameter.device
        self.tail_future = _completed_future(device)
        self.collective_tail = _completed_future(device)
        self.committed_step = 0
        self._active_step: int | None = None
        self._finished_step = False
        self._seen_parameter_ids: set[int] = set()
        self._active_contexts: dict[int, BucketContext] = {}
        self._next_context_id = 0
        self._context_lock = threading.Lock()
        self._execution_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._preparation_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._reconstruction_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._validated_checkpoint_committed_step: int | None = None

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

    def _rank_local_tensor_schema(self) -> dict[str, dict[str, Any]]:
        states = sorted(
            self._parameter_states.values(),
            key=lambda parameter_state: parameter_state.spec.stable_name,
        )
        return {
            state.spec.stable_name: {
                "error": {
                    "shape": list(state.error.shape),
                    "dtype": str(state.error.dtype).removeprefix("torch."),
                },
                "q_memory": {
                    "shape": list(state.q_memory.shape),
                    "dtype": str(state.q_memory.dtype).removeprefix("torch."),
                },
                "q_initialized": {
                    "shape": list(state.q_initialized.shape),
                    "dtype": str(state.q_initialized.dtype).removeprefix("torch."),
                },
            }
            for state in states
        }

    def _tensor_schema(self) -> dict[str, Any]:
        rank_local = self._rank_local_tensor_schema()
        return {
            "shared": {
                "schema_version": {"shape": [], "dtype": "int64"},
                "committed_step": {"shape": [], "dtype": "int64"},
            },
            **{f"rank_{rank}": rank_local for rank in self.group_ranks},
        }

    def parameter_state(self, parameter: Parameter) -> PowerSGDParameterState:
        try:
            return self._parameter_states[id(parameter)]
        except KeyError as exc:
            spec = self._specs_by_parameter.get(id(parameter))
            if spec is None:
                raise ValueError(
                    "parameter is not part of the frozen PowerSGD layout"
                ) from exc
            if spec.role == "dense_aux":
                raise ValueError(
                    f"dense auxiliary parameter {spec.stable_name!r} "
                    "has no PowerSGD matrix state"
                ) from exc
            raise ValueError(
                f"matrix parameter {spec.stable_name!r} uses dense fallback "
                "and has no PowerSGD state"
            ) from exc

    def begin_step(self) -> int:
        if self._active_step is not None:
            raise PowerSGDStateError(
                f"PowerSGD optimizer step {self._active_step} is already active"
            )
        if not self.tail_future.done():
            raise PowerSGDStateError(
                "cannot begin a PowerSGD step while the previous tail is in flight"
            )
        self._active_step = self.committed_step + 1
        self._finished_step = False
        self._seen_parameter_ids.clear()
        return self._active_step

    def note_bucket(self, bucket: dist.GradBucket) -> BucketContext:
        if self._active_step is None or self._finished_step:
            raise PowerSGDStateError(
                "note_bucket requires an unfinished active PowerSGD step"
            )
        parameters = tuple(bucket.parameters())
        gradients = tuple(bucket.gradients())
        if len(parameters) != len(gradients):
            raise PowerSGDStateError("DDP bucket parameter and gradient views disagree")

        bucket_ids = [id(parameter) for parameter in parameters]
        for parameter_id in bucket_ids:
            if parameter_id not in self._all_parameter_ids:
                raise PowerSGDStateError(
                    "DDP bucket contains a parameter outside the PowerSGD layout"
                )
        duplicates = set(bucket_ids) & self._seen_parameter_ids
        locally_seen: set[int] = set()
        for parameter_id in bucket_ids:
            if parameter_id in locally_seen:
                duplicates.add(parameter_id)
            locally_seen.add(parameter_id)
        if duplicates:
            parameter_id = next(
                identifier for identifier in bucket_ids if identifier in duplicates
            )
            name = self._specs_by_parameter[parameter_id].stable_name
            raise PowerSGDStateError(
                f"PowerSGD parameter {name!r} appeared more than once"
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
        previous_collective_tail = self.collective_tail
        completion_future = torch.futures.Future(devices=_future_devices(buffer.device))
        is_compressed_bucket = compressed_phase(
            self._active_step, self.config.start_compress_step
        ) is not None and any(
            id(parameter) in self._parameter_states for parameter in parameters
        )
        collective_completion_future = (
            torch.futures.Future(devices=_future_devices(buffer.device))
            if is_compressed_bucket
            else completion_future
        )
        context_id = self._next_context_id
        self._next_context_id += 1
        context = BucketContext(
            context_id=context_id,
            bucket_index=(
                int(bucket.index()) if callable(getattr(bucket, "index", None)) else -1
            ),
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
            previous_collective_tail=previous_collective_tail,
            collective_completion_future=collective_completion_future,
            completion_future=completion_future,
        )
        with self._context_lock:
            self._active_contexts[context_id] = context
            self.tail_future = _join_completion_futures(
                previous_tail, completion_future, buffer.device
            )
            self.collective_tail = collective_completion_future

        def release(_future: torch.futures.Future) -> None:
            with self._context_lock:
                self._active_contexts.pop(context_id, None)

        completion_future.add_done_callback(release)
        return context

    def finish_step(self) -> None:
        if self._active_step is None or self._finished_step:
            raise PowerSGDStateError(
                "finish_step requires an unfinished active PowerSGD step"
            )
        missing_ids = self._all_parameter_ids - self._seen_parameter_ids
        if missing_ids:
            missing_names = sorted(
                self._specs_by_parameter[parameter_id].stable_name
                for parameter_id in missing_ids
            )
            raise PowerSGDStateError(
                "PowerSGD step has missing parameter coverage: "
                + ", ".join(missing_names)
            )
        if not self.tail_future.done():
            raise PowerSGDStateError("PowerSGD bucket tail is still in flight")
        # done() above preserves the nonblocking lifecycle check. The completed
        # Future wait exports every reconstruction to the caller's CUDA stream.
        self.tail_future.wait()
        self._finished_step = True

    def commit_step(self) -> None:
        if self._active_step is None:
            raise PowerSGDStateError("cannot commit with no active PowerSGD step")
        if not self._finished_step:
            raise PowerSGDStateError("cannot commit PowerSGD step before finish_step")
        self.committed_step = self._active_step
        self._active_step = None
        self._finished_step = False

    def _require_committed_boundary(self, operation: str) -> None:
        if self._active_step is not None or not self.tail_future.done():
            raise PowerSGDStateError(
                f"PowerSGD compressor {operation} requires a committed step boundary"
            )

    def checkpoint_metadata(self) -> dict[str, Any]:
        self._require_committed_boundary("checkpoint")
        return {
            "schema_version": POWER_SGD_COMPRESSOR_SCHEMA_VERSION,
            "dp_world_size": self.world_size,
            "group_ranks": list(self.group_ranks),
            "config_fingerprint": self.fingerprint,
            "config": asdict(self.config),
            "seed_scheme_version": self.config.seed_scheme_version,
            "committed_step": self.committed_step,
            "ordered_parameter_table": self._ordered_parameter_table(),
            "tensor_schema": self._tensor_schema(),
        }

    def validate_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        self._require_committed_boundary("load")
        if metadata.get("schema_version") != POWER_SGD_COMPRESSOR_SCHEMA_VERSION:
            raise ValueError("PowerSGD compressor checkpoint schema version mismatch")
        if metadata.get("dp_world_size") != self.world_size:
            raise ValueError("PowerSGD compressor checkpoint world size mismatch")
        if metadata.get("group_ranks") != list(self.group_ranks):
            raise ValueError("PowerSGD compressor checkpoint rank membership mismatch")
        if metadata.get("config_fingerprint") != self.fingerprint:
            raise ValueError("PowerSGD compressor checkpoint fingerprint mismatch")
        if metadata.get("seed_scheme_version") != self.config.seed_scheme_version:
            raise ValueError("PowerSGD compressor checkpoint seed scheme mismatch")
        if metadata.get("config") != asdict(self.config):
            raise ValueError("PowerSGD compressor checkpoint config mismatch")
        if metadata.get("ordered_parameter_table") != self._ordered_parameter_table():
            raise ValueError("PowerSGD compressor checkpoint parameter table mismatch")
        committed_step = metadata.get("committed_step")
        if not isinstance(committed_step, int) or isinstance(committed_step, bool):
            raise ValueError("PowerSGD compressor checkpoint committed step is invalid")
        if committed_step < 0:
            raise ValueError("PowerSGD compressor checkpoint committed step is invalid")
        self._validate_metadata_tensor_schema(metadata.get("tensor_schema"))
        self._validated_checkpoint_committed_step = committed_step

    def _validate_metadata_tensor_schema(self, tensor_schema: Any) -> None:
        expected = self._tensor_schema()
        if not isinstance(tensor_schema, dict):
            raise ValueError("PowerSGD compressor checkpoint tensor schema mismatch")
        missing_roots = sorted(set(expected) - set(tensor_schema))
        extra_roots = sorted(set(tensor_schema) - set(expected))
        if missing_roots or extra_roots:
            raise ValueError(
                "PowerSGD compressor checkpoint tensor schema mismatch; "
                f"missing={missing_roots}, extra={extra_roots}"
            )
        for root, expected_entry in expected.items():
            actual_entry = tensor_schema[root]
            if actual_entry == expected_entry:
                continue
            if root.startswith("rank_"):
                expected_names = set(expected_entry)
                actual_names = (
                    set(actual_entry) if isinstance(actual_entry, dict) else set()
                )
                missing_names = sorted(expected_names - actual_names)
                if missing_names:
                    raise ValueError(
                        "PowerSGD compressor checkpoint tensor schema "
                        f"is missing parameter {missing_names[0]!r}"
                    )
            raise ValueError("PowerSGD compressor checkpoint tensor schema mismatch")

    def state_dict(self) -> dict[str, Any]:
        self._require_committed_boundary("checkpoint")
        states = sorted(
            self._parameter_states.values(),
            key=lambda parameter_state: parameter_state.spec.stable_name,
        )
        device = self.parameter_specs[0].parameter.device
        return {
            "shared": {
                "schema_version": torch.tensor(
                    POWER_SGD_COMPRESSOR_SCHEMA_VERSION,
                    dtype=torch.int64,
                    device=device,
                ),
                "committed_step": torch.tensor(
                    self.committed_step,
                    dtype=torch.int64,
                    device=device,
                ),
            },
            f"rank_{self.global_rank}": {
                state.spec.stable_name: {
                    "error": state.error,
                    "q_memory": state.q_memory,
                    "q_initialized": state.q_initialized,
                }
                for state in states
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._require_committed_boundary("load")
        if not isinstance(state_dict, dict):
            raise ValueError("PowerSGD compressor checkpoint state must be a mapping")
        rank_key = f"rank_{self.global_rank}"
        expected_roots = {"shared", rank_key}
        if set(state_dict) != expected_roots:
            if "shared" not in state_dict:
                raise ValueError(
                    "PowerSGD compressor checkpoint is missing shared state"
                )
            if rank_key not in state_dict:
                raise ValueError(
                    f"PowerSGD compressor checkpoint is missing state for {rank_key}"
                )
            raise ValueError("PowerSGD compressor checkpoint state fields mismatch")

        shared = state_dict["shared"]
        checkpoint_step = self._validate_shared_state(shared)
        rank_state = state_dict[rank_key]
        states = sorted(
            self._parameter_states.values(),
            key=lambda parameter_state: parameter_state.spec.stable_name,
        )
        expected_names = {state.spec.stable_name for state in states}
        actual_names = set(rank_state) if isinstance(rank_state, dict) else set()
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                "PowerSGD compressor rank-local state mismatch; "
                f"missing={missing}, extra={extra}"
            )

        copy_plan: list[tuple[Tensor, Tensor]] = []
        expected_fields = {"error", "q_memory", "q_initialized"}
        for parameter_state in states:
            name = parameter_state.spec.stable_name
            local_entry = rank_state[name]
            if not isinstance(local_entry, dict) or set(local_entry) != expected_fields:
                raise ValueError(
                    "PowerSGD compressor state fields mismatch for "
                    f"parameter {name!r}"
                )
            for field_name in ("error", "q_memory", "q_initialized"):
                source = local_entry[field_name]
                destination = getattr(parameter_state, field_name)
                if (
                    not torch.is_tensor(source)
                    or source.shape != destination.shape
                    or source.dtype != destination.dtype
                ):
                    raise ValueError(
                        "PowerSGD compressor tensor schema mismatch for "
                        f"{name!r}/{field_name}"
                    )
                copy_plan.append((destination, source))

        if (
            self._validated_checkpoint_committed_step is not None
            and checkpoint_step != self._validated_checkpoint_committed_step
        ):
            raise ValueError(
                "PowerSGD compressor checkpoint committed step payload "
                "does not match metadata"
            )

        for destination, source in copy_plan:
            destination.copy_(source)
        self.committed_step = checkpoint_step
        self._validated_checkpoint_committed_step = None

    def _validate_shared_state(self, shared: Any) -> int:
        if not isinstance(shared, dict):
            raise ValueError("PowerSGD compressor checkpoint shared state mismatch")
        if set(shared) != {"schema_version", "committed_step"}:
            raise ValueError("PowerSGD compressor checkpoint shared state mismatch")
        for name, tensor in shared.items():
            if (
                not torch.is_tensor(tensor)
                or tensor.shape != torch.Size([])
                or tensor.dtype != torch.int64
            ):
                raise ValueError(
                    "PowerSGD compressor shared tensor schema mismatch for " f"{name!r}"
                )
        schema_version = int(shared["schema_version"].item())
        if schema_version != POWER_SGD_COMPRESSOR_SCHEMA_VERSION:
            raise ValueError("PowerSGD compressor checkpoint schema version mismatch")
        committed_step = int(shared["committed_step"].item())
        if committed_step < 0:
            raise ValueError("PowerSGD compressor checkpoint committed step is invalid")
        return committed_step


@dataclass
class _MatrixWork:
    gradient: Tensor
    state: PowerSGDParameterState
    corrected: Tensor
    p: Tensor
    q: Tensor


def _profile_range(name: str, **metadata: Any):
    return record_function(name, args=json.dumps(metadata, sort_keys=True))


def _profile_marker(name: str, **metadata: Any):
    suffix = " ".join(
        f"{key}={quote(str(value), safe='')}" for key, value in sorted(metadata.items())
    )
    return _profile_range(f"{name} {suffix}", **metadata)


def _tensor_bytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _bucket_profile_identity(context: BucketContext) -> dict[str, int]:
    return {
        "context_id": context.context_id,
        "bucket_index": context.bucket_index,
    }


@contextmanager
def _collective_profile_range(category: str, tensor: Tensor):
    payload_bytes = _tensor_bytes(tensor)
    observe_collective(category, "all_reduce", tensor)
    with _profile_range(
        category,
        operation="all_reduce",
        bytes=payload_bytes,
        numel=int(tensor.numel()),
    ):
        with _profile_range(
            f"{category}/payload bytes={payload_bytes}",
            operation="all_reduce",
            bytes=payload_bytes,
            numel=int(tensor.numel()),
        ):
            yield


def _all_reduce_future(
    state: PowerSGDDDPState, tensor: Tensor, stage: str
) -> torch.futures.Future:
    if state.world_size > 1:
        category = f"powersgd_hook/{stage}"
        with _collective_profile_range(category, tensor):
            return dist.all_reduce(
                tensor, group=state.process_group, async_op=True
            ).get_future()
    future = torch.futures.Future(devices=_future_devices(tensor.device))
    future.set_result(tensor)
    return future


def _profiled_all_reduce_future(
    state: PowerSGDDDPState,
    context: BucketContext,
    tensor: Tensor,
    stage: str,
) -> torch.futures.Future:
    with _profile_marker(
        "powersgd_hook/collective_launch",
        **_bucket_profile_identity(context),
        collective_category=f"powersgd_hook/{stage}",
        operation="all_reduce",
        bytes=_tensor_bytes(tensor),
    ):
        return _all_reduce_future(state, tensor, stage)


def _orthogonalize_grouped(
    matrices: Sequence[Tensor], epsilon: float
) -> list[Tensor]:
    """Orthogonalize equal-shaped matrices together to reduce kernel launches."""
    grouped: dict[tuple[torch.Size, torch.dtype, torch.device], list[int]] = {}
    for index, matrix in enumerate(matrices):
        key = (matrix.shape, matrix.dtype, matrix.device)
        grouped.setdefault(key, []).append(index)

    results: list[Tensor | None] = [None] * len(matrices)
    for indices in grouped.values():
        if len(indices) == 1:
            index = indices[0]
            results[index] = orthogonalize(matrices[index], epsilon)
            continue
        batch = torch.stack([matrices[index] for index in indices])
        orthogonalized = _orthogonalize_owned(batch, epsilon)
        for index, matrix in zip(indices, orthogonalized.unbind(0)):
            results[index] = matrix
    return [result for result in results if result is not None]


def _mark_future_complete(context: BucketContext) -> None:
    def mark(_completed: torch.futures.Future) -> None:
        with _profile_marker(
            "powersgd_hook/future_complete",
            **_bucket_profile_identity(context),
        ):
            pass

    context.completion_future.add_done_callback(mark)


def _prepare_compressed_bucket(
    state: PowerSGDDDPState, context: BucketContext
) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]], list[_MatrixWork]]:
    """Pack exact entries first, followed by native-orientation P factors."""
    dense = [
        gradient
        for gradient, item in zip(context.gradients, context.parameter_states)
        if item is None
    ]
    entries = [
        (gradient, item)
        for gradient, item in zip(context.gradients, context.parameter_states)
        if item is not None
    ]
    dense_numel = sum(gradient.numel() for gradient in dense)
    first = context.buffer.new_empty(
        dense_numel
        + sum(gradient.shape[0] * item.q_memory.shape[1] for gradient, item in entries)
    )
    second = context.buffer.new_empty(sum(item.q_memory.numel() for _, item in entries))
    context.retained.extend((first, second))
    dense_views = []
    offset = 0
    for gradient in dense:
        packed = first[offset : offset + gradient.numel()].view_as(gradient)
        packed.copy_(gradient)
        dense_views.append((gradient, packed))
        offset += gradient.numel()

    matrix_work = []
    pending = []
    initial_factors = []
    q_offset = 0
    config = state.config
    assert context.phase is not None
    for gradient, item in entries:
        corrected = corrected_gradient(
            gradient, item.error if config.error_feedback == "ef14" else None
        )
        reuse_q = config.warm_start and (
            context.phase > 0 or bool(item.q_initialized)
        )
        if reuse_q:
            initial_q = item.q_memory
        else:
            seed = derive_power_sgd_seed(
                base_seed=config.seed,
                phase=context.phase,
                stable_parameter_id=item.spec.stable_id,
                seed_scheme_version=config.seed_scheme_version,
            )
            initial_q = make_random_factor(
                item.q_memory.shape[0],
                item.q_memory.shape[1],
                seed,
                gradient.device,
                gradient.dtype,
            )
        pending.append((gradient, item, corrected))
        initial_factors.append(initial_q)

    with _profile_range("powersgd_hook/orthogonalization"):
        initial_factors = _orthogonalize_grouped(
            initial_factors, config.orthogonalization_epsilon
        )
    for (gradient, item, corrected), initial_q in zip(pending, initial_factors):
        p_numel = gradient.shape[0] * item.q_memory.shape[1]
        p = first[offset : offset + p_numel].view(gradient.shape[0], -1)
        p.copy_(compute_left_factor(corrected, initial_q))
        q = second[q_offset : q_offset + item.q_memory.numel()].view_as(item.q_memory)
        matrix_work.append(_MatrixWork(gradient, item, corrected, p, q))
        offset += p_numel
        q_offset += item.q_memory.numel()
    context.retained.append(matrix_work)
    return first, second, dense_views, matrix_work


def power_sgd_ddp_hook(
    state: PowerSGDDDPState, bucket: dist.GradBucket
) -> torch.futures.Future[Tensor]:
    """Average a DDP bucket via P+dense, Q, then local EF14 reconstruction.

    Preparation starts at bucket readiness. Only the collective tail orders
    buckets; the aggregate completion tail also includes reconstruction writes.
    """
    context = state.note_bucket(bucket)
    with _profile_marker(
        "powersgd_hook/bucket_ready",
        **_bucket_profile_identity(context),
        bucket_bytes=_tensor_bytes(context.buffer),
        phase="compressed" if context.phase is not None else "warmup",
    ):
        pass
    _mark_future_complete(context)

    def fail(exc: BaseException) -> None:
        # Torch futures accept Exception only, including on cancellation paths.
        if not isinstance(exc, Exception):
            failure = RuntimeError(f"{type(exc).__name__}: {exc}")
            failure.__cause__ = exc
            exc = failure
        for future in (context.collective_completion_future, context.completion_future):
            if not future.done():
                future.set_exception(exc)

    device = context.buffer.device

    def stream_for(streams: dict[torch.device, torch.cuda.Stream]) -> torch.cuda.Stream:
        if device not in streams:
            streams[device] = torch.cuda.Stream(device=device)
        return streams[device]

    def record_inputs(stream: torch.cuda.Stream) -> None:
        context.buffer.record_stream(stream)
        for item in context.parameter_states:
            if item is not None:
                item.error.record_stream(stream)
                item.q_memory.record_stream(stream)
                item.q_initialized.record_stream(stream)

    prepared = None
    try:
        if context.phase is not None and any(context.parameter_states):
            if device.type == "cuda":
                preparation_stream = stream_for(state._preparation_streams)
                assert context.bucket_ready_event is not None
                preparation_stream.wait_event(context.bucket_ready_event)
                record_inputs(preparation_stream)
                with torch.cuda.stream(preparation_stream):
                    with _profile_range("powersgd_hook/preparation"):
                        prepared = _prepare_compressed_bucket(state, context)
                    context.prepare_done = torch.cuda.Event()
                    context.prepare_done.record(preparation_stream)
            else:
                with _profile_range("powersgd_hook/preparation"):
                    prepared = _prepare_compressed_bucket(state, context)
    except BaseException as exc:
        fail(exc)
        return context.completion_future

    def record_prepared(stream: torch.cuda.Stream) -> None:
        record_inputs(stream)
        if prepared is not None:
            first, second, _, matrix_work = prepared
            first.record_stream(stream)
            second.record_stream(stream)
            for work in matrix_work:
                work.corrected.record_stream(stream)

    def guarded(callback: Callable[[], None]) -> Callable[[torch.futures.Future], None]:
        def run(completed: torch.futures.Future) -> None:
            try:
                # The callback runs only after completion; value propagates errors
                # without blocking a worker thread on Work.wait().
                completed.value()
                if device.type == "cuda":
                    callback_stream = torch.cuda.current_stream(device)
                    execution_stream = stream_for(state._execution_streams)
                    execution_stream.wait_stream(callback_stream)
                    ready = context.prepare_done or context.bucket_ready_event
                    assert ready is not None
                    execution_stream.wait_event(ready)
                    record_prepared(execution_stream)
                    with torch.cuda.stream(execution_stream):
                        callback()
                else:
                    callback()
            except BaseException as exc:
                fail(exc)

        return run

    def launch() -> None:
        if prepared is None:

            def finish_dense() -> None:
                context.buffer.div_(state.world_size)
                context.completion_future.set_result(context.buffer)

            _profiled_all_reduce_future(
                state, context, context.buffer, "dense"
            ).add_done_callback(guarded(finish_dense))
            return

        first, second, dense_views, matrix_work = prepared

        def after_p() -> None:
            for gradient, packed in dense_views:
                gradient.copy_(packed.div_(state.world_size))
            # P is a sum. Scaling before this normalization changes epsilon's
            # effect, and is unnecessary for the PowerSGD projection.
            with _profile_range("powersgd_hook/orthogonalization"):
                orthogonalized_p = _orthogonalize_grouped(
                    [work.p for work in matrix_work],
                    state.config.orthogonalization_epsilon,
                )
            for work, normalized_p in zip(matrix_work, orthogonalized_p):
                work.p.copy_(normalized_p)
                work.q.copy_(compute_right_factor(work.corrected, work.p))

            def after_q() -> None:
                # Capture Q visibility before releasing B: its callback may
                # immediately enqueue more work on the shared execution stream.
                q_done = None
                if device.type == "cuda":
                    q_done = torch.cuda.Event()
                    q_done.record(torch.cuda.current_stream(device))
                    context.retained.append(q_done)
                context.collective_completion_future.set_result(None)

                def reconstruct_bucket() -> None:
                    with _profile_range("powersgd_hook/reconstruction_error"):
                        second.div_(state.world_size)
                        for work in matrix_work:
                            approximation = reconstruct(work.p, work.q)
                            if state.config.error_feedback == "ef14":
                                work.state.error.copy_(work.corrected - approximation)
                            work.state.q_memory.copy_(work.q)
                            work.state.q_initialized.fill_(True)
                            work.gradient.copy_(approximation)
                    # CUDA-aware completion exports all reconstruction/error
                    # writes to DDP and the aggregate lifecycle Future.
                    context.completion_future.set_result(context.buffer)

                if device.type == "cuda":
                    reconstruction_stream = stream_for(state._reconstruction_streams)
                    assert q_done is not None
                    reconstruction_stream.wait_event(q_done)
                    record_prepared(reconstruction_stream)
                    with torch.cuda.stream(reconstruction_stream):
                        reconstruct_bucket()
                else:
                    reconstruct_bucket()

            _profiled_all_reduce_future(state, context, second, "q").add_done_callback(
                guarded(after_q)
            )

        _profiled_all_reduce_future(
            state, context, first, "p_plus_aux"
        ).add_done_callback(guarded(after_p))

    with _profile_marker(
        "powersgd_hook/chain_wait_begin",
        **_bucket_profile_identity(context),
    ):
        pass

    def after_previous_profiled(previous: torch.futures.Future) -> None:
        with _profile_marker(
            "powersgd_hook/chain_wait_end",
            **_bucket_profile_identity(context),
        ):
            pass
        guarded(launch)(previous)

    context.previous_collective_tail.add_done_callback(after_previous_profiled)
    return context.completion_future
