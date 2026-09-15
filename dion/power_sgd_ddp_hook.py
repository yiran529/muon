"""Stable state and lifecycle for the PowerSGD DDP communication hook."""

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Parameter

from .power_sgd import PowerSGDConfig, compressed_phase, should_compress

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
) -> torch.futures.Future:
    """Complete after both inputs and propagate either input exception."""

    def finish(completed: torch.futures.Future) -> None:
        for future in completed.value():
            future.value()

    return torch.futures.collect_all([previous, current]).then(finish)


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
                previous_tail, completion_future
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
        self.tail_future.value()
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
