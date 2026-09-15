"""Stable state and lifecycle for the GreedyLore DDP communication hook."""

import json
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.profiler import record_function
from torch.nn import Parameter

from .greedy_lore import (
    GreedyLoreConfig,
    MatrixOrientation,
    approximate_signed_lambda,
    compressed_phase,
    corrected_gradient,
    derive_greedy_lore_seed,
    is_refresh_step,
    make_random_vectors,
    matrix_orientation,
    orient_matrix,
    reconstruct_global,
    refresh_basis,
    select_projector,
    unorient_matrix,
)
from .collective_observer import observe_collective

GREEDY_LORE_COMPRESSOR_SCHEMA_VERSION = 1


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
    previous_collective_tail: torch.futures.Future
    collective_completion_future: torch.futures.Future
    completion_future: torch.futures.Future
    prepared: "PreparedCompressedBucket | None" = None
    retained: list[Any] = field(default_factory=list)


@dataclass
class CompressedMatrixWork:
    gradient: Tensor
    parameter_state: GreedyLoreParameterState
    corrected: Tensor
    signed_lambda: Tensor
    score_offset: int
    factor_offset: int | None = None
    projector: Tensor | None = None
    support: Tensor | None = None
    local_factor: Tensor | None = None


@dataclass
class PreparedCompressedBucket:
    score_plus_aux: Tensor
    matrix_work: list[CompressedMatrixWork]
    dense_ranges: list[tuple[Tensor, int, int]]
    prepare_done: torch.cuda.Event | None


def _record_prepared_tensors_on_stream(
    prepared: PreparedCompressedBucket,
    stream: torch.cuda.Stream,
) -> None:
    """Keep preparation-stream allocations live on their consumer stream."""

    prepared.score_plus_aux.record_stream(stream)
    for work in prepared.matrix_work:
        work.corrected.record_stream(stream)
        work.signed_lambda.record_stream(stream)


def _record_reconstruction_tensors_on_stream(
    context: BucketContext,
    factor_buffer: Tensor,
    matrix_work: list[CompressedMatrixWork],
    stream: torch.cuda.Stream,
) -> None:
    """Keep execution-stream allocations live through reconstruction."""

    context.buffer.record_stream(stream)
    factor_buffer.record_stream(stream)
    for work in matrix_work:
        assert work.projector is not None
        work.projector.record_stream(stream)


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


def _profile_range(name: str, **metadata: Any):
    return record_function(name, args=json.dumps(metadata, sort_keys=True))


def _tensor_bytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@contextmanager
def _collective_profile_range(
    category: str,
    operation: str,
    tensor: Tensor,
    *,
    bytes: int | None = None,
    numel: int | None = None,
):
    payload_bytes = _tensor_bytes(tensor) if bytes is None else int(bytes)
    logical_numel = int(tensor.numel()) if numel is None else int(numel)
    observe_collective(
        category,
        operation,
        tensor,
        bytes=payload_bytes,
        numel=logical_numel,
    )
    with _profile_range(
        category,
        operation=operation,
        bytes=payload_bytes,
        numel=logical_numel,
    ):
        with _profile_range(
            f"{category}/payload bytes={payload_bytes}",
            operation=operation,
            bytes=payload_bytes,
            numel=logical_numel,
        ):
            yield


def _bucket_profile_metadata(state: "GreedyLoreDDPState", context: "BucketContext"):
    matrix_bytes = 0
    dense_bytes = 0
    score_numel = 0
    factor_numel = 0
    basis_bytes = 0
    has_matrix = any(parameter_state is not None for parameter_state in context.parameter_states)
    for gradient, parameter_state in zip(context.gradients, context.parameter_states):
        if parameter_state is None:
            dense_bytes += _tensor_bytes(gradient)
            if has_matrix:
                score_numel += gradient.numel()
            continue
        matrix_bytes += _tensor_bytes(gradient)
        rows, columns = parameter_state.orientation.compressed_shape
        score_numel += rows
        factor_numel += state.config.rank * columns
        if (
            context.phase is not None
            and is_refresh_step(context.step, state.config)
            and state.config.basis_sync == "broadcast"
        ):
            basis_bytes += _tensor_bytes(parameter_state.basis)
    phase = "warmup"
    if context.phase is not None:
        phase = "refresh" if is_refresh_step(context.step, state.config) else "compressed"
    score_element_size = {
        "bucket": context.buffer.element_size(),
        "float32": torch.empty((), dtype=torch.float32).element_size(),
        "bfloat16": torch.empty((), dtype=torch.bfloat16).element_size(),
    }[state.config.dense_aux_communication_dtype]
    bucket_specs = [
        state._specs_by_parameter[id(parameter)] for parameter in context.parameters
    ]
    return {
        "bucket_bytes": _tensor_bytes(context.buffer),
        "matrix_bytes": matrix_bytes,
        "dense_aux_bytes": dense_bytes,
        "score_bytes": score_numel * score_element_size if phase == "compressed" else 0,
        "factor_bytes": (
            factor_numel * context.buffer.element_size()
            if phase == "compressed"
            else 0
        ),
        "basis_bytes": basis_bytes,
        "parameter_count": len(context.parameters),
        "parameter_names": ",".join(spec.stable_name for spec in bucket_specs),
        "parameter_roles": ",".join(spec.role for spec in bucket_specs),
        "phase": phase,
    }


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
            if spec.parameter.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("GreedyLore matrix parameters must be FP32 or BF16")
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
                    dtype=spec.parameter.dtype,
                    device=spec.parameter.device,
                ),
                basis=torch.eye(
                    rows,
                    dtype=spec.parameter.dtype,
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
        matrix_states = sorted(
            self._parameter_states.values(),
            key=lambda parameter_state: parameter_state.spec.stable_name,
        )
        return {
            state.spec.stable_name: {
                "error": {
                    "shape": list(state.error.shape),
                    "dtype": str(state.error.dtype).removeprefix("torch."),
                },
                "basis": {
                    "shape": list(state.basis.shape),
                    "dtype": str(state.basis.dtype).removeprefix("torch."),
                },
                "last_support": {
                    "shape": list(state.last_support.shape),
                    "dtype": str(state.last_support.dtype).removeprefix("torch."),
                },
            }
            for state in matrix_states
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

    def _require_committed_boundary(self, operation: str) -> None:
        if self._active_step is not None or not self.tail_future.done():
            raise GreedyLoreStateError(
                f"GreedyLore compressor {operation} requires a committed step boundary"
            )

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return value-only metadata suitable for validation before DCP load."""

        self._require_committed_boundary("checkpoint")
        return {
            "schema_version": GREEDY_LORE_COMPRESSOR_SCHEMA_VERSION,
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
        """Fail closed before tensor payloads are loaded into this runtime."""

        self._require_committed_boundary("load")
        if metadata.get("schema_version") != GREEDY_LORE_COMPRESSOR_SCHEMA_VERSION:
            raise ValueError("GreedyLore compressor checkpoint schema version mismatch")
        if metadata.get("dp_world_size") != self.world_size:
            raise ValueError("GreedyLore compressor checkpoint world size mismatch")
        if metadata.get("group_ranks") != list(self.group_ranks):
            raise ValueError(
                "GreedyLore compressor checkpoint rank membership mismatch"
            )
        if metadata.get("config_fingerprint") != self.fingerprint:
            raise ValueError("GreedyLore compressor checkpoint fingerprint mismatch")
        if metadata.get("seed_scheme_version") != self.config.seed_scheme_version:
            raise ValueError("GreedyLore compressor checkpoint seed scheme mismatch")
        if metadata.get("config") != asdict(self.config):
            raise ValueError("GreedyLore compressor checkpoint config mismatch")
        if metadata.get("ordered_parameter_table") != self._ordered_parameter_table():
            raise ValueError(
                "GreedyLore compressor checkpoint parameter table mismatch"
            )
        committed_step = metadata.get("committed_step")
        if not isinstance(committed_step, int) or committed_step < 0:
            raise ValueError(
                "GreedyLore compressor checkpoint committed step is invalid"
            )
        self._validate_metadata_tensor_schema(metadata.get("tensor_schema"))
        self._validated_checkpoint_committed_step = committed_step

    def _validate_metadata_tensor_schema(self, tensor_schema: Any) -> None:
        expected = self._tensor_schema()
        if not isinstance(tensor_schema, dict):
            raise ValueError("GreedyLore compressor checkpoint tensor schema mismatch")
        missing_ranks = sorted(set(expected) - set(tensor_schema))
        extra_ranks = sorted(set(tensor_schema) - set(expected))
        if missing_ranks or extra_ranks:
            missing_text = ", ".join(missing_ranks)
            extra_text = ", ".join(extra_ranks)
            raise ValueError(
                "GreedyLore compressor checkpoint tensor schema mismatch; "
                f"missing={missing_text}, extra={extra_text}"
            )
        for root, expected_entry in expected.items():
            actual_entry = tensor_schema[root]
            if actual_entry != expected_entry:
                if root.startswith("rank_"):
                    expected_names = set(expected_entry)
                    actual_names = (
                        set(actual_entry) if isinstance(actual_entry, dict) else set()
                    )
                    missing_names = sorted(expected_names - actual_names)
                    if missing_names:
                        raise ValueError(
                            "GreedyLore compressor checkpoint tensor schema "
                            f"is missing parameter {missing_names[0]!r}"
                        )
                raise ValueError(
                    "GreedyLore compressor checkpoint tensor schema mismatch"
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

    def preparation_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.type != "cuda":
            raise ValueError(
                "GreedyLore preparation streams are only defined for CUDA devices"
            )
        stream = self._preparation_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._preparation_streams[device] = stream
        return stream

    def reconstruction_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.type != "cuda":
            raise ValueError(
                "GreedyLore reconstruction streams are only defined for CUDA devices"
            )
        stream = self._reconstruction_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._reconstruction_streams[device] = stream
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
        previous_collective_tail = self.collective_tail
        completion_future = torch.futures.Future(devices=_future_devices(buffer.device))
        is_pipelined_compressed_bucket = (
            compressed_phase(self._active_step, self.config.start_compress_step)
            is not None
            and not is_refresh_step(self._active_step, self.config)
            and any(
                self._parameter_states.get(id(parameter)) is not None
                for parameter in parameters
            )
        )
        collective_completion_future = (
            torch.futures.Future(devices=_future_devices(buffer.device))
            if is_pipelined_compressed_bucket
            else completion_future
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
                previous_tail,
                completion_future,
            )
            self.collective_tail = collective_completion_future

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
        self.validate_replicated_basis_across_ranks()
        matrix_states = sorted(
            self._parameter_states.values(),
            key=lambda parameter_state: parameter_state.spec.stable_name,
        )
        return {
            "shared": {
                "schema_version": torch.tensor(
                    GREEDY_LORE_COMPRESSOR_SCHEMA_VERSION,
                    dtype=torch.int64,
                    device=self.parameter_specs[0].parameter.device,
                ),
                "committed_step": torch.tensor(
                    self.committed_step,
                    dtype=torch.int64,
                    device=self.parameter_specs[0].parameter.device,
                ),
            },
            f"rank_{self.global_rank}": {
                state.spec.stable_name: {
                    "error": state.error,
                    "basis": state.basis,
                    "last_support": state.last_support,
                }
                for state in matrix_states
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a validated checkpoint into preallocated compressor tensors."""

        self._require_committed_boundary("load")
        try:
            shared = state_dict["shared"]
        except KeyError as exc:
            raise ValueError(
                "GreedyLore compressor checkpoint is missing shared state"
            ) from exc
        self._validate_shared_state(shared)
        rank_key = f"rank_{self.global_rank}"
        try:
            rank_state = state_dict[rank_key]
        except KeyError as exc:
            raise ValueError(
                f"GreedyLore compressor checkpoint is missing state for {rank_key}"
            ) from exc
        expected_names = {
            parameter_state.spec.stable_name
            for parameter_state in self._parameter_states.values()
        }
        actual_names = set(rank_state) if isinstance(rank_state, dict) else set()
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                "GreedyLore compressor rank-local state mismatch; "
                f"missing={missing}, extra={extra}"
            )
        for parameter_state in self._parameter_states.values():
            name = parameter_state.spec.stable_name
            local_entry = rank_state[name]
            if not isinstance(local_entry, dict) or set(local_entry) != {
                "error",
                "basis",
                "last_support",
            }:
                raise ValueError(
                    "GreedyLore compressor state fields mismatch for "
                    f"parameter {name!r}"
                )
            for field_name in ("error", "basis", "last_support"):
                source = local_entry[field_name]
                destination = getattr(parameter_state, field_name)
                if (
                    not torch.is_tensor(source)
                    or source.shape != destination.shape
                    or source.dtype != destination.dtype
                ):
                    raise ValueError(
                        "GreedyLore compressor tensor schema mismatch for "
                        f"{name!r}/{field_name}"
                    )
                destination.copy_(source)
        checkpoint_step = int(shared["committed_step"].item())
        if (
            self._validated_checkpoint_committed_step is not None
            and checkpoint_step != self._validated_checkpoint_committed_step
        ):
            raise ValueError(
                "GreedyLore compressor checkpoint committed step payload "
                "does not match metadata"
            )
        self.committed_step = checkpoint_step
        self._validated_checkpoint_committed_step = None
        self.validate_replicated_basis_across_ranks()

    def _validate_shared_state(self, shared: Any) -> None:
        if not isinstance(shared, dict):
            raise ValueError("GreedyLore compressor checkpoint shared state mismatch")
        if set(shared) != {"schema_version", "committed_step"}:
            raise ValueError("GreedyLore compressor checkpoint shared state mismatch")
        schema_version = shared["schema_version"]
        committed_step = shared["committed_step"]
        for name, tensor in shared.items():
            if (
                not torch.is_tensor(tensor)
                or tensor.shape != torch.Size([])
                or tensor.dtype != torch.int64
            ):
                raise ValueError(
                    "GreedyLore compressor shared tensor schema mismatch for "
                    f"{name!r}"
                )
        if int(schema_version.item()) != GREEDY_LORE_COMPRESSOR_SCHEMA_VERSION:
            raise ValueError("GreedyLore compressor checkpoint schema version mismatch")
        if int(committed_step.item()) < 0:
            raise ValueError(
                "GreedyLore compressor checkpoint committed step is invalid"
            )


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


def enqueue_collective_bucket_chain(
    state: GreedyLoreDDPState,
    context: BucketContext,
    launch: Callable[[BucketContext], torch.futures.Future],
) -> torch.futures.Future:
    """Launch one bucket after prior collectives, independently of reconstruction."""

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
                execution_stream = state.execution_stream(context.buffer.device)
                ready_event = (
                    context.prepared.prepare_done
                    if context.prepared is not None
                    else context.bucket_ready_event
                )
                assert ready_event is not None
                execution_stream.wait_event(ready_event)
                context.buffer.record_stream(execution_stream)
                if context.prepared is not None:
                    _record_prepared_tensors_on_stream(
                        context.prepared,
                        execution_stream,
                    )
                with torch.cuda.stream(execution_stream):
                    launched = launch(context)
            else:
                launched = launch(context)
            bridge_future(launched, destination, export_completion)
        except BaseException as exc:
            if not destination.done():
                destination.set_exception(exc)
            if not context.collective_completion_future.done():
                context.collective_completion_future.set_exception(exc)

    context.previous_collective_tail.add_done_callback(after_previous)
    return destination


def _all_reduce_future(
    state: GreedyLoreDDPState,
    tensor: Tensor,
    category: str,
) -> torch.futures.Future:
    if state.process_group is not None and state.world_size > 1:
        with _collective_profile_range(category, "all_reduce", tensor):
            return dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
                async_op=True,
            ).get_future()
    future = torch.futures.Future(devices=_future_devices(tensor.device))
    future.set_result(tensor)
    return future


def _broadcast_future(
    state: GreedyLoreDDPState,
    tensor: Tensor,
    category: str,
) -> torch.futures.Future:
    if state.process_group is not None and state.world_size > 1:
        with _collective_profile_range(category, "broadcast", tensor):
            return dist.broadcast(
                tensor,
                src=state.group_ranks[0],
                group=state.process_group,
                async_op=True,
            ).get_future()
    future = torch.futures.Future(devices=_future_devices(tensor.device))
    future.set_result(tensor)
    return future


def _on_bucket_execution_stream(
    state: GreedyLoreDDPState,
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
    state: GreedyLoreDDPState,
    context: BucketContext,
    callback: Callable[[torch.futures.Future], Tensor],
) -> Callable[[torch.futures.Future], Tensor]:
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


def _matrix_entries(
    context: BucketContext,
) -> list[tuple[Tensor, GreedyLoreParameterState]]:
    return [
        (gradient, parameter_state)
        for gradient, parameter_state in zip(
            context.gradients,
            context.parameter_states,
        )
        if parameter_state is not None
    ]


def _stable_matrix_entries(
    context: BucketContext,
) -> list[tuple[Tensor, GreedyLoreParameterState]]:
    return sorted(
        _matrix_entries(context),
        key=lambda item: item[1].spec.stable_name,
    )


def _prepare_refresh_buffer(context: BucketContext) -> None:
    for gradient, parameter_state in _matrix_entries(context):
        corrected = corrected_gradient(
            gradient,
            parameter_state.error,
            parameter_state.orientation,
        )
        gradient.copy_(unorient_matrix(corrected, parameter_state.orientation))


def _divide_completed_buffer(state: GreedyLoreDDPState, value: Any) -> Tensor:
    buffer = _future_tensor(value)
    if state.world_size > 1:
        buffer.div_(state.world_size)
    return buffer


def _refresh_local_svd(
    state: GreedyLoreDDPState,
    context: BucketContext,
    completed: torch.futures.Future,
) -> Tensor:
    _divide_completed_buffer(state, completed.value())
    with _profile_range("greedylore_hook/local_svd"):
        for gradient, parameter_state in _matrix_entries(context):
            global_corrected = orient_matrix(gradient, parameter_state.orientation)
            basis, _, support = refresh_basis(global_corrected, state.config.rank)
            parameter_state.basis.copy_(basis)
            parameter_state.last_support.copy_(support)
            parameter_state.error.zero_()
    return context.buffer


def _launch_dense_bucket(
    state: GreedyLoreDDPState,
    context: BucketContext,
) -> torch.futures.Future:
    source = _all_reduce_future(state, context.buffer, "greedylore_hook/dense")

    def finish(completed: torch.futures.Future) -> Tensor:
        _divide_completed_buffer(state, completed.value())
        return context.buffer

    return source.then(_on_bucket_execution_stream_result(state, context, finish))


def _launch_refresh_bucket(
    state: GreedyLoreDDPState,
    context: BucketContext,
) -> torch.futures.Future:
    _prepare_refresh_buffer(context)
    source = _all_reduce_future(state, context.buffer, "greedylore_hook/dense")
    if state.config.basis_sync == "local_svd":
        return source.then(
            _on_bucket_execution_stream_result(
                state,
                context,
                lambda completed: _refresh_local_svd(state, context, completed),
            )
        )

    completion = torch.futures.Future(devices=_future_devices(context.buffer.device))

    def fail(exc: BaseException) -> None:
        if not completion.done():
            completion.set_exception(exc)

    def complete_after_broadcasts(completed: torch.futures.Future) -> Tensor:
        completed.value()
        return context.buffer

    def after_dense(completed: torch.futures.Future) -> None:
        try:
            _divide_completed_buffer(state, completed.value())
            futures = []
            for gradient, parameter_state in _stable_matrix_entries(context):
                range_context = (
                    _profile_range("greedylore_hook/local_svd")
                    if state.global_rank == state.group_ranks[0]
                    else nullcontext()
                )
                with range_context:
                    if state.global_rank == state.group_ranks[0]:
                        global_corrected = orient_matrix(
                            gradient,
                            parameter_state.orientation,
                        )
                        basis, _, _ = refresh_basis(
                            global_corrected,
                            state.config.rank,
                        )
                        parameter_state.basis.copy_(basis)
                parameter_state.last_support.copy_(
                    torch.arange(
                        state.config.rank,
                        dtype=torch.int64,
                        device=parameter_state.last_support.device,
                    )
                )
                parameter_state.error.zero_()
                futures.append(
                    _broadcast_future(
                        state,
                        parameter_state.basis,
                        "greedylore_hook/basis_broadcast",
                    )
                )
            if futures:
                broadcast_completion = torch.futures.collect_all(futures).then(
                    _on_bucket_execution_stream_result(
                        state,
                        context,
                        complete_after_broadcasts,
                    )
                )
                bridge_future(
                    broadcast_completion,
                    completion,
                    lambda value: value,
                )
            else:
                completion.set_result(context.buffer)
        except BaseException as exc:
            fail(exc)

    source.add_done_callback(_on_bucket_execution_stream(state, context, after_dense))
    return completion


def _parameter_spec_for_dense(
    state: GreedyLoreDDPState,
    parameter: Parameter,
) -> GreedyLoreDDPParameterSpec:
    spec = state._specs_by_parameter[id(parameter)]
    if spec.role != "dense_aux":
        raise GreedyLoreStateError(
            f"expected dense auxiliary parameter {spec.stable_name!r}"
        )
    return spec


def _ordered_compressed_entries(
    state: GreedyLoreDDPState,
    context: BucketContext,
) -> list[tuple[Tensor, GreedyLoreParameterState | None, Parameter, int]]:
    entries = []
    for gradient, parameter_state, parameter in zip(
        context.gradients,
        context.parameter_states,
        context.parameters,
    ):
        stable_id = (
            parameter_state.spec.stable_id
            if parameter_state is not None
            else _parameter_spec_for_dense(state, parameter).stable_id
        )
        entries.append((gradient, parameter_state, parameter, stable_id))
    return sorted(entries, key=lambda item: item[3])


def _prepare_score_plus_aux_buffer(
    state: GreedyLoreDDPState,
    context: BucketContext,
) -> tuple[Tensor, list[CompressedMatrixWork], list[tuple[Tensor, int, int]]]:
    with _profile_range("greedylore_hook/score"):
        communication_dtype = {
            "bucket": context.buffer.dtype,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[state.config.dense_aux_communication_dtype]
        chunks = []
        matrix_work = []
        dense_ranges = []
        offset = 0
        if context.phase is None:
            raise GreedyLoreStateError("compressed score packing requires a phase")
        for gradient, parameter_state, _parameter, _stable_id in _ordered_compressed_entries(
            state,
            context,
        ):
            if parameter_state is None:
                flat_dense = gradient.reshape(-1).to(dtype=communication_dtype)
                chunks.append(flat_dense)
                dense_ranges.append((gradient, offset, flat_dense.numel()))
                offset += flat_dense.numel()
                continue
            corrected = corrected_gradient(
                gradient,
                parameter_state.error,
                parameter_state.orientation,
            )
            rows, columns = corrected.shape
            random_vectors = make_random_vectors(
                rows=rows,
                columns=columns,
                seed=derive_greedy_lore_seed(
                    base_seed=state.config.seed,
                    phase=context.phase,
                    stable_parameter_id=parameter_state.spec.stable_id,
                ),
                device=corrected.device,
                dtype=corrected.dtype,
            )
            signed_lambda = approximate_signed_lambda(
                corrected,
                parameter_state.basis,
                random_vectors,
            )
            chunks.append(signed_lambda)
            matrix_work.append(
                CompressedMatrixWork(
                    gradient=gradient,
                    parameter_state=parameter_state,
                    corrected=corrected,
                    signed_lambda=signed_lambda,
                    score_offset=offset,
                )
            )
            offset += signed_lambda.numel()
        if not chunks:
            return (
                torch.empty(
                    0,
                    dtype=communication_dtype,
                    device=context.buffer.device,
                ),
                matrix_work,
                dense_ranges,
            )
        score_plus_aux = torch.cat([chunk.reshape(-1) for chunk in chunks]).to(
            dtype=communication_dtype
        )
        context.retained.extend([score_plus_aux, matrix_work, dense_ranges])
        return score_plus_aux, matrix_work, dense_ranges


def _prepare_compressed_bucket(
    state: GreedyLoreDDPState,
    context: BucketContext,
) -> None:
    """Queue ordinary score preparation independently of prior buckets."""

    if context.prepared is not None:
        raise GreedyLoreStateError("compressed bucket was prepared more than once")
    if context.buffer.device.type != "cuda":
        score_plus_aux, matrix_work, dense_ranges = _prepare_score_plus_aux_buffer(
            state,
            context,
        )
        context.prepared = PreparedCompressedBucket(
            score_plus_aux=score_plus_aux,
            matrix_work=matrix_work,
            dense_ranges=dense_ranges,
            prepare_done=None,
        )
        return

    preparation_stream = state.preparation_stream(context.buffer.device)
    assert context.bucket_ready_event is not None
    preparation_stream.wait_event(context.bucket_ready_event)
    with torch.cuda.stream(preparation_stream):
        score_plus_aux, matrix_work, dense_ranges = _prepare_score_plus_aux_buffer(
            state,
            context,
        )
        prepare_done = torch.cuda.Event()
        prepare_done.record(preparation_stream)
    context.prepared = PreparedCompressedBucket(
        score_plus_aux=score_plus_aux,
        matrix_work=matrix_work,
        dense_ranges=dense_ranges,
        prepare_done=prepare_done,
    )


def _select_projector_profiled(
    state: GreedyLoreDDPState,
    work: CompressedMatrixWork,
    averaged_lambda: Tensor,
) -> None:
    with _profile_range("greedylore_hook/topr"):
        projector, support = select_projector(
            work.parameter_state.basis,
            averaged_lambda,
            state.config.rank,
        )
    work.projector = projector
    work.support = support
    work.parameter_state.last_support.copy_(support)


def _compress_local_profiled(work: CompressedMatrixWork) -> None:
    assert work.projector is not None
    with _profile_range("greedylore_hook/factor"):
        local_factor = work.projector.mT @ work.corrected
    with _profile_range("greedylore_hook/error"):
        next_error = work.corrected - work.projector @ local_factor
        work.parameter_state.error.copy_(next_error)
    work.local_factor = local_factor


def _prepare_factor_buffer(
    state: GreedyLoreDDPState,
    score_plus_aux: Tensor,
    matrix_work: list[CompressedMatrixWork],
) -> Tensor:
    chunks = []
    offset = 0
    for work in matrix_work:
        averaged_lambda = score_plus_aux[
            work.score_offset : work.score_offset + work.signed_lambda.numel()
        ].to(dtype=work.signed_lambda.dtype)
        _select_projector_profiled(state, work, averaged_lambda)
        _compress_local_profiled(work)
        assert work.local_factor is not None
        work.factor_offset = offset
        chunks.append(work.local_factor.reshape(-1).clone())
        offset += work.local_factor.numel()
    factor_buffer = torch.cat(chunks)
    return factor_buffer


def _copy_averaged_dense_aux(
    score_plus_aux: Tensor,
    dense_ranges: list[tuple[Tensor, int, int]],
) -> None:
    for gradient, offset, numel in dense_ranges:
        averaged = score_plus_aux[offset : offset + numel].view_as(gradient)
        gradient.copy_(averaged.to(dtype=gradient.dtype))


def _reconstruct_compressed_matrices(
    state: GreedyLoreDDPState,
    factor_buffer: Tensor,
    matrix_work: list[CompressedMatrixWork],
) -> Tensor:
    if state.world_size > 1:
        factor_buffer.div_(state.world_size)
    with _profile_range("greedylore_hook/reconstruction"):
        for work in matrix_work:
            assert work.projector is not None
            assert work.local_factor is not None
            assert work.factor_offset is not None
            offset = work.factor_offset
            averaged_factor = factor_buffer[
                offset : offset + work.local_factor.numel()
            ].view_as(work.local_factor)
            reconstructed = reconstruct_global(work.projector, averaged_factor)
            work.gradient.copy_(
                unorient_matrix(reconstructed, work.parameter_state.orientation).to(
                    dtype=work.gradient.dtype
                )
            )
    return matrix_work[0].gradient.new_empty(0)


def _launch_compressed_reconstruction(
    state: GreedyLoreDDPState,
    context: BucketContext,
    factor_buffer: Tensor,
    matrix_work: list[CompressedMatrixWork],
) -> torch.futures.Future:
    """Reconstruct independently and export its writes through a CUDA Future."""

    if context.buffer.device.type != "cuda":
        _reconstruct_compressed_matrices(state, factor_buffer, matrix_work)
        completed = torch.futures.Future()
        completed.set_result(context.buffer)
        return completed

    device = context.buffer.device
    callback_stream = torch.cuda.current_stream(device)
    reconstruction_stream = state.reconstruction_stream(device)
    reconstruction_stream.wait_stream(callback_stream)
    _record_reconstruction_tensors_on_stream(
        context,
        factor_buffer,
        matrix_work,
        reconstruction_stream,
    )
    completed = torch.futures.Future(devices=_future_devices(device))
    with torch.cuda.stream(reconstruction_stream):
        _reconstruct_compressed_matrices(state, factor_buffer, matrix_work)
        completed.set_result(context.buffer)
    return completed


def _mark_future_complete(future: torch.futures.Future) -> torch.futures.Future:
    def mark(_completed: torch.futures.Future) -> None:
        with _profile_range("greedylore_hook/future_complete"):
            pass

    future.add_done_callback(mark)
    return future


def _launch_compressed_bucket(
    state: GreedyLoreDDPState,
    context: BucketContext,
) -> torch.futures.Future:
    if not _matrix_entries(context):
        return _launch_dense_bucket(state, context)

    if context.prepared is None:
        _prepare_compressed_bucket(state, context)
    assert context.prepared is not None
    score_plus_aux = context.prepared.score_plus_aux
    matrix_work = context.prepared.matrix_work
    dense_ranges = context.prepared.dense_ranges
    score_source = _all_reduce_future(
        state,
        score_plus_aux,
        "greedylore_hook/score_plus_aux_allreduce",
    )
    completion = torch.futures.Future(devices=_future_devices(context.buffer.device))
    collective_completion = context.collective_completion_future

    def fail(exc: BaseException) -> None:
        if not completion.done():
            completion.set_exception(exc)
        if not collective_completion.done():
            collective_completion.set_exception(exc)

    def after_score(completed: torch.futures.Future) -> None:
        try:
            reduced_score_plus_aux = _future_tensor(completed.value())
            if state.world_size > 1:
                reduced_score_plus_aux.div_(state.world_size)
            _copy_averaged_dense_aux(reduced_score_plus_aux, dense_ranges)
            factor_buffer = _prepare_factor_buffer(
                state,
                reduced_score_plus_aux,
                matrix_work,
            )
            context.retained.append(factor_buffer)
            factor_source = _all_reduce_future(
                state,
                factor_buffer,
                "greedylore_hook/factor_allreduce",
            )

            def after_factor(factor_completed: torch.futures.Future) -> None:
                try:
                    reduced_factor = _future_tensor(factor_completed.value())
                except BaseException as exc:
                    fail(exc)
                    return
                try:
                    reconstruction = _launch_compressed_reconstruction(
                        state,
                        context,
                        reduced_factor,
                        matrix_work,
                    )
                    if not collective_completion.done():
                        collective_completion.set_result(None)
                    bridge_future(reconstruction, completion, lambda value: value)
                except BaseException as exc:
                    if not collective_completion.done():
                        collective_completion.set_result(None)
                    if not completion.done():
                        completion.set_exception(exc)

            factor_source.add_done_callback(
                _on_bucket_execution_stream(state, context, after_factor)
            )
        except BaseException as exc:
            fail(exc)

    score_source.add_done_callback(
        _on_bucket_execution_stream(state, context, after_score)
    )
    return completion


def greedy_lore_ddp_hook(
    state: GreedyLoreDDPState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[Tensor]:
    """Synchronize one DDP bucket through the globally sequenced GreedyLore chain."""

    context = state.note_bucket(bucket)
    bucket_metadata = _bucket_profile_metadata(state, context)
    bucket_metadata_suffix = " ".join(
        f"{key}={value}" for key, value in sorted(bucket_metadata.items())
    )
    with _profile_range(
        f"greedylore_hook/bucket_ready {bucket_metadata_suffix}",
        **bucket_metadata,
    ):
        pass
    if context.phase is None:
        return _mark_future_complete(enqueue_bucket_chain(
            state,
            context,
            lambda current: _launch_dense_bucket(state, current),
        ))
    if is_refresh_step(context.step, state.config):
        return _mark_future_complete(enqueue_bucket_chain(
            state,
            context,
            lambda current: _launch_refresh_bucket(state, current),
        ))
    if _matrix_entries(context):
        try:
            _prepare_compressed_bucket(state, context)
        except BaseException as exc:
            context.collective_completion_future.set_exception(exc)
            context.completion_future.set_exception(exc)
            return _mark_future_complete(context.completion_future)
        return _mark_future_complete(enqueue_collective_bucket_chain(
            state,
            context,
            lambda current: _launch_compressed_bucket(state, current),
        ))
    return _mark_future_complete(enqueue_collective_bucket_chain(
        state,
        context,
        lambda current: _launch_dense_bucket(state, current),
    ))
