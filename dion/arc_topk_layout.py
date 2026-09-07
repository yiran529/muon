"""Canonical ARC layouts and one-time distributed consistency validation."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

import torch.distributed as dist
from torch.distributed import ProcessGroup

from .arc_topk_sync import ArcTopKSyncConfig
from .collective_observer import get_active_observer


ARC_LAYOUT_SCHEMA_VERSION = 1


class ArcTopKLayoutMismatch(RuntimeError):
    """Raised when ARC ranks do not agree on their canonical layout."""


@dataclass(frozen=True)
class ArcParameterDescriptor:
    stable_name: str
    stable_id: int
    shape: tuple[int, ...]
    dtype: str
    role: Literal["arc_matrix", "dense_aux"]


@dataclass(frozen=True)
class ArcOptimizerTaskDescriptor:
    group_id: int
    task_id: int
    ordered_parameter_names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    config: ArcTopKSyncConfig


def _config_payload(config: ArcTopKSyncConfig) -> dict:
    return asdict(config)


def _parameter_payload(descriptor: ArcParameterDescriptor) -> dict:
    return {
        "stable_name": descriptor.stable_name,
        "stable_id": descriptor.stable_id,
        "shape": list(descriptor.shape),
        "dtype": descriptor.dtype,
        "role": descriptor.role,
    }


def _task_payload(descriptor: ArcOptimizerTaskDescriptor) -> dict:
    return {
        "group_id": descriptor.group_id,
        "task_id": descriptor.task_id,
        "ordered_parameter_names": list(descriptor.ordered_parameter_names),
        "shape": list(descriptor.shape),
        "dtype": descriptor.dtype,
        "config": _config_payload(descriptor.config),
    }


def canonical_arc_fingerprint(
    *,
    base_seed: int,
    config: ArcTopKSyncConfig,
    group_ranks: Sequence[int],
    parameters: Sequence[ArcParameterDescriptor],
    optimizer_tasks: Optional[Sequence[ArcOptimizerTaskDescriptor]] = None,
) -> str:
    """Hash a value-only hook or optimizer ARC layout."""

    stable_names = [descriptor.stable_name for descriptor in parameters]
    if len(stable_names) != len(set(stable_names)):
        raise ValueError("ARC parameter stable names must be unique")
    stable_ids = [descriptor.stable_id for descriptor in parameters]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("ARC parameter stable IDs must be unique")

    layout_kind = "hook" if optimizer_tasks is None else "optimizer"
    payload = {
        "schema_version": ARC_LAYOUT_SCHEMA_VERSION,
        "layout_kind": layout_kind,
        "base_seed": int(base_seed),
        "config": _config_payload(config),
        "group_ranks": [int(rank) for rank in group_ranks],
        "parameters": [_parameter_payload(item) for item in parameters],
    }
    if optimizer_tasks is not None:
        payload["optimizer_tasks"] = [
            _task_payload(item) for item in optimizer_tasks
        ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_arc_fingerprint_across_ranks(
    fingerprint: str,
    process_group: ProcessGroup,
) -> None:
    """Require every rank in ``process_group`` to provide the same digest."""

    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return
    observer = get_active_observer()
    if observer is not None:
        observer.record(
            "arc/layout_validation",
            "all_gather_object",
            1,
            "sha256",
            len(fingerprint.encode("ascii")),
        )
    fingerprints = [None] * world_size
    dist.all_gather_object(fingerprints, fingerprint, group=process_group)
    if all(value == fingerprints[0] for value in fingerprints[1:]):
        return

    group_ranks = dist.get_process_group_ranks(process_group)
    details = ", ".join(
        f"rank {rank}={value}"
        for rank, value in zip(group_ranks, fingerprints)
    )
    raise ArcTopKLayoutMismatch(f"ARC layout mismatch across ranks: {details}")
