"""Canonical GreedyLore layouts and distributed consistency validation."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch.distributed as dist
from torch.distributed import ProcessGroup

from .collective_observer import get_active_observer
from .greedy_lore import GreedyLoreConfig

GREEDY_LORE_LAYOUT_SCHEMA_VERSION = 1


class GreedyLoreLayoutMismatch(RuntimeError):
    """Raised when GreedyLore ranks disagree on their canonical layout."""


@dataclass(frozen=True)
class GreedyLoreParameterDescriptor:
    stable_name: str
    stable_id: int
    shape: tuple[int, ...]
    dtype: str
    role: Literal["matrix", "dense_aux"]


def canonical_greedy_lore_fingerprint(
    *,
    config: GreedyLoreConfig,
    group_ranks: Sequence[int],
    parameters: Sequence[GreedyLoreParameterDescriptor],
) -> str:
    """Hash the ordered, value-only GreedyLore runtime layout."""

    stable_names = [descriptor.stable_name for descriptor in parameters]
    if len(stable_names) != len(set(stable_names)):
        raise ValueError("GreedyLore parameter stable names must be unique")
    stable_ids = [descriptor.stable_id for descriptor in parameters]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("GreedyLore parameter stable IDs must be unique")

    payload = {
        "schema_version": GREEDY_LORE_LAYOUT_SCHEMA_VERSION,
        "config": asdict(config),
        "group_ranks": [int(rank) for rank in group_ranks],
        "parameters": [
            {
                "stable_name": descriptor.stable_name,
                "stable_id": descriptor.stable_id,
                "shape": list(descriptor.shape),
                "dtype": descriptor.dtype,
                "role": descriptor.role,
            }
            for descriptor in parameters
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_greedy_lore_fingerprint_across_ranks(
    fingerprint: str,
    process_group: ProcessGroup,
) -> None:
    """Require every process-group rank to provide the same fingerprint."""

    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return
    observer = get_active_observer()
    if observer is not None:
        observer.record(
            "greedylore/layout_validation",
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
        f"rank {rank}={value}" for rank, value in zip(group_ranks, fingerprints)
    )
    raise GreedyLoreLayoutMismatch(
        f"GreedyLore layout mismatch across ranks: {details}"
    )
