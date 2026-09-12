"""Canonical parameter layout and configuration identity for Sparse-K."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch.distributed as dist
from torch.distributed import ProcessGroup

from .sparse_k import SparseKConfig


@dataclass(frozen=True)
class SparseKParameterDescriptor:
    stable_name: str
    stable_id: int
    shape: tuple[int, ...]
    dtype: str
    role: Literal["sparse_matrix", "dense_aux"]


def canonical_sparse_k_fingerprint(
    *,
    config: SparseKConfig,
    group_ranks: Sequence[int],
    parameters: Sequence[SparseKParameterDescriptor],
) -> str:
    stable_names = [item.stable_name for item in parameters]
    stable_ids = [item.stable_id for item in parameters]
    if len(stable_names) != len(set(stable_names)):
        raise ValueError("Sparse-K parameter stable names must be unique")
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("Sparse-K parameter stable IDs must be unique")
    payload = {
        "config": asdict(config),
        "group_ranks": list(group_ranks),
        "parameters": [asdict(item) for item in parameters],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_sparse_k_fingerprint_across_ranks(
    fingerprint: str,
    process_group: ProcessGroup,
) -> None:
    world_size = dist.get_world_size(process_group)
    fingerprints = [None] * world_size
    dist.all_gather_object(fingerprints, fingerprint, group=process_group)
    if all(item == fingerprints[0] for item in fingerprints[1:]):
        return
    group_ranks = dist.get_process_group_ranks(process_group)
    mismatches = ", ".join(
        f"rank {rank}: {value}" for rank, value in zip(group_ranks, fingerprints)
    )
    raise ValueError(f"Sparse-K layout fingerprint mismatch across ranks: {mismatches}")
