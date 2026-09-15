"""Canonical PowerSGD parameter layouts and distributed validation."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .collective_observer import get_active_observer
from .power_sgd import PowerSGDConfig

POWER_SGD_LAYOUT_SCHEMA_VERSION = 1
_DIGEST_SIZE = hashlib.sha256().digest_size


class PowerSGDLayoutMismatch(RuntimeError):
    """Raised when PowerSGD ranks disagree on their canonical layout."""


@dataclass(frozen=True)
class PowerSGDParameterDescriptor:
    stable_name: str
    stable_id: int
    shape: tuple[int, ...]
    dtype: str
    role: Literal["matrix", "dense_aux"]


def canonical_power_sgd_fingerprint(
    *,
    config: PowerSGDConfig,
    group_ranks: Sequence[int],
    parameters: Sequence[PowerSGDParameterDescriptor],
) -> str:
    """Hash the ordered, value-only PowerSGD runtime layout."""

    stable_names = [descriptor.stable_name for descriptor in parameters]
    if len(stable_names) != len(set(stable_names)):
        raise ValueError("PowerSGD parameter stable names must be unique")
    stable_ids = [descriptor.stable_id for descriptor in parameters]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("PowerSGD parameter stable IDs must be unique")

    payload = {
        "schema_version": POWER_SGD_LAYOUT_SCHEMA_VERSION,
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


def _digest_device(process_group: ProcessGroup) -> torch.device:
    backend = dist.get_backend(process_group)
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def validate_power_sgd_fingerprint_across_ranks(
    fingerprint: str,
    process_group: ProcessGroup,
) -> None:
    """Require every process-group rank to provide the same SHA-256 digest."""

    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return
    try:
        digest = bytes.fromhex(fingerprint)
    except ValueError as exc:
        raise ValueError("PowerSGD fingerprint must be a SHA-256 hex digest") from exc
    if len(digest) != _DIGEST_SIZE:
        raise ValueError("PowerSGD fingerprint must be a SHA-256 hex digest")

    device = _digest_device(process_group)
    local = torch.tensor(list(digest), dtype=torch.uint8, device=device)
    gathered = torch.empty(world_size * _DIGEST_SIZE, dtype=torch.uint8, device=device)
    observer = get_active_observer()
    if observer is not None:
        observer.record(
            "powersgd/layout_validation",
            "all_gather_into_tensor",
            _DIGEST_SIZE,
            "uint8",
            _DIGEST_SIZE,
        )
    dist.all_gather_into_tensor(gathered, local, group=process_group)

    chunks = gathered.reshape(world_size, _DIGEST_SIZE).cpu().tolist()
    if all(bytes(chunk) == digest for chunk in chunks):
        return

    group_ranks = dist.get_process_group_ranks(process_group)
    details = ", ".join(
        f"rank {rank}={bytes(chunk).hex()}"
        for rank, chunk in zip(group_ranks, chunks)
    )
    raise PowerSGDLayoutMismatch(f"PowerSGD layout mismatch across ranks: {details}")
