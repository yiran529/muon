"""Standalone tensor-wise Rand-K and Top-K compression primitives."""

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor

SparseKMethod = Literal["randk", "topk"]
SparseKErrorFeedback = Literal["ef14", "noef"]


@dataclass(frozen=True)
class SparseKConfig:
    method: SparseKMethod = "topk"
    ratio: float = 0.2
    seed: int = 42
    start_compress_step: int = 1000
    error_feedback: SparseKErrorFeedback = "ef14"
    seed_scheme_version: int = 1

    def __post_init__(self) -> None:
        if self.method not in ("randk", "topk"):
            raise ValueError(f"method must be 'randk' or 'topk', got {self.method!r}")
        if not math.isfinite(self.ratio) or not 0.0 < self.ratio <= 1.0:
            raise ValueError(f"ratio must be finite and in (0, 1], got {self.ratio!r}")
        if self.start_compress_step < 0:
            raise ValueError("start_compress_step must be non-negative")
        if self.error_feedback not in ("ef14", "noef"):
            raise ValueError(
                "error_feedback must be 'ef14' or 'noef', "
                f"got {self.error_feedback!r}"
            )
        if self.seed_scheme_version != 1:
            raise ValueError("unsupported seed_scheme_version")


def sparse_k_count(numel: int, ratio: float) -> int:
    if numel <= 0:
        return 0
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be finite and in (0, 1], got {ratio!r}")
    return min(numel, max(1, math.floor(numel * ratio)))


def derive_sparse_k_seed(*, base_seed: int, step: int, stable_parameter_id: int) -> int:
    payload = f"sparse-k-v1:{base_seed}:{step}:{stable_parameter_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def select_randk_indices(
    numel: int,
    k: int,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    if not 0 <= k <= numel:
        raise ValueError("k must be in [0, numel]")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randperm(numel, generator=generator, device=device)[:k]


def select_topk_indices(values: Tensor, k: int) -> Tensor:
    flattened = values.reshape(-1)
    if not 0 <= k <= flattened.numel():
        raise ValueError("k must be in [0, values.numel()]")
    if k == 0:
        return torch.empty(0, dtype=torch.int64, device=values.device)
    return torch.topk(flattened.abs(), k, sorted=False).indices


def select_sparse_k_indices(
    values: Tensor,
    *,
    config: SparseKConfig,
    step: int,
    stable_parameter_id: int,
) -> Tensor:
    k = sparse_k_count(values.numel(), config.ratio)
    if config.method == "topk":
        return select_topk_indices(values, k)
    seed = derive_sparse_k_seed(
        base_seed=config.seed,
        step=step,
        stable_parameter_id=stable_parameter_id,
    )
    return select_randk_indices(values.numel(), k, seed=seed, device=values.device)


def compensated_gradient(gradient: Tensor, residual: Optional[Tensor]) -> Tensor:
    return (
        gradient.clone() if residual is None else gradient + residual.to(gradient.dtype)
    )


def update_ef14_residual_(
    residual: Tensor,
    compensated: Tensor,
    indices: Tensor,
    local_values: Tensor,
) -> None:
    local_compressed = torch.zeros_like(compensated).reshape(-1)
    local_compressed.scatter_(0, indices.to(torch.int64), local_values)
    residual.copy_((compensated.reshape(-1) - local_compressed).view_as(residual))
