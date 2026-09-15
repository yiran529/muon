"""Tensor-level PowerSGD primitives used by the DDP gradient hook."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class PowerSGDConfig:
    rank: int = 1
    start_compress_step: int = 1000
    min_compression_rate: float = 2.0
    error_feedback: Literal["ef14", "none"] = "ef14"
    warm_start: bool = True
    seed: int = 42
    orthogonalization_epsilon: float = 1e-8
    seed_scheme_version: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("rank", self.rank),
            ("start_compress_step", self.start_compress_step),
            ("seed", self.seed),
            ("seed_scheme_version", self.seed_scheme_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.start_compress_step < 0:
            raise ValueError("start_compress_step must be non-negative")
        if self.min_compression_rate <= 0:
            raise ValueError("min_compression_rate must be positive")
        if self.error_feedback not in ("ef14", "none"):
            raise ValueError("error_feedback must be 'ef14' or 'none'")
        if not isinstance(self.warm_start, bool):
            raise TypeError("warm_start must be a bool")
        if self.orthogonalization_epsilon < 0:
            raise ValueError("orthogonalization_epsilon must be non-negative")
        if self.seed_scheme_version != 1:
            raise ValueError("seed_scheme_version must be 1")


def should_compress(
    num_rows: int, num_cols: int, rank: int, min_compression_rate: float
) -> bool:
    """Return whether both low-rank factors meet the requested savings."""
    if num_rows <= 0 or num_cols <= 0 or rank <= 0 or min_compression_rate <= 0:
        raise ValueError("matrix dimensions, rank, and compression rate must be positive")
    effective_rank = min(rank, num_rows, num_cols)
    return min_compression_rate * effective_rank * (num_rows + num_cols) < num_rows * num_cols


def compressed_phase(step: int, start_compress_step: int) -> int | None:
    """Return the zero-based compressed phase, or ``None`` during warmup."""
    if step <= start_compress_step:
        return None
    return step - start_compress_step - 1


def derive_power_sgd_seed(
    *, base_seed: int, phase: int, stable_parameter_id: int, seed_scheme_version: int = 1
) -> int:
    """Derive a deterministic per-parameter seed without touching global RNG state."""
    if seed_scheme_version != 1:
        raise ValueError("seed_scheme_version must be 1")
    if phase < 0:
        raise ValueError("phase must be non-negative")
    return (base_seed + phase * 1_000_003 + stable_parameter_id) % (2**63 - 1)


def make_random_factor(
    rows: int,
    rank: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    """Sample a reproducible Gaussian factor in the requested communication dtype."""
    if rows <= 0 or rank <= 0:
        raise ValueError("rows and rank must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn((rows, rank), device=device, dtype=torch.float32, generator=generator).to(dtype)


def orthogonalize(matrix: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Column-orthogonalize a 2-D matrix with FP32 accumulation."""
    if matrix.ndim != 2:
        raise ValueError("orthogonalize requires a two-dimensional matrix")
    if matrix.shape[1] > matrix.shape[0]:
        raise ValueError("orthogonalize requires at least as many rows as columns")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    work = matrix.float().clone()
    for index in range(work.shape[1]):
        column = work[:, index]
        norm = torch.linalg.vector_norm(column)
        work[:, index] = column / (norm + epsilon)
        if index + 1 < work.shape[1]:
            work[:, index + 1 :] -= torch.outer(
                work[:, index], work[:, index] @ work[:, index + 1 :]
            )
    return work.to(dtype=matrix.dtype)


def corrected_gradient(gradient: Tensor, error: Tensor | None = None) -> Tensor:
    """Apply local EF14 error to a native-orientation gradient."""
    if gradient.ndim != 2:
        raise ValueError("PowerSGD requires a two-dimensional gradient")
    if error is None:
        return gradient.clone()
    if error.shape != gradient.shape:
        raise ValueError("gradient and error must have the same shape")
    return gradient + error


def compute_left_factor(corrected: Tensor, right_factor: Tensor) -> Tensor:
    """Compute P = H Q in the first PowerSGD projection."""
    if corrected.ndim != 2 or right_factor.ndim != 2:
        raise ValueError("PowerSGD factors require two-dimensional tensors")
    if corrected.shape[1] != right_factor.shape[0]:
        raise ValueError("right factor has incompatible shape")
    return corrected @ right_factor


def compute_right_factor(corrected: Tensor, left_factor: Tensor) -> Tensor:
    """Compute Q = Hᵀ P in the second PowerSGD projection."""
    if corrected.ndim != 2 or left_factor.ndim != 2:
        raise ValueError("PowerSGD factors require two-dimensional tensors")
    if corrected.shape[0] != left_factor.shape[0]:
        raise ValueError("left factor has incompatible shape")
    return corrected.mT @ left_factor


def reconstruct(left_factor: Tensor, right_factor: Tensor) -> Tensor:
    """Reconstruct a native-orientation matrix as P Qᵀ."""
    if left_factor.ndim != 2 or right_factor.ndim != 2:
        raise ValueError("PowerSGD factors require two-dimensional tensors")
    if left_factor.shape[1] != right_factor.shape[1]:
        raise ValueError("factors must have the same rank")
    return left_factor @ right_factor.mT
