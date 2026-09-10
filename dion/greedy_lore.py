"""Tensor-level foundations for the GreedyLore gradient compressor."""

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GreedyLoreConfig:
    """Configuration shared by GreedyLore compressor instances."""

    rank: int = 32
    update_interval: int = 200
    seed: int = 42
    start_compress_step: int = 1000
    basis_sync: Literal["local_svd", "broadcast"] = "local_svd"
    seed_scheme_version: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            ("rank", self.rank),
            ("update_interval", self.update_interval),
            ("seed", self.seed),
            ("start_compress_step", self.start_compress_step),
            ("seed_scheme_version", self.seed_scheme_version),
        )
        for name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, not {type(value).__name__}"
                )

        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if self.start_compress_step < 0:
            raise ValueError("start_compress_step must be non-negative")
        if self.basis_sync not in ("local_svd", "broadcast"):
            raise ValueError("basis_sync must be 'local_svd' or 'broadcast'")
        if self.seed_scheme_version != 1:
            raise ValueError("seed_scheme_version must be 1")


@dataclass(frozen=True)
class MatrixOrientation:
    """Shape and transpose metadata for a two-dimensional gradient matrix."""

    original_shape: tuple[int, int]
    compressed_shape: tuple[int, int]
    transposed: bool


def matrix_orientation(shape: Sequence[int]) -> MatrixOrientation:
    """Return the canonical orientation with no more rows than columns."""

    if len(shape) != 2:
        raise ValueError("GreedyLore requires a two-dimensional input")

    rows, columns = shape
    original_shape = (rows, columns)
    if rows <= columns:
        return MatrixOrientation(
            original_shape=original_shape,
            compressed_shape=original_shape,
            transposed=False,
        )
    return MatrixOrientation(
        original_shape=original_shape,
        compressed_shape=(columns, rows),
        transposed=True,
    )


def orient_matrix(tensor: Tensor, orientation: MatrixOrientation) -> Tensor:
    """View a matrix in canonical orientation."""

    return tensor.mT if orientation.transposed else tensor


def unorient_matrix(tensor: Tensor, orientation: MatrixOrientation) -> Tensor:
    """View a canonical-orientation matrix in its original orientation."""

    return tensor.mT if orientation.transposed else tensor


def compressed_phase(step: int, start_compress_step: int) -> int | None:
    """Return the zero-based compressed phase, or ``None`` during warmup."""

    if step <= start_compress_step:
        return None
    return step - start_compress_step - 1


def is_refresh_step(step: int, config: GreedyLoreConfig) -> bool:
    """Whether ``step`` refreshes the GreedyLore basis."""

    phase = compressed_phase(step, config.start_compress_step)
    return phase is not None and phase % config.update_interval == 0


def derive_greedy_lore_seed(
    *, base_seed: int, phase: int, stable_parameter_id: int
) -> int:
    """Derive a deterministic per-parameter seed for seed scheme version one."""

    return (base_seed + phase * 1_000_003 + stable_parameter_id) % (2**63 - 1)


def make_random_vectors(
    *, rows: int, columns: int, seed: int, device: torch.device
) -> Tensor:
    """Sample standard-normal FP32 vectors using a device-local generator."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (rows, columns),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


def canonicalize_svd_basis(basis: Tensor) -> Tensor:
    """Choose deterministic signs for the columns of an SVD basis."""

    if basis.ndim != 2:
        raise ValueError("SVD basis must be two-dimensional")
    if basis.shape[1] == 0:
        return basis.clone()

    pivot_indices = basis.abs().argmax(dim=0)
    columns = torch.arange(basis.shape[1], device=basis.device)
    pivots = basis[pivot_indices, columns]
    signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
    return basis * signs.unsqueeze(0)
