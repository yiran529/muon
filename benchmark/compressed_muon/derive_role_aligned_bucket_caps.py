"""Derive role-aligned DDP caps from a calibrated reducer-ready layout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_MIB = 1024 * 1024


def _load_rank_layouts(layout_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(layout_dir.glob("rank-*-step-*.json"))
    if not paths:
        raise SystemExit(f"no calibrated rank layouts found in {layout_dir}")
    layouts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = layouts[0].get("buckets")
    if not isinstance(expected, list) or not expected:
        raise SystemExit("calibrated layout contains no buckets")
    if any(layout.get("buckets") != expected for layout in layouts[1:]):
        raise SystemExit("calibrated reducer bucket layouts differ across ranks")
    return layouts


def derive_role_aligned_caps(
    buckets: list[dict[str, Any]],
    *,
    target_cap_mb: float,
) -> dict[str, Any]:
    if not math.isfinite(target_cap_mb) or target_cap_mb <= 0:
        raise ValueError("target cap must be finite and positive")
    target_bytes = int(target_cap_mb * _MIB)
    parameters = [
        parameter
        for bucket in buckets
        for parameter in bucket.get("parameters", [])
    ]
    names = [parameter.get("stable_name") for parameter in parameters]
    if not parameters or len(names) != len(set(names)):
        raise ValueError("calibrated parameter order must be non-empty and unique")

    groups: list[list[dict[str, Any]]] = []
    current_matrix_group: list[dict[str, Any]] = []
    current_matrix_bytes = 0

    def flush_matrix_group() -> None:
        nonlocal current_matrix_group, current_matrix_bytes
        if current_matrix_group:
            groups.append(current_matrix_group)
            current_matrix_group = []
            current_matrix_bytes = 0

    for parameter in parameters:
        role = parameter.get("role")
        parameter_bytes = parameter.get("bytes")
        if role not in {"matrix", "dense_aux"}:
            raise ValueError(f"unsupported calibrated parameter role: {role!r}")
        if not isinstance(parameter_bytes, int) or parameter_bytes <= 0:
            raise ValueError("calibrated parameter bytes must be positive integers")
        if role == "dense_aux":
            flush_matrix_group()
            groups.append([parameter])
            continue
        if current_matrix_group and current_matrix_bytes + parameter_bytes > target_bytes:
            flush_matrix_group()
        current_matrix_group.append(parameter)
        current_matrix_bytes += parameter_bytes
    flush_matrix_group()

    caps_bytes = [sum(parameter["bytes"] for parameter in group) for group in groups]
    return {
        "schema_version": 1,
        "source_parameter_order": names,
        "target_cap_mb": target_cap_mb,
        "bucket_cap_bytes_list": caps_bytes,
        "bucket_cap_mb_list": [value / _MIB for value in caps_bytes],
        "expected_buckets": [
            {
                "bytes": cap_bytes,
                "role": group[0]["role"],
                "parameter_names": [parameter["stable_name"] for parameter in group],
            }
            for cap_bytes, group in zip(caps_bytes, groups)
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout_dir", type=Path)
    parser.add_argument("--target-cap-mb", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    layouts = _load_rank_layouts(args.layout_dir)
    result = derive_role_aligned_caps(
        layouts[0]["buckets"],
        target_cap_mb=args.target_cap_mb,
    )
    result["calibration_step"] = layouts[0].get("step")
    result["rank_count"] = len(layouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(",".join(f"{cap:.12g}" for cap in result["bucket_cap_mb_list"]))


if __name__ == "__main__":
    main()
