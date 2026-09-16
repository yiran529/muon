"""Summarize the CM087 bucket bridge and CM088 lower-batch diagnostic."""

import argparse
import json
from pathlib import Path


def measurement(root: Path, mode: str) -> dict[str, float | int]:
    payload = json.loads((root / "timing-summary.json").read_text())
    values = payload["modes"][mode]
    return {
        "step_ms": float(values["mean_step_ms"]),
        "peak_allocated_mib": int(values["peak_allocated_mib"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cm087-root", type=Path, required=True)
    parser.add_argument("--cm088-root", type=Path, required=True)
    args = parser.parse_args()

    cm087 = {}
    for bucket in (160, 80):
        dense = measurement(args.cm087_root / f"bucket{bucket}-dense", "dense")
        independent = measurement(
            args.cm087_root / f"bucket{bucket}-independent200",
            "greedylore_local_svd",
        )
        difference = independent["step_ms"] - dense["step_ms"]
        cm087[f"bucket{bucket}_mib"] = {
            "dense": dense,
            "independent_interval200": independent,
            "independent_minus_dense_ms": difference,
            "independent_minus_dense_percent": difference / dense["step_ms"] * 100,
        }
    cm087.update(
        {
            "schema_version": 1,
            "single_cell_diagnostic": True,
            "model": "gpt350m",
            "model_dtype": "float32",
            "sequence_length": 256,
            "device_batch_size": 8,
            "global_batch_size": 32,
            "historical_cm051_percent": -15.972704871743614,
            "notes": ["Each configuration has one cell; no confidence interval is reported."],
        }
    )
    (args.cm087_root / "summary.json").write_text(json.dumps(cm087, indent=2) + "\n")

    dense = measurement(args.cm088_root / "dense", "dense")
    independent = measurement(
        args.cm088_root / "independent-interval200", "greedylore_local_svd"
    )
    shared200 = measurement(
        args.cm088_root / "shared-interval200", "greedylore_local_svd_shared"
    )
    shared800 = measurement(
        args.cm088_root / "shared-interval800", "greedylore_local_svd_shared"
    )
    refresh_extra_ms = (shared200["step_ms"] - shared800["step_ms"]) / (
        1 / 200 - 1 / 800
    )
    ordinary_ms = (4 * shared800["step_ms"] - shared200["step_ms"]) / 3
    cm088 = {
        "schema_version": 1,
        "single_cell_diagnostic": True,
        "model": "gpt720m",
        "parameter_count": 718602240,
        "model_dtype": "float32",
        "sequence_length": 256,
        "device_batch_size": 8,
        "global_batch_size": 32,
        "dense": dense,
        "independent_interval200": independent,
        "shared_interval200": shared200,
        "shared_interval800": shared800,
        "independent_minus_dense_percent": (independent["step_ms"] / dense["step_ms"] - 1) * 100,
        "shared_interval200_minus_dense_percent": (shared200["step_ms"] / dense["step_ms"] - 1) * 100,
        "shared_interval800_minus_dense_percent": (shared800["step_ms"] / dense["step_ms"] - 1) * 100,
        "estimated_shared_ordinary_step_ms": ordinary_ms,
        "estimated_refresh_extra_ms": refresh_extra_ms,
        "estimated_refresh_amortized_interval200_ms": refresh_extra_ms / 200,
        "estimated_refresh_amortized_interval800_ms": refresh_extra_ms / 800,
        "notes": [
            "Each configuration has one cell; no confidence interval is reported.",
            "The interval decomposition assumes T(I) = ordinary + refresh_extra / I.",
        ],
    }
    (args.cm088_root / "summary.json").write_text(json.dumps(cm088, indent=2) + "\n")


if __name__ == "__main__":
    main()
