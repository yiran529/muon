"""Summarize a single-cell GPT-1B FP32 timing comparison."""

import argparse
import json
from pathlib import Path


def mode_mean(root: Path, mode: str) -> float:
    payload = json.loads((root / "timing-summary.json").read_text())
    return float(payload["modes"][mode]["mean_step_ms"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--model-dim", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dense_ms = mode_mean(args.artifact_root / "dense", "dense")
    independent_interval200_ms = mode_mean(
        args.artifact_root / "independent-interval200",
        "greedylore_local_svd",
    )
    interval200_ms = mode_mean(
        args.artifact_root / "shared-interval200",
        "greedylore_local_svd_shared",
    )
    interval800_ms = mode_mean(
        args.artifact_root / "shared-interval800",
        "greedylore_local_svd_shared",
    )
    refresh_extra_ms = (interval200_ms - interval800_ms) / (1 / 200 - 1 / 800)
    ordinary_ms = (4 * interval800_ms - interval200_ms) / 3
    selected_batch = int((args.artifact_root / "selected_device_batch.txt").read_text())
    selected_sequence_length = int(
        (args.artifact_root / "selected_sequence_length.txt").read_text()
    )

    payload = {
        "schema_version": 1,
        "single_cell_diagnostic": True,
        "model_dtype": "float32",
        "model_dim": args.model_dim,
        "layers": args.layers,
        "heads": args.heads,
        "bucket_and_greedylore_state_dtype_policy": "follow model parameters",
        "sequence_length": selected_sequence_length,
        "device_batch_size": selected_batch,
        "global_batch_size": 4 * selected_batch,
        "dense_step_ms": dense_ms,
        "independent_interval200_step_ms": independent_interval200_ms,
        "shared_interval200_step_ms": interval200_ms,
        "shared_interval800_step_ms": interval800_ms,
        "estimated_shared_ordinary_step_ms": ordinary_ms,
        "estimated_refresh_extra_ms": refresh_extra_ms,
        "estimated_refresh_amortized_interval200_ms": refresh_extra_ms / 200,
        "estimated_refresh_amortized_interval800_ms": refresh_extra_ms / 800,
        "independent_interval200_minus_dense_ms": independent_interval200_ms - dense_ms,
        "shared_interval200_minus_independent_interval200_ms": (
            interval200_ms - independent_interval200_ms
        ),
        "shared_interval200_minus_dense_ms": interval200_ms - dense_ms,
        "shared_interval800_minus_dense_ms": interval800_ms - dense_ms,
        "notes": [
            "Each formal configuration has one cell; no confidence interval is reported.",
            "The interval decomposition assumes T(I) = ordinary + refresh_extra / I.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
