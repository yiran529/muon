"""Summarize CM082 shared-score phase timing and gradient-sync ceiling."""

import argparse
import json
import re
from pathlib import Path


FINAL_RE = re.compile(r"step:820/820 .*?step_avg:([0-9.]+)ms")
PEAK_RE = re.compile(r"Peak memory consumption: ([0-9]+) MiB")


def timing_mean(root: Path, mode: str) -> float:
    payload = json.loads((root / "timing-summary.json").read_text())
    return float(payload["modes"][mode]["mean_step_ms"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--shared-interval200-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dense_payload = json.loads(args.dense_summary.read_text())
    dense_ms = float(dense_payload["modes"]["dense"]["mean_step_ms"])
    shared_200_payload = json.loads(args.shared_interval200_summary.read_text())
    shared_200_ms = float(
        shared_200_payload["modes"]["greedylore_local_svd_shared"]["mean_step_ms"]
    )
    shared_800_ms = timing_mean(
        args.artifact_root / "shared-interval800",
        "greedylore_local_svd_shared",
    )
    oracle_log = (args.artifact_root / "no-grad-sync" / "stdout.log").read_text()
    final_matches = FINAL_RE.findall(oracle_log.replace("\r", "\n"))
    peak_matches = PEAK_RE.findall(oracle_log.replace("\r", "\n"))
    if not final_matches or not peak_matches:
        raise SystemExit("no-gradient-sync oracle final timing or peak memory is missing")
    oracle_ms = float(final_matches[-1])

    refresh_extra_ms = (shared_200_ms - shared_800_ms) / (1 / 200 - 1 / 800)
    ordinary_ms = (4 * shared_800_ms - shared_200_ms) / 3
    payload = {
        "schema_version": 1,
        "single_cell_diagnostic": True,
        "historical_dense_summary": str(args.dense_summary),
        "historical_shared_interval200_summary": str(
            args.shared_interval200_summary
        ),
        "dense_step_ms": dense_ms,
        "shared_interval200_step_ms": shared_200_ms,
        "shared_interval800_step_ms": shared_800_ms,
        "estimated_refresh_extra_ms": refresh_extra_ms,
        "estimated_shared_ordinary_step_ms": ordinary_ms,
        "no_gradient_sync_oracle_step_ms": oracle_ms,
        "no_gradient_sync_peak_allocated_mib": int(peak_matches[-1]),
        "estimated_exposed_ddp_gradient_sync_ceiling_ms": dense_ms - oracle_ms,
        "shared_interval200_minus_dense_ms": shared_200_ms - dense_ms,
        "shared_interval800_minus_dense_ms": shared_800_ms - dense_ms,
        "notes": [
            "Each configuration has one cell; values are diagnostic, not confidence estimates.",
            "Refresh estimates assume T(interval) = ordinary + refresh_extra / interval.",
            "The no-gradient-sync oracle allows rank parameters to diverge and is timing-only.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
