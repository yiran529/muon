"""Write a CPU-only summary for dense-Muon attribution cell artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.compressed_muon.dense_muon_attribution import summarize_artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = summarize_artifacts(args.root)
    output = args.output or args.root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
