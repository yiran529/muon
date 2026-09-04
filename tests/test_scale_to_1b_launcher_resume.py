"""Focused fixture checks for bounded scale-out attempt recovery."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "artifacts/compressed_muon/scale_to_1b_launcher.sh"


def _run_fixture(tmp_path: Path, setup: str, command: str) -> subprocess.CompletedProcess[str]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    env = os.environ.copy()
    env["SCALE_TO_1B_SOURCE_ONLY"] = "1"
    env["SCALE_TO_1B_ARTIFACT_ROOT"] = str(fixture)
    script = f"""
set -euo pipefail
source {LAUNCHER!s}
ARTIFACT_ROOT={fixture!s}
export ARTIFACT_ROOT
MANIFEST=$ARTIFACT_ROOT/manifest.jsonl
STATUS_LOG=$ARTIFACT_ROOT/status.log
VALIDATOR=$ARTIFACT_ROOT/validator.py
PYTHON={ROOT / '.venv/bin/python'!s}
TORCHRUN=$ARTIFACT_ROOT/torchrun
{setup}
{command}
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_interrupted_attempt_is_resumed_once_then_terminalized(tmp_path: Path) -> None:
    result = _run_fixture(
        tmp_path,
        """
id=fixture-id; dir=$ARTIFACT_ROOT/$id; mkdir -p "$dir"
touch "$dir/.timing-r1-attempted" "$dir/.timing-r1-interrupted"
cat > "$ARTIFACT_ROOT/torchrun" <<'SH'
#!/usr/bin/env bash
echo $(( $(cat "$ARTIFACT_ROOT/count" 2>/dev/null || echo 0) + 1 )) > "$ARTIFACT_ROOT/count"
exit 0
SH
chmod +x "$ARTIFACT_ROOT/torchrun"
""",
        """
run_timing fixture-id adamw arc gpt130m normal 1 "$ARTIFACT_ROOT/fixture-id"
run_timing fixture-id adamw arc gpt130m normal 1 "$ARTIFACT_ROOT/fixture-id" || true
test "$(cat "$ARTIFACT_ROOT/count")" = 1
grep -q 'status.*invalid' "$MANIFEST"
test -s "$ARTIFACT_ROOT/fixture-id/.timing-r1-terminal"
""",
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_failed_attempt_is_terminal_and_never_retried(tmp_path: Path) -> None:
    result = _run_fixture(
        tmp_path,
        """
id=fixture-id; dir=$ARTIFACT_ROOT/$id; mkdir -p "$dir"
cat > "$ARTIFACT_ROOT/torchrun" <<'SH'
#!/usr/bin/env bash
echo $(( $(cat "$ARTIFACT_ROOT/count" 2>/dev/null || echo 0) + 1 )) > "$ARTIFACT_ROOT/count"
exit 7
SH
chmod +x "$ARTIFACT_ROOT/torchrun"
""",
        """
run_timing fixture-id adamw arc gpt130m normal 1 "$ARTIFACT_ROOT/fixture-id" || true
run_timing fixture-id adamw arc gpt130m normal 1 "$ARTIFACT_ROOT/fixture-id" || true
test "$(cat "$ARTIFACT_ROOT/count")" = 1
test -s "$ARTIFACT_ROOT/fixture-id/.timing-r1-terminal"
grep -q 'failed' "$ARTIFACT_ROOT/fixture-id/.timing-r1-terminal"
""",
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_orphaned_attempt_without_interruption_is_terminalized(tmp_path: Path) -> None:
    result = _run_fixture(
        tmp_path,
        """
id=fixture-id; dir=$ARTIFACT_ROOT/$id; mkdir -p "$dir"
touch "$dir/.timing-r1-attempted"
cat > "$ARTIFACT_ROOT/torchrun" <<'SH'
#!/usr/bin/env bash
echo unexpected > "$ARTIFACT_ROOT/ran"
exit 0
SH
chmod +x "$ARTIFACT_ROOT/torchrun"
""",
        """
run_timing fixture-id adamw arc gpt130m normal 1 "$ARTIFACT_ROOT/fixture-id" || true
test ! -e "$ARTIFACT_ROOT/ran"
test -s "$ARTIFACT_ROOT/fixture-id/.timing-r1-terminal"
grep -q 'invalid' "$ARTIFACT_ROOT/fixture-id/.timing-r1-terminal"
""",
    )
    assert result.returncode == 0, result.stderr + result.stdout
