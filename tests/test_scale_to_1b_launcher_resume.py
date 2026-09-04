"""Focused fixture checks for bounded scale-out attempt recovery."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


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


def test_signal_between_cells_stops_before_next_launch(tmp_path: Path) -> None:
    result = _run_fixture(
        tmp_path,
        """
cat > "$ARTIFACT_ROOT/torchrun" <<'SH'
#!/usr/bin/env bash
echo launched > "$ARTIFACT_ROOT/ran"
exit 0
SH
chmod +x "$ARTIFACT_ROOT/torchrun"
controller_interrupted
""",
        """
set +e
run_timing fixture-id adamw arc gpt130m normal 1
rc=$?
set -e
test "$rc" = 130
test ! -e "$ARTIFACT_ROOT/ran"
test ! -e "$ARTIFACT_ROOT/fixture-id/.timing-r1-attempted"
""",
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize("kind", ["timing", "profile"])
def test_signal_during_active_formal_attempt_is_resumable(
    tmp_path: Path, kind: str
) -> None:
    runner = "run_timing" if kind == "timing" else "run_profile"
    result = _run_fixture(
        tmp_path,
        """
cat > "$ARTIFACT_ROOT/torchrun" <<'SH'
#!/usr/bin/env bash
kill -TERM "$CONTROLLER_PID"
sleep 0.2
exit 0
SH
chmod +x "$ARTIFACT_ROOT/torchrun"
export CONTROLLER_PID=$$
""",
        f"""
set +e
{runner} fixture-id adamw arc gpt130m normal 1
rc=$?
set -e
test "$rc" = 130
test -s "$ARTIFACT_ROOT/fixture-id/.{kind}-r1-interrupted"
test ! -e "$ARTIFACT_ROOT/fixture-id/.{kind}-r1-terminal"
""",
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_signal_during_probe_is_resumed_once_without_overwrite(tmp_path: Path) -> None:
    result = _run_fixture(
        tmp_path,
        """
cat > "$VALIDATOR" <<'PY'
import json
import sys

path = sys.argv[sys.argv.index("--path") + 1]
try:
    json.load(open(path))
except (OSError, ValueError):
    raise SystemExit(2)
PY
cat > "$ARTIFACT_ROOT/torchrun" <<'SH'
#!/usr/bin/env bash
count=$(( $(cat "$ARTIFACT_ROOT/count" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$ARTIFACT_ROOT/count"
for last; do :; done
if [[ "$count" = 1 ]]; then
  printf 'partial\\n' > "$last"
  kill -TERM "$CONTROLLER_PID"
fi
sleep 0.2
if [[ "$count" = 2 ]]; then
  printf '{}\\n' > "$last"
fi
exit 0
SH
chmod +x "$ARTIFACT_ROOT/torchrun"
export CONTROLLER_PID=$$
""",
        """
optimizer=adamw sync=arc world=4 suffix=b
set +e
run_probe adamw arc 4 b
rc=$?
set -e
test "$rc" = 130
probe_dir="$ARTIFACT_ROOT/probes/gpt1b-adamw-arc-ws4"
test -s "$probe_dir/.probe-r1-interrupted"
test ! -e "$probe_dir/.probe-r1-terminal"
# A resumed launcher is a fresh controller process with persisted sentinels.
CONTROLLER_INTERRUPTED=0
run_probe adamw arc 4 b
run_probe adamw arc 4 b
test "$(cat "$ARTIFACT_ROOT/count")" = 2
test -e "$probe_dir/.probe-r1-resume-attempted"
test -s "$probe_dir/.probe-r1-terminal"
test -s "$probe_dir/probe-retry-"*.json
grep -qx partial "$probe_dir/probe.json"
""",
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_legacy_gpt1b_oom_does_not_gate_lower_model_profile(tmp_path: Path) -> None:
    result = _run_fixture(
        tmp_path,
        """
touch "$ARTIFACT_ROOT/scale-to-1b-oom-adamw-arc"
cat > "$MANIFEST" <<'JSON'
{"event":"cell","id":"CM008-probeb-adamw-arc-gpt1b-ddp-ws4-s42","path":"probe.json","status":"oom"}
JSON
cat > "$ARTIFACT_ROOT/torchrun" <<'SH'
#!/usr/bin/env bash
echo $(( $(cat "$ARTIFACT_ROOT/count" 2>/dev/null || echo 0) + 1 )) > "$ARTIFACT_ROOT/count"
exit 0
SH
chmod +x "$ARTIFACT_ROOT/torchrun"
""",
        """
run_profile lower-id adamw arc gpt130m normal 1
run_profile upper-id adamw arc gpt1b normal 1
test "$(cat "$ARTIFACT_ROOT/count")" = 1
test -e "$ARTIFACT_ROOT/lower-id/.profile-r1-attempted"
test ! -e "$ARTIFACT_ROOT/upper-id/.profile-r1-attempted"
grep -q 'upper-id.*skipped' "$MANIFEST"
""",
    )
    assert result.returncode == 0, result.stderr + result.stdout
