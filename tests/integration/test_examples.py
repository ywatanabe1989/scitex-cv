"""Smoke tests: every example script must run to completion."""

import subprocess
import sys
from pathlib import Path

EXAMPLES = list(Path(__file__).resolve().parents[2].joinpath("examples").glob("*.py"))


def test_examples_smoke(tmp_path):
    assert EXAMPLES, "no example scripts found"
    for ex in EXAMPLES:
        r = subprocess.run(
            [sys.executable, str(ex)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, f"{ex.name} failed: {r.stderr}"
