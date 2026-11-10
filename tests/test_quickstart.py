"""The README quickstart must run for a stranger: offline, numpy only, exit 0."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_runs_and_prints_honest_markers():
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "quickstart.py")],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "value=None  state=dark" in out          # unmeasured is dark, not zero
    assert "frame/1: frame/1 violations: none" in out
    assert "save/load hash-equal: True" in out
