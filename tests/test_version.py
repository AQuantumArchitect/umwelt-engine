"""The package version and pyproject version must agree (and CHANGELOG must name it)."""
import re
from pathlib import Path

import umwelt

ROOT = Path(__file__).resolve().parent.parent


def test_version_agrees_with_pyproject_and_changelog():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m and m.group(1) == umwelt.__version__
    assert f"## [{umwelt.__version__}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
