"""docs/API.md is executable: every ```python block runs, top to bottom, in one
namespace (#66). A doc example that drifts from the code fails here."""
from __future__ import annotations

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "API.md"
_BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _blocks() -> list[str]:
    return _BLOCK.findall(DOC.read_text(encoding="utf-8"))


def test_api_doc_has_examples():
    assert len(_blocks()) >= 8


def test_api_doc_examples_run(tmp_path):
    ns: dict = {"__name__": "api_doc", "tmp": tmp_path}
    for i, src in enumerate(_blocks()):
        code = compile(src, f"{DOC.name}[block {i}]", "exec")
        exec(code, ns)  # noqa: S102 — our own docs, under test


def test_every_public_name_is_documented():
    import umwelt

    text = DOC.read_text(encoding="utf-8")
    missing = [n for n in umwelt.__all__ if f"`{n}" not in text]
    assert not missing, f"public names with no section in docs/API.md: {missing}"
