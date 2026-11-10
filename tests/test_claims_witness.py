"""CLAIMS.md witnesses itself.

House law: CLAIMS.md is only claims with a proof behind them — every row points to a
proof in proofs/ or a passing test. This gate makes the pointers themselves checkable:

  * every backticked repo path in an evidence cell exists (``{a,b}`` brace groups are
    expanded; a bare ``umwelt/...`` resolves under ``src/``);
  * every ``tests/x.py::name`` / ``proofs/x.py::name`` node id names a function that the
    file actually defines (parametrized ids like ``::test_x[case]`` are reduced to ``test_x``);
  * rows whose evidence lives in a foreign repo (the origin meerkat, umwelt-market,
    septacrypt-core, yurt) are exempt by an explicit marker, and the exempt count is
    printed so it cannot grow silently;
  * every relative markdown link in README.md and docs/*.md resolves.

Three stale pointers accumulated in the ledger in one week before this existed
(`src/umwelt/kits/` after the kits moved; two more found on re-witnessing). Dark is
not zero: a pointer this test cannot resolve is a claim without a witness, and the
fix is the ledger — mark the row DARK or repoint it — never this test.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLAIMS = ROOT / "CLAIMS.md"

# Evidence cells that name one of these live in another repository by the ledger's own
# design (tiers 2/3/3c cite the origin and the consumer harnesses, "not re-claimed here").
FOREIGN_MARKERS = ("meerkat", "umwelt-market", "septacrypt-core", "septacrypt", "yurt",
                   "the-broadcast", "github.com/")

PATH_RE = re.compile(r"`((?:src|tests|proofs|examples|tools|docs|umwelt|umweltd|umweltforge)/[^`\s()]+)`")
NODE_RE = re.compile(r"`((?:tests|proofs)/[^`\s]+?\.py)::([^`\s\[]+)")


def _rows():
    """Yield (line_no, claim_cell, evidence_cell) for every table row with ≥3 cells."""
    for no, line in enumerate(CLAIMS.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| claim"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3:
            yield no, cells[0], cells[1]


def _expand_braces(path: str) -> list[str]:
    m = re.search(r"\{([^{}]+)\}", path)
    if not m:
        return [path]
    out = []
    for alt in m.group(1).split(","):
        out.extend(_expand_braces(path[:m.start()] + alt.strip() + path[m.end():]))
    return out


def _exists(path: str) -> bool:
    path = path.rstrip("/").split("::")[0]
    return (ROOT / path).exists() or (ROOT / "src" / path).exists()


def _defined_names(py: Path) -> set[str]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_every_claims_evidence_path_exists():
    missing, checked, foreign = [], 0, 0
    for no, _claim, evidence in _rows():
        if any(mark in evidence for mark in FOREIGN_MARKERS):
            foreign += 1                                  # cited, not re-claimed here — by the ledger's design
            continue
        for raw in PATH_RE.findall(evidence):
            for path in _expand_braces(raw):
                checked += 1
                if not _exists(path):
                    missing.append(f"CLAIMS.md:{no}: `{path}`")
    print(f"\nCLAIMS evidence paths checked: {checked}; foreign-repo rows exempt: {foreign}")
    assert checked > 50, "the ledger lost its evidence column?"
    assert not missing, "claims without a witness in this tree (repoint or mark DARK):\n" + "\n".join(missing)


def test_every_claims_test_node_id_is_defined():
    bad, checked = [], 0
    for no, _claim, evidence in _rows():
        for file, name in NODE_RE.findall(evidence):
            if any(mark in evidence for mark in FOREIGN_MARKERS):
                continue                                  # a foreign harness's test id, labelled as such
            checked += 1
            py = ROOT / file
            if not py.exists():
                bad.append(f"CLAIMS.md:{no}: {file} missing")
            elif name not in _defined_names(py):
                bad.append(f"CLAIMS.md:{no}: {file} defines no `{name}`")
    print(f"\nCLAIMS test node ids checked: {checked}")
    assert checked >= 5
    assert not bad, "\n".join(bad)


LINK_RE = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("doc", sorted([ROOT / "README.md", ROOT / "CLAIMS.md", ROOT / "CHANGELOG.md",
                                        *(ROOT / "docs").glob("*.md")]), ids=lambda p: p.name)
def test_relative_markdown_links_resolve(doc: Path):
    broken = []
    for target in LINK_RE.findall(doc.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("mailto:"):
            continue
        if not (doc.parent / target).exists() and not (ROOT / target).exists():
            broken.append(target)
    assert not broken, f"{doc.relative_to(ROOT)} has broken relative links: {broken}"
