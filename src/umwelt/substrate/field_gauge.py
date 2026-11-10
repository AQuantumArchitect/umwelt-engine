"""field_gauge — the FIELD STATE as a first-class content-addressed gauge/git node in the shelf DAG.

The field's canonical state (field_unify) has a content hash = its gauge coordinate. This versions that state
in the SAME append-only shelf ledger the release gauge writes (<ARCHIVE_ROOT>/index.jsonl — content-addressed
artifacts + a parent DAG + the code git-SHA), so a learned brain becomes a lineage of content-addressed nodes
right next to releases. The clock-tape ↔ git contract made literal at the field level:

  • A node is written ONLY when the canon_hash CHANGED from the last field node — an unchanged hash = an empty
    diff = PROVABLE NON-TRAINING (no node, no churn).
  • A learned web (a new cross-cluster coupling) changes the hash = a provable TRAINING event: a new node,
    content-addressed, parent-linked to the prior state, the recoverable canonical pickle stored, and
    (optionally) git-committed — the siesta-strobed commit of the clock tape.

The stored artifact is the CANONICAL pickle (field_unify) — the one form that unpacks into BOTH the clustered
forebrain and the 1-matrix forecast brain. So any historical field node can be fetched and restored either way.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from umwelt.substrate import field_unify
from umwelt.substrate import shelf

_REPO = Path(__file__).resolve().parents[2]


def _short_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO),
                                       text=True, stderr=subprocess.DEVNULL).strip() or "none"
    except Exception:
        return "none"


def latest_field_node(archive_root=None) -> dict | None:
    """The most recent kind=field node in the shelf DAG (the parent of the next one), or None."""
    root = Path(archive_root) if archive_root else shelf.ARCHIVE_ROOT
    sh = shelf.Shelf(root)
    return next((r for r in reversed(sh.artifacts) if r.get("kind") == "field"), None)


def maybe_record_field_node(reservoir, *, archive_root=None, note: str = "",
                            git_commit: bool | None = None, strobe: bool = True) -> dict | None:
    """The strobed commit boundary, made real: record a field node only at the Poincaré section.

    If `strobe` and the field carries a periodic driver's clock cluster, the node is recorded only
    when the driver phase sits in the rest window (projection.gauge.in_rest_window) — the
    once-per-cycle, same-phase sampling that makes two consecutive snapshots comparable. A field
    with no clock records unconditionally (a world without a declared rhythm has no section to
    strobe on). `git_commit=None` reads UMWELT_FIELD_GIT ('1' → commit; default off, so archive
    behavior is unchanged until a deployment opts in)."""
    if git_commit is None:
        git_commit = os.environ.get("UMWELT_FIELD_GIT") == "1"
    if strobe:
        from umwelt.projection.gauge import driver_phase, in_rest_window
        field = getattr(reservoir, "field", reservoir)
        phase = driver_phase(field)
        if phase is not None and not in_rest_window(phase):
            return None                               # mid-cycle — not the commit boundary
    return record_field_node(reservoir, archive_root=archive_root, note=note,
                             git_commit=git_commit)


def record_field_node(reservoir, *, archive_root=None, note: str = "", git_commit: bool = False,
                      ts: str | None = None) -> dict | None:
    """Record the field's current canonical state as a content-addressed gauge node — versioned alongside
    releases. Writes ONLY when the canon_hash changed since the last field node (else None = provable
    non-training). The stored artifact is the canonical pickle (unpacks into clusters OR the manifold).
    Returns the node record, or None if nothing was learned. Accepts a reservoir (has .field) or a
    bare field-shaped object (has .clusters) — the growth service passes the latter, and passing it
    to the reservoir-only signature silently broke the production learning-event record."""
    root = Path(archive_root) if archive_root else shelf.ARCHIVE_ROOT
    canon = field_unify.pack_field(getattr(reservoir, "field", reservoir))
    h = field_unify.canon_hash(canon)
    prev = latest_field_node(root)
    if prev is not None and (prev.get("params") or {}).get("canon_hash") == h:
        return None                                       # empty diff — no learning since the last node
    (root / "field").mkdir(parents=True, exist_ok=True)
    ts = ts or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stored = f"field/field__{ts}__{h}.pkl"
    field_unify.save(canon, root / stored)
    sha = hashlib.sha256((root / stored).read_bytes()).hexdigest()
    N = int(canon["e1"].shape[0])
    ncpl = len(canon["zz"]) + len(canon["xy"])
    rec = {
        "ts": ts, "kind": "field", "name": "field", "stored": stored, "sha256": sha,
        "bytes": (root / stored).stat().st_size, "host": platform.node(), "git": _short_head(),
        "note": note,
        "parents": [prev["sha256"][:8]] if prev and prev.get("sha256") else [],
        "label": f"field.{h}.{N}q.{ncpl}c",                # the parametric gauge-name (coordinate)
        "params": {"canon_hash": h, "n_qubits": N, "n_couplings": ncpl,
                   "canon_hash_v2": field_unify.canon_hash(canon, version=2),
                   "n_clusters": len(canon["partition"]), "step": int(getattr(reservoir, "_step", 0))},
        "topo": {"partition": [[name, len(roles)] for (name, roles, _g, _dt) in canon["partition"]]},
        # gauge-invariant companion: lets a reader decompose any diff between two nodes into
        # gauge motion (hash moved, sheet identical) vs physical motion (sheet rows moved).
        "invariants": field_unify.invariant_sheet(canon),
    }
    with open(root / "index.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    with open(root / (stored + ".meta.json"), "w") as f:
        json.dump(rec, f, indent=2)
    if git_commit:
        _git_commit(root, stored, rec)
    return rec


def _git_commit(root: Path, stored: str, rec: dict) -> None:
    """Siesta-strobed commit: if the archive is a git repo, commit the new node (best-effort, never fatal)."""
    try:
        if subprocess.call(["git", "-C", str(root), "rev-parse", "--git-dir"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            return
        subprocess.call(["git", "-C", str(root), "add", "index.jsonl", stored, stored + ".meta.json"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(["git", "-C", str(root), "commit", "-q", "-m",
                         f"field {rec['label']} @ step {rec['params']['step']}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def field_lineage(archive_root=None) -> list[dict]:
    """The chain of field nodes, oldest→newest — the brain's learning history as content-addressed states."""
    root = Path(archive_root) if archive_root else shelf.ARCHIVE_ROOT
    return [r for r in shelf.Shelf(root).artifacts if r.get("kind") == "field"]
