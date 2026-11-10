"""FRAME EXPORT — Umwelt writes a `frame/1` document, read-only on ρ.

docs/FRAME_SCHEMA.md defines a Frame: one node per measured-or-not thing, each marked
current / stale / dark by its own age against a threshold the producer states. This
proof is Umwelt acting as that producer (umwelt.projection.frame):

  1. boot the blank gridworld (proofs/first_sentence.py's path) and export a Frame —
     every axis is latent, so every node must be dark with a null value (law 1);
  2. replay one deterministic synthetic day through the production ingest path;
  3. export again — nodes that received signal carry the engine's own last-reading
     timestamp (SensorBridge.latest_value) and come out current; what was never
     signalled stays dark; nothing is promoted.

Asserted: validate() returns no violations on either document; the dark count fell and
at least one node is current after the day; engine.field_canon_hash() is byte-identical
across each export (the projection never touches the field). Receipt:
proofs/receipts/frame_export.json. Seeded, offline, numpy only.

Run:  python3 proofs/frame_export.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.gridworld.world import (  # noqa: E402
    SITE, agent_walk, gridworld_spec, runner_batches, synthesize_rows,
)
from proofs.first_sentence import DAYS, SEED, _SphereCodec, _pin_rngs  # noqa: E402
from umwelt.boot import build_engine  # noqa: E402
from umwelt.learning.context import ContextState  # noqa: E402
from umwelt.learning.runner import BrainRunner  # noqa: E402
from umwelt.projection.frame import FORMAT, frame, validate  # noqa: E402

RECEIPT = HERE / "receipts" / "frame_export.json"
SUBJECT = "gridworld"
STALE_AFTER_S = 3600


def _census(doc: dict) -> dict[str, int]:
    c = Counter(n["state"] for n in doc["nodes"])
    return {"current": c.get("current", 0), "stale": c.get("stale", 0), "dark": c.get("dark", 0),
            "n": len(doc["nodes"])}


def _export(engine) -> tuple[dict, str, str]:
    """Export one Frame and return it with the field hash before and after."""
    h0 = engine.field_canon_hash()
    doc = frame(engine, subject=SUBJECT, stale_after_s=STALE_AFTER_S)
    h1 = engine.field_canon_hash()
    return doc, h0, h1


def run() -> dict:
    _pin_rngs()
    spec = gridworld_spec()
    engine = build_engine(spec=spec, population=False,
                          role=ContextState.replay(dt_factor=10.0))
    engine.register_anchor("geo", codec=_SphereCodec())
    engine.delocate_anchor("geo")

    # ── before: a blank engine has measured nothing ──
    f0, h0a, h0b = _export(engine)
    v0 = validate(f0)
    c0 = _census(f0)

    # ── one synthetic day through the production ingest path ──
    engine.ground_anchor("geo", SITE, alpha=1.0)
    walk = agent_walk(seed=SEED, days=DAYS)
    rows = synthesize_rows(spec, walk, seed=SEED)
    n_batches = BrainRunner(engine).replay(runner_batches(rows, flush_secs=30.0))

    # ── after ──
    f1, h1a, h1b = _export(engine)
    v1 = validate(f1)
    c1 = _census(f1)

    # ── assertions ──
    assert f0["schema"] == f1["schema"] == FORMAT
    assert v0 == [], f"blank frame invalid: {v0[:5]}"
    assert v1 == [], f"post-replay frame invalid: {v1[:5]}"
    assert h0a == h0b, "export mutated the field (blank)"
    assert h1a == h1b, "export mutated the field (after replay)"
    assert h0a != h1a, "the day left the field unchanged — replay did nothing"
    assert c0["n"] == c1["n"] > 0, f"node count changed across the day: {c0} → {c1}"
    assert c0["dark"] == c0["n"], f"a blank engine exported a non-dark node: {c0}"
    assert all(n["value"] is None and n["age_s"] is None for n in f0["nodes"]), \
        "a blank engine wrote a number or an age"
    assert c1["dark"] < c0["dark"], f"dark count did not fall: {c0} → {c1}"
    assert c1["current"] >= 1, f"no node is current after a day of signal: {c1}"
    own_ts = sum(1 for n in f1["nodes"] if n["state"] != "dark" and not n["note"])
    assert own_ts >= 1, "no live node carried its own timestamp"

    sentence = (
        f"A blank engine exported a {FORMAT} with {c0['n']} nodes: {c0['dark']} dark, "
        f"{c0['current']} current, {c0['stale']} stale; after one synthetic day "
        f"({n_batches} ingest batches) it exported {c1['dark']} dark, {c1['current']} current, "
        f"{c1['stale']} stale (threshold {STALE_AFTER_S}s), and the field hash was unchanged by both exports."
    )

    return {
        "schema": "umwelt-proof-receipt/0.1",
        "proof": "frame_export",
        "sentence": sentence,
        "frame_schema": FORMAT,
        "subject": SUBJECT,
        "stale_after_s": STALE_AFTER_S,
        "before": {"census": c0, "violations": v0, "field_canon_hash": h0a,
                   "hash_unchanged_by_export": h0a == h0b, "as_of": f0["as_of"]},
        "after": {"census": c1, "violations": v1, "field_canon_hash": h1a,
                  "hash_unchanged_by_export": h1a == h1b, "as_of": f1["as_of"],
                  "nodes_with_own_timestamp": own_ts,
                  "nodes_with_fallback_age": sum(1 for n in f1["nodes"]
                                                 if n["state"] != "dark" and n["note"])},
        "age_source": "SensorBridge.latest_value(node, role) → _last_raw_ts[sensor_id] "
                      "(wall clock at ingest); null when the axis has no recorded reading",
        "determinism": {"seed": SEED, "days": DAYS, "rng_pin": 1234,
                        "wall_clock": "ages only; the field and census are seeded"},
        "read_only_on_rho": True,
        "builds_on": "proofs/first_sentence.py",
        "sample_nodes": {"before": f0["nodes"][:2], "after": f1["nodes"][:2]},
    }


def test_frame_export() -> None:
    receipt = run()
    assert receipt["before"]["violations"] == [] and receipt["after"]["violations"] == []
    assert receipt["after"]["census"]["dark"] < receipt["before"]["census"]["dark"]
    assert receipt["after"]["census"]["current"] >= 1
    assert receipt["before"]["hash_unchanged_by_export"] and receipt["after"]["hash_unchanged_by_export"]


if __name__ == "__main__":
    receipt = run()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["sentence"])
    print(f"receipt: {RECEIPT.relative_to(ROOT)}")
