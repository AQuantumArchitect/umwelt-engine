"""THE FIRST SENTENCE — one measured number through each of the three coupling maps.

The quantum-strand vocabulary around this engine (QAGIS / Quogniton) was ruled fog by the
S3 intake hopper: "flavor until a demo touches a sentence." This proof is that sentence.
It adds no concept. It reads the engine the way docs/COUPLING_MAP.md says the three
coupling maps already exist in code, and prints what each one measurably did when a blank
engine lived one deterministic synthetic day:

  κ  physical → quantum   BrainRunner.replay → engine.ingest → SensorBridge.process →
                          cluster.observe_qubit (the L0 α-blend; the PRODUCTION path).
  ε  quantum → cognitive  projection.nameplates, read-only on ρ: every declared axis is
                          latent / live / dark.
  χ  cognitive → physical membranes.egress.OutputSurface.route, shadow-first: tendrils
                          decide visibly and the dispatcher is never called.

What the day actually did (and the lexicon's intuition it corrects)
--------------------------------------------------------------------
The naive sentence — "observation collapses belief, so entropy falls" — is FALSE here, and
this proof pins why. A blank engine's world qubits start PURE at a default pole (entropy
exactly 0): a definite-but-meaningless point, the same false certainty blank_slate.py
delocates on the anchor. Over the day three different things happen, and a single total
would hide all of them:

  1. the WORLD differentiates: ten `agent_near` beliefs that began as ONE identical point
     end as many distinct points spread across the Bloch range. Comprehension shows up
     as distinguishability between cells, not as purity of any one cell;
  2. the learnable FIBER purifies: the `_param_*` qubits (per-cell decay rates, gains,
     thresholds) lose entropy — they moved off shared priors onto learned values;
  3. the world's total entropy RISES: the engine trades a pole it never earned for honest
     uncertainty (the dissipative `resource` role relaxes toward mixed) plus real structure.

Entropy here is Σ of single-qubit von Neumann entropies in bits (information_geometry.
qubit_entropy_bits), deliberately plain and hand-checkable; correlations are the job of
mutual_information.field_mi_scan and are not claimed. Receipt: proofs/receipts/
first_sentence.json. Seeded, offline, numpy only, no wall clock.

Run:  python3 proofs/first_sentence.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.gridworld.world import (  # noqa: E402
    SITE, agent_walk, grid_cells, gridworld_spec, runner_batches, synthesize_rows,
)
from umwelt.boot import build_engine  # noqa: E402
from umwelt.learning.context import ContextState  # noqa: E402
from umwelt.learning.runner import BrainRunner  # noqa: E402
from umwelt.projection.nameplates import nameplates  # noqa: E402
from umwelt.substrate.bloch import bloch_to_location, location_to_bloch  # noqa: E402
from umwelt.substrate.information_geometry import qubit_entropy_bits  # noqa: E402

RECEIPT = HERE / "receipts" / "first_sentence.json"
SEED = 7
DAYS = 1.0
WORLD_ROLE = "agent_near"          # the unitary, event-driven role the walk is about


class _SphereCodec:
    """Anchor codec registered by the proof, as blank_slate.py does — the engine ships
    no coordinate system of its own."""

    def encode(self, value):
        return location_to_bloch(*value)

    def decode(self, bloch):
        return bloch_to_location(*bloch)


def _pin_rngs() -> None:
    import random
    import numpy as np
    random.seed(1234)
    np.random.seed(1234)


def _is_fiber(role: str) -> bool:
    return role.startswith("_param_")


def entropy_split(engine) -> dict:
    """Σ H(qubit) in bits, split into the learnable fiber (`_param_*` roles) and the
    world (everything else). A read through the substrate's own gauge; ρ is not mutated."""
    out = {"fiber_bits": 0.0, "fiber_qubits": 0, "world_bits": 0.0, "world_qubits": 0}
    for cluster in engine.field.clusters.values():
        for role in getattr(cluster, "role_index", {}):
            b = cluster.role_bloch(role)
            if not all(math.isfinite(float(v)) for v in b):
                raise AssertionError(f"non-finite Bloch vector on {role}")
            h = qubit_entropy_bits(b)
            k = "fiber" if _is_fiber(role) else "world"
            out[f"{k}_bits"] += h
            out[f"{k}_qubits"] += 1
    out["total_bits"] = out["fiber_bits"] + out["world_bits"]
    return out


def world_differentiation(engine, role: str = WORLD_ROLE) -> dict:
    """How many distinct belief points the cells' `role` qubits occupy, and the spread of
    their z. One point / zero spread is the blank floor; comprehension is the cells coming
    apart."""
    pts, zs = set(), []
    for cname, cluster in engine.field.clusters.items():
        if role not in getattr(cluster, "role_index", {}):
            continue
        b = cluster.role_bloch(role)
        pts.add(tuple(round(float(v), 6) for v in b))
        zs.append(float(b[2]))
    return {"role": role, "qubits": len(zs), "distinct_points": len(pts),
            "z_spread": (max(zs) - min(zs)) if zs else 0.0,
            "pure_at_floor": all(abs(abs(z) - 1.0) < 1e-9 for z in zs)}


def axis_census(engine) -> dict[str, int]:
    """ε: the projection's own latent / live / dark classification of every axis."""
    census = {"latent": 0, "live": 0, "dark": 0}
    for row in nameplates(engine).get("axes", []):
        s = str(row.get("status", ""))
        if s in census:
            census[s] += 1
    return census


def _dispatched(engine) -> int:
    return sum(1 for _ts, _a, live in engine.output_surface.history if live)


def run() -> dict:
    _pin_rngs()
    spec = gridworld_spec()
    engine = build_engine(spec=spec, population=False,
                          role=ContextState.replay(dt_factor=10.0))
    engine.register_anchor("geo", codec=_SphereCodec())
    engine.delocate_anchor("geo")

    # ── before: the blank floor, read not asserted ──
    e0 = entropy_split(engine)
    d0 = world_differentiation(engine)
    c0 = axis_census(engine)
    x0 = _dispatched(engine)

    # ── κ: one synthetic day through the production ingest path ──
    engine.ground_anchor("geo", SITE, alpha=1.0)
    walk = agent_walk(seed=SEED, days=DAYS)
    rows = synthesize_rows(spec, walk, seed=SEED)
    n_batches = BrainRunner(engine).replay(runner_batches(rows, flush_secs=30.0))

    # ── after ──
    e1 = entropy_split(engine)
    d1 = world_differentiation(engine)
    c1 = axis_census(engine)
    x1 = _dispatched(engine)
    recommended = len(engine.output_surface.recommendations)

    # ── the sentence, as assertions ──
    assert n_batches > 150, f"suspiciously few batches ({n_batches})"
    assert (e0["fiber_qubits"], e0["world_qubits"]) == (e1["fiber_qubits"], e1["world_qubits"]), \
        "the field grew or shrank during replay"
    # (0) the blank floor is a pure pole, not maximal ignorance — the false certainty.
    assert d0["distinct_points"] == 1 and d0["pure_at_floor"], f"blank floor not one pure pole: {d0}"
    # (1) the world differentiates: cells come apart.
    n_cells = len(grid_cells())
    assert d1["distinct_points"] >= n_cells - 1, f"cells did not come apart: {d1}"
    assert d1["z_spread"] > 1.0, f"z spread too small to call it comprehension: {d1}"
    # (2) the learnable fiber purifies: learning is entropy leaving the fiber.
    assert e1["fiber_bits"] < e0["fiber_bits"], \
        f"fiber did not learn: {e0['fiber_bits']:.3f} → {e1['fiber_bits']:.3f} bits"
    # (3) the world's entropy rises: the pole it never earned is given back.
    assert e1["world_bits"] > e0["world_bits"], \
        f"world entropy did not rise off the blank pole: {e0['world_bits']:.3f} → {e1['world_bits']:.3f}"
    # (ε) axes that were latent are live after a day of signal.
    assert c1["live"] > c0["live"] and c1["latent"] < c0["latent"], f"projection saw no signal: {c0} → {c1}"
    # (χ) shadow law: a world the engine has never seen dispatches nothing.
    assert x0 == x1 == 0, f"shadow law broken: {x1} dispatched"

    sentence = (
        f"A blank engine replayed one synthetic day ({n_batches} ingest batches, {len(rows)} events). "
        f"Its {d1['qubits']} {WORLD_ROLE} beliefs began as one pure pole and ended as "
        f"{d1['distinct_points']} distinct points spanning {d1['z_spread']:.2f} of the 2.00 Bloch range; "
        f"its learnable fiber lost {e0['fiber_bits'] - e1['fiber_bits']:.2f} bits of entropy over "
        f"{e1['fiber_qubits']} qubits (it learned) while its world beliefs gained "
        f"{e1['world_bits'] - e0['world_bits']:.2f} bits over {e1['world_qubits']} qubits (it gave back a "
        f"certainty it never earned); {c1['live']} of {sum(c1.values())} declared axes went live; "
        f"0 actions were dispatched."
    )

    return {
        "schema": "umwelt-proof-receipt/0.1",
        "proof": "first_sentence",
        "sentence": sentence,
        "corrects": "the intuition that observation lowers belief entropy; in an open belief field "
                    "comprehension is differentiation of the world plus purification of the fiber, "
                    "and total entropy may rise",
        "couplings": {
            "kappa_physical_to_quantum": {
                "path": "BrainRunner.replay → engine.ingest → SensorBridge.process → cluster.observe_qubit",
                "estimator": "L0 alpha-blend (default; L4 Belavkin OFF)",
                "events": len(rows), "batches": n_batches,
                "entropy_bits_before": {k: round(v, 6) if isinstance(v, float) else v for k, v in e0.items()},
                "entropy_bits_after": {k: round(v, 6) if isinstance(v, float) else v for k, v in e1.items()},
                "world_differentiation_before": {**d0, "z_spread": round(d0["z_spread"], 6)},
                "world_differentiation_after": {**d1, "z_spread": round(d1["z_spread"], 6)},
            },
            "epsilon_quantum_to_cognitive": {
                "path": "projection.nameplates (read-only on rho)",
                "axes_before": c0, "axes_after": c1,
            },
            "chi_cognitive_to_physical": {
                "path": "membranes.egress.OutputSurface.route (shadow-first)",
                "dispatched": x1, "recommended_pending": recommended,
            },
        },
        "ledger": "events are the truth; the field is a cache of them (umweltd/worldstore.py)",
        "determinism": {"seed": SEED, "days": DAYS, "rng_pin": 1234, "wall_clock": False},
        "eta_note": "a proof is observation-grade on its own outputs and nothing else; "
                    "it raises no claim about any product",
        "earning_usd": None,
    }


def test_first_sentence() -> None:
    receipt = run()
    k = receipt["couplings"]["kappa_physical_to_quantum"]
    assert k["entropy_bits_after"]["fiber_bits"] < k["entropy_bits_before"]["fiber_bits"]
    assert k["world_differentiation_after"]["distinct_points"] > 1
    assert receipt["couplings"]["chi_cognitive_to_physical"]["dispatched"] == 0


if __name__ == "__main__":
    receipt = run()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["sentence"])
    print(f"receipt: {RECEIPT.relative_to(ROOT)}")
