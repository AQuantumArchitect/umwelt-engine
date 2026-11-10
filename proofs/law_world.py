"""THE CONSTITUTION AS A WORLD — docs/LAW.md clause V, run.

"The constitution is a DomainSpec. The hive world in Umwelt declares the parameters as
learnable `_param_*` fibers with these priors, the classes as output specs that start
shadow, and the invariants as assertions. The law is literally a world the engine runs;
its Frame shows what the system currently believes its own thresholds are, and how sure."

This proof boots examples/law_world/spec.law_spec() blank through the host face and
checks that sentence clause by clause:

  1. BLANK  — a fresh boot exports a frame/1 (subject "hive-law") in which every node is
     null and dark (inv.null): nothing has been sat yet. Every constitution parameter is
     already a QubitBackedParam `_param_hive_<id>` whose value is its prior and whose
     prior_sigma is its width; every egress class is a shadow tendril.
  2. THE BOOT SIT — the priors enter as observations on `prior_<id>` (collapse_alpha 1).
     The Frame now shows one current node per parameter and per class, age 0, value =
     the prior's Bloch chart z = r_obs·(2(v−lo)/(hi−lo) − 1); the role and the fiber agree.
     `egress.money` is shadow and its belief sits at the floor (p_yes 0 → z = −r_obs).
  3. THE SIT STREAM — 20 yes-sits on egress.record, 3 no-sits on egress.outreach, and 1
     yes-sit on egress.money (to push on the pin). record's belief rose, outreach's fell,
     money's did not move at all (η=0 is an engine-law no-op). The measured gain of a yes
     and of a no each equal the class binding's single collapse_alpha: the engine is
     symmetric in sign, so the law's "a no adds three" asymmetry is NOT in the engine —
     it is a policy-layer rule (how sits are delivered), and this proof says so.
  4. SHADOW LAW — OutputSurface dispatched 0; host.live_dispatches is empty.
  5. READ-ONLY — engine.field_canon_hash() is identical before and after every export.

Receipt: proofs/receipts/law_world.json. Seeded, offline, numpy only; wall clock only for
the Frame's ages (expected 0 s, asserted ≤ 1 s).

Run:  python3 proofs/law_world.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.law_world.spec import (  # noqa: E402
    NODE, R_OBS, chart, from_z, law_spec, load_law, prior_readings, sit_alpha, to_z,
)
from proofs.first_sentence import _pin_rngs  # noqa: E402
from umwelt.host.api import GameHost  # noqa: E402
from umwelt.projection.frame import FORMAT, frame, validate  # noqa: E402
from umwelt.substrate.qubit_param import QubitBackedParam  # noqa: E402

RECEIPT = HERE / "receipts" / "law_world.json"
SUBJECT = "hive-law"
STALE_AFTER_S = 3600
T0 = datetime(2026, 10, 17, 3, 0, tzinfo=timezone.utc)      # the constitution's as_of
N_YES_RECORD, N_NO_OUTREACH, N_YES_MONEY = 20, 3, 1
RECORD, OUTREACH, MONEY = "egress.record", "egress.outreach", "egress.money"


def _export(engine) -> tuple[dict, bool]:
    h0 = engine.field_canon_hash()
    doc = frame(engine, subject=SUBJECT, stale_after_s=STALE_AFTER_S)
    return doc, engine.field_canon_hash() == h0


def _census(doc: dict) -> dict[str, int]:
    out = {"current": 0, "stale": 0, "dark": 0}
    for n in doc["nodes"]:
        out[n["state"]] += 1
    out["n"] = len(doc["nodes"])
    return out


def _z(engine, role: str) -> float:
    return float(engine.field.clusters[NODE].role_bloch(role)[2])


def _p(engine, cls_id: str) -> float:
    return from_z(_z(engine, f"p_yes_{cls_id}"), 0.0, 1.0)


def _dispatched(engine) -> int:
    return sum(1 for _ts, _a, live in engine.output_surface.history if live)


def run() -> dict:
    _pin_rngs()
    law = load_law()
    spec = law_spec()
    host = GameHost().register_world(spec)
    eng = host.engine
    params, classes = law["parameters"], law["classes"]
    by_class = {c["id"]: c for c in classes}

    # ── 1. blank ──
    f0, ro0 = _export(eng)
    v0, c0 = validate(f0), _census(f0)
    assert f0["schema"] == FORMAT and v0 == [], f"blank frame invalid: {v0[:5]}"
    assert ro0, "export mutated the field (blank)"
    assert c0["n"] == len(params) + len(classes) == 31, c0
    assert c0["dark"] == c0["n"], f"a blank boot exported a non-dark node: {c0}"
    assert all(n["value"] is None and n["age_s"] is None for n in f0["nodes"]), \
        "a blank boot wrote a number or an age (inv.null)"

    hive = eng.graph.find(NODE)
    fiber: dict[str, dict] = {}
    for p in params:
        qp = hive.param_bundle.get_param(p["id"])
        assert isinstance(qp, QubitBackedParam), f"{p['id']} is not on the fiber: {type(qp).__name__}"
        role = f"_param_{NODE}_{p['id']}"
        assert role in getattr(qp.cluster, "role_index", {}), f"{role} missing from {qp.cluster.zone_name}"
        assert abs(qp.value - float(p["prior"])) < 1e-6, f"{p['id']} booted at {qp.value}, prior {p['prior']}"
        assert qp.prior_sigma == float(p["width"]), f"{p['id']} width {qp.prior_sigma} != {p['width']}"
        lo, hi = chart(p)
        assert (qp.lo, qp.hi) == (lo, hi)
        fiber[p["id"]] = {"fiber_role": role, "cluster": qp.cluster.zone_name, "value": qp.value,
                          "prior": p["prior"], "width": p["width"], "lo": lo, "hi": hi,
                          "self_serving": p["self_serving"]}
    tendrils = {t.name: t for t in eng.tendrils}
    assert set(tendrils) == set(by_class), (sorted(tendrils), sorted(by_class))
    assert all(t.shadow for t in tendrils.values()), "a class booted out of shadow"

    # ── 2. the boot sit: priors as observations at T0 ──
    res = host.observe_many("luke", prior_readings(law), t=T0)
    assert res["accepted"]
    f1, ro1 = _export(eng)
    v1, c1 = validate(f1), _census(f1)
    assert v1 == [], f"post-sit frame invalid: {v1[:5]}"
    assert ro1, "export mutated the field (after the boot sit)"
    assert c1["current"] == c1["n"] == c0["n"], f"not every node is current after the boot sit: {c1}"
    max_age = max(n["age_s"] for n in f1["nodes"])
    assert max_age <= 1, f"a just-sat node is already {max_age}s old"
    nodes_by_id = {n["id"]: n for n in f1["nodes"]}
    expected: dict[str, float] = {}
    for p in params:
        lo, hi = chart(p)
        expected[f"law.{p['id']}"] = to_z(p["prior"], lo, hi)
    for c in classes:
        expected[f"law.{c['id']}"] = to_z(c["p_yes"], 0.0, 1.0)
    for axis_id, want in expected.items():
        n = nodes_by_id[axis_id.replace(".", "_")]
        assert n["state"] == "current" and n["value"] is not None, n
        assert abs(float(n["value"]) - want) <= 1e-4, f"{axis_id}: frame {n['value']} vs chart {want}"
    for p in params:                      # the Frame's face and the fiber agree
        lo, hi = chart(p)
        assert abs(from_z(_z(eng, p["id"]), lo, hi) - fiber[p["id"]]["value"]) <= 1e-3 * (hi - lo), p["id"]
    z_floor = to_z(0.0, 0.0, 1.0)
    assert tendrils[MONEY].shadow and by_class[MONEY]["held"]
    assert abs(_z(eng, f"p_yes_{MONEY}") - z_floor) < 1e-9, "egress.money is not at the floor"
    assert sit_alpha(by_class[MONEY]) == 0.0, "egress.money's sit binding is not pinned (η must be 0)"

    # ── 3. the sit stream ──
    p_rec0, p_out0, z_mon0 = _p(eng, RECORD), _p(eng, OUTREACH), _z(eng, f"p_yes_{MONEY}")
    t = T0
    zr = [_z(eng, f"p_yes_{RECORD}")]
    for _ in range(N_YES_RECORD):
        t += timedelta(minutes=1)
        host.observe("luke", f"sit_{RECORD}", +1.0, t=t)
        zr.append(_z(eng, f"p_yes_{RECORD}"))
    zo = [_z(eng, f"p_yes_{OUTREACH}")]
    for _ in range(N_NO_OUTREACH):
        t += timedelta(minutes=1)
        host.observe("luke", f"sit_{OUTREACH}", -1.0, t=t)
        zo.append(_z(eng, f"p_yes_{OUTREACH}"))
    for _ in range(N_YES_MONEY):
        t += timedelta(minutes=1)
        host.observe("luke", f"sit_{MONEY}", +1.0, t=t)
    p_rec1, p_out1, z_mon1 = _p(eng, RECORD), _p(eng, OUTREACH), _z(eng, f"p_yes_{MONEY}")

    assert p_rec1 > p_rec0, f"20 yes-sits did not raise egress.record: {p_rec0} → {p_rec1}"
    assert p_out1 < p_out0, f"3 no-sits did not lower egress.outreach: {p_out0} → {p_out1}"
    assert z_mon1 == z_mon0 == z_floor, f"a yes-sit moved the pinned egress.money: {z_mon0} → {z_mon1}"
    assert all(b > a for a, b in zip(zr, zr[1:])), "record did not rise monotonically"
    assert all(b < a for a, b in zip(zo, zo[1:])), "outreach did not fall monotonically"
    # The engine's gain per sit: Δz over the distance to the observed pole. One α per binding.
    gain_yes = (zr[1] - zr[0]) / (R_OBS - zr[0])
    gain_no = (zo[1] - zo[0]) / (-R_OBS - zo[0])
    a_rec, a_out = sit_alpha(by_class[RECORD]), sit_alpha(by_class[OUTREACH])
    assert abs(gain_yes - a_rec) < 1e-6, f"yes gain {gain_yes} != binding α {a_rec}"
    assert abs(gain_no - a_out) < 1e-6, f"no gain {gain_no} != binding α {a_out}"
    # Symmetric in sign: a no is moved by the same α a yes would be on that binding. The law's
    # 'a no adds three to β' is therefore NOT in the engine; it is the policy layer's to deliver.
    assert abs(gain_yes / a_rec - gain_no / a_out) < 1e-6, "the engine applied a sign-dependent gain"
    asymmetry_in_engine = False

    # ── 4. shadow law ──
    dispatched = _dispatched(eng)
    assert dispatched == 0 and host.live_dispatches == [], f"shadow law broken: {dispatched} dispatched"
    assert all(t.shadow for t in eng.tendrils)

    # ── 5. read-only export, once more, after the stream ──
    f2, ro2 = _export(eng)
    v2 = validate(f2)
    assert v2 == [] and ro2, f"final frame: violations={v2[:3]} read_only={ro2}"

    sentence = (
        f"The constitution booted as a world: {len(params)} parameters at their priors, "
        f"{len(classes)} egress classes in shadow; {N_YES_RECORD} yes-sits moved {RECORD} belief from "
        f"{p_rec0:.3f} to {p_rec1:.3f}; {N_NO_OUTREACH} no-sits moved {OUTREACH} from "
        f"{p_out0:.3f} to {p_out1:.3f}; {dispatched} dispatched."
    )
    return {
        "schema": "umwelt-proof-receipt/0.1",
        "proof": "law_world",
        "sentence": sentence,
        "law": {"schema": law["schema"], "status": law["status"], "ratified_by": law["ratified_by"],
                "as_of": law["as_of"], "sha256": law["_sha256"], "path": str(Path(law["_path"]).relative_to(ROOT)),
                "canonical": "the-broadcast docs/law.spec.json (this copy must stay byte-identical)"},
        "frame": {"schema": FORMAT, "subject": SUBJECT, "stale_after_s": STALE_AFTER_S,
                  "blank": {"census": c0, "violations": v0, "read_only": ro0},
                  "after_boot_sit": {"census": c1, "violations": v1, "read_only": ro1, "max_age_s": max_age},
                  "after_stream": {"census": _census(f2), "violations": v2, "read_only": ro2}},
        "parameters": fiber,
        "classes": {c["id"]: {"p_yes_prior": c["p_yes"], "p_width": c["p_width"], "held": c["held"],
                              "shadow": tendrils[c["id"]].shadow, "sit_alpha": sit_alpha(c),
                              "rest_z": tendrils[c["id"]].commit.rest_z,
                              "p_yes_now": _p(eng, c["id"])} for c in classes},
        "stream": {
            RECORD: {"sits": f"{N_YES_RECORD} x yes", "p_before": p_rec0, "p_after": p_rec1,
                     "z_series": [round(v, 6) for v in zr], "first_sit_dz": zr[1] - zr[0], "gain": gain_yes},
            OUTREACH: {"sits": f"{N_NO_OUTREACH} x no", "p_before": p_out0, "p_after": p_out1,
                       "z_series": [round(v, 6) for v in zo], "first_sit_dz": zo[1] - zo[0], "gain": gain_no},
            MONEY: {"sits": f"{N_YES_MONEY} x yes", "z_before": z_mon0, "z_after": z_mon1,
                    "pinned": "collapse_alpha 0 → η=0 → engine-law no-op"},
        },
        "asymmetry": {
            "in_engine": asymmetry_in_engine,
            "where": "policy layer",
            "note": "one collapse_alpha per class binding (1/(n0+1), the Beta pseudo-count of the "
                    "class's p_yes/p_width); the engine moves a yes and a no by that same gain. The "
                    "law's 'no adds three to β' is delivered by the app that sits (three readings, or "
                    "η scaled), not by this spec or the engine. The raw per-sit Δz differ between "
                    "record and outreach only through each class's own α and (pole − z) distance.",
        },
        "dispatched": dispatched,
        "mapping": {"parameter": "NodeSpec.params[id] → QubitBackedParam on _param_hive_<id> (+ a mirror role "
                                 "<id> on hive so frame/1 can show it)",
                    "class": "OutputSpec(<id>, hive.p_yes_<id>, shadow=True, coupling.rest_z = 2·p_yes − 1)",
                    "sit": "BindingSpec(sit_<id> → hive.p_yes_<id>, range[-1,1], force_observe, α = 1/(n0+1))",
                    "prior": "BindingSpec(prior_<id> → hive.<role>, range[lo,hi], force_observe, α = 1)"},
        "determinism": {"rng_pin": 1234, "t0": T0.isoformat(), "wall_clock": "frame ages only"},
        "read_only_on_rho": True,
        "builds_on": ["proofs/first_sentence.py", "proofs/frame_export.py"],
    }


def test_law_world() -> None:
    receipt = run()
    assert receipt["dispatched"] == 0
    assert receipt["stream"][RECORD]["p_after"] > receipt["stream"][RECORD]["p_before"]
    assert receipt["stream"][OUTREACH]["p_after"] < receipt["stream"][OUTREACH]["p_before"]
    assert receipt["stream"][MONEY]["z_after"] == receipt["stream"][MONEY]["z_before"]
    assert receipt["frame"]["blank"]["violations"] == [] and receipt["frame"]["after_stream"]["violations"] == []
    assert receipt["asymmetry"]["in_engine"] is False


if __name__ == "__main__":
    receipt = run()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["sentence"])
    print(f"receipt: {RECEIPT.relative_to(ROOT)}")
