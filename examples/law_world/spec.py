"""examples/law_world/spec — the hive constitution (docs/LAW.md, `law/1`) as a DomainSpec.

The law's own clause V says: "The constitution is a DomainSpec." This module is that
sentence made literal. It is READ-ONLY on the constitution: it loads `law.spec.json`
(a verbatim copy of the-broadcast's docs/law.spec.json; the receipt records its sha256
so drift is visible) and builds a world from it. It adds no number of its own — every
value below is read from the json or is a chart derived from one.

What maps to what
-----------------
  one node `hive`                 — the world is one seat that believes things about itself
                                    (under a role-less root `constitution`, engine furniture).

  parameter p  →  `_param_hive_<id>` fiber qubit
      NodeSpec.params[id] = (prior, width, lo, hi). At boot the engine binds every
      bounded, non-frozen spec param onto a QubitBackedParam (engine._bind_param_fiber);
      its value is the prior exactly, `width` is kept as the fiber's prior_sigma (a width
      of 0 — the caps — is recorded as 0, honestly: the engine has no "frozen" field on
      NodeSpec.params, so WHO may move a cap is a policy-layer rule, see below).
      Chart (lo, hi): unit "p" → [0, 1] ([-1, 1] when the label says "offset");
      any other unit → [0, 2·prior] so the prior sits at the chart centre z=0. The
      chart is a frame of reference, not a law value. The fiber clamps to it.

  parameter p  →  belief role `<id>` on `hive` (the Frame's face of the same number)
      frame/1 is drawn from graph-node roles (projection.nameplates), and fiber qubits
      are not graph nodes, so the engine cannot yet show a `_param_*` qubit in a Frame
      without an engine change. Each parameter therefore also carries a unitary belief
      role on `hive`, bound to the sensor `prior_<id>` with the SAME chart as the fiber
      and collapse_alpha=1 (a sit is a hard set — the posture-flip idiom of
      learning/actuator_autonomy). The proof asserts the role's z and the fiber's value
      agree. Retiring this mirror needs a Frame projection of the fiber — engine work,
      not done here.

  egress class c  →  OutputSpec `<id>` (egress.record, …) with shadow=True, reading `hive.p_yes_<id>`
      The committed belief rests at the prior (coupling.rest_z = 2·p_yes − 1) and is
      pumped by the role — the engine's two-timescale tendril geometry. Every class
      starts SHADOW (the law of III and the engine's default coincide): it decides
      visibly and dispatches nothing. Flipping a class to auto is the app's call via
      the cutover plane (learning/actuator_autonomy), never this spec's.

  sit on class c  →  observation `sit_<id>` ∈ {yes: +1, no: -1} on `hive.p_yes_<id>`
      One binding per class, force_observe=True, collapse_alpha = 1/(n0+1) where
      n0 = p_yes(1−p_yes)/p_width² − 1 is the Beta pseudo-count moment-matched to the
      class's (p_yes, p_width). So one yes-sit at the prior moves the belief exactly as
      a Beta α+=1 would. A pinned class (held, p_width 0) gets collapse_alpha = 0:
      η=0 is an engine-law no-op, so sits cannot move `egress.money` until Luke unpins
      it by re-sitting the constitution (a new spec).

Bloch chart. Every binding uses the engine's own `range` normalizer over its chart, so a
sit or a prior lands the belief on z = r_obs·(2(v−lo)/(hi−lo) − 1): the fiber's chart
(substrate.bloch.value_to_bloch_z) scaled by the bridge's observation magnitude
r_obs = 0.95 (SensorBridge.register default; BindingSpec does not expose it). A spec-forced
observe binding on a regular role keeps its normalizer's direction (engine.ingest undoes the
device-echo flip), so no domain normalizer is needed. Read back: p_yes = (z/r_obs + 1)/2.
Pole faces: +z = "yes" / upper bound, −z = "no" / lower bound.

Where the yes/no asymmetry lives. The law says a no adds three to β, a yes adds one to α.
A binding carries ONE collapse_alpha and the engine's α-blend is symmetric in it, so the
engine moves a yes and a no by the same gain. The 3× lives in the POLICY LAYER (the app
that delivers sits — it may send a no three times, or scale each yes's η by 1/3), not in
this spec and not in the engine. proofs/law_world.py measures and pins this.

Invariants are not spec fields. proofs/law_world.py checks these as assertions:
  inv.null    a blank boot exports every node null and dark (never zero);
  inv.age     every exported value carries age_s; current only with its own timestamp;
  inv.no_promote  frame.validate() re-derives every state and finds no violation;
  inv.claims  an export (a read) leaves the field hash unchanged; only sits move belief;
  inv.earned  every class is shadow; OutputSurface dispatches 0; egress.money is pinned;
  inv.ratify  the json's status rides along in the receipt — nothing here stands.
The other six (dry, invent, ledger, one_writer, depend, observations) are enforced where
`law.spec.json → invariants[].enforced_by` says, not by this world.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from umwelt.spec.schema import AxisFace, BindingSpec, DomainSpec, NodeSpec, OutputSpec

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "law.spec.json"       # verbatim copy of the-broadcast docs/law.spec.json
ROOT = "constitution"                        # role-less root; engine DNA params live here
NODE = "hive"
R_OBS = 0.95                                 # SensorBridge.register default; not a law value


def load_law(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_PATH
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != "law/1":
        raise ValueError(f"{p} is not a law/1 document (schema={doc.get('schema')!r})")
    doc["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    doc["_path"] = str(p)
    return doc


def chart(param: dict) -> tuple[float, float]:
    """(lo, hi) for a parameter — a frame of reference derived from its unit and prior."""
    if param["unit"] == "p":
        return (-1.0, 1.0) if "offset" in param["label"] else (0.0, 1.0)
    return 0.0, 2.0 * float(param["prior"])


def sit_alpha(cls: dict) -> float:
    """1/(n0+1): the Beta-moment-matched gain of one yes-sit at the prior; 0 when pinned."""
    p, w = float(cls["p_yes"]), float(cls["p_width"])
    if cls.get("held") or w <= 0.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    return w * w / (p * (1.0 - p))


def to_z(value: float, lo: float, hi: float) -> float:
    """Where a chart value lands on the belief after one full-α observation."""
    return R_OBS * (2.0 * (float(value) - lo) / (hi - lo) - 1.0)


def from_z(z: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * (float(z) / R_OBS + 1.0) / 2.0


def prior_readings(law: dict) -> dict[str, float]:
    """The boot sit: every parameter's and class's prior on its `prior_<id>` channel."""
    out = {f"prior_{p['id']}": float(p["prior"]) for p in law["parameters"]}
    out.update({f"prior_{c['id']}": float(c["p_yes"]) for c in law["classes"]})
    return out


def law_spec(path: str | Path | None = None) -> DomainSpec:
    """Build the hive world from a law/1 document. Read-only on the constitution."""
    law = load_law(path)
    params, roles, modes, faces, bindings, outputs = {}, [], {}, [], [], []

    # Hold still between sits: the law's `later` adds nothing, so a belief must not relax
    # toward a pole in silence. Same prior the engine gives its own `_params` memory cells
    # (substrate/param_bundles.py): amplitude damping 0, clamped to ≤0.001.
    params["gamma"] = (0.0, 0.0, 0.0, 0.001)
    for p in law["parameters"]:
        lo, hi = chart(p)
        params[p["id"]] = (float(p["prior"]), float(p["width"]), lo, hi)
        roles.append(p["id"]); modes[p["id"]] = "unitary"
        faces.append(AxisFace(role=p["id"], axis_id=f"law.{p['id']}",
                              pole_0=f"{hi:g} {p['unit']}", pole_1=f"{lo:g} {p['unit']}"))
        bindings.append(BindingSpec(f"prior_{p['id']}", zone=NODE, role=p["id"],
                                    normalizer={"type": "range", "lo": lo, "hi": hi},
                                    force_observe=True, collapse_alpha=1.0,
                                    description=p["label"]))

    for c in law["classes"]:
        role = f"p_yes_{c['id']}"
        roles.append(role); modes[role] = "unitary"
        faces.append(AxisFace(role=role, axis_id=f"law.{c['id']}", pole_0="yes", pole_1="no"))
        bindings.append(BindingSpec(f"prior_{c['id']}", zone=NODE, role=role,
                                    normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
                                    force_observe=True, collapse_alpha=1.0,
                                    description=f"prior p_yes: {c['label']}"))
        bindings.append(BindingSpec(f"sit_{c['id']}", zone=NODE, role=role,
                                    normalizer={"type": "range", "lo": -1.0, "hi": 1.0},
                                    force_observe=True, collapse_alpha=sit_alpha(c),
                                    description=f"Luke's sit, yes:+1 no:-1: {c['label']}"))
        outputs.append(OutputSpec(
            c["id"], node=NODE, role=role, kind="binary", decode="sticky",
            coupling={"rest_z": 2.0 * float(c["p_yes"]) - 1.0},
            gates={"rate_limit_s": 0.0},
            dispatch={"actuator_id": c["id"], "undo": c["undo"],
                      "cost_wrong_min": c["cost_wrong_min"], "rate_per_day": c["rate_per_day"],
                      "held": bool(c["held"])},
            shadow=True,                     # the law: every class starts shadow
        ))

    return DomainSpec(
        name=f"law-world:{law['schema']}:{law['status']}",
        nodes=(
            # A role-less root (the contraption / manifold_game idiom): engine DNA params
            # attach here, so the hive's own `gamma` below is not shadowed by the root's.
            NodeSpec(ROOT, parent=None, kind="root", roles=()),
            NodeSpec(NODE, parent=ROOT, kind="entity", roles=tuple(roles),
                     cluster_backend="cumulant",       # 31 roles: no 2^n density matrix
                     role_modes=modes, params=params, faces=tuple(faces)),
        ),
        bindings=tuple(bindings),
        outputs=tuple(outputs),
        extensions={"law_sha256": law["_sha256"], "law_path": law["_path"],
                    "law_status": law["status"], "ratified_by": law["ratified_by"]},
    )
