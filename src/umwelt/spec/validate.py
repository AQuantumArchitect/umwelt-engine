"""The deterministic spec gate — every check a DomainSpec must pass before anyone runs it.

One reusable, domain-free validation path for ANY spec, packaging what was previously
scattered (and partly swallowed): `build_graph_from_spec` raises hard on topology, but
`apply_spec_bindings` is membrane-guarded — a binding with a typo'd node/role/normalizer
is skipped with a warning and the world boots without it. Fine for a running deployment,
fatal for authoring: a new spec needs those errors surfaced loudly, plus proof that every
declared binding actually drives the field (the blank-slate proof's "no dead vocabulary"
law, generalized here to arbitrary specs).

Library use:

    from umwelt.spec.validate import validate_spec
    report = validate_spec("my_module:SPEC")     # or a DomainSpec instance
    assert report.ok, report.summary()

CLI use (exit code 0 iff every check passes):

    python -m umwelt.spec.validate my_module:SPEC [--json] [--allow-live-outputs]

Caveat for repeated in-process calls: role/normalizer registries are process-global and
a domain's vocabulary module registers at import, so validating many specs (or the same
spec after editing its module) in ONE process can see stale or double registration.
Harnesses that iterate on a spec (umweltforge does) must run this gate in a fresh
subprocess per attempt — the CLI form exists for exactly that.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from umwelt.spec.schema import DomainSpec

# Fixed synthetic epoch — the gate must be deterministic (no wall clock, ever).
_EPOCH = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)

# The fallback probe ladder for normalizers the gate can't read declaratively
# (a callable, or a domain-registered type): a spread of magnitudes and signs
# that distinguishes most monotone / thresholded / cyclic edge functions.
_GENERIC_LADDER = (0.0, 1.0, -1.0, 0.5, 2.0, 10.0, 100.0, -10.0, 0.1, 1000.0)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False
    warning: bool = False       # ok=True advisory: never gates, always teaches


@dataclass
class ValidationReport:
    """What `validate_spec` found: one named check per gate step, in run order.

    `ok` is the verdict; `summary()` is the human read-out; `to_dict()` the JSON one."""
    spec_name: str
    checks: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every check that ran passed (skipped checks do not count)."""
        return all(c.ok for c in self.checks if not c.skipped)

    def failures(self) -> list:
        """The checks that ran and failed."""
        return [c for c in self.checks if not c.ok and not c.skipped]

    def to_dict(self) -> dict:
        """JSON-ready form: spec name, verdict, and every check."""
        return {
            "spec": self.spec_name,
            "ok": self.ok,
            "checks": [{"name": c.name, "ok": c.ok, "skipped": c.skipped,
                        "warning": c.warning, "detail": c.detail}
                       for c in self.checks],
        }

    def warnings(self) -> list:
        """Advisory checks: reported, never failing the verdict."""
        return [c for c in self.checks if c.warning]

    def summary(self) -> str:
        """One line per check — `ok`, `FAIL`, `warn` or `skip` — under a verdict line."""
        lines = [f"spec {self.spec_name!r}: {'OK' if self.ok else 'FAILED'}"]
        for c in self.checks:
            mark = ("skip" if c.skipped else "warn" if c.warning
                    else "ok " if c.ok else "FAIL")
            lines.append(f"  [{mark}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        return "\n".join(lines)


def _pin_rngs() -> None:
    # build_engine does not take a substrate seed; pin the process RNGs so collapse
    # sampling — and therefore this gate — is reproducible run to run (the
    # blank-slate proof's idiom).
    import random
    import numpy as np
    random.seed(1234)
    np.random.seed(1234)


def _declared_probes(cfg) -> "list[float] | None":
    """Probe candidates read off a DECLARATIVE normalizer config, when recognizable."""
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    if not isinstance(cfg, dict):
        return None                      # a callable — nothing declarative to read
    t = cfg.get("type")
    try:
        if t == "binary":
            return [0.0, 1.0, 0.0]
        if t == "forecast_zflip":
            return [-1.0, 1.0, 0.0]
        if t == "range":
            lo, hi = float(cfg["lo"]), float(cfg["hi"])
            return [lo, (lo + hi) / 2.0, hi]
        if t == "threshold":
            th = float(cfg["threshold"])
            return [th - 2.0, th + 2.0, th - 1.0]
        if t == "regime":
            c, w = float(cfg["center"]), float(cfg["width"])
            return [c - 2.0 * w, c + 2.0 * w, c]
        if t == "cyclic":
            period = float(cfg["period"])
            peak = float(cfg.get("peak", 0.0))
            return [peak, peak + period / 4.0, peak + period / 2.0]
    except (KeyError, TypeError, ValueError):
        return None                      # malformed params — bindings_strict reports it
    # Unknown (domain-registered) type: the generic ladder, widened by hints from the
    # config's own numeric params (a period-like scale, a center-like offset).
    hints: list[float] = []
    for v in cfg.values():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            v = float(v)
            hints.extend([v, v / 2.0, v * 2.0, v + 1.0, v - 1.0])
    return list(_GENERIC_LADDER) + hints


def _select_probes(binding) -> "tuple[list[float] | None, str]":
    """Pick ≤3 raw probe values that the binding's normalizer provably distinguishes.

    Returns (probes, error). A normalizer that maps every candidate to the same output
    (or to nothing ≥0.05 in magnitude) is dead vocabulary: no synthetic reading could
    ever move the field through it, so the exercise below would pass vacuously.
    """
    candidates = _declared_probes(binding.normalizer)
    if candidates is None:
        candidates = list(_GENERIC_LADDER)
    try:
        fn = binding.build_normalizer()
    except Exception as exc:             # bindings_strict already reported the cause
        return None, f"normalizer failed to build: {exc}"

    seen_out: dict[float, float] = {}    # rounded output -> raw value that produced it
    for raw in candidates:
        try:
            out = float(fn(float(raw)))
        except Exception:
            continue
        if not math.isfinite(out):
            continue
        key = round(out, 6)
        if key not in seen_out:
            seen_out[key] = float(raw)
    outs = sorted(seen_out, key=abs, reverse=True)
    if len(outs) < 2 or abs(outs[0]) < 0.05:
        return None, (
            f"normalizer yields no distinguishable outputs over probes "
            f"{candidates[:8]!r} (dead vocabulary)")
    picked = [seen_out[o] for o in outs[:3]]
    if len(picked) == 2:
        picked.append(picked[0])         # 3 readings, alternating — dedup-safe
    return picked, ""


def _real(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def _check_refs(spec, mspec, node_roles: dict) -> "list[str]":
    """Dangling references and out-of-range declarations — every name a spec uses must
    resolve to something it declares, every bounded number must sit in its bounds.

    Each of these used to pass the gate and then fail SILENTLY downstream: an unknown
    kind or reducer boots as something else, a face/role_mode/projection on an
    undeclared role is never read, a forecast leaf at a nonexistent node is skipped
    forever, an inverted rest window is never entered, an exact `ignored` id that is
    also bound contradicts itself. Runs inside schema_sanity, so
    `--resolve-only` (umweltd's front door) refuses them before a worker spawns."""
    from umwelt.spec.derived import REDUCERS
    from umwelt.spec.schema import (BRIDGE_KINDS, NODE_KIND_ALIASES, NODE_KINDS,
                                    canonical_bridge_kind, canonical_node_kind)

    out: list[str] = []
    parent_of = {n.name: n.parent for n in mspec.nodes}

    for n in mspec.nodes:
        roles = node_roles.get(n.name, set())
        if canonical_node_kind(n.kind) not in NODE_KINDS:
            out.append(f"node {n.name!r}: unknown kind {n.kind!r} (known: "
                       f"{sorted(NODE_KINDS)!r}, aliases "
                       f"{sorted(NODE_KIND_ALIASES)!r})")
        if n.reduce is not None and n.reduce not in REDUCERS:
            out.append(f"node {n.name!r}: unknown reduce {n.reduce!r} "
                       f"(registered: {sorted(REDUCERS)!r})")
        for f in (n.faces or ()):
            r = getattr(f, "role", None)
            if r not in roles:
                out.append(f"node {n.name!r}: face for undeclared role {r!r}")
        for r, mode in (n.role_modes or {}).items():
            if r not in roles:
                out.append(f"node {n.name!r}: role_modes names undeclared role {r!r}")
            elif mode not in ("unitary", "dissipative"):
                out.append(f"node {n.name!r}: role_modes[{r!r}] = {mode!r}, expected "
                           f"'unitary' or 'dissipative'")
        parent_roles = node_roles.get(parent_of.get(n.name), set())
        for r, pr in (n.projection or {}).items():
            if r not in roles:
                out.append(f"node {n.name!r}: projection from undeclared role {r!r}")
            if n.parent is None:
                out.append(f"node {n.name!r}: projection declared on the root "
                           f"(no parent to project into)")
            elif pr not in parent_roles:
                out.append(f"node {n.name!r}: projection {r!r}->{pr!r} names a role "
                           f"parent {n.parent!r} does not declare")

    learned_shapes = {}
    for br in (spec.bridges or ()):
        if canonical_bridge_kind(br.kind) not in BRIDGE_KINDS:
            out.append(f"bridge {br.source!r}->{br.target!r}: unknown kind {br.kind!r} "
                       f"(known: {sorted(BRIDGE_KINDS)!r})")
        if br.learn is not None:
            from umwelt.substrate.role_map_learner import validate_learn
            label = f"bridge {br.source!r}->{br.target!r} learn"
            if br.role_map or canonical_bridge_kind(br.kind) == "tendril":
                out.append(f"{label}: not allowed on a tendril")
            try:
                map_id, inputs, target, ridge = validate_learn(br.learn)
            except ValueError as exc:
                out.append(f"{label}: {exc}")
                continue
            missing = set(inputs) - node_roles.get(br.source, set())
            if missing:
                out.append(f"{label}: input roles {sorted(missing)!r} absent on source")
            if target not in node_roles.get(br.target, set()):
                out.append(f"{label}: target role {target!r} absent on target node")
            shape = (inputs, target, ridge)
            if map_id in learned_shapes and learned_shapes[map_id] != shape:
                out.append(f"{label}: conflicting declaration for map_id {map_id!r}")
            learned_shapes[map_id] = shape

    for b in (spec.bindings or ()):
        if b.zone not in node_roles:
            out.append(f"binding {b.sensor_id!r} targets unknown node {b.zone!r}")
        elif b.role not in node_roles[b.zone]:
            out.append(f"binding {b.sensor_id!r} targets role {b.role!r} on {b.zone!r}, "
                       f"which only declares {sorted(node_roles[b.zone])!r}")
        if b.efficiency is not None and not (_real(b.efficiency)
                                             and 0.0 <= float(b.efficiency) <= 1.0):
            out.append(f"binding {b.sensor_id!r}: efficiency {b.efficiency!r} outside "
                       f"[0, 1] (detector efficiency η is a probability)")

    for entry in (spec.ignored or ()):
        if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
            out.append(f"ignored entry {entry!r}: expected (pattern, reason)")
            continue
        pat = entry[0]
        if any(b.sensor_id == pat for b in (spec.bindings or ())):
            out.append(f"ignored pattern {pat!r} is also a bound signal — a signal is "
                       f"either bound or ignored, not both")
        # A `prefix_*` that also covers bound ids is the "the rest of this family"
        # idiom (ingest consults `ignored` only for UNMATCHED ids, so the binding
        # wins) — advisory, see the ignored_overlap warning in validate_spec.

    for leaf in (spec.forecast_leaves or ()):
        if not (isinstance(leaf, (tuple, list)) and len(leaf) == 2):
            out.append(f"forecast_leaves entry {leaf!r}: expected (node, role)")
            continue
        node, role = leaf
        if node not in node_roles:
            out.append(f"forecast_leaves names unknown node {node!r}")
        elif role not in node_roles[node]:
            out.append(f"forecast_leaves names role {role!r} on {node!r}, which only "
                       f"declares {sorted(node_roles[node])!r}")

    hz = tuple(spec.forecast_horizons_min or ())
    lateness = spec.forecast_label_lateness_min
    if not _real(lateness) or float(lateness) < 0:
        out.append("forecast_label_lateness_min must be a finite nonnegative number")
    if hz:
        if not all(_real(h) and float(h) > 0 for h in hz):
            out.append(f"forecast_horizons_min {hz!r}: every horizon must be a finite "
                       f"number > 0 (minutes)")
        elif any(float(a) >= float(b) for a, b in zip(hz, hz[1:])):
            out.append(f"forecast_horizons_min {hz!r}: horizons must be strictly "
                       f"increasing (no duplicates)")

    for d in (spec.drivers or ()):
        w = d.rest_window
        if w is None:
            continue
        if not (isinstance(w, (tuple, list)) and len(w) == 2 and all(_real(x) for x in w)
                and 0.0 <= float(w[0]) < float(w[1]) <= 1.0):
            out.append(f"driver {d.name!r}: rest_window {w!r} must be (lo, hi) phase "
                       f"fractions with 0 <= lo < hi <= 1 (an inverted or out-of-range "
                       f"window is never entered)")
    return out


def _ignored_overlaps(spec) -> "list[str]":
    """`prefix_*` ignored patterns that also cover bound signal ids (binding wins)."""
    from umwelt.spec.schema import match_ignored
    rows: list[str] = []
    for entry in (spec.ignored or ()):
        if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
            continue
        bound = [b.sensor_id for b in (spec.bindings or ())
                 if b.sensor_id != entry[0]
                 and match_ignored(b.sensor_id, (tuple(entry),)) is not None]
        if bound:
            rows.append(f"{entry[0]!r} covers bound {bound!r}")
    return rows


def validate_spec(spec_or_ref: DomainSpec | str, *, require_shadow: bool = True,
                  exercise_rounds: int = 3,
                  resolve_only: bool = False,
                  declarative_only: bool = False) -> ValidationReport:
    """Run every gate check against a DomainSpec (or a 'module:ATTR' ref).

    require_shadow=True (the default, and what the CLI enforces unless
    --allow-live-outputs) fails any OutputSpec with shadow=False — the shadow law: a
    freshly authored world decides visibly and dispatches nothing until a human
    promotes it.

    resolve_only=True stops after resolve + schema_sanity (+ advisory warnings) —
    no engine boot, no synthetic exercise. The cheap front-door form: umweltd's
    supervisor runs it in a fresh subprocess before a worker ever spawns, so a
    manifest that would kill the worker fails as a 400 with this report's exact
    error text instead of a worker-exit 500.
    """
    from umwelt.spec.schema import DomainSpec, load_spec

    checks: list[CheckResult] = []

    def _skip(name: str, why: str) -> None:
        checks.append(CheckResult(name, ok=True, skipped=True, detail=why))

    # ── 0. declarative-only (BEFORE any import — the module has not executed yet) ──
    # The forge judges agent-authored modules; import executes code, and a module that
    # patches the judge at import time used to turn this gate green (2026-11-01). The
    # audit refuses, on source text alone, anything not shaped like data. See
    # umwelt.spec.declarative for the exact allowlist and why it is static, not a sandbox.
    if declarative_only and isinstance(spec_or_ref, str):
        from umwelt.spec.declarative import audit_module
        module_name = spec_or_ref.partition(":")[0]
        problems = audit_module(module_name) if module_name else ["empty module name"]
        if problems:
            shown = "; ".join(problems[:8]) + (f"; … {len(problems) - 8} more" if len(problems) > 8 else "")
            checks.append(CheckResult("declarative_only", False, shown))
            return ValidationReport(spec_name=str(spec_or_ref), checks=checks)
        checks.append(CheckResult("declarative_only", True, "module is data-shaped; import permitted"))
    elif declarative_only:
        _skip("declarative_only", "in-memory DomainSpec: no module to audit")

    # ── 1. resolve ────────────────────────────────────────────────────────────────
    spec = spec_or_ref
    if isinstance(spec_or_ref, str):
        try:
            spec = load_spec(spec_or_ref)
            checks.append(CheckResult("resolve", True))
        except Exception as exc:
            checks.append(CheckResult("resolve", False, f"{type(exc).__name__}: {exc}"))
            return ValidationReport(spec_name=str(spec_or_ref), checks=checks)
    elif isinstance(spec, DomainSpec):
        checks.append(CheckResult("resolve", True, "in-memory DomainSpec"))
    else:
        checks.append(CheckResult(
            "resolve", False, f"expected DomainSpec or 'module:ATTR' ref, got "
            f"{type(spec_or_ref).__name__}"))
        return ValidationReport(spec_name=repr(spec_or_ref), checks=checks)

    report = ValidationReport(spec_name=spec.name, checks=checks)

    # Driver anchor nodes materialize exactly as boot does, so a spec that (correctly)
    # leaves `_clock` out of its topology is judged on what actually gets built.
    from umwelt.boot import _materialize_driver_nodes
    mspec = _materialize_driver_nodes(spec)
    node_roles = {n.name: set(n.roles or ()) for n in mspec.nodes}

    # ── 2. schema_sanity ─────────────────────────────────────────────────────────
    problems: list[str] = []
    roots = [n.name for n in mspec.nodes if n.parent is None]
    if len(roots) == 0:
        problems.append("no root node (a NodeSpec with parent=None)")
    elif len(roots) > 1:
        problems.append(f"multiple roots: {roots!r}")
    names = [n.name for n in mspec.nodes]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(f"duplicate node names: {dupes!r}")
    sids = [b.sensor_id for b in (spec.bindings or ())]
    dupes = sorted({s for s in sids if sids.count(s) > 1})
    if dupes:
        problems.append(f"duplicate binding sensor_ids: {dupes!r}")
    for br in (spec.bridges or ()):
        for end in (br.source, br.target):
            if end not in node_roles:
                problems.append(f"bridge {br.source!r}->{br.target!r} names unknown "
                                f"node {end!r}")
    onames = [o.name for o in (spec.outputs or ())]
    dupes = sorted({o for o in onames if onames.count(o) > 1})
    if dupes:
        problems.append(f"duplicate output names: {dupes!r}")
    for o in (spec.outputs or ()):
        if o.node not in node_roles:
            problems.append(f"output {o.name!r} targets unknown node {o.node!r}")
        elif o.role not in node_roles[o.node]:
            problems.append(f"output {o.name!r} targets role {o.role!r} on {o.node!r}, "
                            f"which only declares {sorted(node_roles[o.node])!r}")
        if require_shadow and not o.shadow:
            problems.append(f"output {o.name!r} declares shadow=False — the shadow law: "
                            f"a new world's outputs decide visibly and dispatch nothing "
                            f"until explicitly promoted (pass --allow-live-outputs to "
                            f"waive for a hand-audited spec)")
    # NodeSpec.params shape: each value must be what ParameterBundle.from_dict accepts
    # — (default, sigma) or (default, sigma, lo, hi). A bare float here used to die as
    # a TypeError deep in the worker (the first external hive deployment's scar #2);
    # fail it HERE, naming the node and the key.
    for n in mspec.nodes:
        for key, val in (n.params or {}).items():
            if (not isinstance(val, (tuple, list)) or len(val) not in (2, 4)
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                               for x in val)):
                problems.append(
                    f"node {n.name!r} param {key!r}: expected (default, sigma, lo, hi) "
                    f"or (default, sigma), got "
                    f"{type(val).__name__} {val!r}")
    from umwelt.spec.initial_state import initial_bloch_errors
    problems.extend(initial_bloch_errors(mspec))
    problems.extend(_check_refs(spec, mspec, node_roles))
    checks.append(CheckResult("schema_sanity", not problems, "; ".join(problems)))

    # ── 2a. ignored_overlap (WARNING, never a failure) ───────────────────────────
    overlaps = _ignored_overlaps(spec)
    if overlaps:
        checks.append(CheckResult(
            "ignored_overlap", True, warning=True, detail=(
                "ignored wildcard(s) also cover bound signals — the binding wins and "
                "the pattern explains only the unbound remainder: " + "; ".join(overlaps)
                + ". If that is the intent, nothing to do; if not, narrow the pattern.")))

    # ── 2b. gamma_vs_hold (WARNING, never a failure) ─────────────────────────────
    # The mute-world scar (external hive deployment #3): a spec with ingest_hold_s
    # set and dissipative gammas fast enough that gamma × hold > 3 relaxes ~95%+ of
    # any belief across ONE hold — sparse session batches land on a field that has
    # already forgotten everything, with no error anywhere. Only spec-DECLARED
    # gammas (gamma_diss / gamma_diss_{role}) are judged; a spec that declares none
    # is left to the engine's defaults without noise.
    hold = getattr(spec, "ingest_hold_s", None)
    if hold:
        from umwelt.spec.roles import role_input_mode
        slow: list[str] = []
        for n in mspec.nodes:
            params = {k: v for k, v in (n.params or {}).items()
                      if isinstance(v, (tuple, list)) and v
                      and isinstance(v[0], (int, float))}
            modes = n.role_modes or {}
            for role in (n.roles or ()):
                if (modes.get(role) or role_input_mode(role)) != "dissipative":
                    continue
                g = params.get(f"gamma_diss_{role}", params.get("gamma_diss"))
                if g is None:
                    continue
                gamma = float(g[0])
                if gamma > 0 and gamma * float(hold) > 3.0:
                    slow.append(f"{n.name}.{role} (gamma_diss {gamma:g}/s → belief "
                                f"half-life {math.log(2) / gamma:.0f}s vs hold "
                                f"{float(hold):g}s)")
        if slow:
            checks.append(CheckResult(
                "gamma_vs_hold", True, warning=True, detail=(
                    "beliefs will fully relax between batches (gamma × ingest_hold_s "
                    "> 3): " + "; ".join(slow) + ". For session/report worlds "
                    "consider slower gammas or observe-collapse bindings "
                    "(force_observe=True + collapse_alpha as the reporter's honest "
                    "η) — see docs/NEW_DOMAIN.md, 'Worlds of reports'.")))

    # ── 2c. one_sided_risk (WARNING, never a failure) ────────────────────────────
    # The saturation scar: a unitary role fed evidence that can only ever push one
    # way pins at |z|≈1 and stays there — one real incident held z≈+0.95 for 13
    # days with nothing anywhere reporting it. A dissipative role relaxes back to
    # ground between readings, so the danger is specific to unitary: no relaxation,
    # no counter-evidence, no floor.
    #
    # WARNING, never a gate, for a reason that is not politeness: a unitary role
    # fed genuinely two-sided evidence is correct and common, and a spec cannot
    # know what its stream will contain. Only the dynamic auditor
    # (umwelt.tools.saturation_audit) sees the actual polarity. This half flags
    # the shape that makes the failure possible; that half measures whether it
    # happened.
    from umwelt.spec.normalizers import one_signed_on_nonnegative
    from umwelt.spec.roles import role_input_mode as _rim

    _by_target: dict[tuple[str, str], list] = {}
    for b in (mspec.bindings or ()):
        _by_target.setdefault((b.zone, b.role), []).append(b)

    _modes = {n.name: (n.role_modes or {}) for n in mspec.nodes}
    risky: list[str] = []
    for (zone, role), bs in sorted(_by_target.items()):
        if (_modes.get(zone, {}).get(role) or _rim(role)) != "unitary":
            continue
        reasons = [one_signed_on_nonnegative(b.normalizer) for b in bs]
        if not all(reasons):
            continue                        # some binding can still push the other way
        risky.append(f"{zone}.{role} ({len(bs)} binding(s): {reasons[0]})")
    if risky:
        checks.append(CheckResult(
            "one_sided_risk", True, warning=True, detail=(
                "unitary role(s) whose every binding is one-signed for non-negative "
                "readings — belief can saturate at |z|≈1 with no relaxation to bring "
                "it back: " + "; ".join(risky) + ". If the stream really is one-sided, "
                "either declare the role dissipative so it relaxes, or add a binding "
                "carrying the other polarity. Run umwelt.tools.saturation_audit "
                "against a live event log to see which it is.")))

    if resolve_only:
        return report

    # ── 3. topology_build ────────────────────────────────────────────────────────
    from umwelt.spec.build import build_graph_from_spec
    graph = None
    try:
        graph = build_graph_from_spec(mspec)
        checks.append(CheckResult("topology_build", True))
    except Exception as exc:
        checks.append(CheckResult("topology_build", False,
                                  f"{type(exc).__name__}: {exc}"))

    # ── 4. bindings_strict ───────────────────────────────────────────────────────
    # The check apply_spec_bindings deliberately swallows: register every binding on a
    # fresh bridge with NO membrane guard, mirroring its exact kwargs, and report every
    # failure — a typo'd node/role/normalizer must fail authoring, not boot.
    if graph is None:
        _skip("bindings_strict", "topology_build failed")
    else:
        from umwelt.membranes.ingress import SensorBridge
        bridge = SensorBridge(graph)
        bad: list[str] = []
        for b in (spec.bindings or ()):
            try:
                _alpha = (b.measurement_alpha() if hasattr(b, "measurement_alpha")
                          else b.collapse_alpha)
                bridge.register(
                    b.sensor_id, zone=b.zone, qubit_role=b.role,
                    normalize=b.build_normalizer(), weight=b.weight,
                    event_type=(b.event_type or None),
                    **({"collapse_alpha": _alpha} if _alpha is not None else {}),
                    **({"force_observe": True} if b.force_observe else {}),
                )
            except Exception as exc:
                bad.append(f"{b.sensor_id!r}: {exc}")
        checks.append(CheckResult("bindings_strict", not bad, "; ".join(bad)))
    bindings_ok = checks[-1].ok and not checks[-1].skipped

    # ── 5. boot_blank ────────────────────────────────────────────────────────────
    engine = None
    if graph is None:
        _skip("boot_blank", "topology_build failed")
    else:
        from umwelt.boot import build_engine
        _pin_rngs()
        try:
            engine = build_engine(spec=spec, population=False)
            checks.append(CheckResult("boot_blank", True))
        except Exception as exc:
            checks.append(CheckResult("boot_blank", False,
                                      f"{type(exc).__name__}: {exc}"))

    # ── 6. synthetic_exercise ────────────────────────────────────────────────────
    # Feed every declared signal a few readings its own normalizer provably
    # distinguishes, then hold the blank-slate coverage law: every non-driver binding
    # must have touched its (node, role) — no dead vocabulary. Driver-anchor bindings
    # are exempt: the ingest path routes them out of band before the touch is recorded
    # (they exist for trust learning, not field drive).
    if engine is None:
        _skip("synthetic_exercise", "boot_blank failed or skipped")
    elif not bindings_ok:
        _skip("synthetic_exercise", "bindings_strict failed")
    else:
        from umwelt.spec.roles import is_driver_role
        probes: dict[str, list[float]] = {}
        dead: list[str] = []
        exempt: list[str] = []
        for b in (spec.bindings or ()):
            if is_driver_role(b.role):
                exempt.append(b.sensor_id)
                continue
            picked, err = _select_probes(b)
            if picked is None:
                dead.append(f"{b.sensor_id!r}: {err}")
            else:
                probes[b.sensor_id] = picked
        if dead:
            checks.append(CheckResult("synthetic_exercise", False, "; ".join(dead)))
        else:
            try:
                for r in range(max(1, int(exercise_rounds))):
                    readings = {sid: vals[r % len(vals)] for sid, vals in probes.items()}
                    engine.ingest(sensor_readings=readings,
                                  now=_EPOCH + timedelta(seconds=60 * r))
                touched = engine.sensor_bridge.touched_roles
                missed = [f"{b.sensor_id!r} -> ({b.zone!r}, {b.role!r})"
                          for b in (spec.bindings or ())
                          if b.sensor_id in probes and (b.zone, b.role) not in touched]
                detail = "; ".join(missed) if missed else (
                    f"exempt driver-role bindings: {exempt!r}" if exempt else "")
                checks.append(CheckResult("synthetic_exercise", not missed, detail))
            except Exception as exc:
                checks.append(CheckResult("synthetic_exercise", False,
                                          f"ingest raised {type(exc).__name__}: {exc}"))

    # ── 7. save_load_roundtrip ───────────────────────────────────────────────────
    if engine is None:
        _skip("save_load_roundtrip", "boot_blank failed or skipped")
    else:
        import tempfile
        from pathlib import Path
        from umwelt.boot import build_engine
        try:
            h_before = engine.field_canon_hash()
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "engine_state.pkl"
                engine.save(str(p))
                _pin_rngs()
                fresh = build_engine(spec=spec, population=False)
                fresh.load(str(p))
                h_after = fresh.field_canon_hash()
            ok = h_after == h_before
            checks.append(CheckResult(
                "save_load_roundtrip", ok,
                "" if ok else f"canon hash changed across save/load: "
                              f"{h_before} != {h_after}"))
        except Exception as exc:
            checks.append(CheckResult("save_load_roundtrip", False,
                                      f"{type(exc).__name__}: {exc}"))

    return report


_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP",
             "SYSTEMROOT", "SystemRoot", "PYTHONIOENCODING", "USERPROFILE", "APPDATA",
             "LOCALAPPDATA", "COMSPEC", "PATHEXT")
_ENV_SECRET_MARKS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def gate_env(base: "dict | None" = None) -> dict:
    """The environment a gate subprocess is allowed to see: the process essentials plus
    the engine's own ``UMWELT_*`` / ``UMWELTD_*`` dials (so the judge runs under the same
    switches the worker will), minus anything secret-shaped. ``ANTHROPIC_API_KEY`` and its
    kin never ride into a process that is about to import an agent-authored module."""
    import os
    src = os.environ if base is None else base
    out = {}
    for k, v in src.items():
        upper = k.upper()
        if any(mark in upper for mark in _ENV_SECRET_MARKS):
            continue
        if k in _ENV_KEEP or upper.startswith("UMWELT"):
            out[k] = v
    return out


def gate_command(ref: str, *, extra_paths: "list[str]" = (), json_out: bool = True,
                 resolve_only: bool = False, declarative_only: bool = False,
                 allow_live_outputs: bool = False, vocabulary: "str | None" = None) -> "list[str]":
    """The argv for running this gate in an ISOLATED interpreter: ``python -I`` (no
    ``PYTHON*`` env, no user site) with the parent's import path first and the untrusted
    ``extra_paths`` LAST — so an authored module can never shadow ``umwelt`` or the stdlib,
    yet the engine resolves exactly as it does for the caller. Pair with gate_env()."""
    import os as _os
    import sys as _sys
    parent = [p or _os.getcwd() for p in _sys.path]       # '' means cwd — make it explicit for -I
    extra = [str(p) for p in extra_paths]
    args = [ref]
    if json_out:
        args.append("--json")
    if resolve_only:
        args.append("--resolve-only")
    if declarative_only:
        args.append("--declarative-only")
    if allow_live_outputs:
        args.append("--allow-live-outputs")
    if vocabulary:
        args += ["--vocabulary", vocabulary]
    code = ("import sys, json; "
            f"sys.path[:] = json.loads({json.dumps(json.dumps(parent))}) + json.loads({json.dumps(json.dumps(extra))}); "
            "from umwelt.spec.validate import main; "
            f"raise SystemExit(main(json.loads({json.dumps(json.dumps(args))})))")
    return [_sys.executable, "-I", "-c", code]


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m umwelt.spec.validate",
        description="Run the deterministic spec gate against a 'module:ATTR' ref.")
    parser.add_argument("ref", help="spec reference, e.g. examples.gridworld.world:SPEC")
    parser.add_argument("--json", action="store_true",
                        help="emit the report as JSON instead of the summary")
    parser.add_argument("--allow-live-outputs", action="store_true",
                        help="waive the shadow law (outputs with shadow=False pass)")
    parser.add_argument("--resolve-only", action="store_true",
                        help="stop after resolve + schema_sanity (no engine boot) — "
                             "the cheap front-door gate umweltd runs before spawning "
                             "a worker")
    parser.add_argument("--declarative-only", action="store_true",
                        help="refuse, before importing, any spec module that is not "
                             "shaped like data (umwelt.spec.declarative) — the forge "
                             "pipeline always sets this for agent-authored modules")
    parser.add_argument("--vocabulary", default=None, metavar="MODULE:FN",
                        help="optional vocabulary ref, imported and CALLED before "
                             "the spec resolves (mirrors worker boot order)")
    args = parser.parse_args(argv)

    if args.vocabulary:
        err = _call_vocabulary_ref(args.vocabulary)
        if err:
            report = ValidationReport(spec_name=args.ref, checks=[
                CheckResult("vocabulary", False, err)])
            print(json.dumps(report.to_dict(), indent=1) if args.json
                  else report.summary())
            return 1

    report = validate_spec(args.ref, require_shadow=not args.allow_live_outputs,
                           resolve_only=args.resolve_only,
                           declarative_only=args.declarative_only)
    print(json.dumps(report.to_dict(), indent=1) if args.json else report.summary())
    return 0 if report.ok else 1


def _call_vocabulary_ref(ref: str) -> str:
    """Resolve and CALL a 'module:function' vocabulary ref exactly the way the
    umweltd worker boots one. Returns "" on success, else the precise error text
    (the front-door gate's 400 body)."""
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        return (f"vocabulary ref {ref!r} must be 'module:function' — a bare module "
                f"name gives the worker nothing to call (did you mean "
                f"{module_name or ref}:register_vocabulary?)")
    try:
        import importlib
        fn = getattr(importlib.import_module(module_name), attr)
        fn()
    except Exception as exc:
        return f"vocabulary ref {ref!r} failed: {type(exc).__name__}: {exc}"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
