"""The three decision policies, all over the SAME forecast ledger.

PREEXISTING: built 2026-11-08 before the Apart AI x Epistemics sprint; contains no sprint result.

`build_cases` replays one stream through a real engine (the world's spec auto-attaches the
ForecastSurface) and writes a ledger: at every tick after burn-in, for every leaf, one Case
holding both candidate forecasts, both confidence numbers, and the outcome that resolves
`horizon_ticks` later. Everything in a Case at tick t uses only information available at t
(the surface learns online from labels that have already resolved); the outcome is attached
afterwards. No state is shared across seeds - each seed gets a fresh engine.

Arms (each maps the ledger to per-case Decisions):
  A always_act              acts on the engine's forecast, always (no signal).
  B persistence_selfconf    forecast = persistence (predict the last value); its SELF-STATED
                            confidence is the absolute-skill-style number max(0, 1 - EMA|err|)
                            (the mirage: a belief that barely moves scores ~1 for free). Acts when
                            that number exceeds a fixed cut.
  C evidence_conf           forecast = the engine's; confidence = ForecastSurface's
                            max(0, skill_vs_persistence) x purity. None (DARK, no evidence)
                            => abstain. Acts when not None and above the cut.
  C' evidence_conformal     forecast = the engine's; confidence = 1 - q_hat, where q_hat is the
                            empirical (1 - alpha) quantile of the engine's OWN recent absolute
                            residuals |forecast - outcome| on RESOLVED cases of the same leaf
                            (rolling window W), scaled to [0, 1] by the outcome range. Keeps C's
                            frozen-belief rule (engine_conf is None => abstain, reason "dark").
                            Fewer than k resolved cases => abstain, reason "warming". It is an
                            ABSOLUTE signal (how wrong has the engine actually been here), where
                            C's is RELATIVE (skill over persistence x purity).

Every Decision records WHY it abstained (reason: "dark" | "warming" | "lowconf"; acted cases
carry "acted") and the engine's own absolute error, so the dark-rises-error table (score.py) can
be computed for any arm that abstains.

WARM-UP (declared, optional): `build_cases(..., warmup_seeds=...)` first streams the warm-up seeds
through the SAME engine (time kept continuous, labels resolving causally) and only then the
scored seed; no case from a warm-up seed is ever scored. The engine-based arms (A, C, C') all read
that one warmed ledger, so C and C' are warmed identically (and A, which acts on the same engine
forecast, gets it too - conservative toward the baseline). B (persistence) has no learned state
and needs no warm-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

from . import world as W

from umwelt.boot import build_engine

PERSIST_EMA = 0.02          # same smoothing the engine's regressor uses for its error EMA

# Arm C' constants - FIXED BEFORE ANY PILOT RESULT (rationale: README "Why W=50, k=10, alpha=0.2").
CONF_W = 50                 # rolling window of resolved residuals per (seed, leaf)
CONF_K = 10                 # minimum resolved residuals before the arm may act
CONF_ALPHA = 0.2            # target miscoverage: the (1 - alpha) = 80% quantile
OUTCOME_RANGE = 2.0         # outcome = belief-z in [-1, 1] (every source is normalised to [-1, 1]),
                            # so |error| <= 2.0; q_hat is divided by this to land in [0, 1].


@dataclass(frozen=True)
class Case:
    seed: int
    leaf: str
    t: int
    engine_forecast: float
    engine_conf: float | None        # None = DARK (no evidence)
    persist_forecast: float
    persist_selfconf: float | None   # self-stated, absolute-skill style
    outcome: float
    resolve_t: int = 0               # tick at which `outcome` becomes known (t + horizon)


@dataclass(frozen=True)
class Decision:
    seed: int
    leaf: str
    t: int
    act: bool
    forecast: float
    outcome: float
    conf: float | None
    reason: str = ""                 # "acted" | "dark" | "warming" | "lowconf"
    pred_q: float | None = None      # C' only: predicted (1-alpha) |error| quantile, outcome units
    engine_err: float | None = None  # |ENGINE forecast - outcome|, whatever the arm did


def _read_z(field, leaf: str) -> float | None:
    c = field.clusters.get(leaf)
    if c is None:
        return None
    idx = c.role_index.get(W.ROLE)
    return None if idx is None else float(c.qubit_bloch(idx)[2])


def _chain(source: W.Source, warmup_seeds, seed: int, ticks: int):
    """Yield (now, readings, scored) over the warm-up seeds then the scored seed, with time kept
    continuous (each segment starts one step after the previous one ended)."""
    step = timedelta(minutes=source.dt_min)
    base = None
    for sd, scored in [(w, False) for w in warmup_seeds] + [(seed, True)]:
        first = last = None
        for now, readings in W.stream(source, sd, ticks):
            if first is None:
                first = now
                if base is None:
                    base = now
            last = base + (now - first)
            yield last, readings, scored
        if last is not None:
            base = last + step


def build_cases(source: W.Source, seed: int, ticks: int, burn_in: int,
                warmup_seeds: tuple = ()) -> list[Case]:
    warmup_seeds = tuple(warmup_seeds)
    if seed in set(warmup_seeds):
        raise ValueError("a scored seed may not also be a warm-up seed")
    eng = build_engine(spec=W.make_spec(source), population=False, calibration=False, fractal=False)
    surf = eng.forecast_surface
    assert surf is not None, "spec.forecast_leaves did not auto-attach the forecast organ"
    h_min = min(surf.horizons)
    H = source.horizon_ticks
    zs: dict[str, list] = {l: [] for l in source.leaves}
    pending: list[dict] = []
    persist_err: dict[str, float | None] = {l: None for l in source.leaves}
    k = -1
    for now, readings, scored in _chain(source, warmup_seeds, seed, ticks):
        eng.ingest(sensor_readings=readings, now=now)
        if not scored:
            continue                      # warm-up: the engine learns; nothing is recorded or scored
        k += 1
        preds = surf.predictions(now)
        for leaf in source.leaves:
            z = _read_z(eng.field, leaf)
            zs[leaf].append(z)
            # self-stated persistence confidence: update with the error of the forecast issued at k-H
            if k >= H and z is not None and zs[leaf][k - H] is not None:
                e = abs(z - zs[leaf][k - H])
                prev = persist_err[leaf]
                persist_err[leaf] = e if prev is None else PERSIST_EMA * e + (1 - PERSIST_EMA) * prev
            p = preds.get((leaf, W.ROLE, h_min))
            if k < burn_in or z is None or p is None:
                continue
            pe = persist_err[leaf]
            pending.append(dict(leaf=leaf, t=k, ef=float(p["z_pred"]), ec=p["confidence"],
                                pf=z, pc=None if pe is None else max(0.0, 1.0 - pe)))
    cases: list[Case] = []
    for r in pending:
        out = zs[r["leaf"]][r["t"] + H] if r["t"] + H < len(zs[r["leaf"]]) else None
        if out is None:
            continue
        cases.append(Case(seed=seed, leaf=r["leaf"], t=r["t"], engine_forecast=r["ef"],
                          engine_conf=r["ec"], persist_forecast=r["pf"],
                          persist_selfconf=r["pc"], outcome=float(out), resolve_t=r["t"] + H))
    return cases


# --------------------------------------------------------------------------- arms
def _dec(c: Case, act: bool, forecast: float, conf: float | None, reason: str,
         pred_q: float | None = None) -> Decision:
    return Decision(c.seed, c.leaf, c.t, bool(act), forecast, c.outcome, conf, reason, pred_q,
                    abs(c.engine_forecast - c.outcome))


def always_act(cases: list[Case]) -> list[Decision]:
    """Arm A: no signal - act on the engine's forecast every time."""
    return [_dec(c, True, c.engine_forecast, None, "acted") for c in cases]


def persistence_selfconf(cases: list[Case], cut: float) -> list[Decision]:
    """Arm B: persistence forecast, act when its self-stated confidence exceeds `cut`.
    No self-stated number yet (fewer than one horizon of history) => abstain, reason "warming";
    B has no DARK concept (that is the point: a frozen belief is maximally 'confident' to it)."""
    out = []
    for c in cases:
        if c.persist_selfconf is None:
            out.append(_dec(c, False, c.persist_forecast, None, "warming"))
        else:
            act = c.persist_selfconf > cut
            out.append(_dec(c, act, c.persist_forecast, c.persist_selfconf, "acted" if act else "lowconf"))
    return out


def evidence_conf(cases: list[Case], cut: float) -> list[Decision]:
    """Arm C: engine forecast; confidence None (DARK) => abstain, else act when above `cut`."""
    out = []
    for c in cases:
        if c.engine_conf is None:
            out.append(_dec(c, False, c.engine_forecast, None, "dark"))
        else:
            act = c.engine_conf > cut
            out.append(_dec(c, act, c.engine_forecast, c.engine_conf, "acted" if act else "lowconf"))
    return out


def conformal_rank(n: int, alpha: float) -> int:
    """1-based order statistic used as the empirical (1 - alpha) quantile of n residuals:
    ceil((n + 1)(1 - alpha)), capped at n (the split-conformal finite-sample rank)."""
    return min(n, max(1, math.ceil(round((n + 1) * (1.0 - alpha), 9))))


def conformal_trace(cases: list[Case], window: int = CONF_W, k: int = CONF_K,
                    alpha: float = CONF_ALPHA) -> list[tuple]:
    """Per case (aligned with `cases`): (state, conf, pred_q) with state in
    {"dark", "warming", None}. Causal: the window for a case issued at tick t holds only
    residuals of cases of the same (seed, leaf) whose outcomes had resolved by t
    (resolve_t <= t), the most recent `window` of them. Residuals are the ENGINE's
    |forecast - outcome|, scaled by OUTCOME_RANGE into [0, 1]."""
    groups: dict = {}
    for i, c in enumerate(cases):
        groups.setdefault((c.seed, c.leaf), []).append(i)
    res: list = [None] * len(cases)
    for idxs in groups.values():
        idxs.sort(key=lambda i: cases[i].t)
        resid: list[float] = []
        j = 0
        for i in idxs:
            c = cases[i]
            while j < len(idxs) and cases[idxs[j]].resolve_t <= c.t:
                cj = cases[idxs[j]]
                resid.append(min(1.0, abs(cj.engine_forecast - cj.outcome) / OUTCOME_RANGE))
                j += 1
            if c.engine_conf is None:
                res[i] = ("dark", None, None)
                continue
            win = resid[-window:]
            if len(win) < k:
                res[i] = ("warming", None, None)
                continue
            q_hat = sorted(win)[conformal_rank(len(win), alpha) - 1]
            res[i] = (None, 1.0 - q_hat, q_hat * OUTCOME_RANGE)
    return res


def evidence_conformal(cases: list[Case], cut: float, window: int = CONF_W, k: int = CONF_K,
                       alpha: float = CONF_ALPHA, trace: list | None = None) -> list[Decision]:
    """Arm C': engine forecast; act iff (not dark) and (window has >= k resolved cases) and
    1 - q_hat >= cut. `trace` lets a cut sweep reuse one conformal_trace (cut-independent)."""
    trace = conformal_trace(cases, window, k, alpha) if trace is None else trace
    out = []
    for c, (state, conf, pq) in zip(cases, trace):
        if state is not None:
            out.append(_dec(c, False, c.engine_forecast, None, state))
        else:
            act = conf >= cut
            out.append(_dec(c, act, c.engine_forecast, conf, "acted" if act else "lowconf", pq))
    return out


ARM_NAMES = ("A_always_act", "B_persistence_selfconf", "C_evidence_conf", "Cprime_evidence_conformal")
ABSTAINING_ARMS = ("B_persistence_selfconf", "C_evidence_conf", "Cprime_evidence_conformal")


def run_arms(cases: list[Case], cut_b: float, cut_c: float, cut_cprime: float) -> dict[str, list[Decision]]:
    return {"A_always_act": always_act(cases),
            "B_persistence_selfconf": persistence_selfconf(cases, cut_b),
            "C_evidence_conf": evidence_conf(cases, cut_c),
            "Cprime_evidence_conformal": evidence_conformal(cases, cut_cprime)}
