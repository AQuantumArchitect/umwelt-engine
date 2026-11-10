"""Cost-weighted decision scoring with a coverage floor.

PREEXISTING: built 2026-11-08 before the Apart AI x Epistemics sprint; contains no sprint result.

Per case: a wrong act costs C_WRONG; a right act earns 0; an abstain costs C_ABSTAIN
(C_ABSTAIN < C_WRONG, so abstaining is better than a wrong act but worse than a right one).
A forecast is "right" when |forecast - outcome| <= TOL. Loss = mean per-case cost over ALL
cases (abstains included). Coverage = fraction of cases acted on.

COVERAGE FLOOR: an arm whose coverage is below COVERAGE_FLOOR is marked FAIL_COVERAGE_FLOOR,
whatever its loss - winning by abstaining on everything does not count.

Every number carries n. Also reported: hit rate and mean squared error of ACTED forecasts, and
a Brier score of stated confidence against hit/miss on acted cases that carry a confidence.
These constants are the defaults; the values actually used are copied into prereg.json.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

C_WRONG = 1.0
C_ABSTAIN = 0.3
TOL = 0.25
COVERAGE_FLOOR = 0.30


@dataclass(frozen=True)
class Costs:
    c_wrong: float = C_WRONG
    c_abstain: float = C_ABSTAIN
    tol: float = TOL
    coverage_floor: float = COVERAGE_FLOOR

    def __post_init__(self):
        if not self.c_abstain < self.c_wrong:
            raise ValueError("c_abstain must be < c_wrong")

    def as_dict(self) -> dict:
        return asdict(self)


def _hit(d, tol: float) -> bool:
    return abs(d.forecast - d.outcome) <= tol


def case_costs(decisions, costs: Costs) -> np.ndarray:
    return np.array([costs.c_abstain if not d.act else (0.0 if _hit(d, costs.tol) else costs.c_wrong)
                     for d in decisions], dtype=float)


def score_arm(decisions, costs: Costs = Costs()) -> dict:
    n = len(decisions)
    if n == 0:
        return {"n": 0, "n_acted": 0, "coverage": None, "loss": None, "hit_rate": None,
                "mse_acted": None, "conf_brier": None, "status": "NO_CASES"}
    acted = [d for d in decisions if d.act]
    cost = case_costs(decisions, costs)
    hits = [_hit(d, costs.tol) for d in acted]
    cov = len(acted) / n
    with_conf = [(d.conf, h) for d, h in zip(acted, hits) if d.conf is not None]
    per_seed = {}
    for s in sorted({d.seed for d in decisions}):
        per_seed[s] = float(np.mean([c for c, d in zip(cost, decisions) if d.seed == s]))
    return {
        "n": n,
        "n_acted": len(acted),
        "coverage": cov,
        "loss": float(cost.mean()),
        "hit_rate": (sum(hits) / len(hits)) if hits else None,
        "mse_acted": float(np.mean([(d.forecast - d.outcome) ** 2 for d in acted])) if acted else None,
        "conf_brier": float(np.mean([(c - float(h)) ** 2 for c, h in with_conf])) if with_conf else None,
        "n_conf_brier": len(with_conf),
        "per_seed_loss": per_seed,
        "status": "ok" if cov >= costs.coverage_floor else "FAIL_COVERAGE_FLOOR",
    }


# --------------------------------------------------------------------------- dark rises error
BUCKETS = ("acted", "dark", "warming", "lowconf")
DARK_MIN_N = 30      # a dark-rises-error verdict needs >= this many cases in BOTH the dark and acted buckets


def bucket_of(d) -> str:
    """acted | dark | warming | lowconf. An abstain with no recognised reason is a bug, not a bucket."""
    if d.act:
        return "acted"
    if d.reason in BUCKETS[1:]:
        return d.reason
    raise ValueError(f"abstained decision without a recognised reason: {d.reason!r}")


def bucket_table(decisions) -> dict:
    """Split decisions into acted / abstained-dark / abstained-warming / abstained-lowconf and
    report, per bucket, n and the mean realized |error| of the ENGINE's forecast (not of the arm's
    own decision forecast - for arm B those differ). n over the buckets sums to the total."""
    errs = {b: [] for b in BUCKETS}
    for d in decisions:
        if d.engine_err is None:
            raise ValueError("decision carries no engine_err")
        errs[bucket_of(d)].append(d.engine_err)
    out = {b: {"n": len(v), "mean_abs_err": float(np.mean(v)) if v else None} for b, v in errs.items()}
    out["total_n"] = len(decisions)
    return out


def dark_effect(decisions) -> dict:
    """mean engine |error| on abstained-dark minus mean on acted, pooled and per seed (seeds
    lacking either bucket are skipped in the per-seed list). Positive = 'dark rises error'."""
    t = bucket_table(decisions)
    dk, ac = t["dark"], t["acted"]
    eff = None if dk["n"] == 0 or ac["n"] == 0 else dk["mean_abs_err"] - ac["mean_abs_err"]
    per_seed = []
    for s in sorted({d.seed for d in decisions}):
        ts = bucket_table([d for d in decisions if d.seed == s])
        if ts["dark"]["n"] and ts["acted"]["n"]:
            per_seed.append(ts["dark"]["mean_abs_err"] - ts["acted"]["mean_abs_err"])
    return {"effect": eff, "n_dark": dk["n"], "n_acted": ac["n"], "per_seed_effect": per_seed}


def calibration(decisions, alpha: float) -> dict:
    """Arm C' calibration: among ACTED decisions that carry a predicted quantile `pred_q`, the
    fraction whose realized engine |error| <= pred_q (target 1 - alpha if the quantile is
    calibrated). Also reported for every decision that carried a quantile (acted or not), which
    isolates calibration of the quantile from the selection effect of acting only on tight ones."""
    def frac(ds):
        ds = [d for d in ds if d.pred_q is not None]
        return {"n": len(ds), "fraction_within": (sum(d.engine_err <= d.pred_q for d in ds) / len(ds)) if ds else None}
    return {"target": 1.0 - alpha, "acted": frac([d for d in decisions if d.act]), "all_issued": frac(decisions)}
