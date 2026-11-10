"""Pre-registration: pilot on SYNTHETIC data, choose the cuts and the margin by a stated
rule, then freeze everything in prereg.json.

PREEXISTING: built 2026-11-08 before the Apart AI x Epistemics sprint; contains no sprint result.

Run:  python examples/epistemics_act_or_abstain/preregister.py [--heldout-csv PATH] [--out-dir DIR]

THE RULE (stated here, copied into prereg.json):
  1. Cut selection, for arm B and arm C separately, on PILOT seeds only: over the grid
     CUT_GRID, keep the cuts whose pilot coverage >= the coverage floor; among those pick the
     cut with the lowest pilot loss; ties go to the LOWEST cut (more coverage). If no grid cut
     meets the floor, take the lowest cut and record pilot_floor_met=false.
     (Arm B's cut is tuned too, so the persistence baseline is as strong as C is allowed to be.)
  2. Margin (minimum improvement in loss, same units as loss, required of C over A AND over B):
       margin = max( MARGIN_FLOOR,
                     SHRINK * max(0, pilot_improvement),
                     2 * SE_pilot )
     pilot_improvement = pilot loss of the better of A/B minus pilot loss of C (at the chosen
     cuts); SE_pilot = standard error across pilot seeds of the per-seed difference
     (better-of-A/B minus C). SHRINK < 1 discounts the in-sample winner's curse.
  3. Pilot seeds and held-out seeds must be disjoint; the pilot is always synthetic. The
     held-out set is synthetic (default) or a csv file whose sha256 is recorded here, before
     any held-out case is generated.
  4. Freeze = prereg.json exists. run.py refuses to run without it or when the hash of
     arms.py + score.py + world.py differs from the recorded code_hash. This script refuses to
     overwrite an existing prereg.json (unless --overwrite, which is itself refused once a
     results.json exists for a non-synthetic held-out set).
  5. ARM C' (evidence_conformal) is pre-registered exactly like C: its cut is chosen by rule 1
     (it acts iff confidence >= cut) and its margin by rule 2 with C' in place of C. Its
     constants W (window), k (min resolved cases), alpha are FIXED in arms.py before any pilot
     (README gives the reason); they are recorded here, never tuned on pilot results.
  6. WARM-UP (optional, declared): --warmup-seeds. The engine streams these seeds before every
     scored seed (pilot and held-out alike), identically for C and C'. They must be disjoint from
     pilot AND held-out seeds, else this script refuses. They are recorded in prereg.json.
  7. DARK-RISES-ERROR margin (per abstaining arm that has a dark bucket, i.e. C and C'): on the
     pilot, effect = mean engine |error| on abstained-dark minus mean on acted; margin =
     max(MARGIN_FLOOR, SHRINK * max(0, effect), 2 * SE_pilot), SE over the per-seed effects.
     Same rule as the main margin, units are |error| in outcome units. A held-out verdict needs
     >= DARK_MIN_N cases in both buckets, else it is inconclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from examples.epistemics_act_or_abstain import arms, score, world  # noqa: E402

HERE = Path(__file__).resolve().parent
CODE_FILES = ("arms.py", "score.py", "world.py")

PILOT_SEEDS = (9001, 9002, 9003)
# Fresh held-out seeds: 101-105 were scored once (arms A/B/C only) BEFORE C' existed, and C' was
# designed in response to that outcome, so they are no longer unseen for the C' design.
HELDOUT_SEEDS = (201, 202, 203, 204, 205)
# `frozen` is included so the DARK bucket is non-empty (without it no leaf is ever DARK and the
# dark-rises-error table would be vacuous). Decided from the dark COUNT alone (0 without it).
SYNTH_LEAVES = ("frozen",) + world.DEFAULT_LEAVES
TICKS = 500
BURN_IN = 150
# 0.00 .. 0.99 step 0.01, one grid for B, C and C' (C' confidence is 1 - q_hat/2, which
# compresses toward 1; a 0.05 grid would leave it three cuts above the tolerance-equivalent 0.875)
CUT_GRID = tuple(round(i / 100, 2) for i in range(0, 100))
MARGIN_FLOOR = 0.02
SHRINK = 0.5

FAILURE_CRITERION = (
    "If arm C (evidence-derived confidence) does not beat BOTH always-act (A) and persistence "
    "with self-stated confidence (B) by at least the pre-registered margin in cost-weighted "
    "loss on the held-out cases, or achieves its loss only with coverage below the "
    "pre-registered coverage floor, the premise is falsified (or, below the floor, the result "
    "is inconclusive and counts as a failure to meet the criterion) and is reported as a "
    "negative result."
)


def code_hash(directory: Path = HERE) -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode() + b"\0" + (directory / name).read_bytes() + b"\0")
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def pick_cut(case_sets: dict, arm_fn, costs: score.Costs, **kw) -> tuple[float, bool, list]:
    """Rule step 1. `case_sets` = {seed: [Case]}; returns (cut, floor_met, grid_table)."""
    flat = [c for cs in case_sets.values() for c in cs]
    table = []
    for cut in CUT_GRID:
        s = score.score_arm(arm_fn(flat, cut, **kw), costs)
        table.append({"cut": cut, "loss": s["loss"], "coverage": s["coverage"], "n": s["n"]})
    ok = [t for t in table if t["coverage"] is not None and t["coverage"] >= costs.coverage_floor]
    if not ok:
        return CUT_GRID[0], False, table
    best = min(ok, key=lambda t: (t["loss"], t["cut"]))
    return best["cut"], True, table


def margin_vs_baselines(pilot_scores: dict, arm: str, seeds_sorted: list) -> tuple[float, dict]:
    """Rule step 2 for `arm` (C or C') against the better of A/B on the pilot."""
    best_base = min(("A_always_act", "B_persistence_selfconf"), key=lambda n: pilot_scores[n]["loss"])
    diffs = np.array([pilot_scores[best_base]["per_seed_loss"][s] - pilot_scores[arm]["per_seed_loss"][s]
                      for s in seeds_sorted])
    improvement = float(pilot_scores[best_base]["loss"] - pilot_scores[arm]["loss"])
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    margin = max(MARGIN_FLOOR, SHRINK * max(0.0, improvement), 2.0 * se)
    return margin, {"margin_floor": MARGIN_FLOOR, "shrink": SHRINK,
                    "pilot_improvement_over_better_baseline": improvement,
                    "better_baseline": best_base, "SE_pilot": se, "n_pilot_seeds": len(diffs)}


def dark_margin(decisions) -> tuple[float, dict]:
    """Rule step 7 for one arm's pilot decisions."""
    eff = score.dark_effect(decisions)
    ps = np.array(eff["per_seed_effect"])
    se = float(ps.std(ddof=1) / np.sqrt(len(ps))) if len(ps) > 1 else 0.0
    e = eff["effect"]
    margin = max(MARGIN_FLOOR, SHRINK * max(0.0, e if e is not None else 0.0), 2.0 * se)
    return margin, {"pilot_effect": e, "SE_pilot": se, "n_dark_pilot": eff["n_dark"],
                    "n_acted_pilot": eff["n_acted"], "n_seeds_with_both": len(ps),
                    "margin_floor": MARGIN_FLOOR, "shrink": SHRINK}


def preregister(out_dir: Path = HERE, heldout_csv: str | None = None, ticks: int = TICKS,
                burn_in: int = BURN_IN, pilot_seeds=PILOT_SEEDS, heldout_seeds=HELDOUT_SEEDS,
                costs: score.Costs = score.Costs(), overwrite: bool = False,
                code_dir: Path = HERE, warmup_seeds=(), leaves=SYNTH_LEAVES) -> dict:
    out_dir = Path(out_dir)
    path = out_dir / "prereg.json"
    if path.exists():
        if not overwrite:
            raise SystemExit(f"refusing: {path} already exists (the pre-registration is frozen)")
        if heldout_csv is not None and (out_dir / "results.json").exists():
            raise SystemExit("refusing to overwrite a pre-registration after a held-out run exists")
    warmup_seeds = tuple(warmup_seeds)
    if set(pilot_seeds) & set(heldout_seeds):
        raise SystemExit("pilot and held-out seeds must be disjoint")
    if set(warmup_seeds) & (set(pilot_seeds) | set(heldout_seeds)):
        raise SystemExit("warm-up seeds must be disjoint from pilot and held-out seeds")
    if len(set(warmup_seeds)) != len(warmup_seeds):
        raise SystemExit("warm-up seeds must be distinct")

    pilot_src = world.synthetic_source(tuple(leaves))
    held_src = world.csv_source(heldout_csv) if heldout_csv else world.synthetic_source(tuple(leaves))

    pilot = {s: arms.build_cases(pilot_src, s, ticks, burn_in, warmup_seeds) for s in pilot_seeds}
    flat = [c for cs in pilot.values() for c in cs]
    cut_b, b_floor, grid_b = pick_cut(pilot, arms.persistence_selfconf, costs)
    cut_c, c_floor, grid_c = pick_cut(pilot, arms.evidence_conf, costs)
    trace = arms.conformal_trace(flat)                       # cut-independent; reused across the sweep
    cut_cp, cp_floor, grid_cp = pick_cut(pilot, lambda cs, cut: arms.evidence_conformal(cs, cut, trace=trace), costs)

    all_decs = arms.run_arms(flat, cut_b, cut_c, cut_cp)
    pilot_scores = {name: score.score_arm(decs, costs) for name, decs in all_decs.items()}
    seeds_sorted = sorted(pilot_seeds)
    margin, margin_inputs = margin_vs_baselines(pilot_scores, "C_evidence_conf", seeds_sorted)
    margin_cp, margin_inputs_cp = margin_vs_baselines(pilot_scores, "Cprime_evidence_conformal", seeds_sorted)
    dark_check = {}
    for arm in ("C_evidence_conf", "Cprime_evidence_conformal"):
        m, inp = dark_margin(all_decs[arm])
        dark_check[arm] = {"margin": m, **inp}
    pilot_dark_tables = {a: score.bucket_table(all_decs[a]) for a in arms.ABSTAINING_ARMS}

    synthetic_heldout = held_src.kind == "synthetic"
    prereg = {
        "schema": 1,
        "preexisting": "built 2026-11-08 before the sprint; the harness is the instrument, not a result",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty_code_files": bool(_git("status", "--porcelain", "--",
                                          *[f"examples/epistemics_act_or_abstain/{n}" for n in CODE_FILES])),
        "code_hash": code_hash(code_dir),
        "code_files": list(CODE_FILES),
        "pilot_source": pilot_src.describe(),
        "heldout_source": held_src.describe(),
        "label": ("synthetic pilot - not the sprint result" if synthetic_heldout
                  else "held-out dataset declared before any held-out case was generated"),
        "ticks": ticks,
        "burn_in": burn_in,
        "seeds": {"pilot": list(pilot_seeds), "heldout": list(heldout_seeds),
                  "warmup": list(warmup_seeds), "disjoint": True},
        "warmup": ("the engine streams the warm-up seeds (online, causal, time continuous) before every "
                   "scored seed, pilot and held-out alike; engine-based arms A, C, C' read that one warmed "
                   "ledger; B (persistence) has no learned state and needs none"
                   if warmup_seeds else "none"),
        "conformal": {"window": arms.CONF_W, "k_min_resolved": arms.CONF_K, "alpha": arms.CONF_ALPHA,
                      "outcome_range": arms.OUTCOME_RANGE,
                      "confidence": "1 - q_hat, q_hat = conformal rank ceil((n+1)(1-alpha)) order statistic "
                                    "of the last W resolved |forecast-outcome| / outcome_range",
                      "acts_iff": "not dark and window >= k and confidence >= cut"},
        "cuts": {"B_persistence_selfconf": cut_b, "C_evidence_conf": cut_c,
                 "Cprime_evidence_conformal": cut_cp},
        "cut_rule": ("per arm, on pilot seeds only: among CUT_GRID cuts with pilot coverage >= floor, "
                     "the lowest pilot loss; ties to the lowest cut"),
        "cut_grid": list(CUT_GRID),
        "pilot_floor_met": {"B_persistence_selfconf": b_floor, "C_evidence_conf": c_floor,
                            "Cprime_evidence_conformal": cp_floor},
        "margin": margin,
        "margin_cprime": margin_cp,
        "margin_rule": "max(margin_floor, shrink * max(0, pilot_improvement), 2 * SE_pilot)",
        "margin_inputs": margin_inputs,
        "margin_inputs_cprime": margin_inputs_cp,
        "dark_check": {"rule": ("PRE-REGISTERED: mean engine |error| on abstained-dark > mean on acted, by "
                                "at least margin = max(margin_floor, shrink * max(0, pilot_effect), 2 * SE_pilot); "
                                "needs >= min_n cases in both buckets else inconclusive"),
                       "min_n": score.DARK_MIN_N, "arms": dark_check},
        "pilot_bucket_tables": pilot_dark_tables,
        "coverage_floor": costs.coverage_floor,
        "cost_weights": {"c_wrong": costs.c_wrong, "c_abstain": costs.c_abstain, "tol": costs.tol},
        "failure_criterion": FAILURE_CRITERION,
        "failure_criterion_cprime": FAILURE_CRITERION.replace("arm C (evidence-derived confidence)",
                                                              "arm C' (evidence-conformal confidence)"),
        "pilot_scores": {n: {k: v for k, v in s.items() if k != "per_seed_loss"}
                         for n, s in pilot_scores.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prereg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout-csv", default=None, help="held-out dataset (timestamp,leaf,value); omit for synthetic")
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--warmup-seeds", type=int, nargs="*", default=[],
                    help="seeds the engine is trained on before any scored case (disjoint from pilot and held-out)")
    a = ap.parse_args(argv)
    p = preregister(Path(a.out_dir), a.heldout_csv, a.ticks, a.burn_in, overwrite=a.overwrite,
                    warmup_seeds=tuple(a.warmup_seeds))
    print(json.dumps({k: p[k] for k in ("label", "cuts", "margin", "margin_cprime", "coverage_floor", "code_hash")}, indent=2))
    print(f"wrote {Path(a.out_dir) / 'prereg.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
