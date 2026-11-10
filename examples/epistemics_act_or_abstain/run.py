"""Held-out evaluation. Refuses to run without a matching prereg.json.

PREEXISTING: built 2026-11-08 before the Apart AI x Epistemics sprint; contains no sprint result.

Run:  python examples/epistemics_act_or_abstain/run.py [--out-dir DIR]

Everything (seeds, ticks, source, cuts, margin, floor, costs) is read from prereg.json; the
pilot seeds are never evaluated here. Writes results.json and RESULTS.md. The verdict is
computed mechanically:
  C below the coverage floor                        -> inconclusive (below coverage floor)
  C beats A and B each by >= margin                 -> premise supported
  otherwise                                         -> premise falsified
The same rule is applied to arm C' with its own pre-registered margin (verdict_cprime). The
dark-rises-error check (arms C and C') is reported as its own table with its own mechanical
verdict from prereg.json["dark_check"]. --warmup-seeds, if given, must equal the pre-registered
list (the run never chooses its own warm-up).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from examples.epistemics_act_or_abstain import arms, score, world  # noqa: E402
from examples.epistemics_act_or_abstain.preregister import HERE, code_hash  # noqa: E402

LABELS = {"A_always_act": "A always_act", "B_persistence_selfconf": "B persistence + self-stated conf",
          "C_evidence_conf": "C evidence-derived conf (relative)",
          "Cprime_evidence_conformal": "C' evidence-conformal (absolute, calibrated)"}


def load_prereg(out_dir: Path, code_dir: Path = HERE) -> dict:
    path = Path(out_dir) / "prereg.json"
    if not path.exists():
        raise SystemExit(f"refusing to run held-out: {path} is missing (run preregister.py first)")
    pre = json.loads(path.read_text(encoding="utf-8"))
    if pre.get("code_hash") != code_hash(code_dir):
        raise SystemExit("refusing to run held-out: arms.py/score.py/world.py changed since pre-registration "
                         "(code_hash mismatch)")
    if set(pre["seeds"]["pilot"]) & set(pre["seeds"]["heldout"]):
        raise SystemExit("refusing to run held-out: pilot and held-out seeds overlap")
    return pre


def verdict(pre: dict, scores: dict, arm: str = "C_evidence_conf", margin_key: str = "margin") -> tuple[str, dict]:
    a, b, c = (scores[k] for k in ("A_always_act", "B_persistence_selfconf", arm))
    if c["n"] == 0 or a["n"] == 0 or b["n"] == 0:
        return "inconclusive (no held-out cases)", {}
    margin = pre[margin_key]
    detail = {"improvement_over_A": a["loss"] - c["loss"], "improvement_over_B": b["loss"] - c["loss"],
              "margin": margin}
    if c["status"] == "FAIL_COVERAGE_FLOOR":
        return "inconclusive (below coverage floor)", detail
    if detail["improvement_over_A"] >= margin and detail["improvement_over_B"] >= margin:
        return "premise supported", detail
    return "premise falsified", detail


def verdict_cprime(pre: dict, scores: dict) -> tuple[str, dict]:
    return verdict(pre, scores, "Cprime_evidence_conformal", "margin_cprime")


def dark_verdict(pre: dict, decisions: dict, arm: str) -> tuple[str, dict]:
    chk = pre["dark_check"]
    eff = score.dark_effect(decisions[arm])
    margin = chk["arms"][arm]["margin"]
    d = {"effect": eff["effect"], "margin": margin, "n_dark": eff["n_dark"], "n_acted": eff["n_acted"]}
    if eff["n_dark"] < chk["min_n"] or eff["n_acted"] < chk["min_n"]:
        return "inconclusive (too few cases in the dark or acted bucket)", d
    return ("dark rises error: supported" if eff["effect"] >= margin else "dark rises error: NOT supported"), d


def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def render(pre: dict, scores: dict, v: str, detail: dict, extra: dict | None = None) -> str:
    synth = pre["heldout_source"]["kind"] == "synthetic"
    lines = []
    if synth:
        lines += ["> **SYNTHETIC PILOT - NOT THE SPRINT RESULT.** PREEXISTING: pipeline demonstration "
                  "on synthetic data, generated before the sprint.", ""]
    lines += ["# Results (held-out)", "",
              "| arm | loss | coverage | hit rate | n |", "|---|---|---|---|---|"]
    for k, s in scores.items():
        flag = " (FAILS coverage floor)" if s["status"] == "FAIL_COVERAGE_FLOOR" else ""
        lines.append(f"| {LABELS[k]}{flag} | {_f(s['loss'])} | {_f(s['coverage'])} | "
                     f"{_f(s['hit_rate'])} | {s['n']} |")
    lines += ["", f"**Verdict (mechanical, from prereg.json): {v}**", ""]
    if detail:
        lines += [f"- pre-registered margin: {_f(detail['margin'], 4)}; observed improvement of C over A: "
                  f"{_f(detail['improvement_over_A'], 4)}, over B: {_f(detail['improvement_over_B'], 4)}",
                  f"- coverage floor: {pre['coverage_floor']}; cuts B/C: "
                  f"{pre['cuts']['B_persistence_selfconf']}/{pre['cuts']['C_evidence_conf']}",
                  f"- pre-registered at {pre['timestamp_utc']}, git {pre['git_sha'][:12]}, "
                  f"held-out seeds {pre['seeds']['heldout']}, warm-up seeds {pre['seeds'].get('warmup', [])}"]
    if extra:
        vc, dc = extra["verdict_cprime"], extra["verdict_cprime_detail"]
        lines += ["", f"**Verdict for C' (the arm that carries the sprint question; same rule, its own "
                  f"pre-registered margin {_f(pre['margin_cprime'], 4)}): {vc}**", ""]
        if dc:
            lines += [f"- observed improvement of C' over A: {_f(dc['improvement_over_A'], 4)}, over B: "
                      f"{_f(dc['improvement_over_B'], 4)}; cuts C/C': {pre['cuts']['C_evidence_conf']}/"
                      f"{pre['cuts']['Cprime_evidence_conformal']}"]
        cf = pre["conformal"]
        lines += ["", f"C' constants (fixed before the pilot): W={cf['window']}, k={cf['k_min_resolved']}, "
                  f"alpha={cf['alpha']}, outcome range {cf['outcome_range']}.", ""]
        lines += ["## Dark rises error (pre-registered check)", "",
                  "Realized mean |error| of the ENGINE's forecast, by what the arm did with the case. "
                  "Arm A never abstains (it is the `acted` row only). B has no dark concept (its "
                  "`warming` = no self-stated confidence yet).", "",
                  "| arm | bucket | n | mean engine abs err |", "|---|---|---|---|"]
        for arm, tbl in extra["bucket_tables"].items():
            for b in score.BUCKETS:
                lines.append(f"| {LABELS[arm]} | {b} | {tbl[b]['n']} | {_f(tbl[b]['mean_abs_err'])} |")
            lines.append(f"| {LABELS[arm]} | total | {tbl['total_n']} | |")
        lines += ["", f"Pre-registered check (arms C, C'): mean |error| on abstained-dark exceeds mean on acted "
                  f"by at least the pilot-set margin; needs >= {pre['dark_check']['min_n']} cases in both buckets.", ""]
        for arm, (dv, dd) in extra["dark_verdicts"].items():
            lines.append(f"- {LABELS[arm]}: effect (dark - acted) = {_f(dd['effect'], 4)}, margin = "
                         f"{_f(dd['margin'], 4)}, n_dark = {dd['n_dark']}, n_acted = {dd['n_acted']} -> **{dv}**")
        cal = extra["calibration_cprime"]
        lines += ["", "### C' calibration", "",
                  f"Fraction of cases whose realized engine |error| <= the predicted (1-alpha) quantile "
                  f"(target {cal['target']:.2f} if calibrated):", "",
                  "| cases | n | fraction within predicted quantile |", "|---|---|---|",
                  f"| acted | {cal['acted']['n']} | {_f(cal['acted']['fraction_within'])} |",
                  f"| all issued (acted or low-conf; excludes dark and warming) | {cal['all_issued']['n']} | "
                  f"{_f(cal['all_issued']['fraction_within'])} |"]
    return "\n".join(lines) + "\n"


def run_heldout(out_dir: Path = HERE, code_dir: Path = HERE, warmup_seeds=None) -> dict:
    out_dir = Path(out_dir)
    pre = load_prereg(out_dir, code_dir)
    warm = tuple(pre["seeds"].get("warmup", []))
    if warmup_seeds is not None and tuple(warmup_seeds) != warm:
        raise SystemExit(f"refusing: --warmup-seeds {list(warmup_seeds)} differs from the pre-registered {list(warm)}")
    if set(warm) & (set(pre["seeds"]["pilot"]) | set(pre["seeds"]["heldout"])):
        raise SystemExit("refusing to run held-out: warm-up seeds overlap pilot or held-out seeds")
    src = world.source_from_description(pre["heldout_source"])
    cw = pre["cost_weights"]
    costs = score.Costs(c_wrong=cw["c_wrong"], c_abstain=cw["c_abstain"], tol=cw["tol"],
                        coverage_floor=pre["coverage_floor"])
    cases = []
    for s in pre["seeds"]["heldout"]:
        cases += arms.build_cases(src, s, pre["ticks"], pre["burn_in"], warm)
    cuts = pre["cuts"]
    decisions = arms.run_arms(cases, cuts["B_persistence_selfconf"], cuts["C_evidence_conf"],
                              cuts["Cprime_evidence_conformal"])
    scores = {k: score.score_arm(d, costs) for k, d in decisions.items()}
    v, detail = verdict(pre, scores)
    vc, detail_c = verdict_cprime(pre, scores)
    tables = {a: score.bucket_table(decisions[a]) for a in arms.ABSTAINING_ARMS}
    dark_verdicts = {a: dark_verdict(pre, decisions, a) for a in ("C_evidence_conf", "Cprime_evidence_conformal")}
    cal = score.calibration(decisions["Cprime_evidence_conformal"], pre["conformal"]["alpha"])
    extra = {"verdict_cprime": vc, "verdict_cprime_detail": detail_c, "bucket_tables": tables,
             "dark_verdicts": dark_verdicts, "calibration_cprime": cal}
    results = {"label": pre["label"], "verdict": v, "verdict_detail": detail,
               "verdict_cprime": vc, "verdict_cprime_detail": detail_c, "arms": scores,
               "dark_rises_error": {"bucket_tables": tables,
                                    "verdicts": {a: {"verdict": dv, **dd} for a, (dv, dd) in dark_verdicts.items()}},
               "calibration_cprime": cal,
               "n_cases": len(cases), "heldout_seeds": pre["seeds"]["heldout"],
               "warmup_seeds": list(warm),
               "prereg_code_hash": pre["code_hash"], "prereg_timestamp_utc": pre["timestamp_utc"]}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "RESULTS.md").write_text(render(pre, scores, v, detail, extra), encoding="utf-8")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--warmup-seeds", type=int, nargs="*", default=None,
                    help="must equal the pre-registered warm-up seeds (the list lives in prereg.json)")
    a = ap.parse_args(argv)
    r = run_heldout(Path(a.out_dir), warmup_seeds=a.warmup_seeds)
    print((Path(a.out_dir) / "RESULTS.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
