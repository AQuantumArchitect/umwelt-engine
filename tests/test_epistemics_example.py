"""Epistemics act-or-abstain harness: the PIPELINE is pinned on synthetic data.

PREEXISTING (2026-11-08). These tests pin mechanics - determinism, DARK => abstain, the coverage
floor, the pre-registration freeze, the prereg -> run round trip. They pin NO result about
whether evidence-derived confidence helps; that is owed at the sprint on real data.
"""
import json
import shutil
from pathlib import Path

import pytest

from examples.epistemics_act_or_abstain import arms, score, world
from examples.epistemics_act_or_abstain import preregister as prereg_mod
from examples.epistemics_act_or_abstain import run as run_mod

TICKS, BURN = 160, 60
SRC = world.synthetic_source(("frozen", "steady", "noisy", "parent", "dependent"))


@pytest.fixture(scope="module")
def cases():
    return arms.build_cases(SRC, 7, TICKS, BURN)


def test_deterministic_by_seed(cases):
    again = arms.build_cases(SRC, 7, TICKS, BURN)
    assert cases == again and len(cases) > 0
    other = arms.build_cases(SRC, 8, TICKS, BURN)
    assert other != cases
    a1 = arms.run_arms(cases, 0.5, 0.1, 0.9)
    a2 = arms.run_arms(again, 0.5, 0.1, 0.9)
    assert a1 == a2


def test_arm_c_abstains_on_frozen_leaf(cases):
    frozen = [c for c in cases if c.leaf == "frozen"]
    assert frozen and all(c.engine_conf is None for c in frozen)       # DARK, not a number
    dec = [d for d in arms.evidence_conf(cases, cut=0.0) if d.leaf == "frozen"]
    assert dec and not any(d.act for d in dec)
    # the mirage arm B, by contrast, is maximally 'confident' about the frozen leaf
    assert all(c.persist_selfconf is not None and c.persist_selfconf > 0.9 for c in frozen[-20:])
    assert all(d.act for d in arms.persistence_selfconf(cases, cut=0.5) if d.leaf == "frozen"
               and d.conf is not None and d.conf > 0.5)


def test_scoring_costs_and_coverage_floor():
    costs = score.Costs(c_wrong=1.0, c_abstain=0.3, tol=0.1, coverage_floor=0.5)
    mk = lambda act, f, o: arms.Decision(0, "x", 0, act, f, o, None)
    # right act = 0, wrong act = c_wrong, abstain = c_abstain
    s = score.score_arm([mk(True, 0.0, 0.0), mk(True, 0.0, 1.0), mk(False, 0.0, 0.0), mk(False, 0.0, 0.0)], costs)
    assert s["n"] == 4 and s["n_acted"] == 2
    assert s["loss"] == pytest.approx((0 + 1.0 + 0.3 + 0.3) / 4) and s["hit_rate"] == 0.5
    assert s["status"] == "ok"
    # abstaining on (almost) everything has the lowest possible loss here but FAILS the floor
    lazy = score.score_arm([mk(False, 0.0, 1.0)] * 9 + [mk(True, 0.0, 0.0)], costs)
    assert lazy["coverage"] == pytest.approx(0.1) and lazy["status"] == "FAIL_COVERAGE_FLOOR"
    with pytest.raises(ValueError):
        score.Costs(c_wrong=0.2, c_abstain=0.5)


def _paths(tmp_path):
    # a private copy of the code files so freezing/tampering never touches the real harness
    d = tmp_path / "code"
    d.mkdir()
    for n in prereg_mod.CODE_FILES:
        shutil.copy(prereg_mod.HERE / n, d / n)
    return d


def test_run_refuses_without_prereg_and_on_hash_mismatch(tmp_path):
    with pytest.raises(SystemExit, match="prereg.json is missing"):
        run_mod.run_heldout(tmp_path)
    assert not (tmp_path / "results.json").exists()
    (tmp_path / "prereg.json").write_text(json.dumps({"code_hash": "0" * 64, "seeds": {"pilot": [1], "heldout": [2]}}))
    with pytest.raises(SystemExit, match="code_hash mismatch"):
        run_mod.run_heldout(tmp_path)


def test_prereg_refuses_overlapping_seeds_and_overwrite(tmp_path):
    with pytest.raises(SystemExit, match="disjoint"):
        prereg_mod.preregister(tmp_path, ticks=TICKS, burn_in=BURN, pilot_seeds=(1, 2), heldout_seeds=(2, 3))
    (tmp_path / "prereg.json").write_text("{}")
    with pytest.raises(SystemExit, match="frozen"):
        prereg_mod.preregister(tmp_path, ticks=TICKS, burn_in=BURN, pilot_seeds=(1, 2), heldout_seeds=(3, 4))


def test_prereg_to_run_round_trip(tmp_path):
    pre = prereg_mod.preregister(tmp_path, ticks=TICKS, burn_in=BURN, pilot_seeds=(7001, 7002),
                                 heldout_seeds=(8001, 8002), warmup_seeds=(6001,))
    assert (tmp_path / "prereg.json").exists()
    assert pre["seeds"]["warmup"] == [6001] and "Cprime_evidence_conformal" in pre["cuts"]
    assert pre["conformal"]["window"] == arms.CONF_W == 50 and pre["conformal"]["alpha"] == 0.2
    assert pre["conformal"]["k_min_resolved"] == arms.CONF_K == 10
    assert pre["seeds"]["disjoint"] and "synthetic" in pre["label"]
    for key in ("timestamp_utc", "git_sha", "cuts", "margin", "coverage_floor", "cost_weights",
                "failure_criterion", "code_hash"):
        assert pre[key]
    res = run_mod.run_heldout(tmp_path)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert set(saved["arms"]) == set(arms.ARM_NAMES)
    assert all(saved["arms"][a]["n"] > 0 for a in arms.ARM_NAMES)
    assert saved["verdict"] in ("premise supported", "premise falsified",
                                "inconclusive (below coverage floor)")
    md = (tmp_path / "RESULTS.md").read_text()
    assert "SYNTHETIC PILOT - NOT THE SPRINT RESULT" in md and saved["verdict"] in md
    assert res["n_cases"] == saved["n_cases"] > 0 and saved["warmup_seeds"] == [6001]
    assert saved["verdict_cprime"] in ("premise supported", "premise falsified",
                                       "inconclusive (below coverage floor)")
    assert "## Dark rises error (pre-registered check)" in md
    with pytest.raises(SystemExit, match="differs from the pre-registered"):
        run_mod.run_heldout(tmp_path, warmup_seeds=[1, 2])
    # tampering with the frozen code after pre-registration is refused
    code = _paths(tmp_path)
    (code / "score.py").write_text((code / "score.py").read_text() + "\n# tampered\n")
    with pytest.raises(SystemExit, match="code_hash mismatch"):
        run_mod.run_heldout(tmp_path, code_dir=code)


def test_mechanical_verdicts():
    pre = {"margin": 0.05}
    mk = lambda loss, status="ok", n=10: {"loss": loss, "status": status, "n": n}
    v = lambda a, b, c: run_mod.verdict(pre, {"A_always_act": a, "B_persistence_selfconf": b, "C_evidence_conf": c})[0]
    assert v(mk(0.6), mk(0.5), mk(0.3)) == "premise supported"
    assert v(mk(0.6), mk(0.5), mk(0.48)) == "premise falsified"
    assert v(mk(0.6), mk(0.5), mk(0.1, "FAIL_COVERAGE_FLOOR")) == "inconclusive (below coverage floor)"


def test_csv_adapter_reads_rows(tmp_path):
    p = tmp_path / "d.csv"
    rows = ["timestamp,leaf,value"]
    for i in range(30):
        ts = f"2026-03-01T00:{i:02d}:00"
        rows += [f"{ts},aa,{0.5 if i % 2 else -0.5}", f"{ts},bb,0.1"]
    p.write_text("\n".join(rows))
    src = world.csv_source(str(p))
    assert src.leaves == ("aa", "bb") and src.dt_min == 1.0
    got = list(world.stream(src, 1, 10))
    assert len(got) == 10 and set(got[0][1]) == {"aa_value", "bb_value"}
    assert src.describe()["csv_sha256"]


# --------------------------------------------------------------------------- arm C' / warm-up / dark table
def _stream(resids, conf=0.5, leaf="x", seed=0, H=1):
    """Hand-built ledger: engine forecasts 0, outcome t = resids[t]; so |error| at t = resids[t]."""
    return [arms.Case(seed, leaf, t, 0.0, conf, 0.0, 0.5, float(r), resolve_t=t + H)
            for t, r in enumerate(resids)]


def test_cprime_abstains_dark_on_frozen_leaf(cases):
    frozen = [c for c in cases if c.leaf == "frozen"]
    assert frozen and all(c.engine_conf is None for c in frozen)
    dec = [d for d in arms.evidence_conformal(cases, cut=0.0) if d.leaf == "frozen"]
    assert dec and not any(d.act for d in dec) and {d.reason for d in dec} == {"dark"}
    # dark wins over warming: even the very first frozen case is "dark", not "warming"
    assert dec[0].reason == "dark"


def test_cprime_warming_until_k_resolved_cases():
    cs = _stream([0.02] * 30)
    dec = arms.evidence_conformal(cs, cut=0.0, window=50, k=10, alpha=0.2)
    # case t sees residuals of cases 0..t-1 (resolve_t = t'+1 <= t)  => t of them
    assert [d.reason for d in dec[:10]] == ["warming"] * 10 and not any(d.act for d in dec[:10])
    assert dec[10].act and dec[10].reason == "acted"          # exactly k resolved cases: may act
    assert all(d.reason == "acted" for d in dec[10:])
    # a longer horizon delays resolution: residuals are NOT visible before their outcome resolves
    slow = arms.evidence_conformal(_stream([0.02] * 30, H=5), cut=0.0, k=10)
    assert [d.reason for d in slow[:14]] == ["warming"] * 14 and slow[15].reason == "acted"


def test_conformal_rank_and_calibration_bookkeeping_by_hand():
    assert [arms.conformal_rank(n, 0.2) for n in (9, 10, 20, 50)] == [8, 9, 17, 41]
    assert arms.conformal_rank(2, 0.2) == 2          # capped at n
    # residual stream cycling 0.02,0.04,...,0.20 (two of each by t=20)
    resids = [0.02 * (t % 10 + 1) for t in range(60)]
    cs = _stream(resids)
    dec = arms.evidence_conformal(cs, cut=0.0)
    d20 = dec[20]
    # window = residuals 0..19 -> rank 17 of 20 sorted -> 0.18 ; scaled by range 2 -> conf = 1 - 0.09
    assert d20.pred_q == pytest.approx(0.18) and d20.conf == pytest.approx(1 - 0.09)
    # window caps at W=50: case 59 uses residuals 9..58 only
    win = sorted(resids[9:59])
    exp = win[arms.conformal_rank(50, 0.2) - 1]
    assert dec[59].pred_q == pytest.approx(exp)
    # calibration = fraction of acted with |err| <= pred_q, recomputed independently
    cal = score.calibration(dec, 0.2)
    ok = sum(d.engine_err <= d.pred_q for d in dec if d.act and d.pred_q is not None)
    n = sum(1 for d in dec if d.act and d.pred_q is not None)
    assert cal["target"] == pytest.approx(0.8) and cal["acted"]["n"] == n == 50
    assert cal["acted"]["fraction_within"] == pytest.approx(ok / n)
    # pure hand-built: 3 of 4 within => 0.75
    mk = lambda e, q: arms.Decision(0, "x", 0, True, 0.0, 0.0, 0.9, "acted", q, e)
    c2 = score.calibration([mk(0.1, 0.2), mk(0.2, 0.2), mk(0.3, 0.2), mk(0.0, 0.2)], 0.2)
    assert c2["acted"] == {"n": 4, "fraction_within": 0.75}
    # abstained cases with a quantile count only in all_issued
    ab = arms.Decision(0, "x", 0, False, 0.0, 0.0, 0.1, "lowconf", 0.2, 0.5)
    c3 = score.calibration([mk(0.1, 0.2), ab], 0.2)
    assert c3["acted"]["n"] == 1 and c3["all_issued"] == {"n": 2, "fraction_within": 0.5}


def test_cprime_high_error_regime_lowers_confidence_and_abstains():
    cs = _stream([0.02] * 40 + [0.9] * 40)
    dec = arms.evidence_conformal(cs, cut=0.8)
    assert dec[39].act and not dec[79].act and dec[79].reason == "lowconf"


def test_warmup_overlap_refused(tmp_path):
    kw = dict(ticks=TICKS, burn_in=BURN, pilot_seeds=(1, 2), heldout_seeds=(3, 4))
    with pytest.raises(SystemExit, match="warm-up seeds must be disjoint"):
        prereg_mod.preregister(tmp_path, warmup_seeds=(2, 9), **kw)
    with pytest.raises(SystemExit, match="warm-up seeds must be disjoint"):
        prereg_mod.preregister(tmp_path, warmup_seeds=(9, 3), **kw)
    assert not (tmp_path / "prereg.json").exists()
    with pytest.raises(ValueError, match="warm-up"):
        arms.build_cases(SRC, 5, 40, 10, warmup_seeds=(5,))
    # run.py also refuses a prereg whose warm-up was tampered into an overlap
    pre = prereg_mod.preregister(tmp_path, warmup_seeds=(11,), **kw)
    pre["seeds"]["warmup"] = [3]
    (tmp_path / "prereg.json").write_text(json.dumps(pre))
    with pytest.raises(SystemExit, match="warm-up seeds overlap"):
        run_mod.run_heldout(tmp_path)


def test_warmup_changes_engine_state_but_not_determinism_or_default():
    base = arms.build_cases(SRC, 7, TICKS, BURN)
    assert base == arms.build_cases(SRC, 7, TICKS, BURN, ())          # no warm-up == previous behaviour
    w1 = arms.build_cases(SRC, 7, TICKS, BURN, (3,))
    assert w1 == arms.build_cases(SRC, 7, TICKS, BURN, (3,)) and len(w1) == len(base)
    assert [c.outcome for c in w1] == [c.outcome for c in base]       # same stream, scored cases only
    assert [c.engine_forecast for c in w1] != [c.engine_forecast for c in base]   # the engine really was trained
    assert all(c.seed == 7 for c in w1)                               # nothing from the warm-up seed is scored


def test_dark_rises_error_table_has_all_buckets_and_sums_to_total(cases):
    decs = arms.run_arms(cases, 0.5, 0.1, 0.9)
    for arm in arms.ABSTAINING_ARMS:
        t = score.bucket_table(decs[arm])
        assert set(t) == set(score.BUCKETS) | {"total_n"}
        assert sum(t[b]["n"] for b in score.BUCKETS) == t["total_n"] == len(cases)
        assert all((t[b]["n"] == 0) == (t[b]["mean_abs_err"] is None) for b in score.BUCKETS)
    tc = score.bucket_table(decs["C_evidence_conf"])
    assert tc["dark"]["n"] == sum(1 for c in cases if c.leaf == "frozen") > 0
    assert tc["warming"]["n"] == 0                                    # C has no warming state
    assert score.bucket_table(decs["Cprime_evidence_conformal"])["warming"]["n"] > 0
    assert score.bucket_table(decs["B_persistence_selfconf"])["dark"]["n"] == 0   # B has no dark concept
    # error is the ENGINE's, even for arm B whose decision forecast is persistence
    b = decs["B_persistence_selfconf"]
    assert all(d.engine_err == pytest.approx(abs(c.engine_forecast - c.outcome)) for d, c in zip(b, cases))
    with pytest.raises(ValueError):
        score.bucket_table([arms.Decision(0, "x", 0, False, 0.0, 0.0, None, "", None, 0.1)])


def test_dark_effect_hand_built():
    mk = lambda seed, act, reason, e: arms.Decision(seed, "x", 0, act, 0.0, 0.0, None, reason, None, e)
    ds = [mk(1, True, "acted", 0.1), mk(1, False, "dark", 0.5), mk(2, True, "acted", 0.2), mk(2, False, "dark", 0.3)]
    eff = score.dark_effect(ds)
    assert eff["effect"] == pytest.approx(0.4 - 0.15) and eff["per_seed_effect"] == pytest.approx([0.4, 0.1])
