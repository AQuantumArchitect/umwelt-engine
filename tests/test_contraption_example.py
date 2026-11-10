"""The contraption: bots collapse beliefs → surprise → forecast, tested falsifiably.

Two pins:
  (1) the world cures the persistence mirage — the beliefs genuinely MOVE (collapse), so purity is
      not frozen at 0.5 and the forecast has something real to anticipate;
  (2) after correcting independent-seed boundaries, the decisive held-out leaf
      misses the original +0.10 margin. This withdraws the earlier capability pin;
      it does not lower that margin. The protocol still reports all three leaves.
"""
import numpy as np

from examples.contraption.world import CONTRAPTION_SPEC, THINGS, bot_collapse_stream
from proofs.contraption_walk import run
from umwelt.boot import build_engine


def test_beliefs_move_no_persistence_mirage():
    """The cure for the frozen-belief mirage: driven by collapses, each thing swings the full ±1
    and holds high purity — never the purity≈0.5 / z≈0 frozen point that inflates absolute skill."""
    eng = build_engine(spec=CONTRAPTION_SPEC, population=False, calibration=False, fractal=False)
    assert eng.forecast_surface is not None
    assert eng.forecast_surface.n_context > 0                 # higher-order monomial context is on
    z = {t: [] for t in THINGS}
    p = {t: [] for t in THINGS}
    for now, readings in bot_collapse_stream(seed=1, ticks=400):
        eng.ingest(sensor_readings=readings, now=now)
        for t in THINGS:
            c = eng.field.clusters[t]
            idx = c.role_index["state"]
            z[t].append(float(c.qubit_bloch(idx)[2]))
            p[t].append(float(c.purity))
    for t in THINGS:
        assert np.std(z[t]) > 0.3, f"{t} barely moved — mirage risk"      # genuinely moving
        assert np.median(p[t]) > 0.7, f"{t} decohered to mixed"           # not the frozen point


def test_corrected_sequence_protocol_misses_original_decisive_margin():
    """Keep the negative result: repeated clock origins must not suppress later training."""
    summary = run(train_seeds=(1, 2, 3), test_seeds=(101,), ticks=900)
    rows = summary["rows"]
    assert summary["protocol"] == "independent-sequence/1"
    assert all(n == 3 * (900 - 8) for group in summary["training_updates"].values()
               for n in group.values())  # every seed,16-minute horizon at2-minute observations

    # decisive leaf: thing_2's collapse tracks thing_1's jittered flip — only watching thing_1 helps.
    t2 = rows["thing_2"]
    assert t2["surprise_beats_clock"], t2
    assert 0 < t2["skill_vs_persist_surprise"] <= 0.10, t2  # corrected protocol: +0.0975

    # periodic control: thing_1 also beats both baselines.
    t1 = rows["thing_1"]
    assert t1["surprise_beats_clock"], t1
    assert t1["skill_vs_persist_surprise"] > 0.10, t1          # observed ≈ +0.26

    # documented higher-order limit: the conjunctive AND-chained thing_3 is NOT a decisive win like
    # thing_2 — it only hovers near persistence (the rich order-2 context lifts it from a clear loss
    # to roughly a tie, but a two-parent AND is a much harder, low-base-rate trigger than a single
    # watchable parent).
    t3 = rows["thing_3"]
    assert t3["skill_vs_persist_surprise"] < 0.10, t3

    assert not summary["decisive_win"]
    assert summary["control_win"]
    assert summary["verdict"] == "PARTIAL"
