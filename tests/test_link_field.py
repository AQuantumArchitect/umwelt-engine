"""LinkField pins — the four gauge-theory properties the ledger claims, plus the adapters.

1. Wilson phases are invariant under arbitrary local gauge transformations.
2. β₁ = E − V + C counts exactly the independent cycles.
3. gauge_fix_tree drives every spanning-forest edge to exactly 0, preserves every
   Wilson phase, and leaves exactly β₁ residual numbers (the fundamental-cycle phases).
4. Closure honesty: open or broken walks REFUSE (NaN / None) — no loop invariant
   without a loop.
5. from_couplings reads the engine's exchange/antiexchange sign structure: a frustrated
   triangle reads W = −1.
"""
import math
import random

import pytest

from umwelt.substrate.link_field import LinkField, edge_pairs, from_canon, from_couplings


def _ring(n, phases=None):
    lf = LinkField()
    for i in range(n):
        lf.add_edge(i, (i + 1) % n, 0.0 if phases is None else phases[i])
    return lf


def _random_field(rng, n_nodes=12, n_extra=8):
    """Connected random graph: a spanning path + random chords, random phases."""
    lf = LinkField()
    for i in range(n_nodes - 1):
        lf.add_edge(i, i + 1, rng.uniform(-math.pi, math.pi))
    added = 0
    while added < n_extra:
        a, b = rng.randrange(n_nodes), rng.randrange(n_nodes)
        if a == b or lf.has_edge(a, b):
            continue
        lf.add_edge(a, b, rng.uniform(-math.pi, math.pi))
        added += 1
    return lf


class TestGaugeInvariance:
    def test_wilson_phase_survives_random_gauge(self):
        for seed in range(5):
            rng = random.Random(seed)
            lf = _random_field(rng)
            cycles = lf.fundamental_cycles()
            assert len(cycles) == lf.betti_1() > 0
            before = [lf.wilson_phase(c) for c in cycles]
            for node in lf.nodes():
                lf.gauge_transform(node, rng.uniform(-math.pi, math.pi))
            after = [lf.wilson_phase(c) for c in cycles]
            for x, y in zip(before, after):
                assert abs(_wrap_diff(x, y)) < 1e-12

    def test_single_transform_changes_incident_edges_only(self):
        lf = _ring(4, [0.3, -0.7, 1.1, 2.0])
        others_before = lf.edge_phase(1, 2)
        lf.gauge_transform(0, 1.234)
        assert abs(lf.edge_phase(0, 1) - _wrap(0.3 + 1.234)) < 1e-12
        assert abs(lf.edge_phase(3, 0) - _wrap(2.0 - 1.234)) < 1e-12
        assert lf.edge_phase(1, 2) == others_before

    def test_gauge_flip_is_z2(self):
        lf = _ring(3)
        lf.set_edge_sign(0, 1, -1)
        w = lf.wilson_sign([0, 1, 2])
        lf.gauge_flip(1)
        assert lf.edge_sign(0, 1) == 1          # the flipped convention moved the link...
        assert lf.wilson_sign([0, 1, 2]) == w   # ...and the loop product survived
        lf.gauge_flip(1)
        assert lf.edge_sign(0, 1) == -1         # g^2 = 1


class TestBetti:
    def test_tree_has_no_cycles(self):
        lf = LinkField()
        for i in range(1, 6):
            lf.add_edge((i - 1) // 2, i, 0.5)   # a small binary tree
        assert lf.betti_1() == 0
        assert lf.fundamental_cycles() == []

    def test_one_chord_one_cycle(self):
        lf = _ring(5)
        assert lf.betti_1() == 1
        assert len(lf.fundamental_cycles()) == 1

    def test_disjoint_components(self):
        lf = LinkField()
        for base in (0, 10):                    # two disjoint triangles
            lf.add_edge(base, base + 1)
            lf.add_edge(base + 1, base + 2)
            lf.add_edge(base + 2, base)
        lf.add_node(99)                          # an isolated node
        assert lf.betti_1() == 6 - 7 + 3

    def test_isolated_nodes_dont_break_forest(self):
        lf = LinkField()
        lf.add_node("a")
        lf.add_node("b")
        assert lf.betti_1() == 0


class TestGaugeFixTree:
    def test_comb_flat_residue_on_chords(self):
        for seed in range(5):
            rng = random.Random(100 + seed)
            lf = _random_field(rng)
            b1 = lf.betti_1()
            cycles = lf.fundamental_cycles()
            wilson_before = [lf.wilson_phase(c) for c in cycles]
            forest = lf._spanning_forest()
            out = lf.gauge_fix_tree()
            # every spanning-forest edge is exactly zero
            for child, parent in forest["parents"].items():
                assert abs(lf.edge_phase(parent, child)) < 1e-9
            # exactly beta_1 residual numbers, each equal to its cycle's Wilson phase
            assert len(out["residual"]) == b1
            wilson_after = [lf.wilson_phase(c) for c in cycles]
            for x, y in zip(wilson_before, wilson_after):
                assert abs(_wrap_diff(x, y)) < 1e-9
            # fundamental_cycles and residual iterate the same edge order; the cycle
            # [a … b] closes over the chord b→a, so W(C) = −(residual stored a→b)
            for cyc_w, res in zip(wilson_after, out["residual"]):
                assert abs(_wrap_diff(cyc_w, -res["phase"])) < 1e-9


class TestClosureHonesty:
    def test_open_walk_refuses(self):
        lf = _ring(5)
        assert math.isnan(lf.wilson_phase([0, 1, 2]))       # 2→0 hop is not an edge
        assert lf.wilson_sign([0, 1, 2]) is None

    def test_broken_walk_refuses(self):
        lf = _ring(5)
        assert math.isnan(lf.wilson_phase([0, 1, 3, 4]))    # 1→3 hop is not an edge
        assert lf.wilson_sign([0, 1, 3, 4]) is None

    def test_too_short_refuses(self):
        lf = _ring(5)
        assert math.isnan(lf.wilson_phase([0, 1]))
        assert math.isnan(lf.wilson_phase([]))

    def test_explicit_and_implicit_closure_agree(self):
        lf = _ring(5, [0.1, 0.2, 0.3, 0.4, 0.5])
        w_implicit = lf.wilson_phase([0, 1, 2, 3, 4])
        w_explicit = lf.wilson_phase([0, 1, 2, 3, 4, 0])
        assert not math.isnan(w_implicit)
        assert abs(w_implicit - w_explicit) < 1e-12

    def test_absent_edge_phase_is_nan_not_zero(self):
        lf = _ring(3)
        assert math.isnan(lf.edge_phase(0, 99))


class TestAdapters:
    def test_antiexchange_triangle_is_frustrated(self):
        # exchange (J, J) on two links, antiexchange (J, -J) on the third
        xy = {(0, 1): (0.2, 0.2), (1, 2): (0.2, 0.2), (0, 2): (0.2, -0.2)}
        lf = from_couplings({}, xy, 3)
        assert lf.wilson_sign([0, 1, 2]) == -1
        assert lf.betti_1() == 1

    def test_zero_couplings_are_placeholders_not_links(self):
        # cluster dicts pre-populate every pair with 0.0 — no edges should appear
        zz = {(0, 1): 0.0, (1, 2): 0.0, (0, 2): 0.0}
        lf = from_couplings(zz, {}, 3)
        assert lf.edge_count() == 0
        assert lf.betti_1() == 0

    def test_zz_sign_seeds_when_no_xy(self):
        zz = {(0, 1): 0.4, (1, 2): -0.4, (0, 2): 0.4}
        lf = from_couplings(zz, {}, 3)
        assert lf.edge_sign(0, 1) == 1
        assert lf.edge_sign(1, 2) == -1
        assert lf.wilson_sign([0, 1, 2]) == -1

    def test_from_canon_and_edge_pairs(self):
        canon = {"e1": [[0, 0, 0]] * 4,
                 "zz": {(0, 1): 0.3, (2, 3): -0.2},
                 "xy": {(1, 2): (0.1, 0.1)}}
        lf = from_canon(canon)
        assert lf.node_count() == 4
        assert edge_pairs(lf) == [(0, 1), (1, 2), (2, 3)]
        assert lf.edge_sign(2, 3) == -1


def _wrap(p):
    q = math.fmod(p + math.pi, 2.0 * math.pi)
    if q <= 0.0:
        q += 2.0 * math.pi
    return q - math.pi


def _wrap_diff(x, y):
    return _wrap(x - y)
