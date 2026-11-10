"""Closure honesty for the Berry machinery — no loop number without a loop.

1. Open paths REFUSE: winding is NaN, berry_holonomy is NaN, the tape yields None.
2. A closed loop reports finite invariants with the textbook values (equatorial loop:
   W = 1, Γ = π = half the enclosed solid angle; polar-hugging loop: Γ → 0).
3. The SAME closed loop traversed at three different speeds reports IDENTICAL
   invariants — the traversal-speed invariance the gauge snapshot's header promises,
   now earned rather than asserted.
4. field_gauge emits the `_loops` block only when a tape has certified a return;
   absence of the key is the refusal.
"""
import math

import numpy as np

from umwelt.clocks.berry_tape import BerryTape
from umwelt.projection.gauge import field_gauge
from umwelt.substrate.continuous_geometry import (
    berry_holonomy, is_closed, loop_invariants, path_invariants)


class TestOpenPathsRefuse:
    def test_open_walk_has_no_winding(self):
        theta = np.linspace(0.0, 2.1, 50)              # wanders, never comes home
        assert not is_closed(theta)
        inv = path_invariants(theta)
        assert not inv["closed"]
        assert math.isnan(inv["winding"])
        assert inv["arc_length_loops"] > 0             # the honest open-path functionals report
        assert 0.0 <= inv["directedness"] <= 1.0
        assert abs(inv["net_loops"] - 2.1 / (2 * math.pi)) < 1e-12

    def test_open_walk_has_no_holonomy(self):
        theta = np.linspace(0.0, 3.0, 40)
        assert math.isnan(berry_holonomy(theta, math.pi / 2))
        out = loop_invariants(theta, polar=math.pi / 2)
        assert not out["closed"] and math.isnan(out["holonomy"])

    def test_diagnostic_escape_hatch_is_explicit(self):
        theta = np.linspace(0.0, 3.0, 40)
        raw = berry_holonomy(theta, math.pi / 2, require_closed=False)
        assert not math.isnan(raw)                     # bookkeeping, labeled as such


class TestClosedLoopsReport:
    def test_equatorial_loop_textbook_values(self):
        theta = np.linspace(0.0, 2 * math.pi, 200)     # one full loop, closes mod 2pi
        assert is_closed(theta)
        inv = loop_invariants(theta, polar=math.pi / 2)
        assert inv["closed"]
        assert abs(inv["winding"] - 1.0) < 1e-9
        # equator: Gamma = (1/2)(1 - cos(pi/2)) * 2pi = pi (half the hemisphere's 2pi)
        assert abs(inv["holonomy"] - math.pi) < 1e-9

    def test_pole_hugging_loop_encloses_nothing(self):
        theta = np.linspace(0.0, 2 * math.pi, 200)
        g = berry_holonomy(theta, 0.01)                # polar ~ 0: tiny solid angle
        assert abs(g) < 1e-3

    def test_two_loops_wind_twice(self):
        theta = np.linspace(0.0, 4 * math.pi, 400)
        inv = path_invariants(theta)
        assert inv["closed"] and abs(inv["winding"] - 2.0) < 1e-9


class TestSpeedInvariance:
    def test_same_loop_three_speeds_identical_invariants(self):
        # the same closed equatorial loop, walked uniformly / accelerating / lurching
        u = np.linspace(0.0, 1.0, 300)
        walks = [u, u ** 3, 0.5 * (1 - np.cos(math.pi * u))]
        results = []
        for w in walks:
            theta = 2 * math.pi * w
            assert is_closed(theta)
            inv = loop_invariants(theta, polar=math.pi / 2)
            results.append((inv["winding"], inv["holonomy"], inv["directedness"]))
        for r in results[1:]:
            assert all(abs(a - b) < 1e-9 for a, b in zip(results[0], r))


def _tape_with_return():
    """Drive a tape's ticker out and back so the same event recurs Berry-adjacent."""
    tape = BerryTape()
    phases = [0.0, 0.4, 0.9, 1.4, 1.8, 1.1, 0.5, 0.05]
    ret = None
    for i, p in enumerate(phases):
        tape.ticker.phase = p                          # walk the global phase directly
        r = tape.stamp_custom("pulse", "src", f"tick {i}")
        ret = r or ret
    return tape, ret


class TestTapeGate:
    def test_no_return_no_invariants(self):
        tape = BerryTape()
        tape.ticker.phase = 0.0
        r = tape.stamp_custom("pulse", "src", "first")  # first stamp: nothing to return to
        assert r is None
        assert tape.loop_invariants(r) is None
        assert tape.closed_segment(None) is None

    def test_return_yields_closed_walk_invariants(self):
        tape, ret = _tape_with_return()
        assert ret is not None                          # the walk came home
        inv = tape.loop_invariants(ret)
        assert inv is not None
        assert inv["closed"]
        assert inv["closure_defect"] == ret.phase_delta
        assert inv["arc_length_loops"] > 0
        assert inv["occurrences"] >= 1


class _FakeField:
    def __init__(self):
        self.clusters = {}
        self.graph = None


class TestSnapshotLoopsBlock:
    def test_absent_without_tape_or_returns(self):
        f = _FakeField()
        assert "_loops" not in field_gauge(f)
        assert "_loops" not in field_gauge(f, tape=BerryTape())

    def test_present_at_certified_returns_only(self):
        tape, ret = _tape_with_return()
        snap = field_gauge(_FakeField(), tape=tape)
        assert "_loops" in snap
        (key, inv), = snap["_loops"].items()
        assert key == "pulse/src"
        assert inv["closed"] and inv["occurrences"] >= 1
