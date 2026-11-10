"""Signed bridges — the Z₂ link variable restored to the lateral web.

The repaired defect (CLAIMS.md tier-4 row): calibration mapped correlation [−1,1] →
(corr+1)/2 ∈ [0,1], bridge params were minted lo=0.0, and bridge transport was pinned
to U = 1 — so an anti-correlated bridge learned as weakly-positive, frustration was
unrepresentable, and every bridge Wilson loop read +1.

1. Calibration now learns the SIGNED correlation: an anti-aligned bridge drives its
   coupling negative.
2. A negative link's transport (partner conjugated by σ_x through the link) makes an
   honestly anti-aligned pair a FIXED POINT, where identity transport drags both ends
   toward the mixed middle.
3. The learned web grows sign-seeded edges, keeps a healthy anti-correlated edge, and
   decays a sign-mismatched one.
4. A frustrated bridge triangle finally reads W = −1 (link_field.from_bridges).
"""
from types import SimpleNamespace

import numpy as np

from umwelt.learning.calibration import CalibrationLoop
from umwelt.substrate.field import QuantumField
from umwelt.substrate.graph import Bridge, WorldGraph, WorldNode
from umwelt.substrate.link_field import from_bridges
from umwelt.substrate.params import ScalarParam
from umwelt.substrate.web_topology import WebTopology, WebTopologyConfig, make_learned_bridge


def _coupling_param(value):
    return ScalarParam(name="coupling_na_nb", value=value, sigma=0.1, lo=-1.5, hi=1.5)


def _bridged_field(coupling):
    root = WorldNode("root")
    root.add_child(WorldNode("na", roles=["shared", "pad_a"]))
    root.add_child(WorldNode("nb", roles=["shared", "pad_b"]))
    g = WorldGraph(root=root, bridges=[Bridge(source="na", target="nb",
                                              shared_roles=["shared"],
                                              coupling_param=_coupling_param(coupling))])
    f = QuantumField(g)
    ia = f.clusters["na"].role_index["shared"]
    ib = f.clusters["nb"].role_index["shared"]
    f.clusters["na"].observe_qubit(ia, (0.0, 0.0, 0.8), alpha=1.0)
    f.clusters["nb"].observe_qubit(ib, (0.0, 0.0, -0.8), alpha=1.0)
    return f, g, ia, ib


def _z(field, name, idx):
    return float(field.clusters[name].qubit_bloch(idx)[2])


class TestCalibrationLearnsTheSign:
    def test_anti_aligned_bridge_learns_negative_coupling(self):
        f, g, _, _ = _bridged_field(0.5)
        cal = CalibrationLoop(g, f)
        for _ in range(60):
            cal._calibrate_coupling()
        assert g.bridges[0].coupling_param.value < 0        # was clamped to [0,1] before

    def test_aligned_bridge_still_learns_positive(self):
        f, g, _, ib = _bridged_field(0.5)
        f.clusters["nb"].observe_qubit(ib, (0.0, 0.0, 0.8), alpha=1.0)   # re-align
        cal = CalibrationLoop(g, f)
        for _ in range(60):
            cal._calibrate_coupling()
        assert g.bridges[0].coupling_param.value > 0.5


class TestZ2Transport:
    def test_negative_link_fixes_frustration(self):
        f, _, ia, ib = _bridged_field(-0.7)
        za0, zb0 = _z(f, "na", ia), _z(f, "nb", ib)
        for _ in range(30):
            f._reconcile_bridges()
        # the anti-aligned pair is a fixed point of the -1 link's transport
        assert abs(_z(f, "na", ia) - za0) < 1e-9
        assert abs(_z(f, "nb", ib) - zb0) < 1e-9

    def test_identity_transport_would_have_averaged_it_away(self):
        f, _, ia, ib = _bridged_field(+0.7)
        for _ in range(30):
            f._reconcile_bridges()
        # same states, positive link: both ends dragged toward the mixed middle
        assert abs(_z(f, "na", ia)) < 0.1
        assert abs(_z(f, "nb", ib)) < 0.1


def _web_setup(series_b):
    """A two-cluster world with hand-fed correlation history on the shared role."""
    root = WorldNode("root")
    g = WorldGraph(root=root)
    mk = lambda: SimpleNamespace(qubit_roles=["r"], role_index={"r": 0})
    field = SimpleNamespace(clusters={"ca": mk(), "cb": mk()}, graph=g)
    web = WebTopology(WebTopologyConfig(min_samples=8, grow_threshold=0.6,
                                        keep_threshold=0.3, floor=0.02))
    series_a = [0.8, -0.8, 0.7, -0.7, 0.8, -0.6, 0.75, -0.8, 0.7, -0.75]
    from collections import deque
    web._buf[("ca", "r")] = deque(series_a, maxlen=16)
    web._buf[("cb", "r")] = deque(series_b, maxlen=16)
    return web, field, g


class TestSignAwareWeb:
    def test_grow_seeds_negative_for_anti_comovers(self):
        web, field, g = _web_setup([-v for v in [0.8, -0.8, 0.7, -0.7, 0.8,
                                                 -0.6, 0.75, -0.8, 0.7, -0.75]])
        out = web.maybe_evolve(field, g)
        assert out["grown"] == [("ca", "cb")]
        assert g.bridges[0].coupling_param.value < 0

    def test_healthy_negative_edge_is_kept(self):
        web, field, g = _web_setup([-v for v in [0.8, -0.8, 0.7, -0.7, 0.8,
                                                 -0.6, 0.75, -0.8, 0.7, -0.75]])
        g.bridges.append(make_learned_bridge("ca", "cb", ["r"], -0.4))
        web.maybe_evolve(field, g)
        assert g.bridges[0].coupling_param.value == -0.4     # no decay: sign agrees

    def test_sign_mismatched_edge_decays(self):
        # strongly POSITIVE co-movement, but the edge carries a negative coupling
        web, field, g = _web_setup([0.8, -0.8, 0.7, -0.7, 0.8, -0.6, 0.75, -0.8, 0.7, -0.75])
        g.bridges.append(make_learned_bridge("ca", "cb", ["r"], -0.4))
        web.maybe_evolve(field, g)
        assert abs(g.bridges[0].coupling_param.value) < 0.4  # decayed toward removal


class TestBridgeWilsonLoops:
    def test_frustrated_triangle_reads_minus_one(self):
        bridges = [
            Bridge(source="x", target="y", shared_roles=["r"],
                   coupling_param=ScalarParam(name="c_xy", value=0.6, sigma=0.1, lo=-1.5, hi=1.5)),
            Bridge(source="y", target="z", shared_roles=["r"],
                   coupling_param=ScalarParam(name="c_yz", value=0.5, sigma=0.1, lo=-1.5, hi=1.5)),
            Bridge(source="x", target="z", shared_roles=["r"],
                   coupling_param=ScalarParam(name="c_xz", value=-0.4, sigma=0.1, lo=-1.5, hi=1.5)),
        ]
        lf = from_bridges(bridges)
        assert lf.betti_1() == 1
        assert lf.wilson_sign(["x", "y", "z"]) == -1          # frustration, finally representable
        # and it is gauge-invariant, not bookkeeping: a local flip can't clear it
        lf.gauge_flip("y")
        assert lf.wilson_sign(["x", "y", "z"]) == -1
