"""The engine's frame conventions, recorded as CHECKABLE gauge fixings.

Until this file existed, no test asserted any result unchanged under a relabeling,
a bookkeeping swap, or a frame rotation. These pins name the conventions:

1. VALUE-AXIS ROTATION: a learned parameter's value and purity are invariant under
   rotation of its qubit about the value (z) axis. The azimuth is residual gauge —
   fixed BY CONVENTION to the +x meridian in exactly four places:
     - substrate/qubit_param.py   (kalman_update's surface target)
     - substrate/qubit_ema.py     (_surface_point)
     - foresight/qubit_trust_web.py
     - learning/universal_learner.py (observe_raw)
   A fifth copy of that convention should be caught here and added to this list.
2. BRIDGE ORDER: Bridge(source, target) ordering is bookkeeping — the physics of
   reconciliation is symmetric, so swapping the ends leaves the field snapshot
   byte-identical.
3. PHASE ALIGN IS A GAUGE MOVE: the global X/Y frame flip (phase_align) moves raw
   coordinates (v1 hash) but cannot move the invariant sheet — |h| magnitudes,
   Bloch radii, and every Z₂ Wilson sign survive. And the flip fires only on a
   DOMAIN-declared reference label; the engine ships no default (vocabulary law).
4. FROZEN ROWS: a subsystem nothing touched has byte-stable rows in the canonical
   form while its neighbor learns — the mechanism under the empty-diff certificate.
"""
import numpy as np

from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.field import QuantumField
from umwelt.substrate.field_unify import canon_hash, canonicalize, invariant_sheet, pack
from umwelt.substrate.fractal_stack import FractalScale
from umwelt.substrate.graph import Bridge, WorldGraph, WorldNode
from umwelt.substrate.params import ParameterBundle
from umwelt.substrate.product_cluster import ProductQubitCluster
from umwelt.substrate.qubit_param import QubitBackedParam
from umwelt.projection.gauge import field_gauge


class TestValueAxisRotation:
    def test_param_value_and_purity_survive_z_rotation(self):
        c = ProductQubitCluster("_fiber_test", ["_param_n_k"])
        p = QubitBackedParam("k", c, 0, lo=0.0, hi=1.0)
        for _ in range(4):
            p.kalman_update(0.7, 0.1)       # builds confidence on the +x meridian
        v, r = p.value, p.purity_r
        assert r > 0.3                       # the convention parked real coherence on x
        for phi in (0.3, 1.2, -2.5, np.pi):
            rz = np.array([[np.exp(-0.5j * phi), 0.0], [0.0, np.exp(0.5j * phi)]])
            c._mats[0] = rz @ c._mats[0] @ rz.conj().T
            assert abs(p.value - v) < 1e-12
            assert abs(p.purity_r - r) < 1e-12


def _bridged_field(src, tgt):
    root = WorldNode("root")
    a = WorldNode("na", roles=["shared", "extra_a"])
    b = WorldNode("nb", roles=["shared", "extra_b"])
    root.add_child(a)
    root.add_child(b)
    g = WorldGraph(root=root,
                   bridges=[Bridge(source=src, target=tgt, shared_roles=["shared"])])
    f = QuantumField(g)
    f.clusters["na"].observe_qubit(f.clusters["na"].role_index["shared"],
                                   (0.3, 0.1, 0.8), alpha=1.0)
    f.clusters["nb"].observe_qubit(f.clusters["nb"].role_index["shared"],
                                   (0.0, -0.2, -0.5), alpha=1.0)
    for _ in range(5):
        f.step()
    return f


class TestBridgeOrderIsBookkeeping:
    def test_source_target_swap_is_physicless(self):
        snap_ab = field_gauge(_bridged_field("na", "nb"))
        snap_ba = field_gauge(_bridged_field("nb", "na"))
        assert snap_ab == snap_ba


def _canon_fixture():
    c = CumulantCluster("m", ["a", "b", "c"], gamma=0.1, dt=0.01,
                        role_modes={r: "unitary" for r in "abc"})
    c.e1[0] = (0.2, 0.3, 0.4)
    c.e1[1] = (-0.1, 0.5, 0.0)
    c.e1[2] = (0.0, 0.0, 0.8)
    c.set_couplings(
        h_fields=np.array([[0.3, -0.4, 0.5], [0.0, 0.2, 0.0], [-0.1, 0.1, 0.9]]),
        zz={(0, 1): 0.3, (1, 2): -0.2, (0, 2): 0.25},
        xy={(0, 1): (0.1, -0.1)})
    return pack({"m": c})


class TestPhaseAlignIsAGaugeMove:
    def test_global_xy_flip_moves_the_frame_not_the_sheet(self):
        canon = _canon_fixture()
        flipped = dict(canon)
        flipped["h"] = np.asarray(canon["h"], float).copy()
        flipped["h"][:, 0] *= -1.0           # the phase_align move: negate every X...
        flipped["h"][:, 1] *= -1.0           # ...and Y coefficient, Z untouched
        assert canon_hash(canon) != canon_hash(flipped)          # the frame moved
        assert invariant_sheet(canon) == invariant_sheet(flipped)  # nothing physical did

    def test_flip_requires_a_domain_declared_label(self):
        scale = FractalScale.__new__(FractalScale)
        scale._h_bundles = {"n": ParameterBundle.from_dict({
            "Y_foo": (-0.4, 0.1), "X_foo": (0.3, 0.1), "Z_foo": (0.7, 0.1)})}
        bundle = scale._h_bundles["n"]
        scale.phase_align({})                                    # no label -> no flip
        assert bundle.get_param("Y_foo").value == -0.4
        scale.phase_align({}, reference_label="Y_foo")           # declared -> flips X/Y only
        assert bundle.get_param("Y_foo").value == 0.4
        assert bundle.get_param("X_foo").value == -0.3
        assert bundle.get_param("Z_foo").value == 0.7
        scale.phase_align({}, reference_label="Y_foo")           # now positive -> no reflip
        assert bundle.get_param("Y_foo").value == 0.4


class TestFrozenRowsAreByteStable:
    def test_untouched_cluster_rows_survive_a_neighbors_learning(self):
        active = CumulantCluster("active", ["r1", "r2"], gamma=0.1, dt=0.01,
                                 role_modes={"r1": "unitary", "r2": "unitary"})
        frozen = CumulantCluster("frozen", ["s1", "s2"], gamma=0.1, dt=0.01,
                                 role_modes={"s1": "unitary", "s2": "unitary"})
        frozen.e1[0] = (0.1, 0.2, 0.3)
        frozen.e1[1] = (0.0, -0.4, 0.5)
        active.set_couplings(h_fields=np.array([[0.5, 0.0, 0.3], [0.0, 0.4, 0.0]]),
                             zz={(0, 1): 0.2})
        before = canonicalize(pack({"active": active, "frozen": frozen}))
        sheet_before = invariant_sheet(before)
        for _ in range(20):
            active.step()                    # only the active cluster evolves
        after = canonicalize(pack({"active": active, "frozen": frozen}))
        assert canon_hash(before) != canon_hash(after)           # the brain DID change...
        gi = {}
        off = 0
        for (name, roles, _g, _dt) in after["partition"]:
            gi[name] = (off, off + len(roles))
            off += len(roles)
        lo, hi = gi["frozen"]
        assert np.array_equal(before["e1"][lo:hi], after["e1"][lo:hi])   # ...but not here
        assert np.array_equal(before["h"][lo:hi], after["h"][lo:hi])
        assert np.array_equal(before["e2"][lo:hi, lo:hi], after["e2"][lo:hi, lo:hi])
        assert invariant_sheet(after)["per_cluster"]["frozen"] == \
            sheet_before["per_cluster"]["frozen"]
