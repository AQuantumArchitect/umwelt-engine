"""measure_axis / measure_pair / measurement receipts (glyph §16 engine-side facts)."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from umwelt.projection.frame import frame, validate
from umwelt.substrate.cluster import QubitCluster
from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.classical_reservoir import ClassicalReservoirCluster
from umwelt.substrate.product_cluster import ProductQubitCluster
from umwelt.substrate.receipts import receipts_of, to_glyph_receipt

Z = (0.0, 0.0, 1.0)
BACKENDS = [QubitCluster, CumulantCluster]
PAULI = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]


def _plus(C, roles=("a", "b")):
    """Both qubits in |+> (Bloch (1,0,0)), uncorrelated, on any backend."""
    c = C("m", list(roles))
    for q in range(len(roles)):
        if C is CumulantCluster:
            c.e1[q] = (1.0, 0.0, 0.0)
        c_ = c
    if C is CumulantCluster:
        c._sync_e2_product()
    else:
        plus = np.array([1.0, 1.0]) / np.sqrt(2)
        v = plus
        for _ in range(len(roles) - 1):
            v = np.kron(v, plus)
        c.evolver.dm.rho[:] = np.outer(v, v.conj())
    return c


def _moments(rho):
    e1 = np.zeros((2, 3))
    for q in range(2):
        for a in range(3):
            op = np.kron(PAULI[a], np.eye(2)) if q == 0 else np.kron(np.eye(2), PAULI[a])
            e1[q, a] = np.real(np.trace(rho @ op))
    e2 = np.zeros((2, 2, 3, 3))
    for a in range(3):
        for b in range(3):
            v = np.real(np.trace(rho @ np.kron(PAULI[a], PAULI[b])))
            e2[0, 1, a, b] = v
            e2[1, 0, b, a] = v
    return e1, e2


def _zz(c):
    if isinstance(c, CumulantCluster):
        return float(c.e2[0, 1, 2, 2])
    return float(np.real(np.trace(c.rho @ np.kron(PAULI[2], PAULI[2]))))


# (a) z axis == old measure_qubit, bit for bit
@pytest.mark.parametrize("C", BACKENDS)
def test_axis_z_is_bit_for_bit_measure_qubit(C):
    a, b = C("m", ["p", "q"]), C("m", ["p", "q"])
    for c in (a, b):
        c.observe_qubit(0, (0.6, 0.0, 0.3), 0.5)
    a.measure_qubit(0, 0.4, 0.7)
    b.measure_axis(0, Z, 0.4, 0.7)
    for q in (0, 1):
        assert np.array_equal(a.qubit_bloch(q), b.qubit_bloch(q))
    if C is CumulantCluster:
        assert np.array_equal(a.e2, b.e2)
    else:
        assert np.array_equal(a.rho, b.rho)


# (b) strong measurement along x / y of |+> / |+i> drives to the axis
@pytest.mark.parametrize("C", BACKENDS)
def test_strong_x_measurement_drives_plus_to_x(C):
    c = _plus(C)
    c.measure_axis(0, "x", 1.0, 40.0)
    assert np.allclose(c.qubit_bloch(0), (1, 0, 0), atol=1e-6)


@pytest.mark.parametrize("C", BACKENDS)
def test_strong_y_measurement_drives_to_y(C):
    c = C("m", ["a"])
    if C is CumulantCluster:
        c.e1[0] = (0.0, 0.6, 0.0)
    else:
        c.measure_axis(0, "y", 1.0, 0.0)           # no-op; state set via observe below
        c.observe_qubit(0, (0.0, 0.6, 0.0), 1.0)
    c.measure_axis(0, "y", 1.0, 40.0)
    assert np.allclose(c.qubit_bloch(0), (0, 1, 0), atol=1e-6)


# (c) order dependence is real
@pytest.mark.parametrize("C", BACKENDS)
def test_noncommuting_measurements_are_order_dependent(C):
    def run(first, second):
        c = C("m", ["a"])
        if C is CumulantCluster:
            c.e1[0] = (0.0, 0.0, 0.0)
        else:
            c.observe_qubit(0, (0.0, 0.0, 0.0), 1.0)
        for ax, rec in (first, second):
            c.measure_axis(0, ax, rec, 1.0)
        return c.qubit_bloch(0)
    zx = run(("z", 1.0), ("x", 1.0))
    xz = run(("x", 1.0), ("z", 1.0))
    assert np.linalg.norm(zx - xz) > 0.1
    zz = run(("z", 1.0), ("z", 1.0))
    zz2 = run(("z", 1.0), ("z", 1.0))
    # a commuting pair is order-free: the same two records in either order agree
    za = run(("z", 1.0), ("z", -0.5))
    zb = run(("z", -0.5), ("z", 1.0))
    assert np.allclose(za, zb, atol=1e-9) and np.allclose(zz, zz2)


def test_order_dependence_numbers_are_stable():
    c1, c2 = QubitCluster("m", ["a"]), QubitCluster("m", ["a"])
    for c in (c1, c2):
        c.observe_qubit(0, (0.0, 0.0, 0.0), 1.0)
    c1.measure_axis(0, "z", 1.0, 1.0); c1.measure_axis(0, "x", 1.0, 1.0)
    c2.measure_axis(0, "x", 1.0, 1.0); c2.measure_axis(0, "z", 1.0, 1.0)
    zx, xz = c1.qubit_bloch(0), c2.qubit_bloch(0)
    assert zx[0] > xz[0] and xz[2] > zx[2]     # last measurement dominates its own axis


# (d) measure_pair raises <ZZ> toward the record
@pytest.mark.parametrize("C", BACKENDS)
def test_pair_zz_moves_toward_record(C):
    c = _plus(C)
    before = _zz(c)
    assert abs(before) < 1e-9
    c.measure_pair(0, 1, 1.0, 1.0)
    assert _zz(c) > before + 0.2
    c.measure_pair(0, 1, 1.0, 40.0)
    assert _zz(c) > 0.99


def test_pair_cumulant_matches_dense_in_pair_block_exactly():
    rng = np.random.default_rng(1)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    r = 0.7 * np.outer(psi, psi.conj()) + 0.3 * np.eye(4) / 4
    for basis in (("z", "z"), ("x", "y"), ((1, 1, 0), (0, 1, 1))):
        d, c = QubitCluster("m", ["a", "b"]), CumulantCluster("m", ["a", "b"])
        d.evolver.dm.rho[:] = r
        c.e1, c.e2 = _moments(r)
        d.measure_pair(0, 1, 0.6, 0.9, basis)
        c.measure_pair(0, 1, 0.6, 0.9, basis)
        e1, e2 = _moments(d.rho)
        assert np.allclose(e1, c.e1, atol=1e-6)
        assert np.allclose(e2[0, 1], c.e2[0, 1], atol=1e-6)


# (e) strength 0 is a no-op
@pytest.mark.parametrize("C", BACKENDS)
def test_strength_zero_is_exact_noop(C):
    c = _plus(C)
    snap = (c.e1.copy(), c.e2.copy()) if C is CumulantCluster else c.rho.copy()
    c.measure_axis(0, (1, 2, 3), 0.5, 0.0)
    c.measure_pair(0, 1, 0.5, 0.0, ("x", "y"))
    if C is CumulantCluster:
        assert np.array_equal(snap[0], c.e1) and np.array_equal(snap[1], c.e2)
    else:
        assert np.array_equal(snap, c.rho)


def test_unrepresentable_backends():
    for C in (ProductQubitCluster, ClassicalReservoirCluster):
        c = C("m", ["a", "b"])
        c.measure_pair(0, 1, 1.0, 0.0)
        with pytest.raises(NotImplementedError):
            c.measure_pair(0, 1, 1.0, 1.0)
        c.measure_axis(0, "x", 1.0, 0.5)   # axis works everywhere


# (f) receipts
def test_receipts_logged_exported_and_glyph_shaped():
    c = QubitCluster("house", ["occupied", "warm"])
    c.measure_axis(0, "z", 1.0, 0.5, confidence=0.9)
    c.measure_axis(0, "x", -1.0, 0.5)
    rs = receipts_of(c)
    assert [r.order_index for r in rs] == [0, 1]
    assert rs[0].axis == Z and rs[1].axis == (1.0, 0.0, 0.0)
    assert rs[0].confidence == 0.9 and rs[0].register == "occupied"

    c.measure_pair(0, 1, 0.5, 0.5, ("z", "z"))
    assert receipts_of(c)[2].joint and receipts_of(c)[2].registers == ("occupied", "warm")

    plates = {"axes": [{"node": "house", "role": "occupied", "status": "live", "z": 0.9},
                       {"node": "house", "role": "warm", "status": "live", "z": 0.1}]}
    bridge = SimpleNamespace(latest_value=lambda *_: (0.9, 100), _last_raw_ts={})
    eng = SimpleNamespace(sensor_bridge=bridge, field=SimpleNamespace(clusters={"house": c}))
    with patch("umwelt.projection.frame.nameplates", return_value=plates):
        doc = frame(eng, subject="house", now=1000)
    assert validate(doc) == []
    occ, warm = doc["nodes"]
    assert [r["order_index"] for r in occ["receipts"]] == [0, 1, 2]
    assert [r["order_index"] for r in warm["receipts"]] == [2]       # only the joint one
    assert occ["receipts"][1]["axis"] == [1.0, 0.0, 0.0]

    # validate() passes without receipts, and rejects a malformed one
    bare = {k: v for k, v in doc.items()}
    bare["nodes"] = [{k: v for k, v in n.items() if k != "receipts"} for n in doc["nodes"]]
    assert validate(bare) == []
    bad = dict(doc); bad["nodes"] = [dict(occ, receipts=[{"register": "x"}]), warm]
    assert validate(bad)

    g = to_glyph_receipt(receipts_of(c))
    assert set(g) == {"measurements", "outcomes"}
    m0, m1, m2 = g["measurements"]
    assert set(m0) == {"register", "direction", "eta", "order_index", "joint", "registers"}
    assert m0["direction"] == Z and 0.0 < m0["eta"] < 1.0
    assert m1["direction"] == (1.0, 0.0, 0.0) and m1["order_index"] == 1
    assert m2["joint"] and m2["registers"] == ("occupied", "warm")


def test_no_receipts_without_measurement():
    c = CumulantCluster("m", ["a"])
    c.observe_qubit(0, (0, 0, 1), 0.5)
    assert receipts_of(c) == [] and "_measurement_receipts" not in c.__dict__


# (g) dense == cumulant parity for the measured qubit's marginal
def test_axis_marginal_parity_dense_vs_cumulant():
    rng = np.random.default_rng(2)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    r = 0.7 * np.outer(psi, psi.conj()) + 0.3 * np.eye(4) / 4
    for axis in ("x", "y", "z", (1, 2, 3), (-1, 0.5, -2)):
        d, c = QubitCluster("m", ["a", "b"]), CumulantCluster("m", ["a", "b"])
        d.evolver.dm.rho[:] = r
        c.e1, c.e2 = _moments(r)
        d.measure_axis(0, axis, -0.4, 0.8)
        c.measure_axis(0, axis, -0.4, 0.8)
        assert np.allclose(d.qubit_bloch(0), c.qubit_bloch(0), atol=1e-6)


# ── cumulant vs dense on correlated 3-/4-qubit clusters (peer cross-update) ───────────────
def _op(n, d):
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, d.get(q, np.eye(2)))
    return m


def _all_moments(rho, n):
    e1 = np.array([[np.real(np.trace(rho @ _op(n, {q: PAULI[a]}))) for a in range(3)]
                   for q in range(n)])
    e2 = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            if i != j:
                for a in range(3):
                    for b in range(3):
                        e2[i, j, a, b] = np.real(np.trace(rho @ _op(n, {i: PAULI[a], j: PAULI[b]})))
    return e1, e2


def _correlated(n, seed, mix):
    """Random product state, three Pauli-pair couplings exp(-iθ σ_aσ_a), mixed with identity."""
    r = np.random.default_rng(seed)
    v = np.array([1.0 + 0j])
    for _ in range(n):
        t, ph = r.uniform(0, np.pi), r.uniform(0, 2 * np.pi)
        v = np.kron(v, np.array([np.cos(t / 2), np.exp(1j * ph) * np.sin(t / 2)]))
    for _ in range(3):
        i, j = r.choice(n, 2, replace=False)
        a, th = PAULI[r.integers(3)], r.uniform(0.4, 1.2)
        v = np.cos(th) * v - 1j * np.sin(th) * (_op(n, {int(i): a, int(j): a}) @ v)
    return mix * np.outer(v, v.conj()) + (1 - mix) * np.eye(2 ** n) / 2 ** n


def _peer_err(mix, strength, legacy=False, n_list=(3, 4), seeds=range(8)):
    from umwelt.substrate import belavkin as B
    worst = 0.0
    for n in n_list:
        for sd in seeds:
            r = np.random.default_rng(1000 + sd)
            rho = _correlated(n, sd, mix)
            axis, i, rec = r.normal(size=3), int(r.integers(n)), float(r.uniform(-1, 1))
            d = QubitCluster("m", [str(k) for k in range(n)])
            d.evolver.dm.rho[:] = rho
            d.measure_axis(i, axis, rec, strength)
            e1d, _ = _all_moments(d.rho, n)
            e1, e2 = _all_moments(rho, n)
            if legacy:      # z-only peer update (the pre-fix behaviour)
                with patch.object(B, "measure_cumulant",
                                  lambda *a, _o=B.measure_cumulant, **k: _o(*a[:5])):
                    B.measure_cumulant_axis(e1, e2, i, axis, rec, strength)
            else:
                B.measure_cumulant_axis(e1, e2, i, axis, rec, strength)
            worst = max(worst, max(np.abs(e1[j] - e1d[j]).max() for j in range(n) if j != i))
    return worst


@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_offaxis_peer_update_matches_dense_on_mixed_correlated_states(strength):
    # measured numbers (8 seeds x n in {3,4}): before 0.042/0.207/0.385, after ~1e-7
    assert _peer_err(0.9, strength) < 5e-7
    assert _peer_err(0.9, strength, legacy=True) > 0.03 * strength ** 0.5  # the gap was real


@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_offaxis_peer_update_on_pure_states_is_pole_floor_limited(strength):
    # Pure correlated states: residual 0.0124/0.0134/0.0146 (strength .1/.5/1). It is NOT the
    # closure (the mean update is exact given exact e2): it is the deliberate Z_CAP=0.98 pole
    # floor that measure_cumulant applies to qubits with |z|>0.98 before measuring.
    assert _peer_err(1.0, strength) < 2e-2


def test_offaxis_peers_get_transverse_updates():
    c, d = CumulantCluster("m", list("abc")), QubitCluster("m", list("abc"))
    rho = _correlated(3, 3, 0.9)
    d.evolver.dm.rho[:] = rho
    c.e1, c.e2 = _all_moments(rho, 3)
    x0 = c.e1[:, :2].copy()
    c.measure_axis(0, (1, 1, 0.5), 0.8, 0.8)
    d.measure_axis(0, (1, 1, 0.5), 0.8, 0.8)
    assert np.abs(c.e1[1:, :2] - x0[1:]).max() > 1e-3        # x/y peers now move
    for q in (1, 2):
        assert np.allclose(c.qubit_bloch(q), d.qubit_bloch(q), atol=1e-6)


# ── measure_pair: third qubits are beyond the second-order closure ───────────────────────
def test_pair_third_qubit_discrepancy_is_pinned():
    """Cumulant leaves qubit k untouched; dense moves it via the 3-body <σ_k A>. Measured on
    8 correlated 3-qubit states, basis (x,z): max |Δe1[k]| (= the discrepancy, since cumulant
    does not move k) is 0.180 at strength .5 and 0.335 at 1.0 (mix .9); 0.200/0.371 pure.
    Pinned as an UPPER bound and as non-trivial, so a closure change shows up here."""
    from umwelt.substrate import belavkin as B
    worst = {}
    for strength in (0.5, 1.0):
        w = 0.0
        for sd in range(8):
            r = np.random.default_rng(1000 + sd)
            rho = _correlated(3, sd, 0.9)
            i, j = (int(x) for x in r.choice(3, 2, replace=False))
            k = ({0, 1, 2} - {i, j}).pop()
            rec = float(r.uniform(-1, 1))
            d = QubitCluster("m", list("abc"))
            d.evolver.dm.rho[:] = rho
            d.measure_pair(i, j, rec, strength, ("x", "z"))
            e1d, _ = _all_moments(d.rho, 3)
            e1, e2 = _all_moments(rho, 3)
            k0, k2 = e1[k].copy(), e2[k].copy()
            B.measure_cumulant_pair(e1, e2, i, j, rec, strength, ("x", "z"))
            assert np.array_equal(e1[k], k0) and np.array_equal(e2[k], k2)   # k untouched
            w = max(w, np.abs(e1[k] - e1d[k]).max())
        worst[strength] = w
    assert 0.1 < worst[0.5] < 0.25 and 0.2 < worst[1.0] < 0.40
