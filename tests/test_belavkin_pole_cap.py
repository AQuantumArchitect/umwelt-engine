"""Pole floor (Z_CAP) + e2 clamp in the library Belavkin path (THEORY §4, #43 item 7)."""
import numpy as np

from umwelt.substrate.belavkin import Z_CAP, measure_cumulant
from umwelt.substrate.cumulant_cluster import CumulantCluster


def _cluster():
    return CumulantCluster("m", ["a", "b"], gamma=0.1, dt=0.01)


def test_one_sided_evidence_stays_bounded():
    c = _cluster()
    for _ in range(500):
        c.measure_qubit(0, 1.0, strength=2.0)
        c.clamp_belavkin()   # once per bin
        assert np.max(np.abs(c.e2)) <= 1.0
        assert np.max(np.abs(c.e1[:, 2])) <= Z_CAP + 1e-12
    assert np.isfinite(c.e1).all() and np.isfinite(c.e2).all()


def test_variance_floor_gain_not_explosive():
    # Pole state with a nonzero z-z covariance: the old v=_EPS floor gave K ~ 1e9.
    e1 = np.zeros((2, 3)); e1[0, 2] = 1.0
    e2 = np.zeros((2, 2, 3, 3)); e2[0, 1, 2, 2] = 0.3; e2[1, 0] = e2[0, 1].T
    before = e1.copy()
    dz = measure_cumulant(e1, e2, 0, 1.0, 1.0)
    assert abs(dz) <= 2.0
    assert abs(e1[1, 2] - before[1, 2]) < 1.0


def test_clamp_belavkin_is_the_post_bin_clip():
    c = _cluster()
    c.e1[:, 2] = 1.0
    c.e2[0, 1, 2, 2] = 5.0
    c.clamp_belavkin()
    assert np.max(np.abs(c.e2)) <= 1.0
    assert np.max(np.abs(c.e1[:, 2])) <= Z_CAP


def test_measure_cumulant_does_not_clip_e2_per_measurement():
    e1 = np.zeros((2, 3)); e2 = np.zeros((2, 2, 3, 3)); e2[0, 1, 0, 0] = 5.0
    measure_cumulant(e1, e2, 0, 1.0, 0.5)
    assert np.max(np.abs(e2)) > 1.0   # clipping is the caller's once-per-bin job


def test_ladder_walk_outputs_unchanged():
    """ladder_walk._run (library helper) is byte-identical to the original caller-side
    clips (z pre-measurement, z+e2 once per bin)."""
    from proofs import ladder_walk as lw
    reports, _ = lw.gridworld_stream(days=0.4)
    for rule in ("l0", "l4", "l4eta"):
        for coupled in (False, True):
            got = lw._run(rule, reports, coupled, 0.6)
            assert np.array_equal(got, _reference_run(lw, rule, reports, coupled, 0.6))


def _reference_run(lw, rule, reports, coupled, strength):
    # verbatim copy of the pre-#56 caller-side loop
    from umwelt.learning.observation_trust import ObservationTrust
    T, R = reports.shape
    c = lw._cluster(coupled)
    trust = ObservationTrust() if rule == "l4eta" else None
    est = np.zeros((T, R))
    for t in range(T):
        c.step(dt_scale=lw.DT_SCALE)
        c.clamp_physical()
        for k in range(R):
            z = reports[t, k]
            if np.isnan(z):
                continue
            np.clip(c.e1[:, 2], -Z_CAP, Z_CAP, out=c.e1[:, 2])
            if rule == "l0":
                c.observe_qubit(k, (0.0, 0.0, float(z)), alpha=min(1.0, strength))
            elif rule == "l4":
                c.measure_qubit(k, float(z), strength=strength)
            else:
                base = trust.learned_alpha((lw.NODES[k],), float(z), float(c.e1[k, 2]))
                c.measure_qubit(k, float(z), strength=strength * base)
        np.clip(c.e1[:, 2], -Z_CAP, Z_CAP, out=c.e1[:, 2])
        np.clip(c.e2, -1.0, 1.0, out=c.e2)
        est[t] = c.e1[:, 2]
    return est
