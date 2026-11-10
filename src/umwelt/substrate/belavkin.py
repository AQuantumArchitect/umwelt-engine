"""Belavkin weak-measurement update — the principled sibling of observe_qubit.

The α-blend in observe_qubit is an ad-hoc convex mix (NOVELTY.md §A2 names the
simplification: "no measurement back-action noise term… dropped the stochastic
innovation"). This module implements the real thing at the single-qubit level — the
discrete Kraus form of the Belavkin/quantum-filtering update for a weak σ_z
measurement — plus the second-cumulant cross-update that lets a measurement of one
qubit move its correlated peers. See docs/QUANTUM_KALMAN.md (the estimator ladder);
this is rung L4 at cumulant order 2.

The measurement model
─────────────────────
A sensor reading on an event/discrete leaf is a noisy draw around the eigenvalues ±1
(an occupancy detector says "occupied-ish", not "the mean is 0.3"). The conditioned update
for record y with effective strength s ≥ 0 is the two-outcome Gaussian Kraus

    M(y) = diag(c₊, c₋),   c± = exp(−s·(y ∓ 1)² / 4),   ρ' = MρM† / Tr(MρM†)

which on the Bloch vector (x, y_b, z) works out to closed form with w± = c±²:

    N   = w₊(1+z) + w₋(1−z)              (2 × the normalization)
    z'  = (w₊(1+z) − w₋(1−z)) / N
    x', y'_b = (2·c₊c₋ / N) · (x, y_b)   (coherence factor ≤ 1: back-action)

Equivalently z' = tanh(atanh(z) + s·y): **the update is a shift in log-odds space by
s·y** — classical Bayesian conditioning of a two-state belief, with the coherence
damping riding along. Everything the ladder promises is visible here:

  • Wonham gain — weak limit Δz ≈ s·(1−z²)·y: the gain carries the bounded binary
    variance (1−z²); poles stop listening, the equator listens hardest.
  • Provable no-op — s = 0 ⇒ w₊ = w₋ ⇒ z' = z and the coherence factor is exactly 1.
    The confidence contract (confidence 0 → nothing happens) is a theorem of the
    formula, not a guard branch.
  • Back-action — an informative record (|y| > 0) shrinks x/y_b: pinning WHERE damps
    the phase that carries WHEN. A pure state stays pure (Kraus of one record).
  • α ↔ s mapping — at the equator, Δz = tanh(s·y) ≈ s·y while the α-blend gives
    α·y: **strength ≈ alpha** to first order at maximum uncertainty; away from the
    equator Belavkin saturates where the blend overshoots. Callers reuse their
    existing collapse_alpha (confidence pre-folded, same convention as observe_qubit).

The cumulant cross-update (the joint-Kalman payoff)
───────────────────────────────────────────────────
Exact conditioning on one qubit moves the means of correlated qubits (the full-ρ
Kraus does this automatically through the off-diagonal blocks). At cumulant order 2
we apply the linear-Gaussian conditioning:

    K_j = cov_zz(j, i) / (1 − z_i² + ε)          (regression gain onto the record)
    z_j += K_j · Δz_i                            (mean transfer)
    C(i,·) ← (1 − γ)·C(i,·),  γ = s·v/(1 + s·v)  (connected-covariance shrink, v=1−z_i²)

On a classically z-correlated pair this mean transfer is EXACT (test-pinned against
the full-ρ Kraus); the covariance shrink is the Kalman (1−K) form, approximate under
the closure. Note the old observe_qubit path only DEcorrelates on observation — the
cross-update is strictly more informative and is the reason to climb the ladder.

Gating: reservoir.ingest branches on UMWELT_BELAVKIN (default OFF → the observe
path is byte-unchanged). Substrates expose `measure_qubit(idx, record_z, strength,
confidence=None)` built on the helpers here; `confidence` is recorded as a gauge
quantity only — the caller folds it into `strength`, the same convention as
observe_qubit (see the contract note there).
"""
from __future__ import annotations

import math
import os

import numpy as np

_EPS = 1e-9

Z_CAP = 0.98   # the pole floor (purity floor), moved from proofs/ladder_walk.py. Repeated
               # one-sided evidence drives log-odds unbounded: the Wonham gain (1−z²) → 0
               # and the cross-update gain cov/(1−z²) explodes. Load-bearing for any L4
               # use (docs/THEORY.md §4). Keeps v ≥ 1 − Z_CAP² ≈ 0.04, so gain ≤ ~25·|cov|.


def env_belavkin_enabled() -> bool:
    """UMWELT_BELAVKIN=1 switches the reservoir's sensor-observe path from the
    α-blend to the Belavkin measurement. Unset/other → off, byte-identical."""
    return os.environ.get("UMWELT_BELAVKIN", "") == "1"


def kraus_weights(record_z: float, strength: float) -> tuple[float, float]:
    """(w₊, w₋) = squared Kraus amplitudes for record `record_z` at strength s.
    Computed in a shifted exponent so w's stay well-scaled for large s."""
    y = float(record_z)
    s = max(0.0, float(strength))
    e_plus = -s * (y - 1.0) ** 2 / 2.0
    e_minus = -s * (y + 1.0) ** 2 / 2.0
    m = max(e_plus, e_minus)
    return math.exp(e_plus - m), math.exp(e_minus - m)


def measure_bloch(
    bloch, record_z: float, strength: float
) -> tuple[np.ndarray, float, float]:
    """Apply the conditioned weak σ_z measurement to one Bloch vector.

    Returns (bloch', gain_dz, coherence_factor):
      bloch'            — the post-measurement (x, y, z)
      gain_dz           — z' − z (the innovation actually applied)
      coherence_factor  — the 2·c₊c₋/N transverse damping ∈ (0, 1]

    strength ≤ 0 is the exact no-op (returns a copy, gain 0, factor 1).
    """
    b = np.asarray(bloch, dtype=float).copy()
    s = float(strength)
    if s <= 0.0:
        return b, 0.0, 1.0
    x, yb, z = float(b[0]), float(b[1]), float(b[2])
    z = max(-1.0, min(1.0, z))
    w_plus, w_minus = kraus_weights(record_z, s)
    n = w_plus * (1.0 + z) + w_minus * (1.0 - z)
    if n <= _EPS:                      # unreachable for w's from kraus_weights; belt+braces
        return b, 0.0, 1.0
    z_new = (w_plus * (1.0 + z) - w_minus * (1.0 - z)) / n
    coh = 2.0 * math.sqrt(w_plus * w_minus) / n
    b[0], b[1], b[2] = coh * x, coh * yb, z_new
    # Numerical guard: keep the result inside the Bloch ball.
    r = float(np.linalg.norm(b))
    if r > 1.0:
        b /= r
    return b, z_new - z, coh


def kalman_shrink(strength: float, z: float) -> float:
    """γ = s·v/(1 + s·v) with v = 1 − z² — the Kalman-gain-shaped fraction by which
    the measured qubit's connected covariance shrinks (cov' = (1−γ)·cov)."""
    v = max(0.0, 1.0 - float(z) ** 2)
    sv = max(0.0, float(strength)) * v
    return sv / (1.0 + sv)


def measure_cumulant(e1: np.ndarray, e2: np.ndarray, idx: int,
                     record_z: float, strength: float, peer_full: bool = False) -> float:
    """In-place Belavkin measurement of qubit `idx` on a cumulant state (e1, e2).

    1. Conditioned single-qubit update on e1[idx] (measure_bloch).
    2. Mean transfer to every peer j via the regression gain K_j = cov_zz(j,i)/v_i.
    3. Connected covariance C(i,j) shrinks by (1−γ) (Kalman form) and i's
       transverse rows additionally damp by the coherence factor.

    `peer_full=True` (used by the off-axis path) generalises step 2 to all three Pauli
    components of every peer: Δ⟨σ_a^j⟩ = cov(σ_z^i, σ_a^j)/v_i · Δz for a ∈ {x,y,z}. The default
    (False) is the legacy z-only peer update, kept so measure_qubit stays bit-for-bit.

    Returns the applied Δz on the measured qubit. strength ≤ 0 touches nothing. Only the
    pre-measurement z clip lives here; call `clamp_belavkin` once per bin afterwards.
    """
    if strength <= 0.0:
        return 0.0
    n = e1.shape[0]
    i = int(idx)
    # Pole floor BEFORE the measurement keeps v = 1−z² off zero (the 1e9-gain bug).
    np.clip(e1[:, 2], -Z_CAP, Z_CAP, out=e1[:, 2])
    old_i = e1[i].copy()
    z_i = float(old_i[2])
    v_i = max(_EPS, 1.0 - z_i * z_i)

    new_i, dz, coh = measure_bloch(old_i, record_z, strength)
    gamma = kalman_shrink(strength, z_i)

    # Peers first (they read the PRE-measurement covariance), then the qubit itself.
    for j in range(n):
        if j == i:
            continue
        old_j = e1[j].copy()
        conn = e2[i, j] - np.outer(old_i, old_j)      # connected C(i,j), 3×3
        k_j = float(conn[2, 2]) / v_i                  # cov_zz(j,i)/var_i
        new_j = old_j.copy()
        new_j[2] = old_j[2] + k_j * dz
        if peer_full:
            new_j[0] = old_j[0] + float(conn[2, 0]) / v_i * dz
            new_j[1] = old_j[1] + float(conn[2, 1]) / v_i * dz
        rj = float(np.linalg.norm(new_j))
        if rj > 1.0:
            new_j /= rj
        conn_new = (1.0 - gamma) * conn
        conn_new[0, :] *= coh                          # i's transverse correlations damp
        conn_new[1, :] *= coh
        e1[j] = new_j
        e2[i, j] = conn_new + np.outer(new_i, new_j)
        e2[j, i] = e2[i, j].T
    e1[i] = new_i
    return dz


def clamp_belavkin(e1: np.ndarray, e2: np.ndarray) -> None:
    """The once-per-bin post-measurement clip: z → ±Z_CAP on every qubit and e2 → [−1, 1]
    (pair correlators obey |⟨σσ⟩| ≤ 1). Opt-in, NOT run per measurement: later
    measurements in a bin must see the un-clipped e2, as the ladder-walk proof does."""
    np.clip(e1[:, 2], -Z_CAP, Z_CAP, out=e1[:, 2])
    np.clip(e2, -1.0, 1.0, out=e2)


def measure_rho(rho: np.ndarray, idx: int, n_qubits: int,
                record_z: float, strength: float) -> np.ndarray:
    """Exact full-ρ Kraus update: ρ' = MρM†/Tr with M = I⊗…⊗diag(c₊,c₋)⊗…⊗I.
    The multipartite ground truth the cumulant path approximates (and the tests
    compare against). Returns a NEW matrix; strength ≤ 0 returns rho unchanged."""
    if strength <= 0.0:
        return rho
    w_plus, w_minus = kraus_weights(record_z, strength)
    c_plus, c_minus = math.sqrt(w_plus), math.sqrt(w_minus)
    m1 = np.array([[c_plus, 0.0], [0.0, c_minus]], dtype=rho.dtype)
    op = np.array([[1.0]], dtype=rho.dtype)
    for q in range(n_qubits):
        op = np.kron(op, m1 if q == idx else np.eye(2, dtype=rho.dtype))
    out = op @ rho @ op.conj().T
    tr = float(np.real(np.trace(out)))
    if tr <= _EPS:
        return rho
    return out / tr


# ── Measurement along an arbitrary axis (measure_axis) ──────────────────────────────────
#
# A weak measurement of σ·n is the σ_z measurement conjugated by the SO(3) rotation R that
# carries n to ẑ (a local unitary on the measured qubit). Every representation below uses
# that identity, so it is exact wherever the z measurement is:
#   dense      — Kraus M = c₊P₊ + c₋P₋ with P± = (I ± n·σ)/2 on the qubit (exact, multipartite)
#   Bloch      — rotate b → Rb, apply measure_bloch, rotate back (exact for a single qubit)
#   cumulant   — rotate qubit i's rows of (e1, e2) into the n frame, run the z routine
#                `measure_cumulant`, rotate back. Same closure as the z path, nothing new.
# axis == (0,0,1) delegates to the z routines untouched (bit-for-bit with measure_qubit).

_AXIS_LETTERS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def unit_axis(axis) -> tuple[float, float, float]:
    """'x'|'y'|'z' or any non-zero 3-vector → a unit 3-tuple (ValueError otherwise)."""
    if isinstance(axis, str):
        try:
            return _AXIS_LETTERS[axis.lower()]
        except KeyError:
            raise ValueError(f"axis {axis!r} is not 'x', 'y', 'z' or a 3-vector") from None
    try:
        x, y, z = (float(c) for c in axis)
    except (TypeError, ValueError):
        raise ValueError(f"axis {axis!r} is not 'x', 'y', 'z' or a 3-vector") from None
    n = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"axis {axis!r} is not a finite non-zero vector")
    return (x / n, y / n, z / n)


def is_z_axis(axis) -> bool:
    return tuple(unit_axis(axis)) == (0.0, 0.0, 1.0)


def frame_rotation(axis) -> np.ndarray:
    """The proper rotation R (3×3) with R·n̂ = ẑ. Identity for n̂ = ẑ exactly."""
    n = np.array(unit_axis(axis), dtype=float)
    c = float(n[2])
    if n[0] == 0.0 and n[1] == 0.0:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    v = np.cross(n, np.array([0.0, 0.0, 1.0]))
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def measure_bloch_axis(bloch, axis, record: float, strength: float):
    """measure_bloch along n̂: (bloch', gain_dn, coherence_factor); dn = Δ⟨σ·n̂⟩."""
    if is_z_axis(axis):
        return measure_bloch(bloch, record, strength)
    R = frame_rotation(axis)
    b_n, dn, coh = measure_bloch(R @ np.asarray(bloch, dtype=float), record, strength)
    return R.T @ b_n, dn, coh


def measure_cumulant_axis(e1: np.ndarray, e2: np.ndarray, idx: int, axis,
                          record: float, strength: float) -> float:
    """In-place weak σ·n̂ measurement of qubit `idx` on a cumulant state (e1, e2).

    Derivation (second order, the z routine's closure): rotate qubit i's rows of (e1, e2) by R
    (R·n̂ = ẑ; FIRST Pauli index of each e2[i,j] only) so n̂ = ẑ. With v = 1−⟨σ_n^i⟩², innovation
    Δn and connected C_a = e2[i,j][z,a] − ⟨σ_n^i⟩⟨σ_a^j⟩:
      Δ⟨σ_a^j⟩ = (C_a / v)·Δn  for every peer j and a ∈ {x,y,z}   (Kalman regression gain),
      C(i,j) → (1−γ)·C(i,j), i's transverse rows also × coh; e2[i,j] = C' + ⟨σ^i⟩'⊗⟨σ^j⟩';
      e2[j,k] between peers is left untouched, as the z routine does (no new closure). Rotate back.
    Peer means are exact to the stored moments (Δ⟨σ_a^j⟩ = d·C_a/N identically) up to the pole
    floor Z_CAP on near-pure states; e2[i,j] (the (1−γ) shrink) and e2[j,k] are NOT exact.
    The z axis delegates to measure_cumulant UNCHANGED (z-only peer update, bit-for-bit).
    Returns the applied Δ⟨σ·n̂⟩. strength ≤ 0 touches nothing."""
    if strength <= 0.0:
        return 0.0
    if is_z_axis(axis):
        return measure_cumulant(e1, e2, idx, record, strength)
    R = frame_rotation(axis)
    i, n = int(idx), e1.shape[0]

    def _to_frame(M):                      # qubit i's index is the FIRST Pauli index of e2[i,j]
        e1[i] = M @ e1[i]
        for j in range(n):
            if j != i:
                e2[i, j] = M @ e2[i, j]
                e2[j, i] = e2[i, j].T
    _to_frame(R)
    try:
        dn = measure_cumulant(e1, e2, i, record, strength, peer_full=True)
    finally:
        _to_frame(R.T)
    return dn


def measure_rho_axis(rho: np.ndarray, idx: int, n_qubits: int, axis,
                     record: float, strength: float) -> np.ndarray:
    """Exact full-ρ Kraus update for a weak σ·n̂ measurement of qubit `idx`:
    ρ' = MρM†/Tr, M = c₊P₊ + c₋P₋ on that qubit. n̂ = ẑ is `measure_rho` exactly."""
    if strength <= 0.0:
        return rho
    if is_z_axis(axis):
        return measure_rho(rho, idx, n_qubits, record, strength)
    nx, ny, nz = unit_axis(axis)
    w_plus, w_minus = kraus_weights(record, strength)
    c_plus, c_minus = math.sqrt(w_plus), math.sqrt(w_minus)
    na = np.array([[nz, nx - 1j * ny], [nx + 1j * ny, -nz]], dtype=complex)   # n̂·σ
    m1 = ((c_plus + c_minus) / 2.0) * np.eye(2) + ((c_plus - c_minus) / 2.0) * na
    m1 = m1.astype(rho.dtype)
    op = np.array([[1.0]], dtype=rho.dtype)
    for q in range(n_qubits):
        op = np.kron(op, m1 if q == idx else np.eye(2, dtype=rho.dtype))
    out = op @ rho @ op.conj().T
    tr = float(np.real(np.trace(out)))
    if tr <= _EPS:
        return rho
    return out / tr


# ── Joint (pair) observable: measure_pair ───────────────────────────────────────────────
#
# A weak measurement of the two-qubit correlator A = (a·σ)^(i) (b·σ)^(j), eigenvalues ±1,
# with a record in [−1,1]. Same Kraus weights as the single-qubit case (A has the same ±1
# spectrum): M = c₊P₊ + c₋P₋ = ((c₊+c₋)/2)·I + ((c₊−c₋)/2)·A, P± = (I ± A)/2.

def measure_rho_pair(rho: np.ndarray, i: int, j: int, n_qubits: int, record: float,
                     strength: float, basis=("z", "z")) -> np.ndarray:
    """Exact full-ρ Kraus update for the joint observable; returns a NEW matrix
    (strength ≤ 0 returns rho unchanged)."""
    if strength <= 0.0:
        return rho
    if int(i) == int(j):
        raise ValueError("measure_pair needs two distinct qubits")
    w_plus, w_minus = kraus_weights(record, strength)
    c_plus, c_minus = math.sqrt(w_plus), math.sqrt(w_minus)

    def _pauli(axis):
        nx, ny, nz = unit_axis(axis)
        return np.array([[nz, nx - 1j * ny], [nx + 1j * ny, -nz]], dtype=rho.dtype)
    ops = {int(i): _pauli(basis[0]), int(j): _pauli(basis[1])}
    A = np.array([[1.0]], dtype=rho.dtype)
    for q in range(n_qubits):
        A = np.kron(A, ops.get(q, np.eye(2, dtype=rho.dtype)))
    M = ((c_plus + c_minus) / 2.0) * np.eye(rho.shape[0], dtype=rho.dtype) \
        + ((c_plus - c_minus) / 2.0) * A
    out = M @ rho @ M.conj().T
    tr = float(np.real(np.trace(out)))
    if tr <= _EPS:
        return rho
    return out / tr


# σ_c·σ_z = i·_PZ_SIGN[c]·σ_{_PZ_OTHER[c]}  for c ∈ {x, y}
_PZ_SIGN = {0: -1.0, 1: 1.0}
_PZ_OTHER = {0: 1, 1: 0}


def measure_cumulant_pair(e1: np.ndarray, e2: np.ndarray, i: int, j: int, record: float,
                          strength: float, basis=("z", "z")) -> float:
    """In-place joint measurement of σ_a^(i)σ_b^(j) on a cumulant state (e1, e2).

    Both qubits are rotated into their measurement frames (so A = σ_z σ_z), the update is
    applied, then rotated back. In that frame every Pauli B on {i,j} either commutes with A
    (then ⟨B⟩' = [s⟨B⟩ + d⟨BA⟩]/N, with ⟨BA⟩ another stored 1- or 2-body moment) or
    anticommutes (then ⟨B⟩' = (c₊c₋/N)⟨B⟩), where s=(w₊+w₋)/2, d=(w₊−w₋)/2,
    N = s + d⟨A⟩. So e1[i], e1[j] and the whole 3×3 block e2[i,j] update EXACTLY, from the
    stored moments alone (this includes ⟨A⟩ itself — the e₂ "toward the record" move).
    APPROXIMATE / NOT DONE: every other qubit k is left UNTOUCHED (e1[k], e2[k,·]). Its exact
    change is ⟨σ_k^a⟩' − ⟨σ_k^a⟩ = d·(⟨σ_k^a A⟩ − ⟨σ_k^a⟩⟨A⟩)/N, which needs the 3-body moment
    ⟨σ_k^a σ_i σ_j⟩; the second-order state stores none (the closure zeroes the 3rd cumulant),
    and inventing one would be a new closure. Measured dense-vs-cumulant gap on correlated
    3-qubit states: up to ~0.18 (s=.5) / ~0.34 (s=1) in a third qubit's Bloch component
    (pinned in tests/test_measure_axis_pair.py::test_pair_third_qubit_discrepancy_is_pinned).
    Returns the applied Δ⟨A⟩. strength ≤ 0 touches nothing."""
    if strength <= 0.0:
        return 0.0
    i, j, n = int(i), int(j), e1.shape[0]
    if i == j:
        raise ValueError("measure_pair needs two distinct qubits")
    Ra, Rb = frame_rotation(basis[0]), frame_rotation(basis[1])

    def _rot(Mi, Mj):                       # apply local frame changes to every moment touching i, j
        e1[i] = Mi @ e1[i]
        e1[j] = Mj @ e1[j]
        for k in range(n):
            if k in (i, j):
                continue
            e2[i, k] = Mi @ e2[i, k]; e2[k, i] = e2[i, k].T
            e2[j, k] = Mj @ e2[j, k]; e2[k, j] = e2[j, k].T
        e2[i, j] = Mi @ e2[i, j] @ Mj.T
        e2[j, i] = e2[i, j].T

    _rot(Ra, Rb)
    try:
        w_plus, w_minus = kraus_weights(record, strength)
        s, d = 0.5 * (w_plus + w_minus), 0.5 * (w_plus - w_minus)
        a, b, E = e1[i].copy(), e1[j].copy(), e2[i, j].copy()
        m = float(E[2, 2])
        N = s + d * m
        if N <= _EPS:
            return 0.0
        coh = math.sqrt(w_plus * w_minus) / N
        na, nb, nE = a.copy(), b.copy(), E.copy()
        na[0], na[1] = coh * a[0], coh * a[1]
        nb[0], nb[1] = coh * b[0], coh * b[1]
        na[2] = (s * a[2] + d * b[2]) / N
        nb[2] = (s * b[2] + d * a[2]) / N
        nE[2, 2] = (s * m + d) / N
        for c in range(3):
            for dd in range(3):
                if (c == 2) != (dd == 2):          # exactly one factor is σ_z → anticommutes with A
                    nE[c, dd] = coh * E[c, dd]
        for c in (0, 1):
            for dd in (0, 1):                      # both transverse → commutes with A
                ba = -_PZ_SIGN[c] * _PZ_SIGN[dd] * E[_PZ_OTHER[c], _PZ_OTHER[dd]]
                nE[c, dd] = (s * E[c, dd] + d * ba) / N
        e1[i], e1[j], e2[i, j] = na, nb, nE
        e2[j, i] = nE.T
        dA = float(nE[2, 2] - m)
    finally:
        _rot(Ra.T, Rb.T)
    return dA
