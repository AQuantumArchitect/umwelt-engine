"""Resonance and tension on mixed states and ensembles (grammar §18). numpy only.

* ``〰`` resonance = Uhlmann fidelity  F(ρ,σ) = (tr √(√ρ σ √ρ))²   (in [0,1]; 1 identical,
  0 orthogonal supports).
* ``⚡`` tension  = Bures distance      d_B = √(2(1 − √F))        (0 identical; √2 for
  orthogonal states).
* ensemble disagreement = Holevo χ = S(Σ pᵢρᵢ) − Σ pᵢ S(ρᵢ), von Neumann entropy in nats; it
  equals the Jensen–Shannon divergence (nats) when the states commute.

Density matrices are n×n (single registers are 2×2). ``resonance``/``tension`` also take
bound sentences: a ``BPopulation`` or ``BBloch`` (or a ``Bound`` whose leaf is one), read in
the register's own frame. A dark or evidence leaf has no state and is refused: dark is not
a half. A Bloch node with a dark phase (``·``) is read DEPHASED (transverse part dropped,
nothing invented); numeric and model phases are used as written.

Provenance (v0.4 §26, least-of-inputs): every metric returns a ``Num``. It is measured only
if every input number was; one forecast (``~``) input (a population, an r, θ or φ~, a
probability weight) makes the result forecast; a dark input makes it dark. Raw arrays and
plain floats are measured. A dephased Bloch reading (dark phase) is the documented §18
convention, not an invented number, so the dark phase alone does not darken the result.
"""
from __future__ import annotations

import math

import numpy as np

from umwelt.glyph.bind import BBloch, BPopulation, Bound
from umwelt.glyph.num import DARK, Num, least
from umwelt.glyph.scope import GlyphError

_TOL = 1e-9


class MetricError(GlyphError):
    """A state a metric cannot be taken of."""


def rho_from_bloch(r: float, theta: float, phi: float | None = 0.0) -> np.ndarray:
    """ρ = ½(I + r·n̂·σ), n̂ = (sinθ cosφ, sinθ sinφ, cosθ). ``phi=None`` (dark phase)
    drops the transverse part: ρ = ½(I + r cosθ σz)."""
    if not 0.0 <= r <= 1.0 + _TOL:
        raise MetricError(f"Bloch r={r} is outside [0,1]")
    z = r * math.cos(theta)
    if phi is None:
        x = y = 0.0
    else:
        x, y = r * math.sin(theta) * math.cos(phi), r * math.sin(theta) * math.sin(phi)
    return 0.5 * np.array([[1.0 + z, x - 1j * y], [x + 1j * y, 1.0 - z]], dtype=complex)


def rho_from_population(p: float) -> np.ndarray:
    """The diagonal state diag(p, 1−p): population p along pole 0."""
    if not 0.0 <= p <= 1.0:
        raise MetricError(f"population p={p} is outside [0,1]")
    return np.array([[p, 0.0], [0.0, 1.0 - p]], dtype=complex)


def to_rho(x) -> np.ndarray:
    """A density matrix from a bound population/Bloch node, a ``Bound`` of one, or an array."""
    if isinstance(x, Bound):
        x = x.leaf
    if isinstance(x, BPopulation):
        return rho_from_population(x.p0)
    if isinstance(x, BBloch):
        return rho_from_bloch(x.r, x.theta, x.phi)
    if isinstance(x, np.ndarray) or (isinstance(x, (list, tuple)) and x and
                                     isinstance(x[0], (list, tuple, np.ndarray))):
        rho = np.asarray(x, dtype=complex)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise MetricError("a density matrix must be square")
        return rho
    raise MetricError(f"{type(x).__name__} carries no state (only a population or a Bloch "
                      f"statement does; dark is not a half)")


def prov_of(x) -> str:
    """Least-of-inputs provenance of a state argument (see the module docstring)."""
    if isinstance(x, Bound):
        x = x.leaf
    if isinstance(x, BPopulation):
        return least(x.p0)
    if isinstance(x, BBloch):
        return least(x.r, x.theta, *(() if x.phi is None else (x.phi,)))
    return "measured"


def _psd_sqrt(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((a + a.conj().T) / 2)
    w = np.where(w > 1e-14, w, 0.0)        # drop rounding noise on rank-deficient states
    return (v * np.sqrt(w)) @ v.conj().T


def uhlmann_fidelity(rho1, rho2) -> float:
    """F = (tr √(√ρ₁ ρ₂ √ρ₁))², clipped to [0,1]."""
    a, b = to_rho(rho1), to_rho(rho2)
    if a.shape != b.shape:
        raise MetricError(f"state dimensions differ: {a.shape} vs {b.shape}")
    # tr √(√a b √a) = the trace norm of √a √b (sum of its singular values)
    f = float(np.sum(np.linalg.svd(_psd_sqrt(a) @ _psd_sqrt(b), compute_uv=False))) ** 2
    return Num(min(1.0, max(0.0, f)), least(prov_of(rho1), prov_of(rho2)))


def bures_distance(rho1, rho2) -> float:
    """d_B = √(2(1 − √F)); 0 for identical states, √2 for orthogonal ones."""
    f = uhlmann_fidelity(rho1, rho2)
    return Num(math.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(float(f))))), f.prov)


def von_neumann_entropy(rho) -> float:
    """S(ρ) = −tr ρ ln ρ, in nats."""
    w = np.linalg.eigvalsh((to_rho(rho) + to_rho(rho).conj().T) / 2)
    w = w[w > 1e-15]
    return Num(float(-np.sum(w * np.log(w))), prov_of(rho))


def holevo_chi(probs, rhos) -> float:
    """χ = S(Σ pᵢρᵢ) − Σ pᵢ S(ρᵢ) (nats). ≥ 0; 0 for identical states."""
    p = np.asarray(probs, dtype=float)
    states = [to_rho(r) for r in rhos]
    if p.ndim != 1 or len(p) != len(states) or len(states) == 0:
        raise MetricError("probs and rhos must be non-empty and the same length")
    if np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
        raise MetricError("probs must be non-negative and sum to 1")
    mean = sum(pi * s for pi, s in zip(p, states))
    prov = least(*(prov_of(r) for r in rhos), *(getattr(x, "prov", "measured") for x in probs))
    return Num(max(0.0, float(von_neumann_entropy(mean)) - float(sum(
        pi * float(von_neumann_entropy(s)) for pi, s in zip(p, states)))), prov)


def resonance(a, b) -> float:
    """〰 between two bound states: Uhlmann fidelity."""
    return uhlmann_fidelity(a, b)


def tension(a, b) -> float:
    """⚡ between two bound states: Bures distance."""
    return bures_distance(a, b)
