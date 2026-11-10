"""v0.3 §18: Uhlmann / Bures / Holevo on mixed states and ensembles."""
import math

import numpy as np
import pytest

from glyph_fixtures import ebind, terrarium, two_basis_receipt
from umwelt.glyph import parse
from umwelt.glyph.metrics import (MetricError, bures_distance, holevo_chi, resonance,
                                  rho_from_bloch, rho_from_population, tension,
                                  uhlmann_fidelity, von_neumann_entropy)

RNG = np.random.default_rng(7)


def rand_rho(n):
    a = RNG.normal(size=(n, n)) + 1j * RNG.normal(size=(n, n))
    r = a @ a.conj().T
    return r / np.trace(r).real


def test_helpers_are_density_matrices():
    for r, t, p in [(1, 0, 0), (0.5, 1.0, 2.0), (0, 0, 0), (0.9, math.pi, 1.0)]:
        rho = rho_from_bloch(r, t, p)
        assert np.allclose(rho, rho.conj().T) and np.trace(rho).real == pytest.approx(1)
        assert np.linalg.eigvalsh(rho).min() >= -1e-12
    assert np.allclose(rho_from_bloch(1, 0, 0), [[1, 0], [0, 0]])
    assert np.allclose(rho_from_bloch(1, math.pi / 2, 0), 0.5 * np.ones((2, 2)))
    assert np.allclose(rho_from_population(0.3), np.diag([0.3, 0.7]))
    assert np.allclose(rho_from_bloch(0.6, 0.7, None), np.diag(np.diag(rho_from_bloch(0.6, 0.7, 0.4))))
    with pytest.raises(MetricError):
        rho_from_population(1.2)
    with pytest.raises(MetricError):
        rho_from_bloch(1.5, 0, 0)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_pin_identical_states(n):
    rho = rand_rho(n)
    assert uhlmann_fidelity(rho, rho) == pytest.approx(1.0, abs=1e-9)
    assert bures_distance(rho, rho) == pytest.approx(0.0, abs=1e-7)
    assert holevo_chi([0.3, 0.7], [rho, rho]) == pytest.approx(0.0, abs=1e-9)


def test_pin_orthogonal_poles():
    up, down = rho_from_bloch(1, 0, 0), rho_from_bloch(1, math.pi, 0)
    assert uhlmann_fidelity(up, down) == pytest.approx(0.0, abs=1e-12)
    assert bures_distance(up, down) == pytest.approx(math.sqrt(2), abs=1e-9)
    assert holevo_chi([.5, .5], [up, down]) == pytest.approx(math.log(2), abs=1e-9)


def test_pure_state_fidelity_is_overlap_squared_and_symmetric():
    a, b = rho_from_bloch(1, 0.4, 0.3), rho_from_bloch(1, 1.9, 2.0)
    va = np.array([math.cos(0.2), math.sin(0.2) * np.exp(0.3j)])
    vb = np.array([math.cos(0.95), math.sin(0.95) * np.exp(2.0j)])
    assert uhlmann_fidelity(a, b) == pytest.approx(abs(np.vdot(va, vb)) ** 2, abs=1e-9)
    m, n = rand_rho(3), rand_rho(3)
    assert uhlmann_fidelity(m, n) == pytest.approx(uhlmann_fidelity(n, m), abs=1e-9)
    assert 0 <= uhlmann_fidelity(m, n) <= 1
    assert bures_distance(m, n) <= math.sqrt(2) + 1e-12


def jsd(probs, dists):
    probs, dists = np.asarray(probs), np.asarray(dists)
    h = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    return h(probs @ dists) - sum(p * h(d) for p, d in zip(probs, dists))


@pytest.mark.parametrize("n,k", [(2, 2), (2, 4), (3, 3), (4, 5)])
def test_pin_commuting_chi_equals_jensen_shannon(n, k):
    dists = RNG.dirichlet(np.ones(n), size=k)
    probs = RNG.dirichlet(np.ones(k))
    chi = holevo_chi(probs, [np.diag(d) for d in dists])
    assert chi == pytest.approx(jsd(probs, dists), abs=1e-9)


def test_chi_noncommuting_is_below_entropy_of_mixture_and_positive():
    up, plus = rho_from_bloch(1, 0, 0), rho_from_bloch(1, math.pi / 2, 0)
    chi = holevo_chi([.5, .5], [up, plus])
    assert 0 < chi < math.log(2)
    assert von_neumann_entropy(rho_from_bloch(0, 0, 0)) == pytest.approx(math.log(2))
    for bad in ([.5], [-.5, 1.5], [.4, .4]):
        with pytest.raises(MetricError):
            holevo_chi(bad, [up, plus])


def test_resonance_and_tension_on_bound_nodes():
    sc = terrarium()
    pop = ebind(parse("[.9]💧"), sc)
    pop_flip = ebind(parse("[.1]🏜"), sc)               # same fact, canonical to pole 0
    other = ebind(parse("[.2]💧"), sc)
    assert resonance(pop, pop_flip) == pytest.approx(1.0) and tension(pop, pop_flip) == pytest.approx(0, abs=1e-7)
    # commuting diagonals: fidelity = (√(.9·.2) + √(.1·.8))²
    assert resonance(pop, other) == pytest.approx((math.sqrt(.18) + math.sqrt(.08)) ** 2)
    assert tension(pop, other) > 0
    rc = two_basis_receipt("energy")
    pure_up = ebind(parse("[1,0,0]🔋🪫"), sc, rc)
    pure_dn = ebind(parse("[1,3.141592653589793,0]🔋🪫"), sc, rc)
    assert resonance(pure_up, pure_dn) == pytest.approx(0, abs=1e-12)
    assert tension(pure_up, pure_dn) == pytest.approx(math.sqrt(2))
    mixed = ebind(parse("[0,0,·]🔋🪫"), sc)             # maximally mixed, phase dark
    assert resonance(mixed, mixed) == pytest.approx(1.0)
    assert resonance(pop, pop.leaf) == pytest.approx(1.0)       # leaf or Bound
    model = ebind(parse("[.8,1.0,2.0~]🔋🪫"), sc)
    assert 0 < tension(model, mixed) < math.sqrt(2)


def test_dark_and_evidence_have_no_state():
    sc = terrarium()
    pop = ebind(parse("[.9]💧"), sc)
    for s in ("💧?", "💧!0.5", "|💧⟩"):
        with pytest.raises(MetricError):
            tension(ebind(parse(s), sc), pop)
    with pytest.raises(MetricError):
        uhlmann_fidelity(np.eye(2) / 2, np.eye(3) / 3)
