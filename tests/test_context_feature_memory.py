"""A context feature cap must bound construction, not only the final array."""
import itertools
import math

import numpy as np
import pytest

from umwelt.foresight.forecast_surface import ForecastSurface


def surface(n_leaves, order, cap):
    leaves = tuple((f"node{i}", "value") for i in range(n_leaves))
    return ForecastSurface(leaves, horizons_min=(1.,), n_context=n_leaves,
                           interaction_order=order, max_context_features=cap)


def old_selection(n_atoms, order, cap):
    monos = [c for k in range(1, min(order, n_atoms) + 1)
             for c in itertools.combinations(range(n_atoms), k)]
    if cap and len(monos) > cap:
        singles = [m for m in monos if len(m) == 1]
        higher = [m for m in monos if len(m) > 1]
        rng = np.random.default_rng(0)
        if cap <= len(singles):
            monos = [monos[i] for i in sorted(rng.choice(len(monos), size=cap, replace=False))]
        else:
            monos = singles + [higher[i] for i in sorted(
                rng.choice(len(higher), size=cap-len(singles), replace=False))]
    return tuple(monos)


@pytest.mark.parametrize("leaves", [1, 2, 4, 6])
@pytest.mark.parametrize("order", [1, 2, 4, 12])
@pytest.mark.parametrize("cap", [0, 1, 4, 16, 64])
def test_bounded_selection_preserves_existing_exact_feature_identity(leaves, order, cap):
    assert surface(leaves, order, cap)._monomials == old_selection(2 * leaves, order, cap)


def test_cap_does_not_build_discarded_combinations(monkeypatch):
    original = itertools.combinations
    consumed = 0
    def guarded(*args):
        nonlocal consumed
        for row in original(*args):
            consumed += 1
            if consumed > 64:
                raise AssertionError("constructed discarded context features beyond the cap")
            yield row
    monkeypatch.setattr(itertools, "combinations", guarded)
    result = surface(12, 6, 64)
    assert result.n_context == 64


def test_astronomical_combination_space_is_sampled_without_enumeration(monkeypatch):
    def forbidden(*args):
        raise AssertionError("cannot enumerate the combination space")
    monkeypatch.setattr(itertools, "combinations", forbidden)
    assert sum(math.comb(100, k) for k in range(1, 51)) > np.iinfo(np.int64).max
    first = surface(50, 50, 64)._monomials
    second = surface(50, 50, 64)._monomials
    assert first == second and len(set(first)) == 64
    assert all(tuple(sorted(set(row))) == row and 1 <= len(row) <= 50 for row in first)
    assert all(0 <= atom < 100 for row in first for atom in row)


def test_higher_order_sample_keeps_all_singletons_when_the_cap_fits_them():
    result = surface(35, 35, 80)
    assert result._monomials[:70] == tuple((i,) for i in range(70))
    assert len(result._monomials) == 80
