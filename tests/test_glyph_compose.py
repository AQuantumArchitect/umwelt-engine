"""v0.4 §26 composition: least-of-inputs through every derived number."""
import numpy as np
import pytest

from glyph_fixtures import ebind, terrarium
from umwelt.glyph import (holevo_chi, parse, resonance, rho_from_bloch, rho_from_population,
                          tension, uhlmann_fidelity)
from umwelt.glyph.metrics import MetricError, von_neumann_entropy
from umwelt.glyph.num import DARK, FORECAST, MEASURED, Num


def st(s):
    return ebind(parse(s), terrarium())


@pytest.mark.parametrize("a,b,want", [
    ("[.9]💧", "[.2]💧", MEASURED),
    ("[.9~]💧", "[.2]💧", FORECAST),
    ("[.9]💧", "[.2~]💧", FORECAST),
    ("[.9~]💧", "[.2~]💧", FORECAST),
])
def test_resonance_and_tension_least_of_inputs_populations(a, b, want):
    A, B = st(a), st(b)
    assert resonance(A, B).prov == want and tension(A, B).prov == want
    # the number itself is unchanged by its provenance
    assert float(resonance(A, B)) == pytest.approx(float(resonance(st("[.9]💧"), st("[.2]💧"))))


@pytest.mark.parametrize("s,want", [
    ("[0.8,1.0,·]🔋🪫", MEASURED),               # dephased reading is the §18 convention
    ("[0.8~,1.0,·]🔋🪫", FORECAST),
    ("[0.8,1.0~,·]🔋🪫", FORECAST),
    ("[0.8,1.0,2.0~]🔋🪫", FORECAST),
    ("[0.8~,1.0~,2.0~]🔋🪫", FORECAST),
])
def test_each_bloch_axis_forecasts_the_result(s, want):
    bound = st(s)
    assert von_neumann_entropy(bound).prov == want
    assert resonance(bound, bound).prov == want and tension(bound, st("[.9]💧")).prov == want


def test_numeric_phase_straight_row_with_receipt():
    from glyph_fixtures import two_basis_receipt
    b = ebind(parse("[0.8,1.0,2.0]🔋🪫"), terrarium(), two_basis_receipt("energy"))
    assert von_neumann_entropy(b).prov == MEASURED and tension(b, b).prov == MEASURED
    bf = ebind(parse("[0.8,1.0,2.0~]🔋🪫"), terrarium())
    assert tension(b, bf).prov == FORECAST and resonance(bf, b).prov == FORECAST


def test_entropy_and_chi_inherit_the_forecast():
    assert von_neumann_entropy(st("[.9]💧")).prov == MEASURED
    assert von_neumann_entropy(st("[.9~]💧")).prov == FORECAST
    rhos = [rho_from_population(.9), rho_from_population(.2)]
    assert holevo_chi([.5, .5], rhos).prov == MEASURED
    assert holevo_chi([Num(.5, FORECAST), Num(.5)], rhos).prov == FORECAST          # a forecast weight
    assert holevo_chi([.5, .5], [st("[.9~]💧"), st("[.2]💧")]).prov == FORECAST     # a forecast state
    assert holevo_chi([.5, .5], rhos) == pytest.approx(holevo_chi([.5, .5], [st("[.9~]💧"), st("[.2~]💧")]))


def test_raw_arrays_and_plain_floats_are_measured_and_values_unchanged():
    a, b = rho_from_population(.9), rho_from_bloch(1.0, 0.0, 0.0)
    assert uhlmann_fidelity(a, b).prov == MEASURED and tension(a, b).prov == MEASURED
    assert isinstance(tension(a, b), float)                       # still usable as a float


def test_dark_has_no_state_so_no_metric_and_never_a_silent_number():
    with pytest.raises(MetricError):
        resonance(st("[?]💧"), st("[.2]💧"))                       # binds to BDark: dark is not a half


def test_derived_arithmetic_on_bound_numbers_composes():
    p = st("[.9~]💧").leaf.p0
    q = st("[.2]💧").leaf.p0
    assert (p + q).prov == FORECAST and (p * q).prov == FORECAST and (q * q).prov == MEASURED
    assert (p * Num(0, DARK)).prov == DARK and (p + q) != float(p + q)
