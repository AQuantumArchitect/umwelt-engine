"""v0.4 §25-§27: every number is measured (straight), forecast (~) or dark (?)."""
import copy
import pickle

import pytest

from glyph_fixtures import terrarium, two_basis_receipt
from umwelt.glyph import (BindError, Bloch, Evidence, Measurement, ParseError, Population,
                          Receipt, bind, canonical, parse, parse_number)
from umwelt.glyph.bind import BBloch, BDark, BEvidence, BPopulation
from umwelt.glyph.num import DARK, FORECAST, MEASURED, Num, least

M, F, D = MEASURED, FORECAST, DARK


# ── the Num type ──
def test_num_never_equals_a_number_of_another_provenance():
    assert Num(0.8) == 0.8 and Num(0.8, F) != 0.8 and Num(0.8, F) != Num(0.8)
    assert Num(0.8, F) == Num(0.8, F) and Num(0, D) == Num(5, D) and Num(0, D) != 0.0
    assert hash(Num(0.8)) == hash(0.8) and hash(Num(0.8, F)) != hash(Num(0.8))
    assert copy.deepcopy(Num(0.8, F)).prov == F and pickle.loads(pickle.dumps(Num(1.5, D))).prov == D


def test_num_arithmetic_is_least_of_inputs():
    assert (Num(.5) + Num(.25)).prov == M and (Num(.5) + 0.25).prov == M
    assert (Num(.5, F) + Num(.25)).prov == F and (1 - Num(.25, F)).prov == F
    assert (Num(.5, F) * Num(.25, D)).prov == D and (Num(.5, D) - 1).prov == D
    assert (-Num(.5, F)).prov == F and abs(Num(-.5, F)).prov == F
    assert least(M, F) == F and least(F, D, M) == D and least() == M
    assert (1 - Num(.25, F)) == Num(.75, F) and (1 - Num(.25, F)) != 0.75


# ── parse ──
@pytest.mark.parametrize("s", [
    "[0.8~]☀", "☀!0.25~", "[0.9~,0.3~,1.2~]☀🌙", "[0.9,0.3,1.2~]☀🌙", "[0.9~,0.3,1.2]☀🌙",
    "[0.9,0.3~,·]☀🌙", "[?]☀", "☀!?", "[0.9,?,1.2~]☀🌙", "[0.9,0.3,·]☀🌙",
    "⟨☀|ψ⟩ = [0.8~]☀", "☀›🔥›[0.3~]⚫", "☀›🔥›⚫!0.4~"])
def test_round_trips(s):
    ast = parse(s)
    assert canonical(ast) == s and parse(canonical(ast)) == ast
    assert canonical(ast).count("~") == s.count("~")


def test_provenance_is_in_the_ast():
    assert parse("[0.8~]☀") == Population(Num(0.8, F), "☀") != parse("[0.8]☀")
    assert parse("☀!0.25~").eta.prov == F and parse("☀!0.25").eta.prov == M
    b = parse("[r,θ,φ]☀🌙".replace("r", "0.9~").replace("θ", "0.3").replace("φ", "1.2~"))
    assert (b.r.prov, b.theta.prov, b.phi.prov, b.phase) == (F, M, F, "model")
    assert parse("[?]☀").p.prov == D and parse("[0.9,0.3,?]☀🌙").phase == "dark"
    assert canonical(parse("[0.9,0.3,?]☀🌙")) == "[0.9,0.3,·]☀🌙"      # the v0.3 spelling stays
    assert parse("[ 0.8 ~ ]☀") == parse("[0.8~]☀")
    assert parse_number("0.5~") == Num(0.5, F) and parse_number("?").prov == D


@pytest.mark.parametrize("s", ["[0.8~~]☀", "☀!0.2~~", "[0.8~?]☀", "[?~]☀", "[~]☀", "☀!~",
                               "[0.9,0.3,~]☀🌙", "[0.9,0.3,·~]☀🌙", "[0.9,0.3,1.2~~]☀🌙",
                               "~[0.8]☀", "☀~!0.8", "|☀~⟩", "☀?~", "[0.8]~☀"])
def test_tilde_refusals(s):
    with pytest.raises(ParseError):
        parse(s)


@pytest.mark.parametrize("s", ["~", "~~", "0.5~~", "0.5~?", "?~", "0.5 0.6"])
def test_parse_number_refusals(s):
    with pytest.raises(ParseError):
        parse_number(s)


# ── bind: a straight number must be earned ──
def test_unearned_straight_number_refused_and_squiggled_accepted():
    with pytest.raises(BindError) as e:
        bind(parse("[0.8]💧"), terrarium())
    msg = str(e.value)
    assert "population p" in msg and "'water'" in msg and "0.8~" in msg     # slot + suggestion
    ok = bind(parse("[0.8~]💧"), terrarium()).leaf
    assert ok == BPopulation("water", Num(0.8, F)) and ok.p0.prov == F


@pytest.mark.parametrize("s,slot", [("🥕!0.4", "evidence η"), ("[.7]🔋", "population p"),
                                    ("[0.9,0.3,0.5~]🔋🪫", "Bloch r"), ("[0.9~,0.3,0.5~]🔋🪫", "Bloch θ"),
                                    ("⟨💧|ψ⟩ = [.7]💧", "population p"), ("🏠›🔧›🩹!0.92", "evidence η")])
def test_every_straight_slot_names_itself_when_unearned(s, slot):
    with pytest.raises(BindError) as e:
        bind(parse(s), terrarium())
    assert slot in str(e.value) and "~" in str(e.value)


def test_a_receipt_of_that_register_or_engine_origin_earns_the_straight_form():
    rc = Receipt((Measurement("water", "z", 0.9, 0),))
    assert bind(parse("[0.8]💧"), terrarium(), rc).leaf == BPopulation("water", 0.8)
    assert bind(parse("[0.8]💧"), terrarium(), engine_origin=True).leaf.p0.prov == M
    with pytest.raises(BindError):                       # a receipt of another register earns nothing
        bind(parse("[0.8]💧"), terrarium(), Receipt((Measurement("food", "z", 0.9, 0),)))
    with pytest.raises(BindError):                       # eta 0 is no measurement
        bind(parse("[0.8]💧"), terrarium(), Receipt((Measurement("water", "z", 0.0, 0),)))
    assert bind(parse("🥕!0.4"), terrarium(), Receipt((Measurement("food", "z", .5, 0),))).leaf \
        == BEvidence("food", 1 - 1, 0.4)


def test_engine_origin_never_earns_a_numeric_phase():
    with pytest.raises(BindError):
        bind(parse("[0.9,0.3,1.2]🔋🪫"), terrarium(), engine_origin=True)


def test_provenance_survives_pole_flip_and_canonicalisation():
    assert bind(parse("[0.25~]🏜"), terrarium()).leaf == BPopulation("water", Num(0.75, F))
    fl = bind(parse("[0.9~,0.3~,1.0~]🪫🔋"), terrarium()).leaf
    assert (fl.r.prov, fl.theta.prov, fl.phi.prov, fl.phase) == (F, F, F, "model")
    ev = bind(parse("🚪!0.4~"), terrarium()).leaf
    assert ev == BEvidence("support", 1, Num(0.4, F)) and ev != BEvidence("support", 1, 0.4)
    # the earned phase keeps its own provenance next to forecast r
    num = bind(parse("[0.9~,0.3,1.2]🔋🪫"), terrarium(), two_basis_receipt("energy"), engine_origin=True).leaf
    assert (num.r.prov, num.theta.prov, num.phi.prov, num.phase) == (F, M, M, "num")


def test_dark_numbers_bind_to_dark_not_to_a_half():
    assert bind(parse("[?]💧"), terrarium()).leaf == BDark("water")
    assert bind(parse("[?,0.3,1.0~]🔋🪫"), terrarium()).leaf == BDark("energy")
    with pytest.raises(BindError):
        bind(parse("🥕!?"), terrarium())


def test_range_checks_still_apply_to_forecasts():
    for s in ("[1.5~]💧", "🥕!0~", "[0.9,4~,1.0~]🔋🪫"):
        with pytest.raises(BindError):
            bind(parse(s), terrarium())
