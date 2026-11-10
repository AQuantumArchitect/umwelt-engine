"""v0.3 §16 earned phase: parse, refuse, accept, QQ."""
import pytest

from glyph_fixtures import ebind, terrarium, two_basis_receipt
from umwelt.glyph import (BindError, Bloch, Dark, Ket, Measurement, Outcome, ParseError,
                          Receipt, ReceiptError, canonical, parse, qq_check)
from umwelt.glyph.bind import BBloch


def test_parse_three_phase_kinds_and_canonical_keeps_marks():
    assert parse("[0.9,0.3,·]🔋🪫") == Bloch(0.9, 0.3, None, "🔋", "🪫", "dark")
    assert parse("[0.9,0.3, 1.2~]🔋🪫") == Bloch(0.9, 0.3, 1.2, "🔋", "🪫", "model")
    assert parse("[0.9,0.3,1.2]🔋🪫") == Bloch(0.9, 0.3, 1.2, "🔋", "🪫", "num")
    for s, want in [("[0.9,0.3,·]🔋🪫", "[0.9,0.3,·]🔋🪫"), ("[.9,.3,1.2~]🔋🪫", "[0.9,0.3,1.2~]🔋🪫"),
                    ("[.9,.3,1.2]🔋🪫", "[0.9,0.3,1.2]🔋🪫")]:
        assert canonical(parse(s)) == want and parse(want) == parse(s)


@pytest.mark.parametrize("s", ["[0.9,0.3,~]🔋🪫", "[0.9,0.3,··]🔋🪫", "[0.9,0.3,1.2~~]🔋🪫",
                               "[0.9,0.3,·~]🔋🪫", "[0.9,0.3]🔋🪫"])
def test_phase_syntax_refusals(s):
    with pytest.raises(ParseError):
        parse(s)


def test_dark_and_model_phase_need_no_receipt():
    d = ebind(parse("[0.9,0.3,·]🔋🪫"), terrarium()).leaf
    m = ebind(parse("[0.9,0.3,1.2~]🔋🪫"), terrarium()).leaf
    assert d == BBloch("energy", 0.9, 0.3, None, "dark")
    assert m == BBloch("energy", 0.9, 0.3, 1.2, "model")
    assert d != m


def test_model_phase_still_range_checked_and_pole_flipped():
    with pytest.raises(BindError):
        ebind(parse("[0.9,0.3,7~]🔋🪫"), terrarium())
    flipped = ebind(parse("[0.9,0.3,1.0~]🪫🔋"), terrarium()).leaf
    assert flipped.phase == "model" and flipped.theta == pytest.approx(3.141592653589793 - 0.3)
    assert ebind(parse("[0.9,0.3,·]🪫🔋"), terrarium()).leaf.phi is None


def test_numeric_phase_refused_without_receipt_or_with_one_basis():
    s = "[0.9,0.3,1.2]🔋🪫"
    with pytest.raises(BindError, match="not earned"):
        ebind(parse(s), terrarium())
    with pytest.raises(BindError):
        ebind(parse(s), terrarium(), Receipt(()))
    one_basis_twice = Receipt((Measurement("energy", (0, 0, 1), 0.9, 0),
                               Measurement("energy", (0, 0, 1), 0.9, 1)))
    with pytest.raises(BindError):
        ebind(parse(s), terrarium(), one_basis_twice)
    # the two poles of one register are the SAME axis: not a second basis
    poles = Receipt((Measurement("energy", "🔋", 0.9, 0), Measurement("energy", "🪫", 0.9, 1)))
    with pytest.raises(BindError):
        ebind(parse(s), terrarium(), poles)
    # two bases, but of another register
    with pytest.raises(BindError):
        ebind(parse(s), terrarium(), two_basis_receipt("water"))
    # eta = 0 is a no-op, not a measurement
    null = Receipt((Measurement("energy", "z", 0.0, 0), Measurement("energy", "x", 0.0, 1)))
    with pytest.raises(BindError):
        ebind(parse(s), terrarium(), null)


def test_numeric_phase_accepted_with_two_basis_receipt():
    for rc in (two_basis_receipt("energy"),
               Receipt((Measurement("energy", "🔋", 0.8, 0), Measurement("energy", "y", 0.8, 1))),
               Receipt((Measurement("energy", (1, 1, 0), 0.5, 0), Measurement("energy", "z", 0.5, 1)))):
        leaf = ebind(parse("[0.9,0.3,1.2]🔋🪫"), terrarium(), rc).leaf
        assert leaf == BBloch("energy", 0.9, 0.3, 1.2, "num")


def test_receipt_validation():
    with pytest.raises(ReceiptError):
        Measurement("energy", "z", 1.5, 0)
    with pytest.raises(ReceiptError):
        ebind(parse("[0.9,0.3,1.2]🔋🪫"), terrarium(),
             Receipt((Measurement("energy", (0, 0, 0), 0.5, 0), Measurement("energy", "x", 0.5, 1))))
    with pytest.raises(ReceiptError):                 # a glyph of another register
        ebind(parse("[0.9,0.3,1.2]🔋🪫"), terrarium(),
             Receipt((Measurement("energy", "💧", 0.5, 0), Measurement("energy", "x", 0.5, 1))))


def _qq(a_first, b_first):
    """a_first/b_first: (p(y,n), p(n,y)) in the order asked."""
    return Receipt((), (Outcome("A", True, False, a_first[0]), Outcome("A", False, True, a_first[1]),
                        Outcome("B", True, False, b_first[0]), Outcome("B", False, True, b_first[1])))


def test_qq_holds_fails_and_dark():
    assert qq_check(_qq((0.10, 0.25), (0.20, 0.15))) == Ket(("✅",))     # 0.35 == 0.35
    assert qq_check(_qq((0.10, 0.25), (0.20, 0.25))) == Ket(("❌",))
    assert qq_check(_qq((0.10, 0.25), (0.20, 0.25)), tol=0.2) == Ket(("✅",))
    only_one_order = Receipt((), (Outcome("A", True, False, .1), Outcome("A", False, True, .2)))
    assert qq_check(only_one_order) == Dark("✅")
    assert qq_check(Receipt()) == Dark("✅")
    with pytest.raises(ReceiptError):
        qq_check(Receipt((), (Outcome("A", True, False, .1), Outcome("A", True, False, .1))))
