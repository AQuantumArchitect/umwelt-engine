"""v0.3 §16 earned tie mark."""
import pathlib

import pytest

import umwelt.glyph as glyph_pkg
from glyph_fixtures import ebind, terrarium
from umwelt.glyph import (BindError, Ket, Measurement, ParseError, Receipt, Register, Scope,
                          canonical, parse, sentences_from_frame)
from umwelt.glyph.bind import BKet


def joint(*regs, eta=0.9, flag=True):
    return Receipt((Measurement(regs[0], "z", eta, 0, joint=flag, registers=regs),))


def test_parse_and_canonical_keep_the_tie():
    k = parse("|🏠⌢🤝⟩")
    assert k == Ket(("🏠", "🤝"), True) and canonical(k) == "|🏠⌢🤝⟩"
    assert parse("|🏠 ⌢ 🤝⟩") == k
    assert parse("|🏠🤝⟩") == Ket(("🏠", "🤝")) and parse("|🏠🤝⟩").tied is False
    assert parse("|🏠⌢🤝⌢🔋⟩").tied and parse("🏠›|🔧⌢🩹⟩").leaf.tied


@pytest.mark.parametrize("s", ["|⌢🏠⟩", "|🏠⌢⟩", "|🏠⌢⌢🤝⟩", "|🏠⌢🤝🔋⟩", "|🏠🤝⌢🔋⟩", "|⌢⟩"])
def test_tie_syntax_refusals(s):
    with pytest.raises(ParseError):
        parse(s)


def test_tie_refused_without_joint_receipt():
    ast = parse("|🏠⌢🤝⟩")
    with pytest.raises(BindError, match="not earned"):
        ebind(ast, terrarium())
    with pytest.raises(BindError):
        ebind(ast, terrarium(), Receipt(()))
    with pytest.raises(BindError):                                # joint flag off
        ebind(ast, terrarium(), joint("shelter", "support", flag=False))
    with pytest.raises(BindError):                                # names only one register
        ebind(ast, terrarium(), Receipt((Measurement("shelter", "z", .9, 0, joint=True),)))
    with pytest.raises(BindError):                                # wrong pair
        ebind(ast, terrarium(), joint("shelter", "water"))
    with pytest.raises(BindError):                                # eta 0 measured nothing
        ebind(ast, terrarium(), joint("shelter", "support", eta=0.0))


def test_tie_accepted_with_joint_receipt_and_plain_ket_unaffected():
    b = ebind(parse("|🏠⌢🤝⟩"), terrarium(), joint("support", "shelter")).leaf
    assert isinstance(b, BKet) and b.tied and [r.register for r in b.refs] == ["shelter", "support"]
    assert ebind(parse("|🏠🤝⟩"), terrarium()).leaf.tied is False          # no receipt needed
    assert ebind(parse("|🏠⌢🤝⟩"), terrarium(), joint("shelter", "support")).leaf != \
        ebind(parse("|🏠🤝⟩"), terrarium()).leaf


def _doc(nodes):
    return {"schema": "frame/1", "subject": "x", "nodes": nodes, "edges": []}


def test_render_never_emits_tie_unless_node_carries_joint_receipt():
    sc = Scope("t", {"a": Register("☀", "🌙"), "b": Register("🔥", "❄")})
    plain = [{"id": "a", "value": 0.5, "unit": "p", "state": "current"},
             {"id": "b", "value": 0.5, "unit": "p", "state": "current"}]
    assert not any("⌢" in s for s in sentences_from_frame(_doc(plain), sc))
    tied = [dict(plain[0], joint_receipt={"with": "b", "poles": [0, 1]}), plain[1]]
    out = sentences_from_frame(_doc(tied), sc)
    assert out == ["[0.5]☀", "|☀⌢❄⟩", "[0.5]🔥"]
    both = [dict(tied[0]), dict(plain[1], joint_receipt={"with": "a", "poles": [0, 1]})]
    assert sum("⌢" in s for s in sentences_from_frame(_doc(both), sc)) == 1   # once per pair
    with pytest.raises(Exception):
        sentences_from_frame(_doc([dict(plain[0], joint_receipt={"with": "zzz", "poles": [0, 1]}),
                                   plain[1]]), sc)
    with pytest.raises(Exception):
        sentences_from_frame(_doc([dict(plain[0], joint_receipt=True), plain[1]]), sc)


def test_the_forbidden_word_appears_nowhere_in_the_package():
    root = pathlib.Path(glyph_pkg.__file__).parent
    for f in root.rglob("*.py"):
        assert "entangle" not in f.read_text(encoding="utf-8").lower(), f
