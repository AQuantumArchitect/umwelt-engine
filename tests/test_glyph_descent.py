"""v0.3 §17: two descents (under a pole, under a register)."""
import pytest

from glyph_fixtures import ebind, terrarium
from umwelt.glyph import (Address, Dark, Population, Register, Scope, Unbound, 
                          canonical, parse)
from umwelt.glyph.bind import BEvidence, BPopulation
from umwelt.glyph.scope import Step


def sky_scope():
    cond = Scope("w/cond", {"wind": Register("🌬", "🍃")})
    under_pole = Scope("w/day", {"bird": Register("🐦", "🦉")})
    return Scope("w", {"sky": Register("☀", "🌙", children={0: under_pole}),
                       "mood": Register("😁", "😞")},
                 children={"sky": cond})


def test_parse_both_descents_and_round_trip():
    a = parse("☀›🐦›[.8]🦉")
    assert a == Address(("☀", "🐦"), Population(0.8, "🦉"))
    r = parse("☀🌙›🌬?")
    assert r == Address(("☀🌙",), Dark("🌬"))
    assert parse("🌙☀›🌬›🍃").path_glyphs == ("🌙☀", "🌬", "🍃")        # bare, register first
    for s in ["☀🌙›[.8]🌬", "☀🌙›🌬!0.5", "☀›🐦›🦉", "☀🌙›🌬›🍃", "☀›🐦🦉›x".replace("x", "[.2]🦉")]:
        a = parse(s)
        assert parse(canonical(a)) == a


def test_two_descents_bind_and_say_which_kind():
    sc = sky_scope()
    pole = ebind(parse("☀›🦉!0.4"), sc)
    reg = ebind(parse("☀🌙›🍃!0.4"), sc)
    assert [tuple(s) for s in pole.path] == [Step(sc.ref, "sky", 0, "pole")]
    assert [tuple(s) for s in reg.path] == [Step(sc.ref, "sky", None, "register")]
    assert pole.leaf == BEvidence("bird", 1, 0.4) and reg.leaf == BEvidence("wind", 1, 0.4)
    assert pole.final_scope.startswith("w/day@") and reg.final_scope.startswith("w/cond@")
    assert ebind(parse("🌙☀›[.3]🍃"), sc).leaf == BPopulation("wind", 0.7)     # either pole order


def test_mixed_path_kinds_in_one_address():
    inner = Scope("w/cond/x", {"leaf": Register("🍂", "🌱")})
    cond = Scope("w/cond", {"wind": Register("🌬", "🍃", children={1: inner})})
    sc = Scope("w", {"sky": Register("☀", "🌙")}, children={"sky": cond})
    b = ebind(parse("☀🌙›🍃›🌱?"), sc)
    assert [s.kind for s in b.path] == ["register", "pole"] and b.leaf.register == "leaf"


def test_descent_refusals():
    sc = sky_scope()
    with pytest.raises(Unbound):                          # mood has no conditional child
        ebind(parse("😁😞›🌬?"), sc)
    with pytest.raises(Unbound):                          # the pole 🌙 opens no child scope
        ebind(parse("🌙›🐦?"), sc)
    with pytest.raises(Unbound):                          # two glyphs of two registers
        ebind(parse("☀😁›🌬?"), sc)
    with pytest.raises(Unbound):                          # a register's pole twice
        ebind(parse("☀☀›🌬?"), sc)
    with pytest.raises(Unbound):                          # the register child is not a pole child
        ebind(parse("☀›🌬?"), sc)
    assert ebind(parse("☀🌙›🌬?"), sc).leaf.register == "wind"


def test_children_must_name_a_register_and_change_the_version():
    with pytest.raises(Exception):
        Scope("s", {"a": Register("☀", "🌙")}, children={"zzz": Scope("c", {})})
    base = Scope("s", {"a": Register("☀", "🌙")})
    withkid = Scope("s", {"a": Register("☀", "🌙")},
                    children={"a": Scope("c", {"b": Register("🔥", "❄")})})
    assert base.version != withkid.version
    assert Scope.from_dict(withkid.to_dict()) == withkid


def test_section7_pole_descents_unchanged():
    b = ebind(parse("🏠›🔧›🩹!0.92"), terrarium())
    assert [s.kind for s in b.path] == ["pole", "pole"]
