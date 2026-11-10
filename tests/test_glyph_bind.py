import math

import pytest

from glyph_fixtures import terrarium, two_basis_receipt
from umwelt.glyph import (Ambiguous, BindError, Register, Scope, Unbound, bind, canonical,
                          parse)
from umwelt.glyph.bind import BBloch, BDark, BEvidence, BPopulation


def b(s, scope=None, receipt=None):
    return bind(parse(s), scope or terrarium(), receipt, engine_origin=True)


def test_ambiguity_refused():
    sc = Scope("s", {"sky": Register("☀", "🌙"), "lamp": Register("💡", "☀")})
    with pytest.raises(Ambiguous):
        sc.resolve("☀")
    with pytest.raises(Ambiguous):
        bind(parse("[.8]☀"), sc)
    assert sc.resolve("🌙") == ("sky", 1)          # unambiguous glyphs still resolve


def test_unbound_refused():
    with pytest.raises(Unbound):
        b("[.8]🐙")
    with pytest.raises(Unbound):
        b("🏠›🔋›🪫?" .replace("🪫", "🐙"))
    with pytest.raises(Unbound):                    # a pole that opens no child scope
        b("🥕›🍽!0.5")


def test_population_identity_canonical_to_pole0():
    a, c = b("[.8]💧"), b("[.2]🏜")
    assert a.leaf == c.leaf == BPopulation("water", 0.8)
    assert b("[0]💧").leaf != b("[1]💧").leaf


def test_evidence_not_symmetrized():
    ev = b("💧!0.8").leaf
    assert ev == BEvidence("water", 0, 0.8)
    assert ev != b("🏜!0.2").leaf and ev != b("🏜!0.8").leaf
    # binding yields exactly one fact: nothing about the other pole is created
    assert (ev.register, ev.pole, ev.eta) == ("water", 0, 0.8)
    assert not hasattr(ev, "p0")


def test_dark_is_not_half_nor_zero():
    d = b("💧?").leaf
    assert d == BDark("water") and d != b("[.5]💧").leaf and d != b("[0]💧").leaf
    assert b("🏜?").leaf == d                        # dark belongs to the register


@pytest.mark.parametrize("s", ["[1.2]💧", "[-.1]💧", "💧!0", "💧!1.5", "[1.2,0.3,0~]🔋🪫",
                               "[.5,3.2,0~]🔋🪫", "[.5,0.3,6.3~]🔋🪫", "[.5,.3,-1~]🔋🪫",
                               "[.5,.3,0~]🔋💧", "|💧🏜⟩"])
def test_range_and_contradiction_refused(s):
    with pytest.raises(BindError):
        b(s)


def test_bloch_pole_order_canonicalized():
    rc = two_basis_receipt("energy")
    a = b("[0.9,0.3,1.2]🔋🪫", receipt=rc).leaf
    c = b("[0.9,%r,%r]🪫🔋" % (math.pi - 0.3, (2 * math.pi - 1.2)), receipt=rc).leaf
    assert isinstance(a, BBloch)
    assert c.theta == pytest.approx(a.theta) and c.phi == pytest.approx(a.phi)


def test_version_stamp_and_hash():
    sc = terrarium()
    assert b("💧?", sc).scope_version == sc.version
    edited = Scope.from_dict(sc.to_dict())
    assert edited.version == sc.version and edited == sc
    d = sc.to_dict(); d["registers"]["water"]["pole1"] = "🌵"
    assert Scope.from_dict(d).version != sc.version
    assert canonical(parse("☀️!0.5")) == "☀!0.5"


def test_descent_binds_through_child_scopes():
    bd = b("🏠›🔧›🩹!0.92")
    assert [s[1:3] for s in bd.path] == [("shelter", 0), ("repair", 0)]
    assert bd.leaf == BEvidence("need", 0, 0.92)
    assert bd.final_scope.startswith("terrarium/shelter/repair@")
    bare = b("🏠›🔋›🪫")
    assert bare.leaf is None and [s[1:3] for s in bare.path] == [("shelter", 0), ("energy", 0), ("charge", 0)]
