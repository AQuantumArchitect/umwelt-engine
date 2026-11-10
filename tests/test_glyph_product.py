"""v0.3 §19: the product scope — the only home for "between" — and the worked example."""
import math

import pytest

from glyph_fixtures import ebind
from umwelt.glyph import (Ambiguous, BindError, Ket, Measurement, ParseError, Receipt,
                          Register, Scope, Unbound, canonical, canonical_in, parse,
                          resonance, tension)
from umwelt.glyph.bind import BPairKet, BPopulation
from umwelt.glyph.metrics import rho_from_population, bures_distance


def household(name, *, poles=("☀", "🌙"), extra=None):
    regs = {"sky": Register(*poles)}
    regs.update(extra or {})
    return Scope(f"terrarium/{name}", regs)


@pytest.fixture
def pair():
    a = household("maple", extra={"mood": Register("😁", "😞")})
    b = household("birch", extra={"hearth": Register("🔥", "❄")})
    return a, b, Scope.product(a, b)


def test_declaration_ids_registers_and_poles(pair):
    a, b, p = pair
    assert p.id == "terrarium/maple⊗terrarium/birch" and p.is_product and p.parents == (a, b)
    assert set(p.pair_registers) == {"sky⊗sky", "sky⊗hearth", "mood⊗sky", "mood⊗hearth"}
    assert p.pair_registers["sky⊗hearth"].poles == {(0, 0): ("☀", "🔥"), (0, 1): ("☀", "❄"),
                                                    (1, 0): ("🌙", "🔥"), (1, 1): ("🌙", "❄")}
    assert p.version != Scope.product(b, a).version and p.ref.startswith(p.id + "@")
    assert Scope.from_dict(p.to_dict()) == p and Scope.from_dict(p.to_dict()).version == p.version
    # a parent edit changes the product version
    edited = Scope.product(household("maple", extra={"mood": Register("😁", "😢")}), b)
    assert edited.version != p.version


def test_product_declaration_refusals(pair):
    a, b, _ = pair
    with pytest.raises(Exception):
        Scope("x", {"r": Register("☀", "🌙")}, parents=(a, b))
    with pytest.raises(Exception):
        Scope("x", {}, parents=(a,))
    with pytest.raises(Exception):
        Scope.product(a, Scope("t", {"k": Register("🔥", "❄")}, keep_skin_tones=True))


def test_subscripts_parse_and_print():
    k = parse("|☀₁🌙₂⟩")
    assert k == Ket(("☀", "🌙"), False, (1, 2)) and canonical(k) == "|☀₁🌙₂⟩"
    assert parse("|☀₂ 🌙₁⟩").subs == (2, 1) and parse("|☀🌙⟩").subs == ()
    assert parse("|☀₁⌢🌙₂⟩").tied and canonical(parse("|☀₁⌢🌙₂⟩")) == "|☀₁⌢🌙₂⟩"
    for bad in ("|₁☀⟩", "|☀₁₂⟩", "|☀₃⟩", "[.5]☀₁"):
        with pytest.raises(ParseError):
            parse(bad)


def test_cross_scope_ket_binds_only_in_product_scope(pair):
    a, b, p = pair
    k = ebind(parse("|☀₁🌙₂⟩"), p).leaf
    assert k == BPairKet("sky⊗sky", (0, 1), (k.parts[0], k.parts[1])) and k.parts[0].register == "sky"
    assert ebind(parse("|😁₁🔥₂⟩"), p).leaf.register == "mood⊗hearth"
    assert ebind(parse("|🔥😁⟩"), p).leaf == ebind(parse("|😁₁🔥₂⟩"), p).leaf   # unambiguous, any order
    assert ebind(parse("|🌙₂☀₁⟩"), p).leaf == ebind(parse("|☀₁🌙₂⟩"), p).leaf
    # refusals: outside a product scope the same ket is undefined, naming §19
    for scope in (a, b):
        with pytest.raises(Unbound, match="§19"):
            ebind(parse("|☀₁🌙₂⟩"), scope)
    with pytest.raises(Unbound, match="§19"):
        ebind(parse("|😁🔥⟩"), a)                  # 🔥 is the other parent's glyph
    with pytest.raises(Unbound, match="§19"):
        ebind(parse("|😁₁🔥₂⟩"), a)
    # inside the product scope, shared glyphs need their subscripts; wrong parent is refused
    with pytest.raises(Ambiguous):
        ebind(parse("|☀🌙⟩"), p)
    with pytest.raises(Unbound):
        ebind(parse("|🔥₁☀₂⟩"), p)
    with pytest.raises(BindError):                # two glyphs of one parent
        ebind(parse("|☀₁😁₁⟩"), p)
    with pytest.raises(BindError):                # one glyph is not a pair
        ebind(parse("|😁₁⟩"), p)
    with pytest.raises(Unbound):                  # populations etc. have no pair-register form
        ebind(parse("[.5]😁"), p)


def test_pair_tie_needs_a_joint_receipt(pair):
    _, _, p = pair
    with pytest.raises(BindError, match="not earned"):
        ebind(parse("|☀₁⌢🔥₂⟩"), p)
    rc = Receipt((Measurement("sky", "z", .9, 0, joint=True, registers=("sky", "hearth")),))
    assert ebind(parse("|☀₁⌢🔥₂⟩"), p, rc).leaf.tied


def test_canonical_form_writes_parent_subscripts(pair):
    _, _, p = pair
    assert canonical_in(parse("|🔥😁⟩"), p) == "|😁₁🔥₂⟩"
    assert canonical_in(parse("|🌙₂☀₁⟩"), p) == "|☀₁🌙₂⟩"
    assert canonical_in(parse("|☀₁🌙₂⟩"), p) == "|☀₁🌙₂⟩"
    assert canonical_in(parse("[.5]☀"), household("x")) == "[0.5]☀"
    assert ebind(parse(canonical_in(parse("|🔥😁⟩"), p)), p) == ebind(parse("|🔥😁⟩"), p)


def test_worked_example_two_households_and_the_tension_receipt():
    """SlopNet's first tension meme, in the language (grammar §24 item 4): two toy
    households, one cross-scope ket, and its ⚡ as the Bures distance between the parents."""
    maple, birch = household("maple"), household("birch")
    both = Scope.product(maple, birch)
    ket = parse("|☀₁🌙₂⟩")                       # "maple is sunny while birch is moonlit"
    assert ebind(ket, both).leaf.poles == (0, 1) and canonical(ket) == "|☀₁🌙₂⟩"
    # each parent speaks in its own scope about its own sky
    state_maple = ebind(parse("[.9]☀"), maple)    # mostly sunny
    state_birch = ebind(parse("[.2]☀"), birch)    # mostly not
    assert state_maple.leaf == BPopulation("sky", 0.9) and state_birch.leaf == BPopulation("sky", 0.2)
    bolt = tension(state_maple, state_birch)                      # ⚡ ⟨ρ₁|ρ₂⟩
    expect = bures_distance(rho_from_population(0.9), rho_from_population(0.2))
    assert bolt == pytest.approx(expect) and 0 < bolt < math.sqrt(2)
    # diagonal states: Bures² = 2 − 2(√(p q) + √((1−p)(1−q)))
    assert bolt ** 2 == pytest.approx(2 - 2 * (math.sqrt(.9 * .2) + math.sqrt(.1 * .8)), abs=1e-9)
    assert resonance(state_maple, state_birch) == pytest.approx((math.sqrt(.18) + math.sqrt(.08)) ** 2)
    # the parents agreeing is no tension at all; opposed poles are maximal
    assert tension(state_maple, ebind(parse("[.9]☀"), birch)) == pytest.approx(0, abs=1e-7)
    assert tension(ebind(parse("[1]☀"), maple), ebind(parse("[0]☀"), birch)) == pytest.approx(math.sqrt(2))
