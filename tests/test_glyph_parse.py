import pytest

from glyph_fixtures import ebind, SECTION_7, terrarium, two_basis_receipt
from umwelt.glyph import (Address, Bloch, Bra, Dark, Evidence, Ket, ParseError, Population,
                          canonical, parse)


@pytest.mark.parametrize("sentence", SECTION_7)
def test_section7_round_trips_and_binds(sentence):
    ast = parse(sentence)
    assert parse(canonical(ast)) == ast
    assert canonical(parse(canonical(ast))) == canonical(ast)
    bound = ebind(ast, terrarium(), two_basis_receipt("energy"))
    assert bound.scope.startswith("terrarium/household@")


def test_six_forms_and_types():
    assert parse("|☀😁⟩") == Ket(("☀", "😁"))
    assert parse("[.8]☀") == Population(0.8, "☀")
    assert parse("☀!0.8") == Evidence("☀", 0.8)
    assert parse("[0.9,0.3,1.2]☀🌙") == Bloch(0.9, 0.3, 1.2, "☀", "🌙")
    assert parse("☀?") == Dark("☀")
    assert parse("⟨☀|") == Bra("☀")
    assert parse("⟨☀|ψ⟩") == Bra("☀", None, True)
    assert parse("⟨☀|ψ⟩ = [.8]☀").answer == Population(0.8, "☀")
    assert parse("⟨✅| = |✅⟩").answer == Ket(("✅",))
    assert parse("⟨✅| = ✅?").answer == Dark("✅")


def test_address_leaf_forms():
    assert parse("☀›🔥›⚫") == Address(("☀", "🔥", "⚫"), None)
    assert parse("☀›🔥›[.8]⚫") == Address(("☀", "🔥"), Population(0.8, "⚫"))
    assert parse("☀›🔥›⚫!0.4") == Address(("☀", "🔥"), Evidence("⚫", 0.4))
    assert parse("☀›🔥›⚫?") == Address(("☀", "🔥"), Dark("⚫"))
    assert parse("☀›🔥›⟨⚫|") == Address(("☀", "🔥"), Bra("⚫"))


def test_descent_and_product_are_different_asts():
    assert parse("☀›🔥›⚫") != parse("|☀🔥⚫⟩")
    assert parse("☀›🔥›⚫").path_glyphs == ("☀", "🔥", "⚫")
    assert parse("|☀🔥⚫⟩").glyphs == ("☀", "🔥", "⚫")


def test_variation_selector_and_comment_and_whitespace():
    assert parse("[ .8 ] ☀️  # age 3400s stale") == Population(0.8, "☀")
    assert parse("[0.9, 0.3, 1.2] ☀ 🌙") == Bloch(0.9, 0.3, 1.2, "☀", "🌙")


def test_ascii_fallbacks_only_lenient():
    for s in ("|☀😁>", "<☀|", "☀>🔥>⚫", "<☀|ψ>"):
        with pytest.raises(ParseError):
            parse(s)
    assert parse("|☀😁>", lenient=True) == parse("|☀😁⟩")
    assert parse("<☀|", lenient=True) == parse("⟨☀|")
    assert parse("☀>🔥>⚫", lenient=True) == parse("☀›🔥›⚫")
    assert parse("<☀|ψ>", lenient=True) == parse("⟨☀|ψ⟩")
    assert parse("☀>🔥>[.8]⚫", lenient=True) == parse("☀›🔥›[.8]⚫")
    assert canonical(parse("|☀>", lenient=True)) == "|☀⟩"


@pytest.mark.parametrize("bad,frag", [
    ("", "empty"), ("☀", "bare glyph"), ("|⟩", "empty ket"), ("|☀", "unterminated"),
    ("[.8]", "pole glyph"), ("[.8,.2]☀🌙", "expected ','"), ("[abc]☀", "number"),
    ("☀!", "number"), ("☀›", "after '›'"), ("[.8]☀ junk", "unexpected"),
    ("⟨☀", "expected '|'"), ("⟨☀|ψ", "⟩"), ("[1e999]☀", "finite"),
    ("⟨☀| = ☀!0.5", "answer"),
])
def test_errors_carry_position(bad, frag):
    with pytest.raises(ParseError) as ei:
        parse(bad)
    assert frag in str(ei.value) and isinstance(ei.value.pos, int)
