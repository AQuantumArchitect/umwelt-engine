"""v0.4 §26-§28: no renderer may drop a ~. Glyph string, prose and frame/1 all carry it."""
import re

import pytest

from glyph_fixtures import terrarium
from umwelt.glyph import (Register, Scope, bind, canonical, frame_node, parse, prose,
                          sentences_from_frame)
from umwelt.glyph.num import FORECAST, Num

SENTENCES = [
    "[0.8]💧", "[0.8~]💧", "[0.25~]🏜", "[?]💧", "🥕!0.25", "🥕!0.25~",
    "[0.9,0.3,1.2~]🔋🪫", "[0.9~,0.3,1.2~]🔋🪫", "[0.9~,0.3~,1.2~]🔋🪫", "[0.9,0.3~,·]🔋🪫",
    "[0.9,0.3,·]🔋🪫", "💧?", "[?,0.3,1.2~]🔋🪫", "⟨💧|ψ⟩ = [0.8~]💧", "🏠›🔧›🩹!0.92~",
    "|🏠🤝⟩",
]


def _n_forecast(text):
    return text.count("~")


def check(text, *, glyph=canonical, words=prose):
    """The round-trip check: every renderer must show exactly as many forecasts as the source."""
    ast = parse(text)
    want = _n_forecast(text)
    g = glyph(ast)
    assert _n_forecast(g) == want, f"glyph string dropped a ~: {text!r} -> {g!r}"
    assert parse(g) == ast, f"glyph string changed the sentence: {text!r} -> {g!r}"
    w = words(ast)
    assert w.count("(forecast)") == want and w.count("about ") == want, \
        f"prose dropped a ~: {text!r} -> {w!r}"
    if want == 0:
        assert "forecast" not in w and "about" not in w
    return g, w


@pytest.mark.parametrize("text", SENTENCES)
def test_glyph_and_prose_preserve_every_squiggle(text):
    check(text)


def test_the_check_catches_a_renderer_that_drops_a_squiggle():
    with pytest.raises(AssertionError, match="dropped a ~"):
        check("[0.8~]💧", glyph=lambda a: canonical(a).replace("~", ""))
    with pytest.raises(AssertionError, match="prose dropped"):
        check("[0.9,0.3,1.2~]🔋🪫", words=lambda a: re.sub(r"about |\(forecast\)", "", prose(a)))
    with pytest.raises(AssertionError, match="prose dropped"):        # forecast shown as a measured number
        check("🥕!0.25~", words=lambda a: "an observation of 🥕 with weight 0.25")


def test_prose_wording():
    assert prose("[0.8]💧") == "💧 stands at 0.8"
    assert prose("[0.8~]💧") == "💧 stands at about 0.8 (forecast)"
    assert "unknown (dark)" in prose("[?]💧") and "unknown (dark)" in prose("[0.9,0.3,·]🔋🪫")
    assert prose("💧?") == "💧 is dark (nothing is claimed)"


# ── frame/1: "forecast": true beside the value, never a different value ──
FRAME_CASES = ["[0.8]💧", "[0.8~]💧", "[0.25~]🏜", "🥕!0.25", "🥕!0.25~", "💧?",
               "[0.9,0.3,1.2~]🔋🪫", "[0.9~,0.3,1.2~]🔋🪫", "[0.9~,0.3~,1.2~]🔋🪫",
               "[0.9,0.3,·]🔋🪫", "[0.9,0.3~,·]🔋🪫"]


@pytest.mark.parametrize("text", FRAME_CASES)
def test_frame_round_trip_keeps_value_and_flag(text):
    sc = terrarium()
    b = bind(parse(text), sc, engine_origin=True)
    node = frame_node(b, sc)
    doc = {"schema": "frame/1", "nodes": [node]}
    back = sentences_from_frame(doc, sc, precise=True)
    assert len(back) == 1
    assert bind(parse(back[0]), sc, engine_origin=True).leaf == b.leaf, (text, node, back)
    assert _n_forecast(back[0]) == _n_forecast(canonical(parse(back[0])))
    assert _n_forecast(back[0]) == sum(f if isinstance(f, bool) and f else 0
                                       for f in ([node.get("forecast", False)] if isinstance(node.get("forecast", False), bool)
                                                 else node["forecast"]))


def test_frame_node_shapes():
    sc = terrarium()
    n = frame_node(bind(parse("[0.8~]💧"), sc), sc)
    assert n["value"] == 0.8 and n["forecast"] is True and n["state"] == "current"
    n = frame_node(bind(parse("[0.8]💧"), sc, engine_origin=True), sc)
    assert n["value"] == 0.8 and n["forecast"] is False
    n = frame_node(bind(parse("[0.9,0.3~,1.2~]🔋🪫"), sc, engine_origin=True), sc)
    assert n["forecast"] == [False, True, True] and n["bloch_rtp"] == [0.9, 0.3, 1.2]
    assert frame_node(bind(parse("💧?"), sc), sc) == {"id": "water", "state": "dark"}


def test_a_flagged_engine_node_renders_squiggled_and_the_unflagged_one_straight():
    sc = Scope("t", {"a": Register("🔥", "🧊", node="n")})
    mk = lambda **k: {"schema": "frame/1", "nodes": [{"id": "n", "state": "current",
                                                      "value": 0.5, "unit": "probability", **k}]}
    assert sentences_from_frame(mk(), sc) == ["[0.5]🔥"]
    assert sentences_from_frame(mk(forecast=True), sc) == ["[0.5~]🔥"]
    stale = mk(forecast=True); stale["nodes"][0].update(state="stale", age_s=9)
    assert sentences_from_frame(stale, sc) == ["[0.5~]🔥  # age 9s stale"]
    bz = mk(forecast=True); bz["nodes"][0].update(unit="bloch_z", value=0.0)
    assert sentences_from_frame(bz, sc) == ["[0.5~]🔥"]
    from umwelt.glyph import RenderError
    for bad in (mk(forecast="yes"), {"schema": "frame/1", "nodes": [{"id": "n", "state": "dark", "forecast": True}]}):
        with pytest.raises(RenderError):
            sentences_from_frame(bad, sc)


def test_engine_precise_vector_with_forecast_flag_squiggles_every_axis():
    sc = Scope("t", {"a": Register("🔥", "🧊", node="n")})
    doc = {"schema": "frame/1", "nodes": [{"id": "n", "state": "current", "value": 0.0,
                                           "bloch": [1.0, 0.0, 0.0], "forecast": True}]}
    (s,) = sentences_from_frame(doc, sc, precise=True)
    a = parse(s)
    assert (a.r.prov, a.theta.prov, a.phi.prov) == (FORECAST,) * 3
    doc["nodes"][0]["forecast"] = False                # unflagged: r, θ read off ρ; φ stays model
    a = parse(sentences_from_frame(doc, sc, precise=True)[0])
    assert (a.r.prov, a.theta.prov, a.phi.prov) == ("measured", "measured", FORECAST)
