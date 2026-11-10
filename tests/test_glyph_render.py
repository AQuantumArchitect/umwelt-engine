"""Rendering a frame/1 never writes: dark -> ``?``, current -> ``[p]``, stale -> comment."""
import copy
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.boot import build_engine
from glyph_fixtures import ebind
from umwelt.glyph import Register, Scope, parse, sentences_from_frame
from umwelt.projection.frame import frame, validate
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

SPEC = DomainSpec(
    name="glyph-render",
    nodes=(NodeSpec("root", parent=None, kind="root", roles=("warm",)),
           NodeSpec("den", parent="root", roles=("warm", "lit"),
                    role_modes={"warm": "dissipative", "lit": "unitary"})),
    bindings=(BindingSpec("den_temp", zone="den", role="warm",
                          normalizer={"type": "range", "lo": 0.0, "hi": 40.0}),),
)
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _world(monkeypatch):
    random.seed(5); np.random.seed(5)
    engine = build_engine(spec=SPEC, population=False)
    for i in range(20):
        # The bridge dates arrival with wall time, independently of the engine's
        # simulation clock. Keep both clocks in this fixture's January world;
        # a real November receipt must not appear fresh in a January frame.
        instant = T0 + timedelta(seconds=10 * i)
        monkeypatch.setattr("umwelt.membranes.ingress.time.time", lambda: instant.timestamp())
        engine.ingest(sensor_readings={"den_temp": 30.0}, confidence={"den_temp": 1.0},
                      now=instant)
    return engine


def _nodes(doc):
    return {n["id"]: n for n in doc["nodes"]}


def test_frame_renders_dark_and_current_without_writing(monkeypatch):
    engine = _world(monkeypatch)
    h0 = engine.field_canon_hash()
    now = (T0 + timedelta(seconds=200)).timestamp()
    doc = frame(engine, subject="render", now=now, stale_after_s=3600)
    nodes = _nodes(doc)
    states = {i: n["state"] for i, n in nodes.items()}
    assert "dark" in states.values() and "current" in states.values(), states
    dark_id = next(i for i, s in states.items() if s == "dark")
    cur_id = next(i for i, s in states.items() if s == "current")
    scope = Scope("render/den", {
        "dk": Register("🌑", "🌕", node=dark_id),
        "cu": Register("🔥", "🧊", node=cur_id),
    })
    before = copy.deepcopy(doc)
    out = sentences_from_frame(doc, scope)
    assert "🌑?" in out
    cur = [s for s in out if s.startswith("[") and s.endswith("🔥")]
    assert len(cur) == 1
    p = parse(cur[0]).p
    assert p == (1 + nodes[cur_id]["value"]) / 2 or abs(p - (1 + nodes[cur_id]["value"]) / 2) < 1e-6
    for s in out:
        ebind(parse(s), scope)                 # every rendered line is a bound sentence
    assert validate(doc) == [] and doc == before
    assert engine.field_canon_hash() == h0


def test_future_bridge_arrival_cannot_date_a_historical_frame(monkeypatch):
    engine = _world(monkeypatch)
    doc = frame(engine, subject="render", now=(T0 - timedelta(seconds=1)).timestamp())
    assert all(n["state"] == "dark" and n["value"] is None for n in doc["nodes"])


def test_stale_comment_and_precise_vector():
    doc = {"schema": "frame/1", "subject": "x", "as_of": "2026-01-01T00:00:00Z",
           "stale_after_s": 10, "law": "l", "producer": {"name": "t", "kind": "sample"},
           "nodes": [{"id": "a", "label": "a", "value": 0.7, "state": "stale", "age_s": 3400,
                      "unit": "p", "bloch": [0.0, 0.6, 0.8]},
                     {"id": "b", "label": "b", "value": None, "state": "dark", "age_s": None}],
           "edges": []}
    assert validate(doc) == []
    sc = Scope("t", {"a": Register("☀", "🌙"), "b": Register("🔋", "🪫")})
    assert sentences_from_frame(doc, sc) == ["[0.7]☀  # age 3400s stale", "🔋?"]
    precise = sentences_from_frame(doc, sc, precise=True)
    ast = parse(precise[0])
    assert (ast.pole0, ast.pole1) == ("☀", "🌙")
    assert ast.phase == "model" and abs(ast.r - 1.0) < 1e-9 and abs(ast.phi - np.pi / 2) < 1e-9
    assert abs(ast.theta - np.arccos(0.8)) < 1e-9


def test_epsilon_comment_only_when_scope_declares_it():
    doc = {"schema": "frame/1", "subject": "x", "as_of": "2026-01-01T00:00:00Z",
           "stale_after_s": 10, "law": "l", "producer": {"name": "t", "kind": "sample"},
           "nodes": [{"id": "a", "label": "a", "value": 0.7, "state": "stale", "age_s": 5,
                      "unit": "p"},
                     {"id": "b", "label": "b", "value": None, "state": "dark", "age_s": None}],
           "edges": []}
    regs = {"a": Register("☀", "🌙"), "b": Register("🔋", "🪫")}
    assert sentences_from_frame(doc, Scope("t", regs)) == ["[0.7]☀  # age 5s stale", "🔋?"]
    out = sentences_from_frame(doc, Scope("t", regs, epsilon=0.25))
    assert out == ["[0.7]☀  # age 5s stale; ε=0.25", "🔋?  # ε=0.25"]
    for s in out:                                              # still parse and bind
        ebind(parse(s), Scope("t", regs, epsilon=0.25))
    assert Scope("t", regs, epsilon=0.25).version != Scope("t", regs).version
    assert Scope("t", regs, epsilon=0).epsilon == 0.0
    assert Scope.from_dict(Scope("t", regs, epsilon=0.25).to_dict()).epsilon == 0.25
    import pytest
    for bad in (-1.0, float("nan"), float("inf"), "0.1", True):
        with pytest.raises(Exception):
            Scope("t", regs, epsilon=bad)
