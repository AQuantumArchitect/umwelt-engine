"""Glass door — ≤5 tiles, fog when empty, authored tiles win."""
from __future__ import annotations

from dataclasses import replace

from tests.test_spec_to_field import tiny_grid_spec
from umwelt.boot import build_engine
from umwelt.projection.glass_door import TILES_MAX, glass_door


def test_derived_tiles_cap_and_skip_clock():
    engine = build_engine(spec=tiny_grid_spec(), population=False, fractal=False)
    snap = glass_door(engine)
    assert snap["schema"] == "glass_door/v1"
    assert snap["tiles_max"] == TILES_MAX == 5
    assert 1 <= snap["tiles_n"] <= 5
    names = [t["node"] for t in snap["tiles"]]
    assert all(not n.startswith("_") for n in names)
    assert snap["honesty"]["null_dispatch"] is True
    assert snap["status"] == "live"


def test_authored_tiles_win():
    spec = replace(tiny_grid_spec(), tiles=(("cell_1_1", "resource"),))
    engine = build_engine(spec=spec, population=False, fractal=False)
    snap = glass_door(engine)
    assert snap["tiles_n"] == 1
    t = snap["tiles"][0]
    assert t["node"] == "cell_1_1" and t["role"] == "resource"


def test_unknown_tile_is_dark_not_invented():
    spec = replace(tiny_grid_spec(), tiles=(("no_such", "agent_near"),))
    engine = build_engine(spec=spec, population=False, fractal=False)
    snap = glass_door(engine)
    assert snap["tiles"][0]["status"] == "dark"
    assert snap["tiles"][0]["z"] is None
    assert snap["tiles"][0].get("reason")
