"""Confession join — observed / believed / forecast / decided; missing organs are null."""
from __future__ import annotations

from datetime import datetime, timezone

from tests.test_spec_to_field import tiny_grid_spec
from umwelt.boot import build_engine
from umwelt.projection.confession import recent


def test_blank_confession_has_no_observed():
    engine = build_engine(spec=tiny_grid_spec(), population=False, fractal=False)
    snap = recent(engine)
    assert snap["schema"] == "confession/v1"
    assert snap["honesty"]["guesses_labels"] is False
    assert snap["items"]
    first = snap["items"][0]
    assert first["observed"] is None
    assert first["forecast"] is None
    assert first["believed"]["z"] is not None
    assert first["dispatched"] is False
    assert first["echo"] is None


def test_ingest_fills_observed_and_does_not_invent_forecast():
    engine = build_engine(spec=tiny_grid_spec(), population=False, fractal=False)
    t = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    engine.ingest(sensor_readings={"sight_cell_0_0": 1.0}, now=t)
    snap = recent(engine, n=8)
    by = {(it["node"], it["role"]): it for it in snap["items"]}
    row = by[("cell_0_0", "agent_near")]
    assert row["observed"] is not None
    assert row["observed"]["sensor_id"] == "sight_cell_0_0"
    assert row["observed"]["value"] == 1.0
    assert row["forecast"] is None          # no forecast organ on this spec
    assert row["status"] == "live"
    # live axes sort ahead of latent
    assert snap["items"][0]["status"] in ("live", "dark")
