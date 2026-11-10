"""n_gym_body example: the spec validates, boots an engine, and takes its report-shaped rows."""
from datetime import datetime, timezone

from examples.n_gym_body.world import n_gym_body_spec, synthetic_capacity_rows
from umwelt.boot import build_engine


def test_spec_shape_and_bindings_land_on_nodes():
    spec = n_gym_body_spec()
    names = {n.name for n in spec.nodes}
    assert {"n_gym", "body", "portfolio", "stream_heavy", "stream_light"} <= names
    assert all(b.zone in names for b in spec.bindings)
    assert all(b.force_observe for b in spec.bindings)


def test_engine_ingests_synthetic_rows():
    eng = build_engine(spec=n_gym_body_spec(), population=False, calibration=False, fractal=False)
    readings = {sid: float(v) for _ts, sid, v, _ in synthetic_capacity_rows()}
    eng.ingest(sensor_readings=readings, now=datetime(2026, 8, 3, 12, tzinfo=timezone.utc))
    assert "body" in eng.field.clusters
