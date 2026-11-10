"""Nameplate projection — one face per node.role; status is evidence, not geometry."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dataclasses import replace

from tests.test_spec_to_field import tiny_grid_spec
from umwelt.boot import build_engine
from umwelt.projection.nameplates import nameplates
from umwelt.spec.schema import AxisFace, NodeSpec


def _engine():
    return build_engine(spec=tiny_grid_spec(), population=False, fractal=False)


def test_blank_boot_axes_are_latent():
    snap = nameplates(_engine())
    assert snap["schema"] == "nameplates/v1"
    assert snap["honesty"]["copies_rho"] is False
    assert snap["n"] >= 4
    by = {(a["node"], a["role"]): a for a in snap["axes"]}
    row = by[("cell_0_0", "agent_near")]
    assert row["status"] == "latent"
    assert row["z"] is not None
    assert row["axis_id"] == "cell_0_0.agent_near"
    assert "rho" not in row and "forecast" not in row


def test_ingest_marks_live_and_authored_poles():
    spec = tiny_grid_spec()
    # stamp one node with an authored face (opaque strings, no domain words)
    nodes = []
    for n in spec.nodes:
        if n.name == "cell_0_0":
            n = NodeSpec(
                name=n.name, parent=n.parent, roles=n.roles, kind=n.kind,
                role_modes=n.role_modes, params=n.params,
                faces=(AxisFace(role="agent_near", axis_id="nearness",
                                pole_0="near", pole_1="far"),),
            )
        nodes.append(n)
    spec = replace(spec, nodes=tuple(nodes))
    engine = build_engine(spec=spec, population=False, fractal=False)
    t = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    engine.ingest(sensor_readings={"sight_cell_0_0": 1.0}, now=t)
    snap = nameplates(engine)
    by = {(a["node"], a["role"]): a for a in snap["axes"]}
    live = by[("cell_0_0", "agent_near")]
    assert live["status"] == "live"
    assert live["pole_0"] == "near" and live["pole_1"] == "far"
    assert live["axis_id"] == "nearness"
    other = by[("cell_0_1", "agent_near")]
    assert other["status"] == "latent"


def test_degrades_on_foreign_object():
    snap = nameplates(object())
    assert snap["schema"] == "nameplates/v1"
    assert snap["n"] == 0
    assert snap["axes"] == []
