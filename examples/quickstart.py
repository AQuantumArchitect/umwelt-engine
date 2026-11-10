"""Quickstart: blank world -> readings -> beliefs -> dark -> frame/1 -> save/load.

Run:  python examples/quickstart.py        (numpy only; offline; about a second)
"""
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from umwelt import BindingSpec, BridgeSpec, DomainSpec, DriverSpec, NodeSpec, build_engine
from umwelt.projection.frame import frame, validate
from umwelt.projection.nameplates import nameplates

spec = DomainSpec(                      # a world is data: two rooms, two sensors
    name="tiny-world",
    nodes=(
        NodeSpec("hall", parent=None, kind="root", roles=("occupied",)),
        NodeSpec("den", parent="hall", roles=("occupied", "warmth"),
                 role_modes={"occupied": "unitary", "warmth": "dissipative"}),
    ),
    bridges=(BridgeSpec("hall", "den", shared_roles=("occupied",), kind="open"),),
    bindings=(
        BindingSpec("den_motion", zone="den", role="occupied", normalizer="binary",
                    force_observe=True),
        BindingSpec("den_temp", zone="den", role="warmth",
                    normalizer={"type": "regime", "center": 21, "width": 4}),
    ),
    drivers=(DriverSpec("day", period_s=86400.0),),
)

engine = build_engine(spec=spec)        # boots blank: knows nothing about this world
t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
for i in range(10):                     # only the motion sensor ever reports; temp never does
    engine.ingest(sensor_readings={"den_motion": 1.0}, now=t0 + timedelta(seconds=10 * i))

doc = frame(engine, subject="tiny-world", now=t0 + timedelta(seconds=100))
for n in doc["nodes"]:                  # a value nobody measured is None (dark), never 0
    print(f"{n['id']:<14} value={n['value']}  state={n['state']}  age_s={n['age_s']}")
print("DARK:", [n["id"] for n in doc["nodes"] if n["value"] is None])
print("frame/1:", doc["schema"], "violations:", validate(doc) or "none")
print("purity (how sharp each belief is):", {a["axis_id"]: a["purity"] for a in nameplates(engine)["axes"]})

with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "world.pkl"
    engine.save(path)
    twin = build_engine(spec=spec)
    twin.load(path)
    same = engine.field_canon_hash() == twin.field_canon_hash()
print("save/load hash-equal:", same)
assert same
