"""A world replayed twice from the same stream is the same world.

docs/SNAPSHOT_STAMP_ANCHOR.md guarantee 2: same snapshot + same typed inputs + same
recorded rng draws ⇒ same next snapshot. Before 2026-11-01 the default `build_engine`
broke it: the genetic Population drew its Generator from OS entropy, so two blank boots of
the quickstart world over an identical stream ended at two different field hashes — even
with `random`, `numpy` and PYTHONHASHSEED pinned. The daemon only looked deterministic
because its worker passes population=False and seeds 1234.

Now the population's seed is the engine seed when given, else a stable hash of the world's
topology — never OS entropy. What this does NOT yet pin: an UNSEEDED engine across two
fresh processes, because Thompson sampling in `substrate/params.py` still draws from the
process-global `random` (review issue #43, S8). Pass `seed=` (or pin the process RNGs, as
the daemon does) for cross-process identity.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.boot import build_engine
from umwelt.spec.schema import BindingSpec, BridgeSpec, DomainSpec, DriverSpec, NodeSpec

SPEC = DomainSpec(
    name="tiny-world",
    nodes=(NodeSpec("hall", parent=None, kind="root", roles=("occupied",)),
           NodeSpec("den", parent="hall", roles=("occupied", "warmth"),
                    role_modes={"occupied": "unitary", "warmth": "dissipative"})),
    bridges=(BridgeSpec("hall", "den", shared_roles=("occupied",), kind="open"),),
    bindings=(BindingSpec("den_motion", zone="den", role="occupied", normalizer="binary", force_observe=True),
              BindingSpec("den_temp", zone="den", role="warmth",
                          normalizer={"type": "regime", "center": 21, "width": 4})),
    drivers=(DriverSpec("day", period_s=86400.0),),
)


def _replay(spec, *, steps=12, **kw):
    engine = build_engine(spec=spec, **kw)
    t0 = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
    for i in range(steps):
        engine.ingest(sensor_readings={"den_motion": float(i % 2), "den_temp": 18.2 + 0.1 * i},
                      now=t0 + timedelta(seconds=10 * i))
    return engine


def test_two_blank_boots_over_one_stream_share_a_field_hash_with_the_population_on():
    hashes = []
    for _ in range(2):
        random.seed(0); np.random.seed(0)           # the process RNGs pinned, as the daemon pins them
        hashes.append(_replay(SPEC, population=True).field_canon_hash())
    assert hashes[0] == hashes[1], "the population drew OS entropy — the world is not a function of its stream"


def test_an_explicit_engine_seed_is_enough_on_its_own():
    a = _replay(SPEC, population=True, seed=7).field_canon_hash()
    random.seed(999); np.random.seed(999)            # dirty the process RNGs between the twins
    b = _replay(SPEC, population=True, seed=7).field_canon_hash()
    assert a == b


def test_population_seed_is_derived_from_topology_and_recorded(tmp_path):
    import pickle
    e1 = build_engine(spec=SPEC, population=True)
    e2 = build_engine(spec=SPEC, population=True)
    assert e1.population_seed == e2.population_seed
    other = DomainSpec(name="other", nodes=SPEC.nodes + (NodeSpec("attic", parent="hall", roles=("occupied",)),),
                       bridges=SPEC.bridges, bindings=SPEC.bindings, drivers=SPEC.drivers)
    assert build_engine(spec=other, population=True).population_seed != e1.population_seed
    assert build_engine(spec=SPEC, population=True, seed=42).population_seed == 42
    e1.save(tmp_path / "s.pkl")
    with open(tmp_path / "s.pkl", "rb") as fh:
        assert pickle.load(fh)["population_seed"] == e1.population_seed
