"""η = 0 is a detector that detected nothing — at `engine.ingest`, not only on the host face.

README novel-part #2: a confidence-0 observation is a no-op. Until 2026-11-01 that held for
the belief field (with the population off) but not for the LEARNED state: the ingress
membrane cached the raw value and timestamp before checking confidence, so an η=0 report
read as live and `hebbian_weight_update` learned the signal's weight from it (fog:
`w_scout_place_0` 1.0 → 1.0000007); observe bindings still emitted an alpha=0 target that
fed the observation-trust innovation EMA; calibration never saw confidence at all; and the
training backbone replayed recorded events at η=1. These pins hold the whole chain.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.boot import build_engine
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

SPEC = DomainSpec(
    name="eta-zero",
    nodes=(NodeSpec("root", parent=None, kind="root", roles=("warm",)),
           NodeSpec("den", parent="root", roles=("warm", "lit"),
                    role_modes={"warm": "dissipative", "lit": "unitary"})),
    bindings=(BindingSpec("den_temp", zone="den", role="warm",
                          normalizer={"type": "range", "lo": 0.0, "hi": 40.0}),
              BindingSpec("den_light", zone="den", role="lit", normalizer="binary", force_observe=True)),
)
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _twin():
    random.seed(11); np.random.seed(11)
    return build_engine(spec=SPEC, population=False)


def _run(readings_for, *, steps=40):
    """Build one twin and replay it to the end before the next is built: two engines in one
    process share the global RNG stream (issue #43, S8), so interleaving them would never
    compare equal for a reason that has nothing to do with η."""
    engine = _twin()
    for i in range(steps):
        readings, confidence = readings_for(i)
        engine.ingest(sensor_readings=readings, confidence=confidence, now=T0 + timedelta(seconds=10 * i))
    return engine


def _learned_state(engine):
    bridge = engine.sensor_bridge
    weights = {sid: (b.weight_param.value if b.weight_param is not None else b.weight)
               for sid, b in bridge.bindings.items()}
    trust = getattr(engine, "_obs_trust", None)
    innov = dict(getattr(trust, "innov", {}) or {}) if trust is not None else None
    return weights, innov, set(bridge.touched_roles), dict(bridge._last_normed)


def test_eta_zero_readings_leave_no_trace_anywhere():
    silent = _run(lambda i: ({}, None))
    zero = _run(lambda i: ({"den_temp": 31.0 + i, "den_light": float(i % 2)},
                           {"den_temp": 0.0, "den_light": 0.0}))
    assert zero.field_canon_hash() == silent.field_canon_hash()
    assert _learned_state(zero) == _learned_state(silent)
    assert zero.sensor_bridge.latest_raw("den_temp") is None, "an η=0 report must not read as live"
    assert zero.sensor_bridge.latest_raw("den_light") is None
    assert "den_temp" not in zero.sensor_bridge._last_normed
    assert zero.sensor_bridge.observe_targets({"den_light": 1.0}, {"den_light": 0.0}) == {}


def test_eta_zero_differs_from_eta_one_so_the_pin_is_not_vacuous():
    silent = _run(lambda i: ({}, None))
    one = _run(lambda i: ({"den_temp": 31.0 + i, "den_light": float(i % 2)},
                          {"den_temp": 1.0, "den_light": 1.0}))
    assert one.field_canon_hash() != silent.field_canon_hash()
    assert one.sensor_bridge.latest_raw("den_temp") is not None


def test_admitted_drops_exactly_the_eta_zero_readings():
    from umwelt.membranes.ingress import SensorBridge
    readings = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert SensorBridge.admitted(readings, None) == readings
    assert SensorBridge.admitted(readings, {"b": 0.0, "c": -1.0}) == {"a": 1.0}
    assert SensorBridge.admitted(readings, {"b": 0.3}) == readings
    assert SensorBridge.admitted(None, {"a": 0.0}) == {}


def test_training_replay_carries_the_recorded_confidence():
    from types import SimpleNamespace
    from umwelt.learning.training_backbone import RAW_EVENT_KIND, TrainingSession

    seen = {}

    class Reservoir:
        driver_forecast = None

        def ingest(self, **kw):
            seen.update(kw)
            return {"step": 1, "collapsed": False}

    trainer = TrainingSession.__new__(TrainingSession)
    record = SimpleNamespace(kind=RAW_EVENT_KIND, timestamp=T0,
                             payload={"sensor_readings": {"den_temp": 31.0}, "confidence": {"den_temp": 0.0}})
    trainer._apply_record(Reservoir(), record)
    assert seen["confidence"] == {"den_temp": 0.0}
    record.payload.pop("confidence")
    trainer._apply_record(Reservoir(), record)
    assert seen["confidence"] is None       # no recorded confidence → the engine's default, as before
