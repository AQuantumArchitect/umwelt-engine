"""Sparse evidence must retain its one-gap hold across a snapshot restart."""
from datetime import datetime, timedelta, timezone
import pickle

import numpy as np
import pytest

from umwelt.boot import build_engine
from umwelt.learning.context import ContextState
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec


def boot(hold=1.0):
    spec = DomainSpec(name="cadence-snapshot", nodes=(
        NodeSpec("root", kind="root", roles=("signal",),
                 role_modes={"signal": "dissipative"}),),
        bindings=(BindingSpec("reading", zone="root", role="signal",
                              normalizer={"type":"range", "lo":0.0, "hi":10.0}),),
        ingest_hold_s=hold)
    engine = build_engine(spec=spec, population=False, calibration=False,
                          fractal=False, role=ContextState.forebrain(), seed=123)
    engine._stream_learn = 0  # isolate the hold path from the known learner closure gap
    return engine


def test_snapshot_preserves_one_gap_evolution_and_observed_inputs(tmp_path):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    reference = boot()
    reference.ingest(sensor_readings={"reading":9.0}, now=t0)
    snapshot = tmp_path / "state.pkl"
    reference.save(snapshot)
    before = reference._wall_catchup_steps
    reference.ingest(sensor_readings={}, now=t0 + timedelta(seconds=3))
    expected = reference.field.clusters["root"].rho.copy()
    assert reference._wall_catchup_steps - before == 2

    resumed = boot()
    resumed.load(snapshot)
    resumed.ingest(sensor_readings={}, now=t0 + timedelta(seconds=3))
    assert resumed._wall_catchup_steps - before == 2
    np.testing.assert_allclose(resumed.field.clusters["root"].rho, expected, atol=1e-10, rtol=0)
    assert resumed._last_bridged_inputs is None  # silence ends the previous hold


def test_silence_and_counters_survive_a_second_restart(tmp_path):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine = boot()
    engine.ingest(sensor_readings={"reading": 9}, now=t0)
    engine.ingest(sensor_readings={}, now=t0 + timedelta(seconds=3))
    snapshot = tmp_path / "silent.pkl"
    engine.save(snapshot)
    resumed = boot()
    resumed.load(snapshot)
    assert resumed._wall_catchup_steps == 2
    assert resumed._last_bridged_inputs is None
    resumed.ingest(sensor_readings={}, now=t0 + timedelta(seconds=6))
    assert resumed._wall_catchup_steps == 4
    assert resumed._last_bridged_inputs is None
    # Restart cannot turn the pre-silence reading into evidence for another gap.
    engine.ingest(sensor_readings={}, now=t0 + timedelta(seconds=6))
    np.testing.assert_allclose(resumed.field.clusters["root"].rho,
                               engine.field.clusters["root"].rho, atol=1e-10, rtol=0)


def test_legacy_absence_does_not_invent_hold_memory(tmp_path):
    original = boot()
    original.ingest(sensor_readings={"reading": 9}, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    snapshot = tmp_path / "legacy.pkl"
    original.save(snapshot)
    data = pickle.loads(snapshot.read_bytes())
    del data["ingest_cadence"]
    snapshot.write_bytes(pickle.dumps(data))
    historical = snapshot.read_bytes()
    resumed = boot()
    resumed.load(snapshot)
    assert resumed._last_wall_now is None and resumed._last_bridged_inputs is None
    assert resumed._wall_catchup_steps == 0
    assert resumed._continuation == "approximate"
    assert snapshot.read_bytes() == historical


@pytest.mark.parametrize("key,value", [
    ("schema", "unknown/2"), ("hold_s", True), ("hold_s", float("nan")),
    ("max_substeps", True), ("catchup_steps", -1), ("catchup_steps", True),
    ("last_wall_now", "2026-01-01"), ("held_inputs", []),
    ("held_inputs", {"root": np.array([float("nan")])}),
    ("held_inputs", {"root": np.array([1, 2])}),
    ("input_roles", {"root": ("different",)}),
])
def test_malformed_cadence_refuses_before_mutating_engine(tmp_path, key, value):
    engine = boot()
    engine.ingest(sensor_readings={"reading":9}, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    snapshot = tmp_path / "bad.pkl"
    engine.save(snapshot)
    data = pickle.loads(snapshot.read_bytes())
    data["ingest_cadence"][key] = value
    snapshot.write_bytes(pickle.dumps(data))
    resumed = boot()
    rho = resumed.field.clusters["root"].rho.copy()
    with pytest.raises(ValueError, match="ingest cadence"):
        resumed.load(snapshot)
    assert resumed._step == 0 and resumed._last_wall_now is None
    np.testing.assert_array_equal(resumed.field.clusters["root"].rho, rho)


@pytest.mark.parametrize("hold,cap", [(2.0, 32), (None, 32), (1.0, 31)])
def test_configuration_is_not_silently_restored_from_snapshot(tmp_path, hold, cap):
    source = boot()
    snapshot = tmp_path / "config.pkl"
    source.save(snapshot)
    target = boot(hold)
    target._wall_catchup_max = cap
    with pytest.raises(ValueError, match="configuration changed"):
        target.load(snapshot)
    assert target.ingest_hold_s == hold and target._wall_catchup_max == cap


def test_disabled_cadence_roundtrip_stays_disabled(tmp_path):
    source = boot(None)
    source.ingest(sensor_readings={"reading":9}, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    snapshot = tmp_path / "off.pkl"
    source.save(snapshot)
    target = boot(None)
    target.load(snapshot)
    assert target._last_wall_now == source._last_wall_now
    assert target._last_bridged_inputs is None
    assert target._wall_catchup_steps == 0 and target.ingest_hold_s is None


def test_explicit_null_block_is_not_legacy_absence(tmp_path):
    source = boot()
    snapshot = tmp_path / "null.pkl"
    source.save(snapshot)
    data = pickle.loads(snapshot.read_bytes())
    data["ingest_cadence"] = None
    snapshot.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match="invalid ingest cadence"):
        boot().load(snapshot)


def test_nontext_node_identity_is_refused_as_evidence(tmp_path):
    source = boot()
    snapshot = tmp_path / "identity.pkl"
    source.save(snapshot)
    data = pickle.loads(snapshot.read_bytes())
    data["ingest_cadence"]["held_inputs"] = {1: np.array([0.5])}
    data["ingest_cadence"]["input_roles"] = {1: ("signal",)}
    snapshot.write_bytes(pickle.dumps(data))
    target = boot()
    with pytest.raises(ValueError, match="node identity"):
        target.load(snapshot)
    assert target._step == 0 and target._last_wall_now is None
