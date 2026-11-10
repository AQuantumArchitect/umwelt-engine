"""A snapshot must retain the forecast learning it claims to carry forward."""
from datetime import datetime, timedelta, timezone
import pickle
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from umwelt.boot import build_engine
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
from umwelt.substrate.continuation import walk


START = datetime(2026, 11, 9)


def engine():
    spec = DomainSpec(name="forecast-continuation", nodes=(
        NodeSpec("root", parent=None, kind="root", roles=("value",)),
    ), bindings=(BindingSpec("sensor", zone="root", role="value",
                             normalizer="binary", force_observe=True),),
        forecast_leaves=(("root", "value"),), forecast_horizons_min=(1., 3.),
        forecast_context_surprise=True)
    return build_engine(spec=spec, population=False, calibration=False, fractal=False)


def tick(eng, minute):
    eng.ingest(sensor_readings={"sensor": minute % 2},
               now=START + timedelta(minutes=minute))


def test_engine_snapshot_restores_weights_scores_issued_predictions_and_pending_labels(tmp_path):
    original = engine()
    for minute in range(9):
        tick(original, minute)
    path = tmp_path / "snapshot.pkl"
    original.save(path)
    resumed = engine()
    resumed.load(path)
    assert walk(original.forecast_surface).digest() == walk(resumed.forecast_surface).digest()
    assert resumed.forecast_surface.predictions() == original.forecast_surface.predictions()


def test_resumed_forecast_uses_same_pending_predictions_on_future_field_observations(tmp_path):
    original = engine()
    for minute in range(9):
        tick(original, minute)
    path = tmp_path / "snapshot.pkl"
    original.save(path)
    resumed = engine()
    resumed.load(path)
    # Give both forecast organs the SAME subsequent field observations. This isolates
    # forecast continuation from the separately known wider engine capture gap.
    for minute in range(9, 16):
        tick(original, minute)
        resumed.forecast_surface.step(START + timedelta(minutes=minute), original.field)
        assert walk(original.forecast_surface).digest() == walk(resumed.forecast_surface).digest()


def recorded(tmp_path):
    original = engine()
    for minute in range(9):
        tick(original, minute)
    path = tmp_path / "snapshot.pkl"
    original.save(path)
    with path.open("rb") as stream:
        data = pickle.load(stream)
    return original, path, data


@pytest.mark.parametrize("damage", [
    "version", "registry", "geometry", "missing-model", "extra-key", "nan-weight",
    "weight-width", "nan-error", "negative-counter", "invalid-ema", "wrong-leaf",
    "pending-window", "pending-order", "pending-vector", "prediction-window",
    "null-block", "context", "surprise-decay", "clock-ahead", "mixed-clocks",
])
def test_invalid_forecast_block_rejects_before_any_engine_mutation(tmp_path, damage):
    _, path, data = recorded(tmp_path)
    state = data["forecast_continuation"]
    record = next(iter(state["bank"].values()))
    reg, leaf, pending = record["regressor"], record["leaf"], record["pending"]
    if damage == "version": state["format"] = "forecast-continuation/999"
    elif damage == "registry": state["features"] = (("wrong", "module", "builder"),)
    elif damage == "geometry": state["surface"]["_monomials"] = ()
    elif damage == "missing-model": state["bank"].pop(next(iter(state["bank"])))
    elif damage == "extra-key": reg["unknown"] = 1
    elif damage == "nan-weight": reg["W"][0, 0] = np.nan
    elif damage == "weight-width": reg["W"] = np.zeros((1, 100))
    elif damage == "nan-error": reg["error_ema"] = np.nan
    elif damage == "negative-counter": reg["n_scored"] = -1
    elif damage == "invalid-ema": reg["ema"] = 2.
    elif damage == "wrong-leaf": leaf["role"] = "absent"
    elif damage == "pending-window": pending[0]["expires"] = START
    elif damage == "pending-order": pending.extend([dict(pending[0], due=START)])
    elif damage == "pending-vector": pending[0]["feature"][0] = np.inf
    elif damage == "prediction-window": leaf["prediction_for"] = START
    elif damage == "null-block": data["forecast_continuation"] = None
    elif damage == "context": state["surface"]["_last_z"][("wrong", "role")] = 1.
    elif damage == "surprise-decay": state["surface"]["surprise_decay"] = np.inf
    elif damage == "clock-ahead": leaf["_last_step_at"] += timedelta(days=1)
    elif damage == "mixed-clocks": pending[0]["due"] = pending[0]["due"].replace(tzinfo=timezone.utc)
    path.write_bytes(pickle.dumps(data))
    target = engine()
    tick(target, 20)
    before = walk(target).digest()
    with pytest.raises(ValueError):
        target.load(path)
    assert walk(target).digest() == before


def test_legacy_missing_forecast_block_keeps_old_behavior_and_honest_verdict(tmp_path):
    original, path, data = recorded(tmp_path)
    del data["forecast_continuation"]
    path.write_bytes(pickle.dumps(data))
    resumed = engine()
    resumed.load(path)
    assert all(leaf.fc.W is None for leaf in resumed.forecast_surface.bank.values())
    assert resumed._continuation == "approximate"
    assert walk(resumed.forecast_surface).digest() != walk(original.forecast_surface).digest()


def test_load_preserves_live_references_and_independent_arrays(tmp_path):
    original, path, _ = recorded(tmp_path)
    resumed = engine()
    surface = resumed.forecast_surface
    key = next(iter(surface.bank))
    leaf, reg = surface.bank[key], surface.bank[key].fc
    resumed.load(path)
    assert resumed.forecast_surface is surface
    assert surface.bank[key] is leaf and leaf.fc is reg
    assert not np.shares_memory(reg.W, original.forecast_surface.bank[key].fc.W)
    reg.W[:] = 0
    assert np.any(original.forecast_surface.bank[key].fc.W != 0)


def test_changed_runtime_priors_do_not_rewrite_frozen_windows_on_restore(tmp_path):
    original = engine()
    for minute in range(9): tick(original, minute)
    leaf = next(iter(original.forecast_surface.bank.values()))
    old = (leaf.prediction_for, leaf.label_deadline)
    leaf.set_horizon_minutes(20)
    leaf.label_lateness_minutes = 4.
    path = tmp_path / "snapshot.pkl"
    original.save(path)
    resumed = engine()
    resumed.load(path)
    leaf = next(iter(resumed.forecast_surface.bank.values()))
    assert leaf.horizon_min == 20 and leaf.label_lateness_minutes == 4.
    assert (leaf.prediction_for, leaf.label_deadline) == old


def test_dissolved_rollout_rejects_learned_bank_and_null_roundtrips(tmp_path, monkeypatch):
    _, path, _ = recorded(tmp_path)
    monkeypatch.setenv("UMWELT_FORECAST_DISSOLVE", "1")
    resumed = engine()
    before = walk(resumed).digest()
    with pytest.raises(ValueError, match="matching learned surface"):
        resumed.load(path)
    assert walk(resumed).digest() == before
    resumed.save(path)
    resumed.load(path)


def test_invalid_live_weights_cannot_be_written_as_a_valid_snapshot(tmp_path):
    original = engine()
    for minute in range(3): tick(original, minute)
    next(iter(original.forecast_surface.bank.values())).fc.W[:] = np.nan
    path = tmp_path / "snapshot.pkl"
    path.write_bytes(b"previous retained snapshot")
    with pytest.raises(ValueError): original.save(path)
    assert path.read_bytes() == b"previous retained snapshot"


def test_cold_snapshot_retains_pending_unissued_labels(tmp_path):
    original = engine()
    tick(original, 0)
    path = tmp_path / "snapshot.pkl"
    original.save(path)
    resumed = engine()
    resumed.load(path)
    assert walk(original.forecast_surface).digest() == walk(resumed.forecast_surface).digest()
    for leaf in resumed.forecast_surface.bank.values():
        assert leaf.fc.W is None and leaf.fc.n_scored == 0
        assert len(leaf.buffer) == 1 and leaf.buffer[0].prediction is None


def test_fresh_process_resumes_real_ingest_forecasts(tmp_path):
    original, path, _ = recorded(tmp_path)
    before = walk(original.forecast_surface).digest()
    for minute in range(9, 16): tick(original, minute)
    after = walk(original.forecast_surface).digest()
    script = '''
import json, runpy, sys
helpers = runpy.run_path(sys.argv[1])
eng = helpers['engine']()
eng.load(sys.argv[2])
before = helpers['walk'](eng.forecast_surface).digest()
for minute in range(9, 16): helpers['tick'](eng, minute)
print(json.dumps({'before': before, 'after': helpers['walk'](eng.forecast_surface).digest()}))
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), str(path)],
                            check=True, text=True, capture_output=True, timeout=30)
    assert json.loads(result.stdout) == {"before": before, "after": after}
