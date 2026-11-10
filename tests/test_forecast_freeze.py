"""The engine's frozen context must also reach its attached forecast learner."""
import pickle
from datetime import datetime, timedelta

import pytest

from umwelt.boot import build_engine
from umwelt.learning.context import ContextState
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec


START = datetime(2026, 11, 9)


def engine():
    spec = DomainSpec(name="forecast-freeze", nodes=(
        NodeSpec("root", parent=None, kind="root", roles=("value",)),
    ), bindings=(BindingSpec("sensor", zone="root", role="value",
                             normalizer="binary", force_observe=True),),
        forecast_leaves=(("root", "value"),), forecast_horizons_min=(1.,))
    eng = build_engine(spec=spec, population=False, calibration=False, fractal=False)
    eng._stream_learn = 0
    return eng


def tick(eng, minute):
    eng.ingest(sensor_readings={"sensor": minute % 2},
               now=START + timedelta(minutes=minute))


@pytest.mark.parametrize("mode", [ContextState.forebrain, ContextState.test])
def test_frozen_context_predicts_without_training_scoring_or_queuing_labels(mode):
    eng = engine()
    for minute in range(4):
        tick(eng, minute)
    leaf = next(iter(eng.forecast_surface.bank.values()))
    assert leaf.fc.n_updates > 0 and leaf.fc.n_scored > 0
    before_weights = leaf.fc.W.copy()
    before_counts = (leaf.fc.n_updates, leaf.fc.n_scored)
    before_quality = (leaf.fc.error_ema, leaf.fc.baseline_error_ema)
    before_queue = pickle.dumps(leaf.buffer)
    mode().write_to_bundle(eng.graph.root.param_bundle)
    for minute in range(4, 7):
        tick(eng, minute)
    assert (leaf.fc.W == before_weights).all()
    assert (leaf.fc.n_updates, leaf.fc.n_scored) == before_counts
    assert (leaf.fc.error_ema, leaf.fc.baseline_error_ema) == before_quality
    assert pickle.dumps(leaf.buffer) == before_queue
    assert leaf.issued_at == START + timedelta(minutes=6)
    assert leaf.prediction is not None
    ContextState.live().write_to_bundle(eng.graph.root.param_bundle)
    tick(eng, 7)  # old unresolved labels are expired, not filled from frozen ticks
    assert leaf.fc.n_updates == before_counts[0]
    assert leaf.censored_labels > 0
    tick(eng, 8)
    assert leaf.fc.n_updates == before_counts[0] + 1


def test_cold_frozen_engine_does_not_create_a_trained_model():
    eng = engine()
    ContextState.forebrain().write_to_bundle(eng.graph.root.param_bundle)
    for minute in range(4):
        tick(eng, minute)
    leaf = next(iter(eng.forecast_surface.bank.values()))
    assert leaf.fc.W is None and leaf.fc.n_updates == leaf.fc.n_scored == 0
    assert not leaf.buffer and leaf.prediction is None


def test_explicit_stream_learning_still_reaches_the_forecast_organ():
    eng = engine()
    ContextState.forebrain().write_to_bundle(eng.graph.root.param_bundle)
    eng._stream_learn = 2
    leaf = next(iter(eng.forecast_surface.bank.values()))
    leaf.label_lateness_minutes = 1.
    for minute in range(4):
        tick(eng, minute)
    assert eng._stream_count == 2
    assert leaf.fc.n_updates == 1


def test_dissolved_surface_still_runs_in_frozen_engine(monkeypatch):
    monkeypatch.setenv("UMWELT_FORECAST_DISSOLVE", "1")
    eng = engine()
    ContextState.forebrain().write_to_bundle(eng.graph.root.param_bundle)
    tick(eng, 0)
    surface = eng.forecast_surface
    assert surface._now == START
    assert surface._field is eng.field
    rows = surface.predictions()
    assert rows and all(row["skill"] is None for row in rows.values())
