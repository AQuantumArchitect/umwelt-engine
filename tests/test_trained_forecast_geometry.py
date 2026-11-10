"""Missing geometry cannot authorize a trained forecast's downstream nudge.

Score seeding isolates this admission boundary; it is not calibration evidence.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from umwelt.foresight.forecast_surface import ForecastSurface


START = datetime(2026, 11, 9, tzinfo=timezone.utc)
KEY = ('sample', 'value')


def cluster(purity=1.):
    return SimpleNamespace(role_index={'value': 0}, purity=purity,
                           qubit_bloch=lambda _: (0., 0., .2))


def prepared():
    surface = ForecastSurface(leaves=(KEY,), horizons_min=(1.,))
    surface.pre_train({KEY: [(START - timedelta(minutes=i), .2) for i in range(25, 0, -1)]}, epochs=1)
    surface.step(START, SimpleNamespace(clusters={'sample': cluster()}))
    fc = surface.bank[(*KEY, 1.)]
    assert fc.prediction is not None
    fc.fc.error_ema, fc.fc.baseline_error_ema = .1, .4
    assert surface.consume_targets()[KEY]['confidence'] == .75
    return surface, fc


@pytest.mark.parametrize('bad', [None, True, '1', float('nan'), float('inf'), -.1, 1.1, 10**400])
def test_invalid_declared_purity_never_reuses_previous_valid_confidence(bad):
    surface, fc = prepared()
    surface.step(START + timedelta(seconds=10), SimpleNamespace(clusters={'sample': cluster(bad)}))
    assert surface.predictions()[(*KEY, 1.)]['confidence'] is None
    assert surface.consume_targets() == {}
    assert fc.prediction is not None  # Retain the prior forecast for inspection.


@pytest.mark.parametrize('missing', ['cluster', 'role', 'getter'])
def test_missing_current_leaf_clears_admission_but_retains_original_forecast(missing):
    surface, fc = prepared()
    previous = (fc.prediction, fc.issued_at, fc.prediction_for)
    field = SimpleNamespace(clusters={'sample': cluster()})
    if missing == 'cluster': field.clusters.clear()
    if missing == 'role': field.clusters['sample'].role_index.clear()
    if missing == 'getter':
        def unavailable(_): raise RuntimeError('sample unavailable')
        field.clusters['sample'].qubit_bloch = unavailable
    surface.step(START + timedelta(seconds=10), field)
    assert surface.predictions()[(*KEY, 1.)]['confidence'] is None
    assert surface.consume_targets() == {}
    assert (fc.prediction, fc.issued_at, fc.prediction_for) == previous


@pytest.mark.parametrize('purity', [0., .25, .5, 1.])
def test_valid_joint_purity_keeps_existing_weighting(purity):
    surface, _ = prepared()
    surface.step(START + timedelta(seconds=10), SimpleNamespace(clusters={'sample': cluster(purity)}))
    assert surface.consume_targets()[KEY]['confidence'] == pytest.approx(.75 * purity)


@pytest.mark.parametrize('bad_z', [None, True, '.2', float('nan'), float('inf'), -1.1, 1.1])
def test_invalid_current_bloch_value_cannot_authorize_an_old_forecast(bad_z):
    surface, fc = prepared()
    c = cluster()
    c.qubit_bloch = lambda _: (0., 0., bad_z)
    previous = (fc.prediction, fc.issued_at, fc.prediction_for, fc.fc.n_updates)
    surface.step(START + timedelta(seconds=10), SimpleNamespace(clusters={'sample': c}))
    assert surface.predictions()[(*KEY, 1.)]['confidence'] is None
    assert surface.consume_targets() == {}
    assert (fc.prediction, fc.issued_at, fc.prediction_for, fc.fc.n_updates) == previous


def test_reappearing_valid_leaf_can_supply_a_new_consumable_forecast():
    surface, fc = prepared()
    surface.step(START + timedelta(seconds=10), SimpleNamespace(clusters={}))
    assert surface.consume_targets() == {}
    surface.step(START + timedelta(seconds=20), SimpleNamespace(clusters={'sample': cluster()}))
    assert surface.consume_targets()[KEY]['confidence'] == .75
    assert fc.issued_at == START + timedelta(seconds=20)


def test_actual_engine_trains_then_withdraws_missing_leaf_from_consumption():
    from umwelt.boot import build_engine
    from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
    spec = DomainSpec(name='forecast-geometry-consumer', nodes=(
        NodeSpec('sample', parent=None, kind='root', roles=('value',)),),
        bindings=(BindingSpec('sensor', zone='sample', role='value',
                              normalizer='binary', force_observe=True),),
        forecast_leaves=(KEY,), forecast_horizons_min=(1.,))
    eng = build_engine(spec=spec, population=False, calibration=False, fractal=False)
    for minute in range(200):
        eng.ingest(sensor_readings={'sensor': minute % 2}, now=START + timedelta(minutes=minute))
    surface = eng.forecast_surface
    fc = surface.bank[(*KEY, 1.)]
    # This is a deterministic synthetic consumer fixture, not outside skill or
    # calibration evidence. Unlike the admission isolates, no score is seeded.
    assert fc.fc.n_scored > 0 and surface.consume_targets()[KEY]['confidence'] > 0
    previous = (fc.prediction, fc.issued_at, fc.fc.n_updates, fc.fc.n_scored)
    surface.step(START + timedelta(minutes=200), SimpleNamespace(clusters={}))
    assert surface.consume_targets() == {}
    assert (fc.prediction, fc.issued_at, fc.fc.n_updates, fc.fc.n_scored) == previous
    eng.ingest(sensor_readings={'sensor': 1}, now=START + timedelta(minutes=201))
    assert surface.consume_targets()[KEY]['confidence'] > 0
    assert fc.issued_at == START + timedelta(minutes=201)
