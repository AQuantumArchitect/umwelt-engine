from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from umwelt.foresight.forecast_rollout import DissolvedForecastSurface, _purity_of
from umwelt.substrate.product_cluster import ProductQubitCluster


def broken_geometry(_):
    raise RuntimeError('geometry unavailable')


def surface(cluster):
    s = DissolvedForecastSurface(leaves=(('sample', 'a'),), horizons_min=(10,))
    s.step(datetime(2026, 11, 9, tzinfo=timezone.utc),
           SimpleNamespace(clusters={'sample': cluster}))
    # Isolate evidence admission from forecast arithmetic: a value alone must not
    # grant itself confidence when the model geometry could not be read.
    s._rollout = lambda _: {('sample', 'a'): .2}
    return s


@pytest.mark.parametrize('bad', [None, True, '1', float('nan'), float('inf'), -.1, 1.1, 10**400])
def test_invalid_declared_purity_is_dark_even_if_bloch_fallback_would_be_pure(bad):
    c = SimpleNamespace(role_index={'a': 0}, purity=bad, role_bloch=lambda _: (0., 0., 1.))
    assert _purity_of(c, 'a') is None
    s = surface(c)
    assert next(iter(s.predictions().values()))['confidence'] is None
    assert s.consume_targets() == {}


def test_unreadable_geometry_does_not_become_a_pure_confident_target():
    c = SimpleNamespace(role_index={'a': 0}, role_bloch=broken_geometry)
    assert _purity_of(c, 'a') is None
    s = surface(c)
    row = next(iter(s.predictions().values()))
    assert row['z_pred'] == .2 and row['confidence'] is None
    assert row['skill'] is None
    assert s.consume_targets() == {}


@pytest.mark.parametrize('vec', [None, (0., 1.), (0., 0., float('nan')),
                                (0., 0., True), (2., 0., 0.)])
def test_invalid_local_fallback_stays_unknown(vec):
    c = SimpleNamespace(role_bloch=lambda _: vec)
    assert _purity_of(c, 'a') is None


@pytest.mark.parametrize('purity', [0., .25, .5, 1.])
def test_valid_declared_purity_preserves_existing_positive_behavior(purity):
    c = SimpleNamespace(role_index={'a': 0}, purity=purity)
    assert _purity_of(c, 'a') == purity
    row = next(iter(surface(c).predictions().values()))
    assert row['confidence'] == pytest.approx(round(purity / 2.718281828459045, 5))


def test_real_product_backend_keeps_supported_local_radius_fallback():
    c = ProductQubitCluster('sample', ['a'])
    c.observe_qubit(0, (0., .3, .4), 1.)
    assert _purity_of(c, 'a') == pytest.approx(.5)


def test_single_precision_endpoint_roundoff_is_clamped_not_promoted_arbitrarily():
    assert _purity_of(SimpleNamespace(purity=1.0000001), 'a') == 1.
    assert _purity_of(SimpleNamespace(purity=-.0000001), 'a') == 0.
    assert _purity_of(SimpleNamespace(purity=1.001), 'a') is None


def test_new_step_cannot_reuse_geometry_from_a_now_missing_cluster():
    s = surface(SimpleNamespace(role_index={'a': 0}, purity=1.))
    assert s.consume_targets()
    s.step(datetime(2026, 11, 9, 0, 1, tzinfo=timezone.utc), SimpleNamespace(clusters={}))
    assert next(iter(s.predictions().values()))['confidence'] is None
    assert s.consume_targets() == {}
