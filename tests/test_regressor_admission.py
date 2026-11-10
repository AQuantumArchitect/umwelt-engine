"""Bad observations must not damage the learner that rejects them."""
import pickle
from datetime import datetime, timedelta

import numpy as np
import pytest

from umwelt.learning.regressor import OnlineRegressor
from umwelt.foresight.leaf_forecast import LeafForecaster


def trained():
    reg = OnlineRegressor(1)
    reg.update(np.array([1., .5]), np.array([.4]), baseline_pred=np.array([0.]))
    return reg


@pytest.mark.parametrize("features,label,options", [
    ([np.nan, .5], [.4], {}), ([np.inf, .5], [.4], {}),
    ([1+2j, .5], [.4], {}), ([1., .5], [1+2j], {}),
    ([1., .5], [np.nan], {}), ([1., .5], [np.inf], {}),
    ([[1., .5]], [.4], {}), ([], [.4], {}),
    ([1., .5], [.4, .7], {}), ([1., .5], [[.4]], {}),
    ([1., .5], [.4], {"baseline_pred": np.array([np.nan])}),
    ([1., .5], [.4], {"baseline_pred": np.array([0., 0.])}),
    ([1., .5], [.4], {"lr": np.nan}),
    ([1., .5], [.4], {"lr": 1+2j}),
    ([1., .5], [.4], {"lr": -1}),
    ([1., .5], [.4], {"l2": np.inf}),
    ([1., .5], [.4], {"l2": -1}),
    ([1., .5], [.4], {"ema": 1.1}),
    ([1., .5], [.4], {"ema": -0.1}),
    ([1., .5, .2], [np.nan], {"lr": .7}),
    ([1e308, 1e308], [1.], {}),
    ([1., .5], [1e308], {"lr": 1e308}),
])
def test_refused_update_is_atomic_and_followed_by_valid_learning(features, label, options):
    reg = trained()
    before = pickle.dumps(reg.__dict__)
    with pytest.raises(ValueError):
        reg.update(np.asarray(features), np.asarray(label), **options)
    assert pickle.dumps(reg.__dict__) == before
    reg.update(np.array([.2, 1.]), np.array([.9]))
    assert reg.n_updates == 2 and np.isfinite(reg.W).all()


@pytest.mark.parametrize("features", [[np.nan, 1], [1, 2, np.nan], [[1, 2]], [], [np.inf, 1], [1+2j, 1]])
def test_invalid_prediction_does_not_reset_or_change_an_existing_model(features):
    reg = trained()
    before = pickle.dumps(reg.__dict__)
    with pytest.raises(ValueError):
        reg.predict(np.asarray(features))
    assert pickle.dumps(reg.__dict__) == before


def test_cold_invalid_update_does_not_install_weights_or_controls():
    reg = OnlineRegressor(1)
    before = pickle.dumps(reg.__dict__)
    with pytest.raises(ValueError):
        reg.update(np.array([np.nan]), np.array([1.]), lr=.7)
    assert pickle.dumps(reg.__dict__) == before


def test_scoring_overflow_does_not_poison_diagnostics():
    reg = trained()
    before = pickle.dumps(reg.__dict__)
    with pytest.raises(ValueError):
        reg.record_prediction_error(np.array([-1e308]), np.array([1e308]))
    assert pickle.dumps(reg.__dict__) == before


def test_real_forecast_rejects_bad_context_and_keeps_learning():
    fc = LeafForecaster("probe", "probe", "node", "value", n_context=1, horizon_minutes=10)
    now = datetime(2026, 11, 9)
    fc.fc.W = np.zeros((1, len(fc._feature(now, 0., [0.]))))
    fc.step(now, 0., context=[0.])
    before = pickle.dumps(fc.fc.__dict__)
    with pytest.raises(ValueError):
        fc.step(now + timedelta(minutes=1), 1., context=[np.nan])
    assert pickle.dumps(fc.fc.__dict__) == before
    fc.step(now + timedelta(minutes=10), 1., context=[.5])
    assert fc.fc.n_updates == fc.fc.n_scored == 1
    assert np.isfinite(fc.fc.W).all() and np.isfinite(fc.prediction)


@pytest.mark.parametrize("mode", ["mean_abs", "half_norm"])
@pytest.mark.parametrize("ball", [False, True])
def test_valid_gradient_and_fit_diagnostics_keep_the_existing_mathematics(mode, ball):
    reg = OnlineRegressor(2, lr=.25, l2=.1, ema=.3, error_mode=mode, project_ball=ball)
    x, y, baseline = np.array([2., 1.]), np.array([1., 2.]), np.array([0., .5])
    reg.update(x, y, baseline_pred=baseline)
    expected = .25 * np.outer(y, x)
    np.testing.assert_array_equal(reg.W, expected)
    fitted = expected @ x
    if mode == "half_norm" and ball:
        fitted = fitted / np.linalg.norm(fitted)
    error = .5 * np.linalg.norm(y - fitted) if mode == "half_norm" else np.abs(y - fitted).mean()
    assert reg.error_ema == pytest.approx(error)
    assert reg.baseline_error_ema == pytest.approx(1.25)
    assert reg.n_updates == reg.n_scored == 1
    alias = reg.W
    reg.update(x, y, record_error=False)
    assert reg.W is alias and reg.n_updates == 2 and reg.n_scored == 1
    np.testing.assert_allclose(reg.W, expected + .25 * (np.outer(y - expected @ x, x) - .1 * expected))


def test_valid_geometry_change_still_resets_old_quality():
    reg = trained()
    assert reg.predict(np.ones(3)) is None
    assert reg.W is None and reg.n_updates == reg.n_scored == 0
    assert reg.error_ema is reg.baseline_error_ema is None
    reg.update(np.ones(3), np.ones(1))
    assert reg.W.shape == (1, 3) and reg.n_updates == 1


def test_legacy_corrupt_weights_require_explicit_reset():
    reg = trained()
    reg.W[:] = np.nan
    before = pickle.dumps(reg.__dict__)
    for operation in (lambda: reg.predict(np.ones(2)), lambda: reg.update(np.ones(2), np.ones(1))):
        with pytest.raises(ValueError, match="explicit model reset"):
            operation()
        assert pickle.dumps(reg.__dict__) == before
    reg._reset_weights()
    reg.update(np.ones(2), np.ones(1))
    assert reg.n_updates == 1 and np.isfinite(reg.W).all()
