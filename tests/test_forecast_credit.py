"""Credit belongs to an issued prediction and its declared outcome time."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import numpy as np
import pytest

from umwelt.foresight.leaf_forecast import LeafForecaster
from umwelt.foresight.forecast_surface import ForecastSurface
from umwelt.learning.regressor import OnlineRegressor


START = datetime(2026, 11, 9, tzinfo=timezone.utc)


def leaf(**kwargs):
    return LeafForecaster("probe", "probe", "probe", "value", horizon_minutes=10, **kwargs)


def initialized(**kwargs):
    fc = leaf(l2=0, **kwargs)
    feat = fc._feature(START + timedelta(minutes=10), 0.0, None)
    fc.fc.W = np.zeros((1, len(feat)))
    return fc, feat


def test_hour_late_labels_do_not_train_ten_minute_forecasts():
    fc = leaf()
    fc.step(START, 0.0)
    fc.step(START + timedelta(minutes=5), .25)
    fc.step(START + timedelta(minutes=60), 1.0)
    assert fc.fc.n_updates == 0
    assert fc.fc.error_ema is None and fc.fc.baseline_error_ema is None


def test_perfect_fit_does_not_rewrite_a_wrong_issued_prediction():
    fc, feat = initialized()
    fc.fc.lr = 1 / float(feat @ feat)       # one gradient step fits this label exactly
    fc.step(START, 0.0)
    assert fc.prediction == 0.0
    fc.step(START + timedelta(minutes=10), 1.0)
    assert fc.fc.predict(feat)[0] == pytest.approx(1.0)
    assert fc.fc.error_ema == pytest.approx(1.0)
    assert fc.fc.baseline_error_ema == pytest.approx(1.0)
    assert fc.fc.skill_vs_persistence == pytest.approx(0.0)


def test_offline_warm_start_does_not_earn_prospective_confidence():
    fc = leaf()
    history = [(START + timedelta(minutes=5*i), float(i % 3)) for i in range(40)]
    assert fc.pre_train(history, epochs=3) > 0
    assert fc.fc.n_updates > 0 and fc.fc.W is not None
    assert fc.fc.error_ema is None and fc.fc.baseline_error_ema is None


def test_later_weight_change_cannot_rewrite_an_issued_forecast():
    fc, _ = initialized()
    fc.step(START, 0.0)
    fc.fc.W[:] = 100.0
    fc.fc.lr = 0.0
    fc.step(START + timedelta(minutes=10), 1.0)
    assert fc.fc.error_ema == 1.0
    assert fc.fc.n_scored == 1


def test_first_unissued_prediction_can_train_but_cannot_score():
    fc = leaf()
    fc.step(START, 0.0)
    assert fc.prediction is None
    fc.step(START + timedelta(minutes=10), 1.0)
    assert fc.fc.n_updates == 1
    assert fc.fc.n_scored == 0 and fc.fc.error_ema is None


def test_missing_due_label_is_not_filled_with_last_value():
    fc, _ = initialized()
    fc.step(START, 0.0)
    fc.step(START + timedelta(minutes=10), None)
    fc.step(START + timedelta(minutes=11), 1.0)
    assert fc.fc.n_updates == fc.fc.n_scored == 0
    assert fc.snapshot()["censored_labels"] == 1


@pytest.mark.parametrize("delay,scored", [(1, 1), (2, 1), (2.01, 0)])
def test_explicit_lateness_window_is_frozen_at_issue(delay, scored):
    fc, _ = initialized(label_lateness_minutes=2)
    fc.step(START, 0.0)
    fc.label_lateness_minutes = 100  # does not extend the existing forecast's window
    fc.step(START + timedelta(minutes=10 + delay), 1.0)
    assert fc.fc.n_scored == fc.fc.n_updates == scored
    assert fc.censored_labels == 1 - scored


def test_shorter_horizon_does_not_hide_mature_forecast_behind_old_queue_head():
    fc, _ = initialized()
    fc.step(START, 0.0)             # due10
    fc.set_horizon_minutes(1)
    fc.step(START + timedelta(minutes=1), 0.0)  # due2
    fc.step(START + timedelta(minutes=2), 1.0)
    assert fc.fc.n_scored == fc.fc.n_updates == 1
    assert fc.fc.error_ema == 1.0


def test_predict_only_does_not_train_score_or_add_pending_labels():
    fc, _ = initialized()
    fc.step(START, 0.0)
    before = fc.fc.W.copy()
    fc.step(START + timedelta(minutes=10), 1.0, train=False)
    np.testing.assert_array_equal(fc.fc.W, before)
    assert fc.fc.n_updates == fc.fc.n_scored == 0
    assert len(fc.buffer) == 1


def test_binary_score_uses_the_emitted_clipped_value():
    fc, _ = initialized(binary=True, lr=0)
    fc.fc.W[0, -1] = 2.0
    fc.step(START, 0.0)
    assert fc.prediction == 1.0
    fc.step(START + timedelta(minutes=10), 1.0)
    assert fc.fc.error_ema == 0.0


def test_surface_warm_start_stays_dark_until_prospective_time_matched_outcomes():
    surface = ForecastSurface(leaves=(("probe", "value"),), horizons_min=(10,))
    fc = surface.bank[("probe", "value", 10)]
    surface._purity[("probe", "value")] = 1.0
    history = [(START + timedelta(minutes=5*i), float(i % 2)) for i in range(40)]
    assert surface.pre_train({("probe", "value"): history}) > 0
    fc.step(START + timedelta(minutes=200), 0.0)
    assert surface._confidence(fc) is None
    assert surface.consume_targets() == {}
    assert surface.predictions()[("probe", "value", 10)]["scoring_contract"] == "issued-leaf-forecast/1"


def test_feature_reset_drops_baseline_and_quality_from_old_model():
    reg = OnlineRegressor(1)
    reg.update(np.ones(2), np.ones(1), baseline_pred=np.zeros(1))
    assert reg.n_scored == 1
    assert reg.predict(np.ones(3)) is None
    assert reg.error_ema is reg.baseline_error_ema is None
    assert reg.n_updates == reg.n_scored == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_lateness_prior_refuses_invalid_values(bad):
    with pytest.raises(ValueError, match="lateness"):
        leaf(label_lateness_minutes=bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_outcome_does_not_mutate_forecast_or_score(value):
    fc, _ = initialized()
    fc.step(START, 0.0)
    with pytest.raises(ValueError, match="finite"):
        fc.step(START + timedelta(minutes=10), value)
    assert fc.last_value == 0.0 and fc.issued_at == START
    assert fc.fc.n_updates == fc.fc.n_scored == 0


def test_scores_match_manual_errors_of_every_issued_prediction():
    fc = leaf()
    outcomes = [float((i * 37) % 7) / 6 for i in range(72)]
    issued, model_error, baseline_error = {}, None, None
    for i, value in enumerate(outcomes):
        if i in issued:
            prediction, baseline = issued.pop(i)
            error, naive = abs(prediction - value), abs(baseline - value)
            model_error = error if model_error is None else .02 * error + .98 * model_error
            baseline_error = naive if baseline_error is None else .02 * naive + .98 * baseline_error
        fc.step(START + timedelta(minutes=5*i), value)
        if fc.prediction is not None:
            issued[i + 2] = (fc.prediction, value)
    assert fc.fc.error_ema == pytest.approx(model_error)
    assert fc.fc.baseline_error_ema == pytest.approx(baseline_error)


def test_rewound_clock_cannot_resolve_a_different_trial_without_reset():
    fc, _ = initialized()
    fc.step(START + timedelta(minutes=10), 0.0)
    with pytest.raises(ValueError, match="start_sequence"):
        fc.step(START, 1.0)
    assert fc.last_value == 0.0 and fc.issued_at == START + timedelta(minutes=10)
    fc.start_sequence()
    assert fc.censored_labels == 1
    assert fc.fc.W is not None and fc.fc.n_updates == 0
    fc.step(START, 1.0)
    fc.step(START + timedelta(minutes=10), .5)
    assert fc.fc.n_scored == 1


def test_surface_sequence_reset_removes_cross_trial_context_without_forgetting_weights():
    surface = ForecastSurface(leaves=(("probe", "value"),), horizons_min=(10,), n_context=1)
    fc = surface.bank[("probe", "value", 10)]
    feat = fc._feature(START, 0.0, None)
    fc.fc.W = np.ones((1, len(feat)))
    weights = fc.fc.W.copy()
    fc.step(START, 0.0)
    surface._last_z[("probe", "value")] = .9
    surface._trace[("probe", "value")] = .7
    surface._purity[("probe", "value")] = 1.0
    surface._now = START
    surface.start_sequence()
    np.testing.assert_array_equal(fc.fc.W, weights)
    assert surface._last_z == surface._trace == surface._purity == {}
    assert surface._now is None and fc.last_value is None
    assert fc.censored_labels == 1 and not fc.buffer


@pytest.mark.parametrize("allowance,valid", [(0.0, True), (1.0, True),
    (-1.0, False), (float("inf"), False), (float("nan"), False), (True, False)])
def test_domain_gate_validates_declared_label_window(allowance, valid):
    from examples.shaman_self.world import SHAMAN_SELF_SPEC
    from umwelt.spec.validate import validate_spec
    spec = replace(SHAMAN_SELF_SPEC, forecast_label_lateness_min=allowance)
    report = validate_spec(spec, resolve_only=True)
    assert report.ok is valid, report.to_dict()
    if not valid:
        assert any("forecast_label_lateness_min" in c.detail for c in report.failures())


def test_published_window_cannot_change_when_future_priors_change():
    surface = ForecastSurface(leaves=(("probe", "value"),), horizons_min=(10,),
                              label_lateness_minutes=2)
    fc = surface.bank[("probe", "value", 10)]
    feat = fc._feature(START, 0.0, None)
    fc.fc.W = np.zeros((1, len(feat)))
    fc.step(START, 0.0)
    key = ("probe", "value", 10)
    issued = surface.predictions()[key]
    fc.label_lateness_minutes = 100
    fc.set_horizon_minutes(1)
    assert surface.predictions()[key] == issued
    assert issued["horizon_min"] == 10 and issued["label_lateness_minutes"] == 2
    assert issued["label_deadline"] == (START + timedelta(minutes=12)).isoformat()
