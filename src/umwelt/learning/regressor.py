"""OnlineRegressor — THE one online linear forecaster (dissolution M3, b9.33).

Every forecaster in the system — driver, environment, leaf, location — is the same
learner pointed at different targets: an online, L2-regularized linear map from a
feature vector to a target vector, with a rolling error EMA for skill, a raw-error
SNR proxy for the calibrator, and lr/l2/ema threaded live from the parameter fiber.
It used to be written twice (`CelestialForecast` ~150 lines / `WeatherForecast`
~240 lines, near-verbatim copies) — and the copies had DRIFTED: one carried a
feature-geometry reset guard another copy lacked, so a live feature-layout change
(sparse-feature migration, field rebuild) would crash that path mid-matmul.
The dedup delivers that guard to both.

The two survivors are thin subclasses (kept at their historical module paths so
imports and pickles are undisturbed):

    WeatherForecast  = OnlineRegressor(error="mean_abs")             sensor targets
    CelestialForecast= OnlineRegressor(error="half_norm", ball=True)  Bloch target

`project_ball` is the sphere chart's post-map (keep the prediction on/in the
Bloch ball — one atlas application, see bloch.py); `error_mode` preserves each
class's historical skill metric exactly: the scalar family tracks mean |err| per target,
the sphere family tracks the halved norm of the on-ball error.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np
from numpy.typing import NDArray

from umwelt._util import round_or_none


class OnlineRegressor:
    """Online linear regressor W: features → targets, trained one gradient step at a
    time (ΔW = lr(err ⊗ feat − l2·W)), geometry-guarded, with rolling error EMAs."""

    def __init__(
        self,
        n_targets: int,
        target_ids: list[str] | None = None,
        lr: float = 0.02,
        l2: float = 1e-4,
        ema: float = 0.02,
        project_ball: bool = False,
        error_mode: str = "mean_abs",          # "mean_abs" | "half_norm"
    ):
        self.n_targets = int(n_targets)
        self.target_ids = list(target_ids) if target_ids else [f"t{i}" for i in range(n_targets)]
        self.W: NDArray[np.floating] | None = None    # (n_targets, feat_dim), lazy-init
        self.lr = lr
        self.l2 = l2
        self.ema = ema                                # error EMA smoothing
        self.project_ball = bool(project_ball)
        self.error_mode = str(error_mode)
        self.error_ema: float | None = None           # rolling normalized error
        # Persistence-baseline error EMA (predict-last-value). Only accumulates when
        # update() is called with baseline_pred → skill_vs_persistence measures foresight
        # OVER "predict the current value", killing the frozen-belief mirage where absolute
        # skill reads ~1.0 for a belief that never moves.
        self.baseline_error_ema: float | None = None
        self.per_target_err: NDArray[np.floating] | None = None
        self.n_updates = 0
        self.n_scored = 0
        self._last_pred: NDArray[np.floating] | None = None
        # Raw (un-smoothed) error history — lets the calibrator estimate the
        # signal's SNR and adapt the EMA bandwidth (non-circular: it reads the
        # RAW series, not the smoothed one).
        self._raw_errors: deque[float] = deque(maxlen=64)

    def _reset_weights(self) -> None:
        """Drop the learned mapping when the feature geometry changes. The
        regressor re-learns from live data; cheap relative to a feature-layout
        change, which is rare (migration / field rebuild)."""
        self.W = None
        self.error_ema = None
        self.baseline_error_ema = None
        self.per_target_err = None
        self._last_pred = None
        self.n_updates = 0
        self.n_scored = 0
        self._raw_errors.clear()

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """Forecast the target vector, or None before any training. Feature geometry
        can change under us (sparse-feature migration, a live field rebuild) — W is
        lazily shaped to features, so guard the in-process case: a stale W never
        matmuls against a mismatched feature vector."""
        features = self._vector(features, "features")
        if self.W is None:
            return None
        self._check_weights()
        if self.W.shape[1] != features.shape[0]:
            self._reset_weights()
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            pred = self._project(self.W @ features)
        self._last_pred = pred
        return pred

    def _vector(self, value, name, *, targets=False):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real, not a complex projection")
        try:
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real numeric vector") from exc
        if (vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all()
                or (targets and vector.shape != (self.n_targets,))):
            raise ValueError(f"{name} must be a finite vector with valid geometry")
        return vector

    @staticmethod
    def _control(value, name, *, maximum=None):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real numeric coefficient") from exc
        if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
            raise ValueError(f"{name} must be finite and within its mathematical range")
        return value

    def _check_weights(self):
        if self.W is None:
            return
        try:
            valid = (isinstance(self.W, np.ndarray) and self.W.ndim == 2
                     and self.W.shape[0] == self.n_targets and not np.iscomplexobj(self.W)
                     and np.isfinite(self.W).all())
        except TypeError:
            valid = False
        if not valid:
            raise ValueError("existing weights are invalid; explicit model reset is required")

    def _project(self, prediction):
        if not np.isfinite(prediction).all():
            raise ValueError("prediction overflow; model unchanged")
        if self.project_ball:
            norm = float(np.linalg.norm(prediction))
            if not math.isfinite(norm):
                raise ValueError("prediction norm overflow; model unchanged")
            if norm > 1.0:
                prediction = prediction / norm
        return prediction

    def update(
        self,
        features: NDArray[np.floating],
        label: NDArray[np.floating],
        lr: float | None = None,
        l2: float | None = None,
        ema: float | None = None,
        baseline_pred: NDArray[np.floating] | None = None,
        *, record_error: bool = True,
    ) -> None:
        """One online step: train features → label. lr/l2/ema, when given, come
        live from the parameter fiber (calibrated, not buried constants).

        baseline_pred, when given, is the naive persistence forecast (predict the
        value current at prediction time) in the SAME normalized space as label; its
        abs error is EMA'd into baseline_error_ema so skill_vs_persistence can grade
        the model against "predict-last-value" instead of against zero.

        The default diagnostic is post-fit error, retained for existing callers.
        Delayed forecasters use record_error=False, then score their frozen issued
        prediction separately with record_prediction_error()."""
        # Admission and candidate calculation precede EVERY state mutation,
        # including coefficient changes and feature-geometry resets.
        features = self._vector(features, "features")
        label = self._vector(label, "label", targets=True)
        if baseline_pred is not None:
            baseline_pred = self._vector(baseline_pred, "baseline", targets=True)
        new_lr = self._control(self.lr if lr is None else lr, "lr")
        new_l2 = self._control(self.l2 if l2 is None else l2, "l2")
        new_ema = self._control(self.ema if ema is None else ema, "ema", maximum=1)
        self._check_weights()
        reset = self.W is not None and self.W.shape[1] != features.size
        weights = (np.zeros((self.n_targets, features.size))
                   if self.W is None or reset else self.W)
        with np.errstate(over="ignore", invalid="ignore"):
            err_vec = label - weights @ features
            candidate = weights + new_lr * (np.outer(err_vec, features) - new_l2 * weights)
            if not np.isfinite(err_vec).all() or not np.isfinite(candidate).all():
                raise ValueError("learning update overflow; model unchanged")
            score = None
            if record_error:
                prediction = candidate @ features
                if self.error_mode == "half_norm":
                    prediction = self._project(prediction)
                score = self._score_state(prediction, label, baseline_pred, new_ema, reset=reset)
        if reset:
            self._reset_weights()
        if self.W is None:
            self.W = candidate
        else:
            self.W[:] = candidate       # preserve existing aliases on valid updates
        self.lr, self.l2, self.ema = new_lr, new_l2, new_ema
        self.n_updates += 1
        if score is not None:
            if self.error_mode == "half_norm":
                self._last_pred = prediction
            self._commit_score(score)

    def record_prediction_error(self, prediction, label, *, baseline_pred=None) -> None:
        """Score supplied predictions without fitting or re-reading current weights."""
        prediction = self._vector(prediction, "prediction", targets=True)
        label = self._vector(label, "label", targets=True)
        if baseline_pred is not None:
            baseline_pred = self._vector(baseline_pred, "baseline", targets=True)
        ema = self._control(self.ema, "ema", maximum=1)
        with np.errstate(over="ignore", invalid="ignore"):
            score = self._score_state(prediction, label, baseline_pred, ema)
        self._commit_score(score)

    def _score_state(self, prediction, label, baseline_pred, ema, *, reset=False):
        """Prepare diagnostics without modifying the learner, including on overflow."""
        previous = None if reset else self.error_ema
        per_target = None if reset else self.per_target_err
        baseline = None if reset else self.baseline_error_ema
        if self.error_mode == "half_norm":
            e = 0.5 * float(np.linalg.norm(label - prediction))
        else:
            ae = np.abs(label - prediction)
            per_target = ae if per_target is None else ema * ae + (1.0 - ema) * per_target
            e = float(np.mean(ae))
        error = e if previous is None else ema * e + (1.0 - ema) * previous
        if baseline_pred is not None:
            be = float(np.mean(np.abs(label - baseline_pred)))
            baseline = be if baseline is None else ema * be + (1.0 - ema) * baseline
        if (not math.isfinite(e) or not math.isfinite(error)
                or (per_target is not None and not np.isfinite(per_target).all())
                or (baseline is not None and not math.isfinite(baseline))):
            raise ValueError("scoring overflow or invalid diagnostic state; model unchanged")
        return e, error, per_target, baseline

    def _commit_score(self, score):
        e, self.error_ema, self.per_target_err, self.baseline_error_ema = score
        self._raw_errors.append(e)
        self.n_scored += 1

    @property
    def skill(self) -> float:
        """1 − rolling normalized error, clamped to [0, 1]; 0 until it learns."""
        if self.error_ema is None:
            return 0.0
        return max(0.0, 1.0 - self.error_ema)

    @property
    def skill_vs_persistence(self) -> float:
        """SIGNED foresight over the naive persistence forecast: 1 − error_ema/baseline_error_ema.

        > 0  lower recorded error than persistence; ≈ 0 it merely ties it;
        < 0  it is WORSE than doing nothing. Signed on purpose so a DENIED result shows a
        negative number (the CLAIMS discipline). This measures prospective foresight only
        when the caller scores the frozen issued prediction, not a post-fit diagnostic.
        Returns 0.0 when no baseline has been fed
        OR when persistence is ~perfect (a frozen belief: baseline_error≈0) — the exact case
        the absolute `skill` inflates to ~1.0, here honestly reported as "no foresight to add"."""
        if self.error_ema is None or self.baseline_error_ema is None:
            return 0.0
        if self.baseline_error_ema < 1e-9:
            return 0.0
        return 1.0 - self.error_ema / self.baseline_error_ema

    @property
    def weight_norm(self) -> float:
        """‖W‖ — drives the L2 calibration (regularization ↔ weight magnitude)."""
        return float(np.linalg.norm(self.W)) if self.W is not None else 0.0

    def raw_error_snr(self) -> float | None:
        """SNR proxy of the RAW error series: var(series) / var(first-difference).

        High → slowly-drifting signal (track it: widen bandwidth / larger α).
        Low  → tick-to-tick noise (smooth it: smaller α). Non-circular — it
        reads the un-smoothed series. None until enough samples.
        """
        if len(self._raw_errors) < 16:
            return None
        x = np.asarray(self._raw_errors, dtype=float)
        var_sig = float(np.var(x))
        var_diff = float(np.var(np.diff(x)))
        if var_diff < 1e-12:
            return None
        return var_sig / var_diff

    def _base_snapshot(self) -> dict:
        return {
            "trained": self.W is not None and self.n_updates > 0,
            "n_updates": self.n_updates,
            "n_scored": self.n_scored,
            "error_ema": round_or_none(self.error_ema, 4),
            "skill": round(self.skill, 4),
            "skill_vs_persistence": round(self.skill_vs_persistence, 4),
            "lr": round(self.lr, 5), "l2": round(self.l2, 7), "ema": round(self.ema, 4),
            "weight_norm": round(self.weight_norm, 4),
        }

    def snapshot(self) -> dict:
        return self._base_snapshot()
