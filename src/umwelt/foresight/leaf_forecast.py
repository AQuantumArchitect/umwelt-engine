"""
Generic leaf forecasting — every leaf learns to anticipate itself.

The DRY generalization of a per-signal forward model: ONE capability that every
(node, role) leaf inherits. Each leaf runs a tiny online forward model that
predicts its own future value one (learned, per-leaf) horizon ahead, from
time-as-phase features + a persistence anchor + optional cross-leaf context
(regions predicting each other), and reports a `skill`. Comprehension becomes a
skill per leaf — visible, universal.

Reuses the ONE online regressor (learning.regressor.OnlineRegressor). Continuous
leaves predict a value; binary leaves (on/off) predict P(on). Pre-trainable from
logged history so a leaf boots already skilled where a pattern exists;
constant/patternless leaves simply show ~0 skill (self-pruning).

Periodic-driver features (deterministic cycles a leaf should anticipate against)
are supplied through the register_leaf_feature() registry — empty by default. A
domain registers its own drivers; the origin deployment registered a periodic
astronomical builder, left behind in its example. The universal wall-clock cyclic
features stay built in.
"""
from __future__ import annotations
from typing import Callable
from umwelt._util import clamp01

import bisect
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from umwelt._util import round_or_none
from umwelt.learning.regressor import OnlineRegressor


# Registry of periodic-driver feature builders: name -> (datetime -> fixed-width feature list).
# Empty by default (domain-free). A domain registers the deterministic cycles its leaves should
# anticipate against; every builder must return a stable-width list so the feature geometry is
# fixed across a forecaster's lifetime. Register at import, before any forecaster runs.
_LEAF_FEATURE_BUILDERS: "dict[str, Callable[[datetime], list[float]]]" = {}


def register_leaf_feature(name: str, fn: "Callable[[datetime], list[float]]") -> None:
    """Register a periodic-driver feature builder (datetime -> fixed-width feature list)."""
    _LEAF_FEATURE_BUILDERS[name] = fn


def clear_leaf_features() -> None:
    """Drop all registered driver-feature builders (test/reset hook)."""
    _LEAF_FEATURE_BUILDERS.clear()


def _driver_features(dt: datetime) -> list[float]:
    """Concatenate every registered periodic-driver feature builder's output for `dt`."""
    feats: list[float] = []
    for fn in _LEAF_FEATURE_BUILDERS.values():
        feats.extend(fn(dt))
    return feats


def time_features(dt: datetime) -> list[float]:
    """Cyclic clock features a periodic driver can't carry: hour-of-day + day-of-week.
    Routines are clock+calendar driven (a morning peak, weekday != weekend) — the one
    encoding pure driver-phase features miss."""
    h = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    return [math.cos(2 * math.pi * h / 24.0), math.sin(2 * math.pi * h / 24.0),
            math.cos(2 * math.pi * dow / 7.0), math.sin(2 * math.pi * dow / 7.0)]


@dataclass(frozen=True)
class _Pending:
    due: datetime
    expires: datetime
    feature: np.ndarray
    anchor: float
    prediction: float | None


class LeafForecaster:
    """One leaf's intrinsic forward model: features(t) → its own value at t+H.

    feature = [ *driver_features(t+H), hour/dow-cyclic(t+H)(4),
                anchor=norm(current), <context...>, bias ]
    """

    def __init__(
        self,
        leaf_id: str,
        sensor_id: str,
        node: str,
        role: str,
        binary: bool = False,
        center: float = 0.0,
        scale: float = 1.0,
        horizon_minutes: float = 60.0,
        lr: float = 0.02,
        l2: float = 0.005,
        n_context: int = 0,
        label_lateness_minutes: float = 0.0,
    ):
        self.leaf_id = leaf_id
        self.sensor_id = sensor_id
        self.node, self.role = node, role
        self.binary = binary
        self.center, self.scale = float(center), float(scale or 1.0)
        horizon_minutes = float(horizon_minutes)
        if not math.isfinite(horizon_minutes) or horizon_minutes <= 0:
            raise ValueError("forecast horizon must be finite and positive")
        self.horizon = timedelta(minutes=float(horizon_minutes))
        self.label_lateness_minutes = float(label_lateness_minutes)
        self._label_tolerance()  # validate before accepting observations
        self.n_context = int(n_context)
        self.fc = OnlineRegressor(1, [leaf_id], lr=lr, l2=l2, error_mode="mean_abs")
        self.buffer: deque[_Pending] = deque()
        self.censored_labels = 0
        self.prediction: float | None = None
        self.prediction_for: datetime | None = None
        self.issued_at: datetime | None = None   # the `now` this prediction was made at
        self.label_deadline: datetime | None = None
        self.last_value: float | None = None
        self._last_step_at: datetime | None = None
        self.pretrained_pairs = 0

    def start_sequence(self) -> None:
        """Begin an independent timeline, retaining weights and recorded scores.

        Unresolved labels from the old timeline are censored, never matched to a
        different trial just because it reuses a timestamp.
        """
        self.censored_labels += len(self.buffer)
        self.buffer.clear()
        self.prediction = self.prediction_for = self.issued_at = None
        self.label_deadline = None
        self.last_value = self._last_step_at = None

    # -- normalization (continuous: physical; binary: 0/1 -> -1/+1) ----------
    def _norm(self, v: float) -> float:
        return (v - self.center) / self.scale

    def _denorm(self, z: float) -> float:
        v = z * self.scale + self.center
        return float(clamp01(v)) if self.binary else float(v)

    def _feature(self, future: datetime, anchor_v: float, context: list[float] | None) -> np.ndarray:
        parts = [*_driver_features(future), *time_features(future),
                 self._norm(anchor_v)]
        if self.n_context:
            ctx = list(context or [])[: self.n_context]
            ctx += [0.0] * (self.n_context - len(ctx))
            parts += ctx
        parts.append(1.0)
        return np.asarray(parts, dtype=float)

    def set_horizon_minutes(self, m: float) -> None:
        if not math.isfinite(float(m)) or m <= 0:
            raise ValueError("forecast horizon must be finite and positive")
        self.horizon = timedelta(minutes=float(max(1.0, m)))

    def _label_tolerance(self) -> timedelta:
        minutes = self.label_lateness_minutes
        if not math.isfinite(minutes) or minutes < 0:
            raise ValueError("label lateness must be finite and nonnegative")
        return timedelta(minutes=minutes)

    @property
    def horizon_min(self) -> float:
        return self.horizon.total_seconds() / 60.0

    @property
    def issued_horizon_min(self) -> float | None:
        if self.prediction_for is None or self.issued_at is None:
            return None
        return (self.prediction_for - self.issued_at).total_seconds() / 60.0

    @property
    def issued_lateness_min(self) -> float | None:
        if self.label_deadline is None or self.prediction_for is None:
            return None
        return (self.label_deadline - self.prediction_for).total_seconds() / 60.0

    # -- offline pre-training from history -----------------------------------
    def pre_train(self, history: list[tuple[datetime, float]], tol_minutes: float = 30.0,
                  epochs: int = 2, context_at=None) -> int:
        """history: ordered [(datetime, value)]. Builds (feature(t)->value(t+H))
        pairs by matching each reading to the one nearest t+horizon (within tol).
        This is approximate offline fitting, never prospective forecast credit.
        context_at(dt) -> list[float] supplies historical cross-leaf context."""
        pts = sorted([(dt, float(v)) for dt, v in history if v is not None], key=lambda p: p[0])
        if len(pts) < 20:
            return 0
        dts = [p[0] for p in pts]
        tol = timedelta(minutes=tol_minutes)
        pairs = []
        for dt, v in pts:
            tgt = dt + self.horizon
            k = bisect.bisect_left(dts, tgt)
            best = None
            for cand in (k - 1, k):
                if 0 <= cand < len(pts) and abs(pts[cand][0] - tgt) <= tol:
                    if best is None or abs(pts[cand][0] - tgt) < abs(pts[best][0] - tgt):
                        best = cand
            if best is not None:
                ctx = context_at(dt) if (self.n_context and context_at) else None
                # v is the value current at prediction time → its norm is the persistence baseline.
                pairs.append((self._feature(pts[best][0], v, ctx), pts[best][1], self._norm(v)))
        for _ in range(max(1, epochs)):
            for f, label, _anchor_norm in pairs:
                # Warm-start is fitting, not an issued prospective forecast.
                self.fc.update(f, np.array([self._norm(label)]), record_error=False)
        self.pretrained_pairs = len(pairs)
        return len(pairs)

    # -- live online step (delayed-label) ------------------------------------
    def step(self, now: datetime, current_value: float | None, context: list[float] | None = None,
             train: bool = True) -> None:
        tolerance = self._label_tolerance()
        if self._last_step_at is not None and now < self._last_step_at:
            raise ValueError("forecast clock moved backward; start_sequence for an independent timeline")
        if current_value is not None and not math.isfinite(float(current_value)):
            raise ValueError("forecast observation must be finite or missing")
        self._last_step_at = now
        if current_value is not None:
            self.last_value = float(current_value)
        anchor = self.last_value if self.last_value is not None else self.center
        feat = self._feature(now + self.horizon, anchor, context)
        p = self.fc.predict(feat)
        if p is not None:
            self.prediction = self._denorm(float(p[0]))
            self.prediction_for = now + self.horizon
            self.issued_at = now
            self.label_deadline = self.prediction_for + tolerance
        else:
            self.prediction = self.prediction_for = self.issued_at = None
            self.label_deadline = None
        # train=False → predict-only (frozen weights): held-out evaluation reads self.prediction
        # and scores it externally, so a generalization split never trains on the test schedule.
        if train:
            if current_value is not None:
                due = now + self.horizon
                pending = _Pending(due, due + tolerance, feat, self._norm(anchor),
                                   None if self.prediction is None else self._norm(self.prediction))
                self.buffer.append(pending)
                # A revised horizon can move a newer due time ahead of an older
                # one. Keep the maturity queue ordered without rescanning it on
                # every ordinary tick.
                if len(self.buffer) > 1 and self.buffer[-2].due > due:
                    self.buffer = deque(sorted(self.buffer, key=lambda item: item.due))
            waiting = deque()
            while self.buffer and self.buffer[0].due <= now:
                past = self.buffer.popleft()
                if now > past.expires or past.feature.shape != feat.shape:
                    self.censored_labels += 1
                elif current_value is None:
                    waiting.append(past)  # a missing label is never filled with last_value
                else:
                    y = np.array([self._norm(current_value)])
                    self.fc.update(past.feature, y, record_error=False)
                    if past.prediction is not None:
                        self.fc.record_prediction_error(np.array([past.prediction]), y,
                            baseline_pred=np.array([past.anchor]))
            self.buffer.extendleft(reversed(waiting))

    def snapshot(self) -> dict:
        return {
            "leaf": self.leaf_id, "node": self.node, "role": self.role,
            "binary": self.binary, "skill": round(self.fc.skill, 4),
            "skill_vs_persistence": round(self.fc.skill_vs_persistence, 4),
            "n_updates": self.fc.n_updates, "pretrained_pairs": self.pretrained_pairs,
            "scoring_contract": "issued-leaf-forecast/1",
            "n_scored": self.fc.n_scored, "censored_labels": self.censored_labels,
            "pending_labels": len(self.buffer),
            "label_lateness_minutes": self.label_lateness_minutes,
            "issued_horizon_min": self.issued_horizon_min,
            "issued_lateness_min": self.issued_lateness_min,
            "label_deadline": self.label_deadline.isoformat() if self.label_deadline else None,
            "horizon_min": round(self.horizon_min),
            "prediction": round_or_none(self.prediction, 3),
            "prediction_for": self.prediction_for.isoformat() if self.prediction_for else None,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "last_value": round_or_none(self.last_value, 3),
        }
