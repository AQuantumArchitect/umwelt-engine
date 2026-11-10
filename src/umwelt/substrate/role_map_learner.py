"""Standalone eta-weighted ridge map. No observations or field updates are implicit.

Centered sufficient statistics use O(d**2) memory, independent of stream length.
Ridge penalizes slopes, not the intercept; readout residual is training RMS, not
held-out accuracy or epistemic confidence. There is no quantum-channel claim.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _shape(inputs, target, ridge):
    if (not isinstance(inputs, (list, tuple)) or not inputs
            or any(not isinstance(s, str) or not s.strip() for s in inputs)
            or len(set(inputs)) != len(inputs)):
        raise ValueError("inputs must be distinct nonempty role names")
    if not isinstance(target, str) or not target.strip() or target in inputs:
        raise ValueError("target must be a nonempty role outside inputs")
    ridge = _number(ridge, "ridge")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    return tuple(inputs), target, ridge


def validate_learn(value):
    """Validate a declaration without allocating a learner or observing anything."""
    if not isinstance(value, dict) or set(value) != {"map_id", "family", "inputs", "target", "ridge"}:
        raise ValueError("learn requires map_id, family, inputs, target and ridge")
    if not isinstance(value["map_id"], str) or not value["map_id"].strip():
        raise ValueError("learn map_id must be nonempty")
    if value["family"] != "linear":
        raise ValueError("learn family must be linear")
    inputs, target, ridge = _shape(value["inputs"], value["target"], value["ridge"])
    return value["map_id"], inputs, target, ridge


class LearnedRoleMap:
    def __init__(self, inputs, target, ridge):
        self.inputs, self.target, self.ridge = _shape(inputs, target, ridge)
        d = len(self.inputs)
        self._mean_x = np.zeros(d)
        self._mean_y = 0.0
        self._xx = np.zeros((d, d))
        self._xy = np.zeros(d)
        self._yy = 0.0
        self.n = 0
        self.eta_total = 0.0

    def _features(self, features):
        if not isinstance(features, Mapping) or any(k not in features for k in self.inputs):
            raise ValueError("every declared input must be observed; missing is not zero")
        return np.array([_number(features[k], k) for k in self.inputs])

    def observe(self, features, target_value, eta=1.0):
        x = self._features(features)
        y, eta = _number(target_value, "target_value"), _number(eta, "eta")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0, 1]")
        if eta == 0:
            return  # no evidence weight, no count or learned state
        if self.n == 0:
            self._mean_x, self._mean_y = x, y
            self.n, self.eta_total = 1, eta
            return
        # Compute a complete candidate before committing; refused input cannot
        # poison accumulated evidence. Centering avoids large-offset cancellation.
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                weight = self.eta_total + eta
                dx, dy = x - self._mean_x, y - self._mean_y
                fraction = eta / weight
                cross_weight = eta * (self.eta_total / weight)
                mean_x = self._mean_x + fraction * dx
                mean_y = self._mean_y + fraction * dy
                xx = self._xx + cross_weight * np.outer(dx, dx)
                xy = self._xy + cross_weight * dx * dy
                yy = self._yy + cross_weight * dy * dy
                if not (np.isfinite(xx).all() and np.isfinite(xy).all()
                        and np.isfinite(mean_x).all() and all(map(math.isfinite, (weight, mean_y, yy)))):
                    raise FloatingPointError("nonfinite sufficient statistics")
        except FloatingPointError as exc:
            raise ValueError("observation exceeds finite accumulator range") from exc
        self._mean_x, self._mean_y = mean_x, mean_y
        self._xx, self._xy, self._yy = xx, xy, yy
        self.eta_total, self.n = weight, self.n + 1

    def _fit(self):
        if not self.n:
            raise ValueError("map is DARK: no positive-weight observations")
        try:
            with np.errstate(over="raise", invalid="raise"):
                matrix = self._xx + self.ridge * np.eye(len(self.inputs))
                weights = np.linalg.lstsq(matrix, self._xy, rcond=None)[0]
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("map fit exceeds finite solver range") from exc
        if not np.isfinite(weights).all():
            raise ValueError("map fit is not finite")
        return weights

    def predict(self, features):
        x = self._features(features)
        weights = self._fit()
        result = float(self._mean_y + (x - self._mean_x) @ weights)
        if not math.isfinite(result):
            raise ValueError("prediction exceeds finite range")
        return result  # unbounded linear estimate, not a clipped Bloch belief

    def readout(self):
        if not self.n:
            return {"weights": None, "bias": None, "residual": None, "n": 0, "eta_total": 0.0, "rank": 0}
        w = self._fit()
        cross, quadratic = float(2 * w @ self._xy), float(w @ self._xx @ w)
        squared = self._yy - cross + quadratic
        if not all(map(math.isfinite, (cross, quadratic, squared))):
            raise ValueError("residual exceeds finite range")
        # Permit only floating-point cancellation at zero, not arbitrary negative RMS.
        bound = np.finfo(float).eps * (abs(self._yy) + abs(cross) + abs(quadratic)) * max(1, self.n)
        if squared < -bound:
            raise ValueError("residual lost numerical consistency")
        bias = float(self._mean_y - self._mean_x @ w)
        if not math.isfinite(bias):
            raise ValueError("intercept exceeds finite range")
        return {"weights": dict(zip(self.inputs, map(float, w))), "bias": bias,
                "residual": math.sqrt(max(0.0, squared) / self.eta_total),
                "n": self.n, "eta_total": self.eta_total, "rank": int(np.linalg.matrix_rank(self._xx))}
