"""Versioned continuation of the learned forecast organ, never its rounded display.

This block lives inside the existing trusted engine snapshot. It does not make
pickle safe for untrusted input or establish continuation of the other organs.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
import math

import numpy as np

from .forecast_surface import ForecastSurface
from .leaf_forecast import _Pending, _LEAF_FEATURE_BUILDERS


FORMAT = "forecast-continuation/1"
SURFACE = "leaves horizons surprise_decay interaction_order _monomials n_context _purity _now _last_z _trace".split()
LEAF = "leaf_id sensor_id node role binary center scale horizon label_lateness_minutes n_context censored_labels prediction prediction_for issued_at label_deadline last_value _last_step_at pretrained_pairs".split()
REGRESSOR = "n_targets target_ids W lr l2 ema project_ball error_mode error_ema baseline_error_ema per_target_err n_updates n_scored _last_pred _raw_errors".split()


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("forecast continuation has missing or unknown fields")


def _pick(obj, keys, excluded=()):
    _keys(vars(obj), [*keys, *excluded])
    return {key: getattr(obj, key) for key in keys}


def _features():
    # Identity/order guard, not a proof that arbitrary driver code is unchanged.
    return tuple((name, getattr(fn, "__module__", type(fn).__module__),
                  getattr(fn, "__qualname__", type(fn).__qualname__))
                 for name, fn in _LEAF_FEATURE_BUILDERS.items())


def snapshot(surface):
    if surface is None:
        return None
    if type(surface) is not ForecastSurface:
        # A dissolved rollout has no learned bank and retains no field reference
        # in this block. Do not pretend another surface uses this state contract.
        return None
    state = {"format": FORMAT, "features": _features(),
             "surface": _pick(surface, SURFACE, ("bank",)), "bank": {}}
    for key, leaf in surface.bank.items():
        state["bank"][key] = {
            "leaf": _pick(leaf, LEAF, ("fc", "buffer")),
            "regressor": _pick(leaf.fc, REGRESSOR),
            "pending": [asdict(item) for item in leaf.buffer],
        }
    # Capture independent arrays/deques and reject corrupt state before writing.
    return prepare(surface, state)


def _number(value, *, nullable=False, minimum=None, maximum=None):
    if value is None and nullable:
        return
    if (isinstance(value, bool) or np.iscomplexobj(value) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError("forecast continuation contains an invalid number")


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("forecast continuation contains an invalid count")


def _array(value, shape):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype.kind not in "fi" or not np.isfinite(value).all()):
        raise ValueError("forecast continuation has invalid array geometry or values")


def _time(value, *, nullable=True):
    if not ((nullable and value is None) or isinstance(value, datetime)):
        raise ValueError("forecast continuation has an invalid time")


def _ordered(a, b, *, strict=False):
    try:
        valid = a < b if strict else a <= b
    except TypeError as exc:
        raise ValueError("forecast continuation has incompatible clocks") from exc
    if not valid:
        raise ValueError("forecast continuation times are inconsistent")


def prepare(surface, state):
    """Validate all forecast state before the engine changes anything; return a copy."""
    if state is None:
        if type(surface) is ForecastSurface:
            raise ValueError("learned forecast surface cannot restore a null continuation block")
        return None
    if type(surface) is not ForecastSurface:
        raise ValueError("forecast continuation needs the matching learned surface")
    _keys(state, ("format", "features", "surface", "bank"))
    if state["format"] != FORMAT or state["features"] != _features():
        raise ValueError("forecast continuation format or feature registry mismatch")
    top = state["surface"]
    _keys(top, SURFACE)
    for name in ("leaves", "horizons", "interaction_order", "_monomials", "n_context"):
        if top[name] != getattr(surface, name):
            raise ValueError("forecast continuation surface geometry mismatch")
    _number(top["surprise_decay"], minimum=0, maximum=1)
    _time(top["_now"])
    for name in ("_purity", "_last_z", "_trace"):
        values = top[name]
        if not isinstance(values, dict) or not set(values).issubset(surface.leaves):
            raise ValueError("forecast continuation context identity mismatch")
        for value in values.values():
            _number(value, minimum=0 if name != "_last_z" else None)
    _keys(state["bank"], surface.bank)
    for key, record in state["bank"].items():
        target = surface.bank[key]
        _keys(record, ("leaf", "regressor", "pending"))
        leaf, reg = record["leaf"], record["regressor"]
        _keys(leaf, LEAF)
        _keys(reg, REGRESSOR)
        for name in ("leaf_id", "sensor_id", "node", "role", "binary", "center", "scale", "n_context"):
            if leaf[name] != getattr(target, name):
                raise ValueError("forecast continuation leaf identity or geometry mismatch")
        for name in ("n_targets", "target_ids", "project_ball", "error_mode"):
            if reg[name] != getattr(target.fc, name):
                raise ValueError("forecast continuation regressor identity mismatch")
        if not isinstance(leaf["horizon"], timedelta) or leaf["horizon"] <= timedelta(0):
            raise ValueError("forecast continuation horizon must be positive")
        _number(leaf["label_lateness_minutes"], minimum=0)
        for name in ("prediction", "last_value"):
            _number(leaf[name], nullable=True)
        for name in ("censored_labels", "pretrained_pairs"):
            _count(leaf[name])
        for name in ("issued_at", "prediction_for", "label_deadline", "_last_step_at"):
            _time(leaf[name])
        if leaf["_last_step_at"] is not None:
            if top["_now"] is None:
                raise ValueError("forecast continuation leaf clock lacks a surface clock")
            _ordered(leaf["_last_step_at"], top["_now"])
        times = (leaf["issued_at"], leaf["prediction_for"], leaf["label_deadline"])
        if leaf["prediction"] is None:
            if any(value is not None for value in times):
                raise ValueError("forecast continuation has dates without an issued prediction")
        else:
            if any(value is None for value in times):
                raise ValueError("forecast continuation prediction lacks its frozen dates")
            issued, due, expires = times
            _ordered(issued, due, strict=True)
            _ordered(due, expires)
            if issued != leaf["_last_step_at"]:
                raise ValueError("forecast continuation issued window is inconsistent")
        if reg["W"] is not None:
            # Cold warm-starts have no live timestamp. This date is only a feature
            # width probe, never an invented observation or issued forecast.
            reference = leaf["_last_step_at"] or top["_now"] or datetime(2000, 1, 1)
            width = target._feature(reference, target.center, None).size
            _array(reg["W"], (target.fc.n_targets, width))
        for name in ("lr", "l2", "ema"):
            _number(reg[name], minimum=0, maximum=1 if name == "ema" else None)
        for name in ("error_ema", "baseline_error_ema"):
            _number(reg[name], nullable=True, minimum=0)
        for name in ("per_target_err", "_last_pred"):
            if reg[name] is not None:
                _array(reg[name], (target.fc.n_targets,))
        for name in ("n_updates", "n_scored"):
            _count(reg[name])
        errors = reg["_raw_errors"]
        if not isinstance(errors, deque) or errors.maxlen != target.fc._raw_errors.maxlen:
            raise ValueError("forecast continuation score window mismatch")
        for value in errors:
            _number(value, minimum=0)
        if not isinstance(record["pending"], list):
            raise ValueError("forecast continuation pending labels must be a list")
        previous = None
        for pending in record["pending"]:
            _keys(pending, ("due", "expires", "feature", "anchor", "prediction"))
            _time(pending["due"], nullable=False)
            _time(pending["expires"], nullable=False)
            _ordered(pending["due"], pending["expires"])
            if leaf["_last_step_at"] is None:
                raise ValueError("forecast continuation pending label lacks an observation clock")
            # Pending labels can be expired after a frozen interval. Compare clocks
            # without requiring that every pending label is still in its window.
            try:
                pending["due"] - leaf["_last_step_at"]
            except TypeError as exc:
                raise ValueError("forecast continuation has incompatible clocks") from exc
            if previous is not None:
                _ordered(previous, pending["due"])
            previous = pending["due"]
            vector = pending["feature"]
            if not isinstance(vector, np.ndarray) or vector.ndim != 1 or not vector.size:
                raise ValueError("forecast continuation pending feature is not a vector")
            _array(vector, vector.shape)  # old geometry stays pending until explicitly censored
            _number(pending["anchor"])
            _number(pending["prediction"], nullable=True)
    return deepcopy(state)


def restore(surface, prepared):
    """Commit a prepared block, retaining references to surface, leaf and regressor."""
    if prepared is None:
        return
    for key, record in prepared["bank"].items():
        leaf = surface.bank[key]
        vars(leaf.fc).update(record["regressor"])
        vars(leaf).update(record["leaf"])
        leaf.buffer = deque(_Pending(**row) for row in record["pending"])
    vars(surface).update(prepared["surface"])
