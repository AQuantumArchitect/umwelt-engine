"""Measurement receipts — what the engine measured, in what order, along which axis.

Every supported, valid `measure_axis` / `measure_pair` call appends a `MeasurementReceipt` to
that cluster's log (read with `receipts_of(cluster)`), so a frame can say not just "z = 0.7"
but "z was earned from these measurements, in this order". Receipts are bookkeeping only:
they never feed back into ρ, the cumulants, or any hash. The log is created lazily on the
first measurement, so an engine that never calls these methods carries nothing new (the
default `UMWELT_BELAVKIN=0` ingest path calls neither).
Successful positive-strength updates are marked `applied` only after they return.
No-op and historical receipts are not applied evidence; a failed update cannot
establish freshness. This journal does not make state updates transactionally durable.

`to_glyph_receipt` adapts a list of receipts to the dict shape of the glyph module's
`umwelt.glyph.receipt.Receipt` (the-broadcast docs/GLYPH-LANGUAGE.md §16) WITHOUT importing
it — glyph lives on another branch. The shape:

    {"measurements": [ {"register": str, "direction": (x, y, z), "eta": float in [0, 1],
                        "order_index": int, "joint": bool, "registers": (str, ...)} , ... ],
     "outcomes": []}

so `Receipt(measurements=tuple(Measurement(**m) for m in d["measurements"]), outcomes=())`
loads it. Mapping notes (the glyph side has no field for the rest):
  * register     the role name of the measured qubit (the qubit index if the cluster has none).
  * direction    the unit axis σ·n̂ (a 3-tuple, glyph `Measurement.direction`). For a joint
                 measurement it is the FIRST register's axis; the second is receipt.axis2.
  * eta          1 − exp(−strength): 0 for a no-op, → 1 as the measurement becomes projective.
                 (strength is the Belavkin rate s; glyph eta is a [0,1] efficiency.)
  * order_index  position in this cluster's log (monotone per cluster, not per register).
  * joint        True for `measure_pair`; `registers` then names both qubits' roles.
`record`, `confidence`, `t` stay on the engine-side receipt (and in the frame JSON).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

LOG_CAP = 10_000   # oldest receipts drop first; order_index keeps counting


@dataclass(frozen=True)
class MeasurementReceipt:
    register: str
    axis: tuple                      # unit 3-tuple (the first qubit's axis for a joint measurement)
    record: float
    strength: float
    confidence: Optional[float]
    order_index: int
    t: float
    joint: bool = False
    registers: tuple = ()            # both roles for a joint measurement, else ()
    axis2: Optional[tuple] = None    # the second qubit's axis for a joint measurement
    applied: bool = False           # confirmed successful positive-strength call

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axis"] = list(self.axis)
        d["axis2"] = None if self.axis2 is None else list(self.axis2)
        d["registers"] = list(self.registers)
        return d

    @staticmethod
    def from_dict(d: dict) -> "MeasurementReceipt":
        return MeasurementReceipt(
            register=str(d["register"]), axis=tuple(float(c) for c in d["axis"]),
            record=float(d["record"]), strength=float(d["strength"]),
            confidence=None if d.get("confidence") is None else float(d["confidence"]),
            order_index=int(d["order_index"]), t=float(d["t"]), joint=bool(d.get("joint", False)),
            registers=tuple(d.get("registers") or ()),
            axis2=None if d.get("axis2") is None else tuple(float(c) for c in d["axis2"]),
            applied=d.get("applied") is True)


def _role(cluster, idx: int) -> str:
    roles = getattr(cluster, "qubit_roles", None)
    try:
        return str(roles[idx])
    except (TypeError, IndexError):
        return str(idx)


def _append(cluster, **kw) -> MeasurementReceipt:
    log = cluster.__dict__.setdefault("_measurement_receipts", [])
    n = cluster.__dict__.get("_measurement_receipt_count", 0)
    clock = cluster.__dict__.get("receipt_clock") or time.time
    r = MeasurementReceipt(order_index=n, t=float(clock()), **kw)
    log.append(r)
    cluster.__dict__["_measurement_receipt_count"] = n + 1
    if len(log) > LOG_CAP:
        del log[: len(log) - LOG_CAP]
    return r


def record_axis(cluster, idx: int, axis, record: float, strength: float,
                confidence, *, applied=False) -> MeasurementReceipt:
    from umwelt.substrate.belavkin import unit_axis
    return _append(cluster, register=_role(cluster, idx), axis=unit_axis(axis),
                   record=float(record), strength=float(strength),
                   confidence=None if confidence is None else float(confidence), applied=applied)


def record_pair(cluster, i: int, j: int, basis, record: float, strength: float,
                confidence, *, applied=False) -> MeasurementReceipt:
    from umwelt.substrate.belavkin import unit_axis
    ri, rj = _role(cluster, i), _role(cluster, j)
    return _append(cluster, register=ri, axis=unit_axis(basis[0]), axis2=unit_axis(basis[1]),
                   record=float(record), strength=float(strength),
                   confidence=None if confidence is None else float(confidence),
                   joint=True, registers=(ri, rj), applied=applied)


def validate_call(cluster, indices, axes, record, strength, confidence):
    """Reject invalid calls before state or confidence bookkeeping can change."""
    from numbers import Integral
    from umwelt.substrate.belavkin import unit_axis
    size = len(cluster.qubit_roles)
    if len(indices) != len(axes):
        raise ValueError('one measurement axis required per qubit')
    if any(not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < size for i in indices):
        raise ValueError('measurement index outside cluster')
    if len(set(indices)) != len(indices):
        raise ValueError('measurement needs distinct qubits')
    for axis in axes: unit_axis(axis)
    for value in (record, strength, *(() if confidence is None else (confidence,))):
        if not math.isfinite(float(value)):
            raise ValueError('measurement inputs must be finite')


def receipts_of(cluster) -> list:
    """This cluster's receipts, oldest first (empty if it never measured)."""
    return list(cluster.__dict__.get("_measurement_receipts", ()))


def eta_of(strength: float) -> float:
    return 1.0 - math.exp(-max(0.0, float(strength)))


def to_glyph_receipt(receipts) -> dict:
    """The dict shape of glyph `Receipt` (see module docstring). Pure; imports nothing from glyph."""
    ms = []
    for r in receipts:
        ms.append({"register": r.register, "direction": tuple(r.axis), "eta": eta_of(r.strength),
                   "order_index": r.order_index, "joint": bool(r.joint),
                   "registers": tuple(r.registers)})
    return {"measurements": ms, "outcomes": []}
