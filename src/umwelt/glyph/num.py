"""The number of the language (v0.4 §25-§26): a value plus a provenance.

Every number is **measured** (written straight), **forecast** (trailing ``~``) or **dark**
(``?``). ``Num`` is a ``float`` subclass so range checks and numpy keep working, but it
carries ``prov`` and refuses to compare equal to a number of another provenance: a
``Num(0.8, "forecast")`` is not ``0.8``. A plain ``int``/``float`` is read as measured.

Composition is least-of-inputs: measured < forecast < dark; an operation's result takes
the least certain provenance among its inputs (one ``~`` input => ``~``; one ``?`` => ``?``).
Arithmetic on ``Num`` does this itself; ``math.*``/numpy calls return plain floats, so code
that computes a derived number from ``Num`` inputs must re-wrap with ``Num(v, least(...))``.
"""
from __future__ import annotations

import math

MEASURED, FORECAST, DARK = "measured", "forecast", "dark"
_RANK = {MEASURED: 0, FORECAST: 1, DARK: 2}
_NAMES = tuple(_RANK)


def least(*xs) -> str:
    """Least-of-inputs provenance of numbers/provenance names (plain numbers are measured)."""
    r = 0
    for x in xs:
        p = x if isinstance(x, str) else getattr(x, "prov", MEASURED)
        r = max(r, _RANK[p])
    return _NAMES[r]


class Num(float):
    prov: str

    def __new__(cls, v=0.0, prov: str = MEASURED):
        if prov not in _RANK:
            raise ValueError(f"provenance must be one of {_NAMES}, got {prov!r}")
        self = super().__new__(cls, math.nan if prov == DARK else float(v))
        object.__setattr__(self, "prov", prov)
        return self

    def __getnewargs__(self):
        return (0.0 if self.prov == DARK else float(self), self.prov)

    def __setattr__(self, k, v):
        raise AttributeError("Num is immutable")

    # equality never ignores provenance
    def __eq__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            op = getattr(other, "prov", MEASURED)
            if op != self.prov:
                return False
            return True if op == DARK else float(self) == float(other)
        return NotImplemented

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __hash__(self):
        return hash(float(self)) if self.prov == MEASURED else hash((self.prov, float(self)))

    def __repr__(self):
        return spell(self)

    __str__ = __repr__

    # least-of-inputs arithmetic
    def _wrap(self, other, v):
        if v is NotImplemented:
            return v
        return Num(v, least(self, other))

    def __add__(self, o): return self._wrap(o, float.__add__(self, o))
    def __radd__(self, o): return self._wrap(o, float.__radd__(self, o))
    def __sub__(self, o): return self._wrap(o, float.__sub__(self, o))
    def __rsub__(self, o): return self._wrap(o, float.__rsub__(self, o))
    def __mul__(self, o): return self._wrap(o, float.__mul__(self, o))
    def __rmul__(self, o): return self._wrap(o, float.__rmul__(self, o))
    def __truediv__(self, o): return self._wrap(o, float.__truediv__(self, o))
    def __rtruediv__(self, o): return self._wrap(o, float.__rtruediv__(self, o))
    def __pow__(self, o): return self._wrap(o, float.__pow__(self, o))
    def __neg__(self): return Num(-float(self), self.prov)
    def __abs__(self): return Num(abs(float(self)), self.prov)
    def __mod__(self, o): return self._wrap(o, float.__mod__(self, o))


def as_num(x, default: str = MEASURED) -> Num:
    """``x`` as a Num; a plain number takes ``default`` provenance, a Num keeps its own."""
    if isinstance(x, Num):
        return x
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"not a number: {x!r}")
    return Num(float(x), default)


def spell(x) -> str:
    """Canonical text of a number: ``0.8`` / ``0.8~`` / ``?``."""
    p = getattr(x, "prov", MEASURED)
    if p == DARK:
        return "?"
    return repr(float(x)) + ("~" if p == FORECAST else "")


def say(x) -> str:
    """Prose for a number: bare when measured; 'about N (forecast)' when ``~``; 'unknown' when dark."""
    p = getattr(x, "prov", MEASURED)
    if p == DARK:
        return "unknown (dark)"
    if p == FORECAST:
        return f"about {float(x)!r} (forecast)"
    return repr(float(x))
