"""Receipts — what a sentence's quantity was earned from (grammar §16).

A ``Receipt`` is a list of ``Measurement``s ``(register, direction, eta, order_index)`` plus,
optionally, ordered two-question outcome probabilities for the QQ test (``Outcome``).

Direction is a pole glyph of the register (its declared frame axis), an axis ``(x, y, z)``
in the same coordinates as the scope's frames, or one of ``"x" "y" "z"``. Two measurements
are *non-commuting* when their axes are neither parallel nor antiparallel: ☀ and 🌙 of one
register are the SAME axis, so measuring both is not a second basis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

from dataclasses import replace

from umwelt.glyph.num import DARK, FORECAST, MEASURED, Num
from umwelt.glyph.parse import (Address, Bloch, Bra, Dark, Evidence, Ket, Population,
                                canonical, parse)
from umwelt.glyph.scope import GlyphError, Scope, Unbound

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_PARALLEL_TOL = 1e-9


class ReceiptError(GlyphError):
    """A malformed receipt."""


@dataclass(frozen=True)
class Measurement:
    register: str
    direction: Union[str, tuple]
    eta: float
    order_index: int = 0
    joint: bool = False          # a joint measurement of ``registers`` (§16, bet S5)
    registers: tuple = ()        # the registers a joint measurement names (default: (register,))

    def names(self) -> tuple:
        return tuple(self.registers) or (self.register,)

    def __post_init__(self):
        object.__setattr__(self, "registers", tuple(self.registers))
        if isinstance(self.direction, list):
            object.__setattr__(self, "direction", tuple(self.direction))
        if not (isinstance(self.eta, (int, float)) and 0.0 <= self.eta <= 1.0):
            raise ReceiptError(f"measurement eta={self.eta!r} is outside [0,1]")


@dataclass(frozen=True)
class Outcome:
    """P(answers a, b) when question ``first`` ("A" or "B") was asked first."""
    first: str
    a: bool
    b: bool
    p: float

    def __post_init__(self):
        if self.first not in ("A", "B"):
            raise ReceiptError(f"Outcome.first must be 'A' or 'B', got {self.first!r}")
        if not 0.0 <= self.p <= 1.0:
            raise ReceiptError(f"Outcome.p={self.p!r} is outside [0,1]")


@dataclass(frozen=True)
class Receipt:
    measurements: tuple = ()
    outcomes: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "measurements", tuple(self.measurements))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))


def _unit(v) -> tuple:
    try:
        x, y, z = (float(c) for c in v)
    except (TypeError, ValueError):
        raise ReceiptError(f"direction {v!r} is not a glyph, 'x'/'y'/'z' or a 3-vector") from None
    n = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(n) or n == 0.0:
        raise ReceiptError(f"direction {v!r} is not a finite non-zero vector")
    return (x / n, y / n, z / n)


def axis_of(m: Measurement, scope: Scope) -> tuple:
    """The unit axis a measurement was taken along, read against ``scope``."""
    d = m.direction
    if isinstance(d, tuple):
        return _unit(d)
    if d in _AXES:
        return _AXES[d]
    name, _pole = scope.resolve(d)
    if name != m.register:
        raise ReceiptError(f"direction glyph {d!r} is a pole of register {name!r}, not of "
                           f"{m.register!r}")
    return scope.registers[m.register].frame


def _parallel(u, v) -> bool:
    c = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return math.sqrt(sum(t * t for t in c)) <= _PARALLEL_TOL


def has_two_bases(receipt: Receipt, register: str, scope: Scope) -> bool:
    """True iff the receipt measured ``register`` (eta > 0) along two non-commuting axes."""
    if receipt is None:
        return False
    axes = []
    for m in receipt.measurements:
        if m.register == register and m.eta > 0.0 and not m.joint:
            axes.append(axis_of(m, scope))
    return any(not _parallel(u, v) for i, u in enumerate(axes) for v in axes[i + 1:])


def has_joint(receipt: Receipt, registers) -> bool:
    """True iff the receipt holds a joint measurement (eta > 0) naming every register."""
    if receipt is None:
        return False
    want = set(registers)
    return any(m.joint and m.eta > 0.0 and want <= set(m.names()) for m in receipt.measurements)


def qq_check(receipt: Receipt, *, tol: float = 1e-6):
    """The QQ equality as a referee test (§16, bet S4), for two binary questions A and B:

        p(Ay,Bn) + p(An,By)  =  p(By,An) + p(Bn,Ay)

    (answer pairs in the order asked: left side asked A first, right side asked B first).
    Returns ``Ket(('✅',))`` if it holds within ``tol``, ``Ket(('❌',))`` if not, and
    ``Dark('✅')`` (``✅?``) when the receipt does not hold both orders' discordant cells.
    Only two binary questions are covered."""
    cells: dict = {}
    for o in receipt.outcomes:
        key = (o.first, o.a, o.b)
        if key in cells:
            raise ReceiptError(f"outcome {key!r} appears twice in the receipt")
        cells[key] = o.p
    need = [("A", True, False), ("A", False, True), ("B", True, False), ("B", False, True)]
    if any(k not in cells for k in need):
        return Dark("✅")
    lhs = cells[("A", True, False)] + cells[("A", False, True)]
    rhs = cells[("B", True, False)] + cells[("B", False, True)]
    return Ket(("✅",)) if abs(lhs - rhs) <= tol else Ket(("❌",))


# ── grading a forecast (v0.4 §26) ────────────────────────────────────────────────────────

def _map_numbers(ast, f):
    """Rebuild ``ast`` with every numeric slot passed through ``f`` (in reading order)."""
    if isinstance(ast, Population):
        return replace(ast, p=f(ast.p))
    if isinstance(ast, Evidence):
        return replace(ast, eta=f(ast.eta))
    if isinstance(ast, Bloch):
        r, th = f(ast.r), f(ast.theta)
        phi = f(ast.phi_num)
        return Bloch(r, th, None if phi.prov == DARK else phi, ast.pole0, ast.pole1)
    if isinstance(ast, Bra) and ast.answer is not None:
        return replace(ast, answer=_map_numbers(ast.answer, f))
    if isinstance(ast, Address) and ast.leaf is not None:
        return replace(ast, leaf=_map_numbers(ast.leaf, f))
    return ast


def _numbers(ast) -> list:
    out: list = []
    _map_numbers(ast, lambda n: (out.append(n), n)[1])
    return out


@dataclass(frozen=True)
class Grade:
    """A forecast graded against a measurement: the 🧪 sentence plus a tape record."""
    forecast: str            # canonical text of the forecast sentence
    measured: str            # canonical text of the measured sentence
    tol: float
    max_abs_err: float | None
    answer: Union[Ket, Dark]   # ✅ | ❌ | ✅? (the measurement was dark: ungraded)

    @property
    def verdict(self) -> str | None:
        return None if isinstance(self.answer, Dark) else self.answer.glyphs[0]

    @property
    def sentence(self) -> str:
        return f"🧪 ⟨{self.measured}|{self.forecast}⟩ → {canonical(self.answer)}"

    def to_record(self) -> dict:
        return {"kind": "grade", "sentence": self.sentence, "forecast": self.forecast,
                "measured": self.measured, "tol": self.tol, "max_abs_err": self.max_abs_err,
                "verdict": self.verdict}


def grade(forecast_sentence, measured_sentence, *, tol: float = 1e-6, tape: list | None = None
          ) -> Grade:
    """Grade a forecast sentence (``[0.8~]☀``) against the measured sentence of the same
    shape (``[0.83]☀``): ``🧪 ⟨[0.83]☀|[0.8~]☀⟩ → |✅⟩`` if every numeric slot is within
    ``tol``, ``|❌⟩`` if not, ``✅?`` if a measured slot is dark (nothing to grade against).
    Sentences are text or parsed ASTs. Refuses a forecast with no ``~`` slot (nothing to
    grade), a measurement that is itself a forecast, and sentences of different shape.
    ``tape`` (a list) gets the record appended. Compares slots linearly (no φ wraparound)."""
    if not (isinstance(tol, (int, float)) and math.isfinite(tol) and tol >= 0):
        raise ReceiptError(f"tol must be a finite number >= 0, got {tol!r}")
    f = parse(forecast_sentence) if isinstance(forecast_sentence, str) else forecast_sentence
    m = parse(measured_sentence) if isinstance(measured_sentence, str) else measured_sentence
    zero = lambda n: Num(0.0, MEASURED)          # shape only: the glyph structure
    if canonical(_map_numbers(f, zero)) != canonical(_map_numbers(m, zero)):
        raise ReceiptError(f"cannot grade {canonical(f)!r} against {canonical(m)!r}: the "
                           f"sentences are not the same statement")
    fn, mn = _numbers(f), _numbers(m)
    if not any(n.prov == FORECAST for n in fn):
        raise ReceiptError(f"{canonical(f)!r} holds no forecast ('~') number: nothing to grade")
    if any(n.prov == FORECAST for n in mn):
        raise ReceiptError(f"{canonical(m)!r} is itself a forecast: a forecast is graded only "
                           f"by a measurement (a straight number, earned by a receipt)")
    pairs = [(a, b) for a, b in zip(fn, mn) if a.prov != DARK]
    if any(b.prov == DARK for _, b in pairs):
        ans, err = Dark("✅"), None
    else:
        err = max((abs(float(a) - float(b)) for a, b in pairs), default=0.0)
        ans = Ket(("✅",)) if err <= tol else Ket(("❌",))
    g = Grade(canonical(f), canonical(m), float(tol), err, ans)
    if tape is not None:
        tape.append(g.to_record())
    return g
