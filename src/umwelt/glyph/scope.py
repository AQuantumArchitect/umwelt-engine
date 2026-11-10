"""Scope — the declared space a sentence is read against (grammar §1, §2, §4).

A scope has a stable ``id``, registers (qubits) each with two pole glyphs and a frame
(axis), optional child scopes opened by a pole (descent, §4), and questions. Its
``version`` is a hash of its canonical content, so editing a pole changes the version.

``Register.node`` optionally names the ``frame/1`` node id this register renders from
(default: the register's own name). The engine spec's ``AxisFace`` carries pole labels
but nothing a sentence can resolve against, so scopes are declared separately.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple

from umwelt.glyph.normalize import canon, glyphs

Z_AXIS = (0.0, 0.0, 1.0)
# Symbols the grammar owns; none may be a pole glyph.
RESERVED = frozenset({"⟨", "⟩", "›", "ψ", "·", "⌢"} | set("₀₁₂₃₄₅₆₇₈₉"))


class GlyphError(Exception):
    """Base of every error raised by the glyph package."""


class Ambiguous(GlyphError):
    """A glyph names more than one register inside the scope (§1): refused."""


class Unbound(GlyphError):
    """A glyph names no register inside the scope: outside a scope it is undefined."""


class ScopeError(GlyphError):
    """An invalid scope declaration."""


def _unit(v: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in v)
    n = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(n) or n == 0.0:
        raise ScopeError("a register frame must be a finite non-zero vector")
    return (round(x / n, 12) + 0.0, round(y / n, 12) + 0.0, round(z / n, 12) + 0.0)


class Step(NamedTuple):
    """One step of a resolved descent path (§4, §17)."""
    scope_ref: str
    register: str
    pole: int | None      # the pole descended under; None for a register step
    kind: str             # "pole" (☀›: conditioned on an outcome) | "register" (☀🌙›: conditional state)


@dataclass(frozen=True)
class Register:
    pole0: str
    pole1: str
    frame: tuple = Z_AXIS
    children: dict = field(default_factory=dict)      # pole index (0|1) -> Scope
    node: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "frame", _unit(self.frame))
        kids = {}
        for k, v in dict(self.children).items():
            k = int(k)
            if k not in (0, 1):
                raise ScopeError(f"child key must be pole 0 or 1, got {k!r}")
            kids[k] = v
        object.__setattr__(self, "children", kids)

    def glyph(self, pole: int) -> str:
        return self.pole0 if pole == 0 else self.pole1


@dataclass(frozen=True)
class PairRegister:
    """A register of a product scope (§19): a pair ``a⊗b`` from the two parents. Its
    "poles" are the four glyph pairs ``(reg1.glyph(i), reg2.glyph(j))``."""
    first: str
    second: str
    reg1: Register
    reg2: Register

    @property
    def name(self) -> str:
        return f"{self.first}⊗{self.second}"

    @property
    def poles(self) -> dict:
        return {(i, j): (self.reg1.glyph(i), self.reg2.glyph(j)) for i in (0, 1) for j in (0, 1)}


@dataclass(frozen=True)
class Scope:
    id: str
    registers: dict = field(default_factory=dict)     # name -> Register
    questions: tuple = ()
    keep_skin_tones: bool = False
    children: dict = field(default_factory=dict)      # register name -> Scope (☀🌙›, §17)
    epsilon: float | None = None                      # declared resolution ε (§17); None: undeclared
    parents: tuple = ()                               # () or the two parents of a product scope (§19)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ScopeError("scope id must be a non-empty string")
        keep = self.keep_skin_tones
        regs = {}
        for name, r in dict(self.registers).items():
            p0, p1 = (canon(g, keep_skin_tones=keep) for g in (r.pole0, r.pole1))
            for g in (p0, p1):
                if g in RESERVED:
                    raise ScopeError(f"register {name!r}: {g!r} is a grammar symbol, not a pole")
                if len(glyphs(g, keep_skin_tones=keep)) != 1:
                    raise ScopeError(f"register {name!r}: pole {g!r} is not exactly one glyph")
            if p0 == p1:
                raise ScopeError(f"register {name!r}: both poles are {p0!r}")
            regs[str(name)] = Register(p0, p1, r.frame, r.children, r.node)
        object.__setattr__(self, "registers", regs)
        kids = {}
        for name, child in dict(self.children).items():
            if str(name) not in regs:
                raise ScopeError(f"children: {name!r} is not a register of scope {self.id!r}")
            kids[str(name)] = child
        object.__setattr__(self, "children", kids)
        parents = tuple(self.parents)
        if parents:
            if len(parents) != 2 or not all(isinstance(p, Scope) for p in parents):
                raise ScopeError("a product scope has exactly two parent scopes")
            if regs or kids:
                raise ScopeError("a product scope declares no registers or children of its "
                                 "own: its registers are the pairs of its parents (§19)")
            if parents[0].keep_skin_tones != parents[1].keep_skin_tones:
                raise ScopeError("the parents of a product scope must agree on keep_skin_tones")
            object.__setattr__(self, "keep_skin_tones", parents[0].keep_skin_tones)
        object.__setattr__(self, "parents", parents)
        if self.epsilon is not None:
            eps = self.epsilon
            if isinstance(eps, bool) or not isinstance(eps, (int, float)) \
                    or not math.isfinite(eps) or eps < 0:
                raise ScopeError(f"epsilon must be a finite number >= 0 or None, got {eps!r}")
            object.__setattr__(self, "epsilon", float(eps))
        object.__setattr__(self, "questions", tuple(str(q) for q in self.questions))

    # ── (de)serialization and the version stamp ──
    def to_dict(self) -> dict:
        regs = {}
        for name, r in self.registers.items():
            d: dict[str, Any] = {"pole0": r.pole0, "pole1": r.pole1, "frame": list(r.frame)}
            if r.node is not None:
                d["node"] = r.node
            if r.children:
                d["children"] = {str(k): r.children[k].to_dict() for k in sorted(r.children)}
            regs[name] = d
        out: dict[str, Any] = {"id": self.id, "registers": regs, "questions": list(self.questions)}
        if self.keep_skin_tones:
            out["keep_skin_tones"] = True
        if self.parents:
            out["parents"] = [p.to_dict() for p in self.parents]
        if self.epsilon is not None:
            out["epsilon"] = self.epsilon
        if self.children:
            out["children"] = {n: self.children[n].to_dict() for n in sorted(self.children)}
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Scope":
        regs = {}
        for name, r in d.get("registers", {}).items():
            kids = {int(k): cls.from_dict(v) for k, v in r.get("children", {}).items()}
            regs[name] = Register(r["pole0"], r["pole1"], tuple(r.get("frame", Z_AXIS)),
                                  kids, r.get("node"))
        kids = {n: cls.from_dict(v) for n, v in d.get("children", {}).items()}
        parents = tuple(cls.from_dict(p) for p in d.get("parents", ()))
        return cls(d["id"], regs, tuple(d.get("questions", ())),
                   bool(d.get("keep_skin_tones", False)), kids, d.get("epsilon"), parents)

    # ── the product scope (§19) ──
    @classmethod
    def product(cls, s1: "Scope", s2: "Scope", *, epsilon: float | None = None) -> "Scope":
        """``s1⊗s2``: the only scope in which a ket may take glyphs from both parents."""
        return cls(f"{s1.id}⊗{s2.id}", {}, (), s1.keep_skin_tones, {}, epsilon, (s1, s2))

    @property
    def is_product(self) -> bool:
        return bool(self.parents)

    @property
    def pair_registers(self) -> dict:
        """name ``a⊗b`` -> PairRegister, for a product scope (empty otherwise)."""
        if not self.parents:
            return {}
        s1, s2 = self.parents
        return {f"{a}⊗{b}": PairRegister(a, b, r1, r2)
                for a, r1 in s1.registers.items() for b, r2 in s2.registers.items()}

    def resolve_parent(self, glyph: str, parent: int | None = None):
        """In a product scope: glyph -> (parent index 1|2, register, pole). A subscript
        names the parent; without one the glyph must belong to exactly one parent."""
        hits = []
        for k, ps in enumerate(self.parents, start=1):
            if parent in (None, k):
                try:
                    hits.append((k,) + ps.resolve(glyph))
                except Unbound:
                    pass
        if not hits:
            raise Unbound(f"glyph {glyph!r} names no register in either parent of product "
                          f"scope {self.ref}")
        if len(hits) > 1:
            raise Ambiguous(f"glyph {glyph!r} belongs to both parents of product scope "
                            f"{self.ref}; subscript it ({glyph}₁ or {glyph}₂)")
        return hits[0]

    def content_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def version(self) -> str:
        return self.content_hash()[:12]

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    # ── resolution ──
    def _find(self, glyph: str) -> list[tuple[str, int]]:
        g = canon(glyph, keep_skin_tones=self.keep_skin_tones)
        hits = []
        for name, r in self.registers.items():
            if r.pole0 == g:
                hits.append((name, 0))
            if r.pole1 == g:
                hits.append((name, 1))
        return hits

    def resolve(self, glyph: str) -> tuple[str, int]:
        """glyph -> (register name, pole index). Refuses ambiguity; never guesses."""
        hits = self._find(glyph)
        if not hits:
            hint = (" (a product scope's registers are pairs; write a ket with parent "
                    "subscripts, §19)" if self.parents else "")
            raise Unbound(f"glyph {glyph!r} names no register in scope {self.ref}{hint}")
        if len(hits) > 1:
            names = ", ".join(f"{n}[pole {p}]" for n, p in hits)
            raise Ambiguous(f"glyph {glyph!r} names more than one register in scope "
                            f"{self.ref}: {names}. Sharing a glyph is not a mapping.")
        return hits[0]

    def resolve_path(self, path: Iterable[str], *, descend_last: bool = True):
        """Walk ``path`` step by step, descending as it goes (§4, §17).

        An element is either one pole glyph (descent under a POLE: ``Register.children``)
        or two glyphs, the two poles of one register (descent under the REGISTER, a
        conditional state: ``Scope.children``). Returns ``(steps, final_scope)``: steps is
        a list of ``Step(scope_ref, register, pole, kind)`` (pole is None for a register
        step), final_scope the scope reached. With ``descend_last=False`` the last element
        is only resolved (a bare address ends at a pole, not inside it)."""
        path = list(path)
        scope = self
        steps = []
        for i, g in enumerate(path):
            gs = glyphs(g, keep_skin_tones=scope.keep_skin_tones)
            if len(gs) == 2:
                (n0, p0), (n1, p1) = scope.resolve(gs[0]), scope.resolve(gs[1])
                if n0 != n1 or p0 == p1:
                    raise Unbound(f"{g!r} is not the two poles of one register of scope "
                                  f"{scope.ref}; a register step names both its poles")
                steps.append(Step(scope.ref, n0, None, "register"))
                if i == len(path) - 1 and not descend_last:
                    break
                child = scope.children.get(n0)
                if child is None:
                    raise Unbound(f"register {n0!r} of scope {scope.ref} opens no conditional "
                                  f"child scope (register descent, §17)")
                scope = child
                continue
            if len(gs) != 1:
                raise Unbound(f"path element {g!r} is neither one pole glyph nor a register's "
                              f"two poles")
            name, pole = scope.resolve(g)
            steps.append(Step(scope.ref, name, pole, "pole"))
            if i == len(path) - 1 and not descend_last:
                break
            child = scope.registers[name].children.get(pole)
            if child is None:
                raise Unbound(f"pole {g!r} (register {name!r}, pole {pole}) of scope "
                              f"{scope.ref} opens no child scope")
            scope = child
        return steps, scope
