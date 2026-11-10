"""Binding — resolve every glyph of a parsed sentence against one ``scope@version``.

Rules enforced here (grammar §3):

* ``[p]`` needs p in [0,1]; ``!η`` needs η in (0,1]; Bloch r in [0,1], θ in [0,π],
  φ in [0,2π).
* **Populations are symmetric** (§3b): ``[p]☀ ≡ [1−p]🌙``. Binding canonicalizes every
  population to pole 0 (the flipped value is rounded to 12 places so 1−0.8 is 0.2).
* **Evidence is NOT symmetrized** (§3c): "This is **not** `🌙!0.2`. Observation has a
  direction and a strength; a population does not." ``☀!0.8`` binds to exactly one fact —
  pole of ☀, η 0.8 — and no fact about the other pole is created.
* **Dark is not a half** (§3e): ``Dark`` binds to ``BDark``, never to a population.
  Dark is a property of the register, so ``☀?`` and ``🌙?`` bind to the same fact.
* **Every number has a provenance** (v0.4 §25): straight = measured, ``~`` = forecast,
  ``?`` = dark. A straight number is *earned*: it binds only with a ``receipt`` holding a
  measurement (η > 0) of its register, or with ``engine_origin=True`` (the adapter's flag:
  the numbers were read off a state built from receipts). A ``~`` number needs neither.
  A dark population / r / θ binds to ``BDark`` (dark is not a half); a dark η is refused.
  Provenance rides every bound number and survives the pole flips.
* **Earned phase** (§16): a numeric φ binds only with a ``receipt`` holding two
  non-commuting measurements of that register; ``·`` (dark) and ``φ~`` (model) need none.
* **Earned tie** (§16): ``|☀⌢🔥⟩`` binds only with a receipt entry ``joint=True`` naming
  every register of the ket.
* A Bloch pair written pole-1-first (``🌙☀``) is canonicalized to pole-0-first by the
  half-turn about x (θ → π−θ, φ → −φ mod 2π): same state, one spelling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from umwelt.glyph.num import DARK, FORECAST, MEASURED, Num, as_num, least
from umwelt.glyph.parse import (Address, Bloch, Bra, Dark, Evidence, Ket, Population,
                                Sentence)
from umwelt.glyph.receipt import Receipt, has_joint, has_two_bases
from umwelt.glyph.scope import GlyphError, Scope, Unbound

TWO_PI = 2.0 * math.pi


class BindError(GlyphError):
    """A sentence that parses but is out of range or contradicts itself."""


@dataclass(frozen=True)
class Ref:
    register: str
    pole: int


@dataclass(frozen=True)
class BKet:
    refs: tuple          # one Ref per register, jointly collapsed
    tied: bool = False   # a receipt-backed non-separability claim (⌢)


@dataclass(frozen=True)
class BPairKet:
    """A ket of a product scope (§19): one pole from each parent = a pole pair of the pair
    register ``a⊗b``."""
    register: str        # "a⊗b"
    poles: tuple         # (pole in parent 1, pole in parent 2)
    parts: tuple         # (Ref in parent 1, Ref in parent 2)
    tied: bool = False


@dataclass(frozen=True)
class BPopulation:
    register: str
    p0: float            # support along pole 0 (canonical); a Num

    def __post_init__(self): object.__setattr__(self, "p0", as_num(self.p0))


@dataclass(frozen=True)
class BEvidence:
    register: str
    pole: int
    eta: float           # a Num

    def __post_init__(self): object.__setattr__(self, "eta", as_num(self.eta))


@dataclass(frozen=True)
class BBloch:
    register: str
    r: float
    theta: float         # from pole 0
    phi: float | None    # None: dark phase
    phase: str = "num"   # "num" (earned, receipt-backed) | "model" (φ~) | "dark" (·)
                         # r, theta, phi are Nums; phase is derived from phi's provenance

    def __post_init__(self):
        object.__setattr__(self, "r", as_num(self.r))
        object.__setattr__(self, "theta", as_num(self.theta))
        phi = self.phi
        if self.phase == "dark" or phi is None or getattr(phi, "prov", MEASURED) == DARK:
            phi, phase = None, "dark"
        else:
            phi = as_num(phi, FORECAST if self.phase == "model" else MEASURED)
            phase = "model" if phi.prov == FORECAST else "num"
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "phase", phase)


@dataclass(frozen=True)
class BDark:
    register: str


@dataclass(frozen=True)
class BBra:
    register: str
    pole: int
    psi: bool
    answer: Union[BKet, BPopulation, BDark, None] = None


@dataclass(frozen=True)
class Bound:
    scope_id: str
    scope_version: str
    path: tuple          # (scope_ref, register, pole) per descent step
    leaf: object         # one of the B* forms, or None for a bare address
    final_scope: str     # scope@version of the scope the leaf lives in

    @property
    def scope(self) -> str:
        return f"{self.scope_id}@{self.scope_version}"


def _check_one_scope(ket: Ket, scope: Scope):
    """Glyphs some of which this scope knows and some not are a cross-scope ket: undefined
    outside a product scope (§19)."""
    known = [g for g in ket.glyphs if scope._find(g)]
    if known and len(known) < len(ket.glyphs):
        missing = [g for g in ket.glyphs if g not in known]
        raise Unbound(f"ket mixes glyphs of scope {scope.ref} with {missing!r}, which it does "
                      f"not declare: juxtaposing glyphs from two scopes is undefined except "
                      f"inside a declared product scope s₁⊗s₂ (§19)")


def _bind_pair_ket(ket: Ket, scope: Scope, receipt):
    if len(ket.glyphs) != 2:
        raise BindError(f"a ket of product scope {scope.ref} takes exactly one glyph from "
                        f"each parent, got {len(ket.glyphs)}")
    subs = ket.subs or (0, 0)
    found = {}
    for g, sub in zip(ket.glyphs, subs):
        k, name, pole = scope.resolve_parent(g, sub or None)
        if k in found:
            raise BindError(f"ket names parent {k} of {scope.ref} twice; it takes one glyph "
                            f"from each parent")
        found[k] = Ref(name, pole)
    a, b = found[1], found[2]
    pair = f"{a.register}⊗{b.register}"
    if ket.tied and not (has_joint(receipt, [pair]) or has_joint(receipt, [a.register, b.register])):
        raise BindError(f"the tie '⌢' over {pair!r} is not earned: it needs a receipt entry "
                        f"marked joint=True naming both registers (§16)")
    return BPairKet(pair, (a.pole, b.pole), (a, b), ket.tied)


def _has_measurement(receipt, register: str) -> bool:
    return receipt is not None and any(
        m.eta > 0.0 and register in m.names() for m in receipt.measurements)


def _earn(num, slot: str, register: str, receipt, engine_origin: bool):
    """A straight number must be earned (v0.4 §25); forecast and dark need nothing."""
    if getattr(num, "prov", MEASURED) != MEASURED or engine_origin:
        return
    if not _has_measurement(receipt, register):
        raise BindError(
            f"{slot} {float(num)!r} on register {register!r} is written straight (measured) "
            f"but nothing earns it: it needs a receipt holding a measurement of "
            f"{register!r}, or an engine rendering from receipts. If you did not read it off "
            f"a receipt, write it forecast: {float(num)!r}~ (v0.4 §25).")


def _bind_leaf(leaf, scope: Scope, receipt: Receipt | None = None,
               engine_origin: bool = False):
    if isinstance(leaf, Ket):
        if scope.is_product:
            return _bind_pair_ket(leaf, scope, receipt)
        if leaf.subs:
            raise Unbound(f"parent subscripts {leaf.subs} on a ket bound to scope {scope.ref}: "
                          f"a cross-scope ket is defined only inside a declared product "
                          f"scope s₁⊗s₂ (§19)")
        refs, seen = [], {}
        _check_one_scope(leaf, scope)
        for g in leaf.glyphs:
            name, pole = scope.resolve(g)
            if name in seen:
                raise BindError(f"ket names register {name!r} twice ({g!r}); a ket is "
                                f"siblings read jointly, one pole per register")
            seen[name] = pole
            refs.append(Ref(name, pole))
        if leaf.tied:
            names = [r.register for r in refs]
            if not has_joint(receipt, names):
                raise BindError(f"the tie '⌢' over {names} is not earned: it needs a receipt "
                                f"entry marked joint=True naming every one of those registers "
                                f"(§16). Without it write the plain ket and the field decides.")
        return BKet(tuple(refs), leaf.tied)
    if isinstance(leaf, Population):
        name, pole = scope.resolve(leaf.glyph)
        if leaf.p.prov == DARK:
            return BDark(name)
        if not 0.0 <= leaf.p <= 1.0:
            raise BindError(f"population p={leaf.p} is outside [0,1]")
        _earn(leaf.p, "population p", name, receipt, engine_origin)
        return BPopulation(name, leaf.p if pole == 0 else Num(round(1.0 - float(leaf.p), 12), leaf.p.prov))
    if isinstance(leaf, Evidence):
        name, pole = scope.resolve(leaf.glyph)
        if leaf.eta.prov == DARK:
            raise BindError(f"evidence η is dark ('?') on register {name!r}: an observation "
                            f"without a weight is no observation; write the weight, "
                            f"forecast if expected ('~'), or say the register dark")
        if not 0.0 < leaf.eta <= 1.0:
            raise BindError(f"evidence η={leaf.eta} is outside (0,1]")
        _earn(leaf.eta, "evidence η", name, receipt, engine_origin)
        return BEvidence(name, pole, leaf.eta)
    if isinstance(leaf, Bloch):
        if DARK in (leaf.r.prov, leaf.theta.prov):
            n0, _ = scope.resolve(leaf.pole0)
            return BDark(n0)
        if not 0.0 <= leaf.r <= 1.0:
            raise BindError(f"Bloch r={leaf.r} is outside [0,1]")
        if not 0.0 <= leaf.theta <= math.pi:
            raise BindError(f"Bloch θ={leaf.theta} is outside [0,π]")
        if leaf.phase != "dark" and not 0.0 <= leaf.phi < TWO_PI:
            raise BindError(f"Bloch φ={leaf.phi} is outside [0,2π)")
        n0, p0 = scope.resolve(leaf.pole0)
        n1, p1 = scope.resolve(leaf.pole1)
        if n0 != n1:
            raise BindError(f"Bloch pair {leaf.pole0}{leaf.pole1} spans two registers "
                            f"({n0!r}, {n1!r}); it must be one register's two poles")
        if leaf.phase == "num" and not has_two_bases(receipt, n0, scope):
            raise BindError(f"numeric phase φ={leaf.phi} on register {n0!r} is not earned: "
                            f"a receipt must hold measurements of {n0!r} along two "
                            f"non-commuting directions (§16). Write the phase dark "
                            f"'[r,θ,·]' or, if the model predicts it, 'φ~'.")
        _earn(leaf.r, "Bloch r", n0, receipt, engine_origin)
        _earn(leaf.theta, "Bloch θ", n0, receipt, engine_origin)
        theta, phi = leaf.theta, leaf.phi
        if p0 == 1:                                   # written pole-1-first: flip
            theta = math.pi - theta                   # (Num arithmetic keeps the provenance)
            if phi is not None:
                phi = (-phi) % TWO_PI
                if phi >= TWO_PI:
                    phi = Num(0.0, phi.prov)
        return BBloch(n0, leaf.r, theta, phi, leaf.phase)
    if isinstance(leaf, Dark):
        return BDark(scope.resolve(leaf.glyph)[0])
    if isinstance(leaf, Bra):
        name, pole = scope.resolve(leaf.glyph)
        ans = (None if leaf.answer is None
               else _bind_leaf(leaf.answer, scope, receipt, engine_origin))
        return BBra(name, pole, leaf.psi, ans)
    raise BindError(f"cannot bind {leaf!r}")


def bind(ast: Sentence, scope: Scope, receipt: Receipt | None = None, *,
         engine_origin: bool = False) -> Bound:
    """Bind ``ast`` to ``scope`` (stamping ``scope.version``). Raises ``Ambiguous``,
    ``Unbound`` or ``BindError``; never guesses. ``receipt`` is what earned quantities
    (a numeric φ, §16; every straight number, v0.4 §25) must be backed by.
    ``engine_origin=True`` is the engine adapter's declaration that the sentence was rendered
    from a state built from receipts, so its straight numbers count as measured (it never
    earns a numeric φ: that still needs two bases). Wired by the caller, not here."""
    if isinstance(ast, Address):
        bare = ast.leaf is None
        steps, final = scope.resolve_path(ast.path_glyphs, descend_last=not bare)
        leaf = None if bare else _bind_leaf(ast.leaf, final, receipt, engine_origin)
        return Bound(scope.id, scope.version, tuple(steps), leaf, final.ref)
    return Bound(scope.id, scope.version, (), _bind_leaf(ast, scope, receipt, engine_origin), scope.ref)


def canonical_in(ast: Sentence, scope: Scope) -> str:
    """Canonical text of ``ast`` read against ``scope``. For a ket of a product scope the
    parent subscripts are written and the glyphs ordered parent 1 first: ``|☀₁🌙₂⟩`` (§19);
    otherwise the same as ``canonical``. (Resolution only: it does not check receipts.)"""
    from dataclasses import replace
    from umwelt.glyph.parse import canonical
    if isinstance(ast, Ket) and scope.is_product:
        subs = ast.subs or (0,) * len(ast.glyphs)
        placed = sorted((scope.resolve_parent(g, sub or None)[0], g)
                        for g, sub in zip(ast.glyphs, subs))
        ast = replace(ast, glyphs=tuple(g for _, g in placed), subs=tuple(k for k, _ in placed))
    return canonical(ast)
