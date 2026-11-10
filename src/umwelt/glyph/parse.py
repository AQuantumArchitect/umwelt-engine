"""Parser for the six sentence forms and addresses (grammar §3, §4). Pure syntax: no scope.

    ket         |☀😁⟩            Ket(glyphs)   (|☀⌢🔥⟩: tied=True, §16;
                |☀₁🌙₂⟩: subs=(1, 2), parent subscripts of a product scope, §19)
    population  [.8]☀           Population(p, glyph)
    evidence    ☀!0.8           Evidence(glyph, eta)
    precise     [r,θ,φ]☀🌙      Bloch(r, theta, phi, pole0, pole1, phase)
                [r,θ,·]☀🌙 (dark phase)   [r,θ,φ~]☀🌙 (model phase, §16)
    numbers     every numeric slot is a ``Num`` (v0.4 §25): ``0.8`` measured, ``0.8~``
                forecast, ``?`` dark. ``~~``, ``~?``, ``?~`` and a bare ``~`` are refused.
    dark        ☀?              Dark(glyph)
    bra         ⟨☀|  ⟨☀|ψ⟩  ⟨☀|ψ⟩ = <ket|population|dark>      Bra(glyph, answer, psi)
    address     ☀›🔥›<leaf>     Address(path_glyphs, leaf)   (leaf None: bare ☀›🔥›⚫)
                ☀🌙›<leaf>      a path element of TWO glyphs (stored as "☀🌙") is a descent
                                under the register (a conditional state, §17); one glyph is
                                a descent under a pole.

Decisions where the page is silent are listed in docs/GLYPH.md. A trailing ``# comment``
is ignored. ASCII fallbacks ``|..>`` ``<..|`` and ``>`` for ``›`` need ``lenient=True``;
``canonical`` always writes the Unicode forms.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Union

from umwelt.glyph.normalize import canon, glyphs
from umwelt.glyph.num import DARK, FORECAST, MEASURED, Num, as_num, spell
from umwelt.glyph.scope import RESERVED, GlyphError

LKET, RKET, LBRA, RBRA = "|", "⟩", "⟨", "|"
DESC, PSI = "›", "ψ"
DARK_PHASE, MODEL_MARK, TIE = "·", "~", "⌢"
SUBSCRIPTS = {"₁": 1, "₂": 2}
_SUB_CHAR = {1: "₁", 2: "₂"}
_RESERVED = set(RESERVED)
_NUM_CHARS = set("0123456789.+-eE")
_NUM_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


class ParseError(GlyphError):
    def __init__(self, msg: str, pos: int, text: str):
        self.msg, self.pos, self.text = msg, pos, text
        super().__init__(f"{msg} at position {pos} of {text!r} (positions count the "
                         f"normalized text)")


def _tup(obj, name, v):
    object.__setattr__(obj, name, tuple(v))


@dataclass(frozen=True)
class Ket:
    glyphs: tuple
    tied: bool = False       # |a⌢b⟩: a non-separability claim, earned only by a joint receipt (§16)
    subs: tuple = ()         # parent subscripts per glyph (0 none, 1 ₁, 2 ₂); () when none (§19)

    def __post_init__(self):
        _tup(self, "glyphs", self.glyphs)
        subs = tuple(self.subs)
        if subs and (len(subs) != len(self.glyphs) or any(x not in (0, 1, 2) for x in subs)):
            raise ValueError("Ket.subs must hold one of 0, 1, 2 per glyph")
        object.__setattr__(self, "subs", subs if any(subs) else ())


@dataclass(frozen=True)
class Population:
    p: float                 # a Num: measured / forecast (~) / dark (?)
    glyph: str

    def __post_init__(self): object.__setattr__(self, "p", as_num(self.p))


@dataclass(frozen=True)
class Evidence:
    glyph: str
    eta: float               # a Num

    def __post_init__(self): object.__setattr__(self, "eta", as_num(self.eta))


@dataclass(frozen=True)
class Bloch:
    r: float
    theta: float
    phi: float | None        # None when the phase is dark (``·`` or ``?``)
    pole0: str
    pole1: str
    phase: str = "num"       # DERIVED from phi's provenance: "num" (witnessed), "model"
                             # (φ~, forecast), "dark" (·). A compat argument: "model" on a
                             # plain phi makes it a forecast; "dark" drops phi.

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

    @property
    def phi_num(self) -> Num:
        return Num(0.0, DARK) if self.phi is None else self.phi


@dataclass(frozen=True)
class Dark:
    glyph: str


@dataclass(frozen=True)
class Bra:
    glyph: str
    answer: Union[Ket, Population, Dark, None] = None
    psi: bool = False


@dataclass(frozen=True)
class Address:
    path_glyphs: tuple
    leaf: Union[Ket, Population, Evidence, Bloch, Dark, Bra, None] = None
    def __post_init__(self): _tup(self, "path_glyphs", self.path_glyphs)


Sentence = Union[Ket, Population, Evidence, Bloch, Dark, Bra, Address]


def _is_glyph(tok: str) -> bool:
    if not tok or tok.isspace() or tok in _RESERVED:
        return False
    return ord(tok[0]) > 127 or "⃣" in tok     # non-ASCII, or a keycap sequence


class _Parser:
    def __init__(self, text: str, lenient: bool, keep_skin_tones: bool):
        self.src = canon(text, keep_skin_tones=keep_skin_tones)
        toks = glyphs(self.src, keep_skin_tones=keep_skin_tones)
        self.lenient = lenient
        # drop a trailing comment; remember offsets for error positions
        self.toks, self.offs, off = [], [], 0
        for t in toks:
            if t == "#":
                break
            self.toks.append(t)
            self.offs.append(off)
            off += len(t)
        self.end_off = off
        self.i = 0

    # ── helpers ──
    def err(self, msg: str, i: int | None = None):
        i = self.i if i is None else i
        pos = self.offs[i] if i < len(self.offs) else self.end_off
        raise ParseError(msg, pos, self.src)

    def ws(self):
        while self.i < len(self.toks) and self.toks[self.i].isspace():
            self.i += 1

    def peek(self, k: int = 0) -> str | None:
        """Next non-space token (k-th), without consuming."""
        j, seen = self.i, -1
        while j < len(self.toks):
            if not self.toks[j].isspace():
                seen += 1
                if seen == k:
                    return self.toks[j]
            j += 1
        return None

    def take(self) -> str | None:
        self.ws()
        if self.i >= len(self.toks):
            return None
        t = self.toks[self.i]
        self.i += 1
        return t

    def _ascii_check(self, tok: str | None):
        if tok in ("<", ">") and not self.lenient:
            self.err(f"ASCII {tok!r} is only accepted with lenient=True; "
                     f"the canonical forms are ⟨ ⟩ ›")

    def is_close_ket(self, tok) -> bool:
        return tok == RKET or (self.lenient and tok == ">")

    def is_descent(self, tok) -> bool:
        return tok == DESC or (self.lenient and tok == ">")

    def expect(self, want: str, what: str | None = None):
        self.ws()
        t = self.take()
        if t != want:
            self.err(f"expected {what or repr(want)}, found {t!r}" if t is not None
                     else f"expected {what or repr(want)}, found end of sentence", self.i - 1 if t is not None else self.i)

    def glyph(self, what: str = "a glyph") -> str:
        self.ws()
        t = self.take()
        if t is None:
            self.err(f"expected {what}, found end of sentence")
        if not _is_glyph(t):
            self._ascii_check(t)
            self.err(f"expected {what}, found {t!r}", self.i - 1)
        return t

    def number(self, what: str) -> Num:
        """A numeric literal with its provenance: ``0.8`` / ``0.8~`` / ``?`` (v0.4 §25)."""
        self.ws()
        start = self.i
        if self.peek() == "?":
            self.take()
            if self.peek() == MODEL_MARK:
                self.err("'?~' is not a number: dark has no forecast (write '?' or a number)",
                         self.i)
            return Num(0.0, DARK)
        if self.peek() == MODEL_MARK:
            self.err(f"'~' marks a number as a forecast and must follow one; "
                     f"expected {what} (a number)", start)
        buf = ""
        while self.i < len(self.toks) and self.toks[self.i] in _NUM_CHARS:
            buf += self.toks[self.i]
            self.i += 1
        if not _NUM_RE.fullmatch(buf):
            self.err(f"expected {what} (a number), found {buf or self.peek()!r}", start)
        v = float(buf)
        if not math.isfinite(v):
            self.err(f"{what} {buf!r} is not finite", start)
        if self.peek() == MODEL_MARK:
            self.take()
            nxt = self.peek()
            if nxt == MODEL_MARK:
                self.err("'~~' is not a number: the forecast mark is written once", self.i)
            if nxt == "?":
                self.err("'~?' is not a number: a number is forecast or dark, not both", self.i)
            return Num(v, FORECAST)
        return Num(v, MEASURED)

    # ── grammar ──
    def sentence(self) -> Sentence:
        self.ws()
        if self.peek() is None:
            self.err("empty sentence")
        path: list[str] = []
        while True:
            t, n, m = self.peek(), self.peek(1), self.peek(2)
            if t is not None and _is_glyph(t) and n is not None and self.is_descent(n):
                path.append(self.glyph())            # descent under a pole  ☀›
                self.take()                          # the descent mark
                continue
            if (t is not None and _is_glyph(t) and n is not None and _is_glyph(n)
                    and m is not None and self.is_descent(m)):
                path.append(self.glyph() + self.glyph())   # under a register  ☀🌙›  (§17)
                self.take()
                continue
            break
        t, n = self.peek(), self.peek(1)
        if path and t is not None and _is_glyph(t) and n is None:
            g = self.glyph()
            return Address(tuple(path) + (g,), None)        # bare address
        if path and t is None:
            self.err("expected a sentence form or a pole after '›', found end of sentence")
        leaf = self.leaf()
        self.ws()
        if self.i < len(self.toks):
            self.err(f"unexpected {self.toks[self.i]!r} after the sentence")
        return Address(tuple(path), leaf) if path else leaf

    def leaf(self, answer_only: bool = False):
        self.ws()
        t = self.peek()
        if t is None:
            self.err("expected a sentence form, found end of sentence")
        if t == LKET:
            return self.ket()
        if t == "[":
            return self.bracket()
        if t == LBRA or (self.lenient and t == "<"):
            if answer_only:
                self.err("a bra cannot be the answer of a bra")
            return self.bra()
        if _is_glyph(t):
            g = self.glyph()
            op = self.take()
            if op == "?":
                return Dark(g)
            if op == "!":
                if answer_only:
                    self.err("an answer is a ket, a population or dark; evidence is not", self.i - 1)
                return Evidence(g, self.number("η"))
            self.err(f"expected '!' or '?' after glyph {g!r}" +
                     (f", found {op!r}" if op is not None else ", found end of sentence")
                     + " (a bare glyph says nothing)", self.i - 1 if op is not None else self.i)
        self._ascii_check(t)
        self.err(f"expected a sentence form, found {t!r}")

    def ket(self) -> Ket:
        self.expect(LKET)
        gs, subs, ties = [], [], 0
        while True:
            self.ws()
            t = self.peek()
            if t is None:
                self.err("unterminated ket: expected '⟩'")
            if self.is_close_ket(t):
                self._ascii_check(t)
                self.take()
                break
            if t == TIE:
                self.take()
                if not gs:
                    self.err("a tie '⌢' needs a glyph before it", self.i - 1)
                n = self.peek()
                if n is None or n == TIE or self.is_close_ket(n):
                    self.err("a tie '⌢' must be followed by a glyph")
                ties += 1
                continue
            gs.append(self.glyph("a glyph or '⟩'"))
            sub = 0
            if self.peek() in SUBSCRIPTS:
                sub = SUBSCRIPTS[self.take()]
            subs.append(sub)
            if len(gs) - 1 != ties and ties not in (0, len(gs) - 1):
                self.err("a tie must join every adjacent pair of the ket or none", self.i - 1)
        if not gs:
            self.err("empty ket", self.i - 1)
        if ties not in (0, len(gs) - 1):
            self.err("a tie must join every adjacent pair of the ket or none")
        return Ket(tuple(gs), ties > 0, tuple(subs))

    def bracket(self):
        self.expect("[")
        a = self.number("a number")
        self.ws()
        t = self.take()
        if t == "]":
            return Population(a, self.glyph("the pole glyph after ']'"))
        if t != ",":
            self.err(f"expected ']' or ',', found {t!r}" if t is not None else
                     "unterminated '['", self.i - 1)
        theta = self.number("θ")
        self.expect(",")
        self.ws()
        if self.peek() == DARK_PHASE:
            self.take()
            if self.peek() == MODEL_MARK:
                self.err("'·~' is not a number: a dark phase has no forecast")
            phi = None
        else:
            phi = self.number("φ (a number, or '·' for a dark phase)")
            if phi.prov == DARK:
                phi = None
        self.expect("]")
        p0 = self.glyph("the first pole glyph")
        p1 = self.glyph("the second pole glyph")
        return Bloch(a, theta, phi, p0, p1)

    def bra(self) -> Bra:
        t = self.take()
        self._ascii_check(t)
        g = self.glyph()
        self.expect(RBRA, "'|'")
        psi, answer = False, None
        if self.peek() == PSI:
            self.take()
            close = self.take()
            if not self.is_close_ket(close):
                self.i -= 1
                self.err("expected '⟩' after ψ")
            self._ascii_check(close)
            psi = True
        if self.peek() == "=":
            self.take()
            answer = self.leaf(answer_only=True)
            if isinstance(answer, Evidence):
                self.err("an answer is a ket, a population or dark")
        return Bra(g, answer, psi)


def parse(text: str, *, lenient: bool = False, keep_skin_tones: bool = False) -> Sentence:
    """Parse one sentence. Raises ``ParseError`` (with position) on anything off-grammar.

    ``keep_skin_tones`` must match the scope the sentence will be bound to
    (``Scope.keep_skin_tones``), or the modifiers are stripped before binding."""
    return _Parser(text, lenient, keep_skin_tones).sentence()


_num = spell


def parse_number(text: str) -> Num:
    """Parse a standalone numeric literal: ``0.8`` / ``0.8~`` / ``?``. For the slots the
    grammar does not carry yet (costs, dates, counts, durations); same refusals."""
    p = _Parser(text, False, False)
    n = p.number("a number")
    p.ws()
    if p.i < len(p.toks):
        p.err(f"unexpected {p.toks[p.i]!r} after the number")
    return n


def canonical(ast: Sentence) -> str:
    """Canonical text of an AST; ``parse(canonical(x)) == x`` for every parsed x."""
    if isinstance(ast, Ket):
        subs = ast.subs or (0,) * len(ast.glyphs)
        body = (TIE if ast.tied else "").join(g + _SUB_CHAR.get(k, "") for g, k in zip(ast.glyphs, subs))
        return f"|{body}⟩"
    if isinstance(ast, Population):
        return f"[{_num(ast.p)}]{ast.glyph}"
    if isinstance(ast, Evidence):
        return f"{ast.glyph}!{_num(ast.eta)}"
    if isinstance(ast, Bloch):
        ph = DARK_PHASE if ast.phi is None else _num(ast.phi)
        return f"[{_num(ast.r)},{_num(ast.theta)},{ph}]{ast.pole0}{ast.pole1}"
    if isinstance(ast, Dark):
        return f"{ast.glyph}?"
    if isinstance(ast, Bra):
        s = f"⟨{ast.glyph}|" + (f"{PSI}⟩" if ast.psi else "")
        if ast.answer is not None:
            s += f" = {canonical(ast.answer)}"
        return s
    if isinstance(ast, Address):
        head = DESC.join(ast.path_glyphs)
        return head if ast.leaf is None else f"{head}{DESC}{canonical(ast.leaf)}"
    raise TypeError(f"not a sentence: {ast!r}")
