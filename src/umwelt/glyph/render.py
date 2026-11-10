"""Render a ``frame/1`` document as sentences. Reading only — never writes state (§1).

Mapping (decisions in docs/GLYPH.md):

* a register is matched to a frame node by ``Register.node`` (default: the register name);
  frame nodes with no register are not rendered.
* state ``dark``                  -> ``☀?`` (pole-0 glyph)
* ``current`` / ``stale`` + scalar -> ``[p]☀``. Unit ``bloch_z`` (what the engine exports)
  is a z-component, so p = (1+z)/2 along pole 0; any other unit must already be a
  number in [0,1]. A stale sentence gets a trailing ``  # age <n>s stale`` comment; the
  sentence itself stays pure.
* a scope that declares a resolution ε (§17) appends ``# ε=…`` to each sentence it renders
  (joined to the stale note with ``;``). Nothing else changes.
* a node carrying an explicit ``joint_receipt`` ``{"with": <node id>, "poles": [i, j]}``
  (and only then) also yields the tied ket ``|g_i⌢g_j⟩`` of the two registers; a tie is
  never inferred from the field.
* ``precise`` + a node key ``bloch`` = [x,y,z] -> ``[r,θ,φ~]☀🌙`` in the register's frame
  (the engine computes the phase from dynamics: a model phase, never a witnessed one, §16;
  at r=0 the phase is dark ``·``).
* v0.4 §25: a node with ``"forecast": true`` beside its ``value`` is a forecast: its numbers
  render with ``~`` (``[0.8~]☀``; a precise vector renders ``[r~,θ~,φ~]``). A ``dark`` node
  cannot be a forecast. Numbers without the flag are the engine's, read off ρ, and bind as
  measured (``engine_origin=True``). Also read: ``"evidence": η`` + ``"pole": i`` and
  ``"bloch_rtp": [r,θ,φ|null]`` with ``"forecast": [bool, bool, bool]`` (per-component,
  for mixed vectors). The reverse direction is ``frame_node``; prose is ``prose``. Neither
  ever drops a ``~``: a forecast is a ``"forecast"`` flag in a frame, "about … (forecast)"
  in prose, and ``~`` in a glyph string.
"""
from __future__ import annotations

import math

import numpy as np

from umwelt.glyph.bind import (BBloch, BDark, BEvidence, BPopulation, Bound, bind)
from umwelt.glyph.num import DARK, FORECAST, MEASURED, Num, say
from umwelt.glyph.parse import (Address, Bloch, Bra, Dark, Evidence, Ket, Population, canonical,
                                parse)
from umwelt.glyph.receipt import Measurement, Receipt
from umwelt.glyph.scope import GlyphError, Scope

FRAME_FORMAT = "frame/1"
_Z = (0.0, 0.0, 1.0)


class RenderError(GlyphError):
    pass


def _frame_basis(n):
    n = np.asarray(n, dtype=float)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = ref - n * float(ref @ n)
    ex /= np.linalg.norm(ex)
    return ex, np.cross(n, ex), n


def _clamp01(x: float, what: str) -> float:
    if -1e-9 <= x <= 1.0 + 1e-9:
        return min(1.0, max(0.0, x))
    raise RenderError(f"{what}={x} is outside [0,1]")


def sentences_from_frame(doc: dict, scope: Scope, *, precise: bool = False) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != FRAME_FORMAT:
        raise RenderError(f"not a {FRAME_FORMAT} document")
    by_node = {}
    for name, reg in scope.registers.items():
        by_node.setdefault(reg.node or name, (name, reg))
    out: list[str] = []
    ties: set = set()
    for node in doc.get("nodes", []):
        hit = by_node.get(node.get("id"))
        if hit is None:
            continue
        name, reg = hit
        state, value = node.get("state"), node.get("value")
        fc = node.get("forecast", False)
        if state == "dark":
            if fc:
                raise RenderError(f"node {node.get('id')!r}: a dark node cannot be a forecast "
                                  f"('~?' is not a number)")
            ast = Dark(reg.pole0)
            note = ""
        elif state in ("current", "stale") and ("evidence" in node or "bloch_rtp" in node):
            where = f"node {node.get('id')!r}"
            if "evidence" in node:
                pole = node.get("pole", 0)
                if pole not in (0, 1) or isinstance(pole, bool) or not isinstance(fc, bool):
                    raise RenderError(f"{where}: evidence needs pole in {{0,1}} and a bool forecast")
                ast = Evidence(reg.glyph(pole), Num(node["evidence"], FORECAST if fc else MEASURED))
            else:
                rtp = node["bloch_rtp"]
                if (not isinstance(rtp, (list, tuple)) or len(rtp) != 3
                        or not isinstance(fc, (list, tuple)) or len(fc) != 3
                        or any(not isinstance(b, bool) for b in fc)):
                    raise RenderError(f"{where}: bloch_rtp is [r,θ,φ|null] and its forecast "
                                      f"is a list of three bools")
                r, th, ph = (None if x is None else Num(x, FORECAST if b else MEASURED)
                             for x, b in zip(rtp, fc))
                ast = Bloch(r, th, ph, reg.pole0, reg.pole1)
            note = f"  # age {node.get('age_s')}s stale" if state == "stale" else ""
        elif state in ("current", "stale"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RenderError(f"node {node.get('id')!r}: {state} but value {value!r} is not "
                                  f"a scalar; refusing to promote")
            where = f"node {node.get('id')!r}"
            if not isinstance(fc, bool):
                raise RenderError(f"{where}: 'forecast' must be a bool beside a scalar value")
            kind = FORECAST if fc else MEASURED
            bloch = node.get("bloch") if precise else None
            if bloch is not None:
                ex, ey, ez = _frame_basis(reg.frame)
                v = np.asarray(bloch, dtype=float)
                x, y, z = float(v @ ex), float(v @ ey), float(v @ ez)
                r = math.sqrt(x * x + y * y + z * z)
                if r > 1.0 + 1e-9:
                    raise RenderError(f"{where}: Bloch vector length {r} > 1")
                r = min(1.0, r)
                theta = math.acos(max(-1.0, min(1.0, z / r))) if r > 0 else 0.0
                phi = math.atan2(y, x) % (2 * math.pi) if r > 0 else 0.0
                if phi >= 2 * math.pi:
                    phi = 0.0
                # The engine's phase is a model output (§16): spelled φ~, or dark at r=0.
                ast = (Bloch(Num(r, kind), Num(theta, kind), None, reg.pole0, reg.pole1) if r == 0
                       else Bloch(Num(r, kind), Num(theta, kind), Num(phi, FORECAST),
                                  reg.pole0, reg.pole1))
            else:
                if node.get("unit") == "bloch_z":
                    if tuple(reg.frame) != _Z:
                        raise RenderError(f"{where}: a bloch_z value is the z-component; "
                                          f"register {name!r} declares another frame")
                    p = _clamp01((1.0 + float(value)) / 2.0, "p")
                else:
                    p = _clamp01(float(value), "value")
                ast = Population(Num(p, kind), reg.pole0)
            note = f"  # age {node.get('age_s')}s stale" if state == "stale" else ""
        else:
            raise RenderError(f"node {node.get('id')!r}: unknown state {state!r}")
        text = canonical(ast)
        # the sentence must mean one thing in this scope; its straight numbers were read
        # off the engine's state (engine_origin), its '~' ones need nothing
        bind(parse(text), scope, engine_origin=True)
        if scope.epsilon is not None:
            eps = f"ε={scope.epsilon!r}"
            note = f"{note}; {eps}" if note else f"  # {eps}"
        out.append(text + note)
        if "joint_receipt" in node:
            tied = _tied_sentence(node, by_node, scope)
            key = frozenset((node.get("id"), node["joint_receipt"].get("with")))
            if key not in ties:
                ties.add(key)
                out.append(tied)
    return out


def _tied_sentence(node: dict, by_node: dict, scope: Scope) -> str:
    jr = node["joint_receipt"]
    if not isinstance(jr, dict) or jr.get("with") not in by_node or node["id"] not in by_node:
        raise RenderError(f"node {node.get('id')!r}: joint_receipt must be a dict naming a "
                          f"partner node that has a register in this scope")
    poles = jr.get("poles")
    if (not isinstance(poles, (list, tuple)) or len(poles) != 2
            or any(p not in (0, 1) or isinstance(p, bool) for p in poles)):
        raise RenderError(f"node {node.get('id')!r}: joint_receipt needs poles [i, j] in {{0,1}}")
    (na, ra), (nb, rb) = by_node[node["id"]], by_node[jr["with"]]
    if na == nb:
        raise RenderError(f"node {node.get('id')!r}: joint_receipt partner is the same register")
    text = canonical(Ket((ra.glyph(poles[0]), rb.glyph(poles[1])), True))
    rc = Receipt((Measurement(na, "z", 1.0, 0, joint=True, registers=(na, nb)),))
    bind(parse(text), scope, rc, engine_origin=True)          # the tie is earned by the node's own receipt key
    return text


def _flag(n) -> bool:
    return n.prov == FORECAST


def frame_node(bound: Bound, scope: Scope) -> dict:
    """A frame/1-style node dict for a bound sentence's leaf (population, evidence, Bloch,
    dark): the value as written, never a different one, with ``"forecast"`` beside it —
    a bool for a scalar, a list of three bools for a Bloch vector (``bloch_rtp``).
    ``sentences_from_frame`` reads it back (``precise=True`` for a vector)."""
    leaf = bound.leaf
    if isinstance(leaf, BDark):
        reg = scope.registers[leaf.register]
        return {"id": reg.node or leaf.register, "state": "dark"}
    if not isinstance(leaf, (BPopulation, BEvidence, BBloch)):
        raise RenderError(f"{type(leaf).__name__} has no frame node (only a population, "
                          f"evidence, Bloch or dark statement does)")
    reg = scope.registers[leaf.register]
    node = {"id": reg.node or leaf.register, "state": "current"}
    if isinstance(leaf, BPopulation):
        node.update(value=float(leaf.p0), unit="probability", forecast=_flag(leaf.p0))
    elif isinstance(leaf, BEvidence):
        node.update(evidence=float(leaf.eta), pole=leaf.pole, forecast=_flag(leaf.eta))
    else:
        node.update(bloch_rtp=[float(leaf.r), float(leaf.theta),
                               None if leaf.phi is None else float(leaf.phi)],
                    forecast=[_flag(leaf.r), _flag(leaf.theta),
                              leaf.phi is not None and _flag(leaf.phi)])
    return node


def prose(ast) -> str:
    """The sentence in words. A forecast number reads "about N (forecast)", a measured one
    is the bare number, a dark one "unknown (dark)". Never drops a ``~``."""
    if isinstance(ast, str):
        ast = parse(ast)
    if isinstance(ast, Population):
        return f"{ast.glyph} stands at {say(ast.p)}"
    if isinstance(ast, Evidence):
        return f"an observation of {ast.glyph} with weight {say(ast.eta)}"
    if isinstance(ast, Bloch):
        ph = "unknown (dark)" if ast.phi is None else say(ast.phi)
        return (f"{ast.pole0}{ast.pole1} has Bloch length r = {say(ast.r)}, polar angle "
                f"θ = {say(ast.theta)}, phase φ = {ph}")
    if isinstance(ast, Dark):
        return f"{ast.glyph} is dark (nothing is claimed)"
    if isinstance(ast, Ket):
        return ("the joint " if len(ast.glyphs) > 1 else "") + "ket " + canonical(ast)
    if isinstance(ast, Bra):
        s = f"asking {ast.glyph}" + (" of the current state" if ast.psi else "")
        return s + (f", answered: {prose(ast.answer)}" if ast.answer is not None else "")
    if isinstance(ast, Address):
        head = " then ".join(ast.path_glyphs)
        return f"under {head}" + (f": {prose(ast.leaf)}" if ast.leaf is not None else "")
    raise RenderError(f"cannot put {ast!r} into words")
