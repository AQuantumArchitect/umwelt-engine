"""Embedder ⇄ de-embedder — the two-way symbol membrane between the manifold and words.

The field embeds the world into high-D folds (sensors → qubits); comprehension_notes already de-embeds
a little, but by hand (if occupancy z>0 say "occupied"). This is the principled, TWO-WAY version
(docs/MIND.md, the "doesn't mint symbols" cure):

  de_embed:  a qubit's Bloch position → a SYMBOL (a discrete label + a confidence). The geometry stays
             the mind; the symbol is its expression. The operator/LLM reads symbols, not raw vectors.
  embed:     a symbol → the Bloch ANCHOR it names. The inverse — so the operator can WRITE to the field
             in field terms (drive a qubit toward "occupied"), making the language membrane two-way
             instead of read-only.

The codebook is the Fibonacci sphere — the golden-angle tiling of the Bloch sphere (fib_fractal finding
#3: φ is load-bearing for STATE geometry; the most-irrational angle tiles the sphere with the least
clumping, so each cell is a maximally-distinct symbol). A symbol = a labeled cell. Cells start unlabeled;
declared groundings label the ones we know. When a CONFIDENT belief persistently lands in an
unlabeled cell, the membrane records a *latent* visit (`label=null`) — it does NOT invent a
word. Naming is an app write (`ground` / `embed`). A coordinate may only be named from
evidence, never from geometry (the place-provenance law). The codebook stays finite.

Confidence rides the qubit's PURITY (a mixed belief de-embeds to a low-confidence symbol) times the
angular closeness to the cell centre — both gauge quantities, so a symbol carries its own epistemics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from umwelt._util import clamp01

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))     # 2π/φ² — the golden angle


def fibonacci_sphere(n: int) -> list[tuple]:
    """`n` maximally-spread points on the unit sphere (the golden-angle spiral). The codebook cells —
    each a candidate symbol, as distinct from its neighbours as the sphere allows."""
    pts = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = GOLDEN_ANGLE * i
        pts.append((r * math.cos(theta), r * math.sin(theta), z))
    return pts


def _norm(v: tuple) -> tuple:
    n = math.sqrt(sum(c * c for c in v))
    return (0.0, 0.0, 0.0) if n < 1e-12 else tuple(c / n for c in v)


def _cos(a: tuple, b: tuple) -> float:
    return sum(x * y for x, y in zip(_norm(a), _norm(b)))


@dataclass
class Symbol:
    """A de-embedded symbol: the word, the Bloch cell it names, and how sure the geometry is of it."""
    label: str | None
    anchor: tuple                      # the cell centre on the Bloch sphere
    confidence: float                  # purity × angular-closeness ∈ [0,1]
    minted: bool = False               # kept for callers; always False (engine never invents words)
    latent: bool = False               # True if this call noted a confident unlabeled cell
    cell: int | None = None            # codebook index


class SymbolCodebook:
    """The Fibonacci-sphere codebook + the two-way map. Declared groundings name known cells;
    confident unlabeled visits are latent (label=null) until the app grounds them."""

    def __init__(self, n_cells: int = 64, *, mint_purity: float = 0.6, mint_closeness: float = 0.55):
        self.cells = fibonacci_sphere(n_cells)
        self._label: dict[int, str] = {}            # cell index → symbol label
        self._latent_visits: dict[int, int] = {}
        self.latent_events: list[dict] = []
        self.mint_purity = float(mint_purity)       # only note a state the field is CONFIDENT about
        self.mint_closeness = float(mint_closeness)  # …and that actually sits in this cell (cos sim)

    # --- grounding (declared, known meanings) --------------------------------
    def ground(self, label: str, bloch: tuple) -> int:
        """Pin `label` to the cell nearest `bloch` (a declared grounding, e.g. z=+1 → 'occupied')."""
        idx, _ = self._nearest(bloch)
        self._label[idx] = label
        return idx

    def ground_axis(self, axis_labels: dict) -> None:
        """Convenience: ground the cardinal Bloch directions. `axis_labels` maps any of
        '+z','-z','+x','-x','+y','-y' to a label (the belief poles: +z occupied / −z empty, etc.)."""
        dirs = {"+z": (0, 0, 1), "-z": (0, 0, -1), "+x": (1, 0, 0), "-x": (-1, 0, 0),
                "+y": (0, 1, 0), "-y": (0, -1, 0)}
        for k, label in axis_labels.items():
            if k in dirs:
                self.ground(label, dirs[k])

    # --- the geometry ⇄ symbol map -------------------------------------------
    def _nearest(self, bloch: tuple) -> tuple:
        sims = [(_cos(bloch, c), i) for i, c in enumerate(self.cells)]
        cos, idx = max(sims)
        return idx, cos

    def de_embed(self, bloch: tuple, purity: float = 1.0, *, mint: bool = False) -> Symbol:
        """Geometry → symbol. Nearest grounded cell names it. A confident unlabeled cell
        is recorded as latent (`label=None`); `mint=True` still does not invent a word —
        it only records the visit. Call `ground` to name a cell."""
        idx, cos = self._nearest(bloch)
        closeness = max(0.0, cos)
        confidence = clamp01(purity) * (cos + 1.0) / 2.0
        label = self._label.get(idx)
        latent = False
        if label is None and purity >= self.mint_purity and closeness >= self.mint_closeness:
            self._latent_visits[idx] = self._latent_visits.get(idx, 0) + 1
            event = {
                "cell": idx,
                "purity": round(float(purity), 4),
                "closeness": round(float(closeness), 4),
                "visits": self._latent_visits[idx],
                "label": None,
            }
            self.latent_events.append(event)
            latent = True
            # `mint` used to invent concept_N; that path is gone. The flag is kept so
            # callers don't break; it no longer names anything.
            _ = mint
        return Symbol(label=label, anchor=self.cells[idx], confidence=round(confidence, 4),
                      minted=False, latent=latent, cell=idx)

    def embed(self, label: str) -> tuple | None:
        """Symbol → the Bloch anchor it names (the WRITE direction: drive a qubit toward this). Averages
        if a label spans cells. None if the word isn't in the codebook yet."""
        named = [self.cells[i] for i, lab in self._label.items() if lab == label]
        if not named:
            return None
        if len(named) == 1:
            return named[0]
        c = tuple(sum(p[k] for p in named) / len(named) for k in range(3))
        return _norm(c)

    def vocabulary(self) -> dict:
        """The current grounded symbol set: label → anchor. Latent cells are not in here."""
        return {lab: self.cells[i] for i, lab in self._label.items()}

    def n_minted(self) -> int:
        """Always 0 — the engine never invents labels. Kept for callers."""
        return 0

    def n_latent(self) -> int:
        return len(self._latent_visits)


# ── field-wide de-embedding (the principled qubit→symbol membrane) ───────────

def de_embed_field(field, codebook: SymbolCodebook, *, mint: bool = False) -> dict:
    """Walk every cluster role, de-embed its qubit into a symbol. Returns {`cluster.role`: Symbol}.
    This is the gauge-grounded successor to comprehension_notes' hand-coded mapping — the same source
    (the field's Bloch states + purity), but symbols minted from geometry instead of if/else."""
    out = {}
    clusters = getattr(field, "clusters", {})
    for name in sorted(clusters):
        cl = clusters[name]
        roles = getattr(cl, "role_index", {})
        for role in sorted(roles):
            try:
                bloch = tuple(float(v) for v in cl.role_bloch(role))
            except Exception:
                continue
            purity = _role_purity(cl, role, bloch)
            out[f"{name}.{role}"] = codebook.de_embed(bloch, purity, mint=mint)
    return out


def _role_purity(cluster, role: str, bloch: tuple) -> float:
    """A role's belief confidence: the scalar cluster purity, or |r| for a product cluster (its own
    per-qubit purity), falling back to the Bloch radius."""
    p = getattr(cluster, "purity", None)
    if isinstance(p, (int, float)):
        return float(p)
    return min(1.0, math.sqrt(sum(c * c for c in bloch)))


def symbols_to_notes(symbols: dict, *, min_confidence: float = 0.5) -> str:
    """Render de-embedded symbols into a compact notes block (the LLM-edge text), confidence-gated.
    The membrane's expressive side — what comprehension_notes does, now driven by the codebook."""
    lines = []
    for path, sym in sorted(symbols.items()):
        if sym.label is None or sym.confidence < min_confidence:
            continue
        tag = " (new)" if sym.minted else ""
        lines.append(f"- {path}: {sym.label}{tag} ({sym.confidence:.0%})")
    return "\n".join(lines) if lines else "World state: still settling, limited data."
