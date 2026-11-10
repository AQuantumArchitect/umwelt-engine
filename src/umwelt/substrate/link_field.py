"""LINK FIELD — lattice gauge bookkeeping on an arbitrary graph (pure math, no engine imports).

Each edge (i, j) carries a U(1) phase A_ij ∈ (−π, π] with A_ji = −A_ij. The Z₂ story is
the same field restricted to {0, π} (signs u_ij = e^{iA} ∈ {+1, −1}).

The whole point, executable:

  - A LOCAL gauge transformation at node i (χ_i) shifts every incident edge,
    A_ij → A_ij + χ_i − χ_j. Half the ledger changes. Nothing physical does.
  - A CLOSED-LOOP sum W(C) = Σ_C A_ij is untouched: every χ enters twice with opposite
    signs. The Wilson loop is the survivor — the relationship around the loop was the
    information, never the individual link values.
  - On a TREE there are no loops, so gauge freedom can erase every edge phase
    (gauge_fix_tree drives all spanning-forest edges to exactly zero). Add one edge that
    closes a cycle and one number becomes ineradicable. The count of surviving numbers is
    the first Betti number β₁ = E − V + C: topology creates the places gauge-invariant
    information can live.

Closure honesty: wilson_phase REFUSES open or broken walks (returns NaN; wilson_sign
returns None). A path that does not close carries no loop invariant, and pretending
otherwise is exactly the bookkeeping error this module exists to make impossible.

This is a ledger ABOUT a coupling graph — it never feeds back into dynamics. Adapters
at the bottom read the Z₂ sign structure the engine already has (signed zz couplings;
exchange vs antiexchange xy couplings) into the honest starting ledger. Theory + the
certificate semantics: docs/GAUGE.md.
"""
from __future__ import annotations

import math
import random


def _wrap(phase: float) -> float:
    """Wrap to (−π, π]."""
    p = math.fmod(phase + math.pi, 2.0 * math.pi)
    if p <= 0.0:
        p += 2.0 * math.pi
    return p - math.pi


class LinkField:
    """U(1)/Z₂ link phases on a simple undirected graph, with gauge moves and loop reads."""

    def __init__(self):
        self._nodes: list = []                 # insertion-ordered node ids
        self._node_set: set = set()
        self._edges: list[dict] = []           # {"a", "b", "phase"}; storage orientation a→b
        self._index: dict = {}                 # a -> {b: edge_index}, both orientations
        self._frame: dict = {}                 # node -> accumulated local χ (bookkeeping)

    # -- construction ----------------------------------------------------------

    def add_node(self, i) -> None:
        if i in self._node_set:
            return
        self._node_set.add(i)
        self._nodes.append(i)
        self._index[i] = {}

    def add_edge(self, a, b, phase: float = 0.0) -> None:
        """Simple graph: one edge per pair; re-adding overwrites the phase (oriented a→b)."""
        if a == b:
            return
        self.add_node(a)
        self.add_node(b)
        existing = self._index[a].get(b)
        if existing is not None:
            e = self._edges[existing]
            e["phase"] = _wrap(phase if e["a"] == a else -phase)
            return
        self._edges.append({"a": a, "b": b, "phase": _wrap(phase)})
        idx = len(self._edges) - 1
        self._index[a][b] = idx
        self._index[b][a] = idx

    # -- reads -----------------------------------------------------------------

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def nodes(self) -> list:
        return list(self._nodes)

    def has_edge(self, a, b) -> bool:
        return a in self._index and b in self._index[a]

    def neighbors(self, i) -> list:
        return list(self._index.get(i, {}))

    def degree(self, i) -> int:
        return len(self._index.get(i, {}))

    def edge_phase(self, a, b) -> float:
        """Oriented read: A_ab = −A_ba. NaN for absent edges (there is no phase to report)."""
        if not self.has_edge(a, b):
            return math.nan
        e = self._edges[self._index[a][b]]
        return float(e["phase"]) if e["a"] == a else -float(e["phase"])

    def set_edge_sign(self, a, b, sign_value: int) -> None:
        """Z₂ write: +1 → phase 0, −1 → phase π."""
        self.add_edge(a, b, 0.0 if sign_value >= 0 else math.pi)

    def edge_sign(self, a, b) -> int | None:
        """Z₂ read of a link: which half of the circle the phase sits in. None if absent."""
        p = self.edge_phase(a, b)
        if math.isnan(p):
            return None
        return 1 if math.cos(p) >= 0.0 else -1

    def node_frame(self, i) -> float:
        """Accumulated local frame χ at a node — pure bookkeeping; physics never reads it."""
        return float(self._frame.get(i, 0.0))

    # -- gauge moves -----------------------------------------------------------

    def gauge_transform(self, node, chi: float) -> None:
        """The local move: rezero one node's convention by χ. Every incident edge shifts,
        A_ij → A_ij + χ_i − χ_j. Locally everything changed; physically nothing."""
        if node not in self._index:
            return
        self._frame[node] = _wrap(self._frame.get(node, 0.0) + chi)
        for b in self._index[node]:
            e = self._edges[self._index[node][b]]
            delta = chi if e["a"] == node else -chi
            e["phase"] = _wrap(e["phase"] + delta)

    def gauge_flip(self, node) -> None:
        """The Z₂ move: χ = π. Every incident sign flips; every closed-loop product survives."""
        self.gauge_transform(node, math.pi)

    def random_gauge(self, seed: int) -> None:
        """Scramble every local convention (red-team helper): the test any candidate
        'observable' must survive to call itself gauge-invariant."""
        rng = random.Random(seed)
        for node in list(self._nodes):
            self.gauge_transform(node, rng.uniform(-math.pi, math.pi))

    # -- loop reads (the invariants) -------------------------------------------

    def wilson_phase(self, cycle: list) -> float:
        """Φ_C = Σ A_ij around the node cycle, wrapped to (−π, π] — the gauge-invariant
        residue of the whole ledger. CLOSURE HONESTY: returns NaN unless the walk closes
        (an explicit [a, …, a] or implicit [a, …, z] with the z→a edge present) and every
        hop is an existing edge. An open path has no loop invariant."""
        walk = list(cycle)
        if len(walk) >= 2 and walk[0] == walk[-1]:
            walk = walk[:-1]
        n = len(walk)
        if n < 3:
            return math.nan
        total = 0.0
        for i in range(n):
            p = self.edge_phase(walk[i], walk[(i + 1) % n])
            if math.isnan(p):
                return math.nan
            total += p
        return _wrap(total)

    def wilson_sign(self, cycle: list) -> int | None:
        """W(C) = Π u_ij for the Z₂ game. None on open/broken walks (the refusal)."""
        p = self.wilson_phase(cycle)
        if math.isnan(p):
            return None
        return 1 if math.cos(p) >= 0.0 else -1

    def betti_1(self) -> int:
        """β₁ = E − V + C: independent cycles = places invariant information lives."""
        return len(self._edges) - len(self._nodes) + len(self._spanning_forest()["roots"])

    def fundamental_cycles(self) -> list[list]:
        """One node-cycle per non-tree edge of a BFS spanning forest. Their Wilson phases
        are a complete gauge-invariant description of the field; len(...) == betti_1()."""
        parents = self._spanning_forest()["parents"]
        out = []
        for e in self._edges:
            if self._is_tree_edge(parents, e["a"], e["b"]):
                continue
            out.append(self._splice_cycle(parents, e["a"], e["b"]))
        return out

    def gauge_fix_tree(self) -> dict:
        """The tree lesson, executed: choose χ by BFS so every spanning-forest edge becomes
        exactly 0 (χ_root = 0; χ_child = χ_parent + A_parent→child, all read from the
        pre-transformation field). What cannot be erased is left concentrated on the
        non-tree chords — one surviving number per fundamental cycle, equal to that
        cycle's Wilson phase. Returns {"chi": {node: χ}, "residual": [{a, b, phase}]}."""
        forest = self._spanning_forest()
        parents = forest["parents"]
        chi: dict = {}
        for root in forest["roots"]:
            chi[root] = 0.0
            queue = [root]
            while queue:
                cur = queue.pop(0)
                for b in self._index.get(cur, {}):
                    if parents.get(b) == cur and b not in chi:
                        chi[b] = _wrap(chi[cur] + self.edge_phase(cur, b))
                        queue.append(b)
        # Apply in one pass from the frozen χ table (per-node application would read
        # half-transformed links — χ is settled before anything moves).
        for e in self._edges:
            e["phase"] = _wrap(e["phase"] + chi.get(e["a"], 0.0) - chi.get(e["b"], 0.0))
        for node, c in chi.items():
            self._frame[node] = _wrap(self._frame.get(node, 0.0) + c)
        residual = [
            {"a": e["a"], "b": e["b"], "phase": float(e["phase"])}
            for e in self._edges
            if not self._is_tree_edge(parents, e["a"], e["b"])
        ]
        return {"chi": chi, "residual": residual}

    # -- internals -------------------------------------------------------------

    def _spanning_forest(self) -> dict:
        """BFS forest over insertion order: {"parents": node → parent (roots absent),
        "roots": [...]}. Deterministic for a given construction order."""
        parents: dict = {}
        visited: set = set()
        roots: list = []
        for start in self._nodes:
            if start in visited:
                continue
            roots.append(start)
            visited.add(start)
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for b in self._index.get(cur, {}):
                    if b not in visited:
                        visited.add(b)
                        parents[b] = cur
                        queue.append(b)
        return {"parents": parents, "roots": roots}

    @staticmethod
    def _is_tree_edge(parents: dict, a, b) -> bool:
        return parents.get(a) == b or parents.get(b) == a

    @staticmethod
    def _root_path(parents: dict, node) -> list:
        path = [node]
        cur = node
        while cur in parents:
            cur = parents[cur]
            path.append(cur)
        return path

    @classmethod
    def _splice_cycle(cls, parents: dict, a, b) -> list:
        """Tree path a→…→LCA→…→b; the chord b→a closes it implicitly."""
        pa = cls._root_path(parents, a)
        pb = cls._root_path(parents, b)
        in_pa = {n: i for i, n in enumerate(pa)}
        lca_i, lca_j = len(pa) - 1, len(pb) - 1
        for j, n in enumerate(pb):
            if n in in_pa:
                lca_i, lca_j = in_pa[n], j
                break
        cycle = pa[: lca_i + 1]              # a … LCA inclusive
        cycle.extend(pb[j] for j in range(lca_j - 1, -1, -1))  # LCA-child … b
        return cycle


# -- adapters: the engine's coupling dicts → the honest starting ledger ------------


def from_couplings(zz: dict, xy: dict, n: int, *, tol: float = 1e-12) -> LinkField:
    """Z₂ seed from the sign structure the engine already carries. Nodes are 0..n−1.
    An edge exists where any coupling magnitude exceeds tol (cluster dicts pre-populate
    every pair with 0.0 — zeros are placeholders, not links). Seed sign per edge:
    the xy exchange sign when present (exchange (J, J) → +1, antiexchange (J, −J) → −1 —
    the only two forms the coupling learner mints), else sign(J_zz)."""
    lf = LinkField()
    for i in range(n):
        lf.add_node(i)
    keys = set()
    if zz:
        keys.update(zz.keys())
    if xy:
        keys.update(xy.keys())
    for k in sorted(keys):
        a, b = int(k[0]), int(k[1])
        kxx, kyy = (float(xy[k][0]), float(xy[k][1])) if xy and k in xy else (0.0, 0.0)
        j = float(zz[k]) if zz and k in zz else 0.0
        if abs(kxx) + abs(kyy) > tol:
            prod = kxx * kyy
            sign = (1 if prod > 0 else -1) if abs(prod) > tol * tol \
                else (1 if kxx + kyy >= 0 else -1)
        elif abs(j) > tol:
            sign = 1 if j >= 0 else -1
        else:
            continue
        lf.set_edge_sign(a, b, sign)
    return lf


def from_canon(canon: dict, *, tol: float = 1e-12) -> LinkField:
    """Whole-web link field from a field_unify canonical dict ({e1, zz, xy, ...})."""
    n = len(canon["e1"])
    return from_couplings(canon.get("zz") or {}, canon.get("xy") or {}, n, tol=tol)


def from_bridges(bridges, *, tol: float = 1e-12) -> LinkField:
    """Z₂ links over the LATERAL web: nodes are cluster names, one link per bridge,
    sign = sign(coupling_base) — the signed coupling calibration now learns (an
    anti-correlated bridge carries a −1 link, so a frustrated bridge triangle reads
    W = −1). Bridges with a ~zero coupling mint no link. The canonical-state
    invariant sheet covers the intra-cluster/manifold web (from_canon); this is the
    live read for the bridge sector, whose couplings ride params, not the canon."""
    lf = LinkField()
    for b in bridges:
        base = float(getattr(b, "coupling_base", 0.0))
        if abs(base) <= tol:
            continue
        lf.set_edge_sign(b.source, b.target, 1 if base >= 0 else -1)
    return lf


def edge_pairs(lf: LinkField) -> list[tuple]:
    """The field's edge set in resolve_zz_pairs currency: (i, j) with i < j, sorted."""
    return sorted((e["a"], e["b"]) if e["a"] < e["b"] else (e["b"], e["a"])
                  for e in lf._edges)
