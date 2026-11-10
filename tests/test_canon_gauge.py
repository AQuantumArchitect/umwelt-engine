"""Canon v2 pins — the gauge fixing under the ledger.

1. The v1 hash recipe is FROZEN (a hard-pinned hex over a hand-built canon dict) — existing
   referees and packet contracts compare v1, so no refactor may silently move it.
2. v1 is coordinate-sensitive (the documented defect: insertion/declaration order changes it);
   v2 = canon_hash over canonicalize() is invariant under both.
3. canonicalize is idempotent and permutes e2 blocks to where the role names say.
4. The invariant sheet survives relabeling and moves when a coupling actually moves —
   including the loop content (β₁, Z₂ Wilson signs) read through the link ledger.
"""
import numpy as np

from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.field_unify import (
    canon_hash, canonicalize, invariant_sheet, pack)

# the same physical brain, assigned BY ROLE so declaration order is pure bookkeeping
E1 = {"alpha": {"p": (0.1, 0.2, 0.3), "q": (-0.4, 0.0, 0.5)},
      "beta": {"u": (0.0, 0.6, -0.2), "v": (0.3, -0.3, 0.1), "w": (0.05, 0.0, 0.9)}}
HF = {"alpha": {"p": (0.0, 0.0, 0.7), "q": (0.2, 0.0, -0.1)},
      "beta": {"u": (0.0, 0.1, 0.0), "v": (0.0, 0.0, 0.4), "w": (-0.3, 0.0, 0.0)}}
ZZ = {("alpha", "p", "q"): 0.3,
      ("beta", "u", "v"): -0.25, ("beta", "v", "w"): 0.15, ("beta", "u", "w"): 0.2}
XY = {("beta", "u", "v"): (0.1, -0.1)}          # antiexchange — seeds a -1 link
M_PQ = np.arange(9, dtype=float).reshape(3, 3) / 10.0   # a distinctive (p, q) correlation


def _brain(cluster_order, role_orders):
    clusters = {}
    for name in cluster_order:
        roles = role_orders[name]
        c = CumulantCluster(name, roles, gamma=0.1, dt=0.01,
                            role_modes={r: "unitary" for r in roles})
        for r in roles:
            c.e1[c.role_index[r]] = E1[name][r]
        for r1 in roles:
            for r2 in roles:
                c.e2[c.role_index[r1], c.role_index[r2]] = np.outer(E1[name][r1], E1[name][r2])
        if name == "alpha":
            i, j = c.role_index["p"], c.role_index["q"]
            c.e2[i, j] = c.e2[i, j] + M_PQ
            c.e2[j, i] = c.e2[j, i] + M_PQ.T
        h = np.array([HF[name][r] for r in roles])
        zz = {(c.role_index[a], c.role_index[b]): v
              for (nm, a, b), v in ZZ.items() if nm == name}
        xy = {(c.role_index[a], c.role_index[b]): v
              for (nm, a, b), v in XY.items() if nm == name}
        c.set_couplings(h_fields=h, zz=zz, xy=xy)
        clusters[name] = c
    return pack(clusters)


def _canon_a():
    return _brain(["alpha", "beta"],
                  {"alpha": ["p", "q"], "beta": ["u", "v", "w"]})


def _canon_b():
    """The SAME brain, different bookkeeping: cluster insertion order and role declaration
    order both shuffled."""
    return _brain(["beta", "alpha"],
                  {"alpha": ["q", "p"], "beta": ["w", "u", "v"]})


def test_v1_recipe_is_frozen():
    """Hard pin over a hand-built canon dict — pure arrays, no cluster construction — so this
    pins the HASH RECIPE itself. If this hex moves, a refactor changed what existing referees
    (daemon parity, packet verify) compare, and that needs a coordinated migration, not a drift."""
    canon = {
        "partition": [("z", ["b", "a"], 0.1, 0.01)],
        "modes": {}, "diss_rates": {},
        "e1": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "e2": np.arange(36, dtype=float).reshape(2, 2, 3, 3) / 100.0,
        "h": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "zz": {(0, 1): 0.25}, "xy": {(0, 1): (0.5, -0.5)},
    }
    assert canon_hash(canon) == "971efb080109884c"
    assert canon_hash(canon, version=2) != canon_hash(canon)   # v2 is tagged; never confusable


def test_v1_is_coordinate_sensitive_v2_is_not():
    a, b = _canon_a(), _canon_b()
    # the documented v1 defect, kept as documentation: pure relabeling moves the v1 hash
    assert canon_hash(a) != canon_hash(b)
    # the repair: the gauge-fixed hash agrees
    assert canon_hash(a, version=2) == canon_hash(b, version=2)


def test_canonicalize_idempotent_and_blocks_land_by_role():
    a = canonicalize(_canon_a())
    aa = canonicalize(a)
    assert canon_hash(a) == canon_hash(aa)          # idempotent to the byte recipe
    assert [p[0] for p in a["partition"]] == ["alpha", "beta"]
    assert a["partition"][0][1] == ["p", "q"]        # roles sorted
    assert a["partition"][1][1] == ["u", "v", "w"]
    # e2 block for (p, q) is where the role names say, regardless of source declaration order
    b = canonicalize(_canon_b())
    gi = {}
    off = 0
    for (name, roles, _g, _dt) in b["partition"]:
        for r in roles:
            gi[(name, r)] = off
            off += 1
    i, j = gi[("alpha", "p")], gi[("alpha", "q")]
    expected = np.outer(E1["alpha"]["p"], E1["alpha"]["q"]) + M_PQ
    assert np.allclose(b["e2"][i, j], expected, atol=1e-12)
    assert np.allclose(b["e2"][j, i], expected.T, atol=1e-12)


def test_invariant_sheet_survives_relabeling():
    sa, sb = invariant_sheet(_canon_a()), invariant_sheet(_canon_b())
    assert sa == sb
    # and its content reads the real loop structure: the beta triangle (one independent
    # cycle) is frustrated by the antiexchange (u, v) link
    assert sa["beta_1"] == 1
    assert list(sa["wilson_signs"].values()) == [-1]
    assert sa["n_qubits"] == 5 and sa["n_clusters"] == 2 and sa["n_links"] == 4


def test_invariant_sheet_moves_on_physical_change():
    a = _canon_a()
    moved = dict(a)
    moved["zz"] = dict(a["zz"])
    (k, v), = [(k, v) for k, v in a["zz"].items() if abs(v - 0.3) < 1e-12]
    moved["zz"][k] = 0.31                            # a real coupling change
    sa, sm = invariant_sheet(a), invariant_sheet(moved)
    assert sa != sm
    assert sa["wilson_signs"] == sm["wilson_signs"]  # ...but no loop sign flipped
    # flipping the antiexchange link's sign IS a loop-content change
    flipped = dict(a)
    flipped["xy"] = dict(a["xy"])
    (kk, vv), = flipped["xy"].items()
    flipped["xy"][kk] = (vv[0], -vv[1])              # antiexchange -> exchange
    assert invariant_sheet(flipped)["wilson_signs"] != sa["wilson_signs"]
