"""Glass door — ≤N root tiles, drill is address links, missing face → dark.

Glance surface for a world. Tiles are nameplate axes, not a control panel.
Default cap is 5 (Butler's glass-door law). Authored `DomainSpec.tiles` win;
otherwise the first non-clock children of the root, first role each.

Honesty: fog when the atlas is empty; null_dispatch is always True here —
this projection never claims an action left the process.
"""
from __future__ import annotations

from typing import Any

from umwelt.projection.nameplates import nameplates

FORMAT = "glass_door/v1"
TILES_MAX = 5


def _engine(host_or_engine: Any):
    return getattr(host_or_engine, "engine", host_or_engine)


def _declared_tiles(eng) -> list[tuple[str, str]]:
    graph = getattr(getattr(eng, "field", None), "graph", None) or getattr(eng, "graph", None)
    raw = tuple(getattr(graph, "tiles", ()) or ())
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict) and item.get("node") and item.get("role"):
            out.append((str(item["node"]), str(item["role"])))
    return out


def _derived_tiles(eng, atlas: dict) -> list[tuple[str, str]]:
    graph = getattr(getattr(eng, "field", None), "graph", None) or getattr(eng, "graph", None)
    root = getattr(graph, "root", None) if graph is not None else None
    skip_kinds = {"clock", "anchor", "synthetic"}
    chosen: list[tuple[str, str]] = []
    if root is not None:
        for child in getattr(root, "children", {}) or {}:
            node = root.children[child] if hasattr(root, "children") else None
            if node is None:
                continue
            kind = getattr(node, "kind", "")
            if kind in skip_kinds:
                continue
            roles = list(getattr(node, "roles", []) or [])
            if not roles:
                continue
            chosen.append((node.name, roles[0]))
            if len(chosen) >= TILES_MAX:
                return chosen
    # fallback: first TILES_MAX axes that aren't the root/clock
    root_name = getattr(root, "name", None)
    for axis in atlas.get("axes") or []:
        if axis.get("node") in (root_name, None) or str(axis.get("node", "")).startswith("_"):
            continue
        pair = (axis["node"], axis["role"])
        if pair not in chosen:
            chosen.append(pair)
        if len(chosen) >= TILES_MAX:
            break
    return chosen[:TILES_MAX]


def glass_door(host_or_engine: Any) -> dict:
    eng = _engine(host_or_engine)
    atlas = nameplates(eng)
    by_key = {(a["node"], a["role"]): a for a in atlas.get("axes") or []}
    declared = _declared_tiles(eng)
    keys = declared[:TILES_MAX] if declared else _derived_tiles(eng, atlas)
    tiles: list[dict] = []
    for node, role in keys:
        axis = by_key.get((node, role))
        if axis is None:
            tiles.append({
                "node": node, "role": role, "status": "dark",
                "reason": "axis not in nameplate atlas",
                "z": None, "purity": None, "pole_0": "", "pole_1": "",
            })
            continue
        tiles.append({
            "node": node,
            "role": role,
            "axis_id": axis.get("axis_id"),
            "status": axis.get("status"),
            "pole_0": axis.get("pole_0"),
            "pole_1": axis.get("pole_1"),
            "z": axis.get("z"),
            "purity": axis.get("purity"),
        })
    fog = not tiles
    return {
        "schema": FORMAT,
        "status": "dark" if fog else "live",
        "reason": "no tiles" if fog else None,
        "tiles_n": len(tiles),
        "tiles_max": TILES_MAX,
        "tiles": tiles,
        "honesty": {"fog": True, "null_dispatch": True},
    }
