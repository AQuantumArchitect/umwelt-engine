"""Nameplate projection — one human face per node.role axis.

Roles already *are* the qubits. This serializer is the stable *face* of an axis:
pole labels, live/latent/dark status, Bloch z + purity. Cheap O(1) reads; never
rebuilds a density matrix; never copies ρ, ticks, or forecast arrays into the
row. Join key is `{node}.{role}` — one nameplate per axis, not per instance.

Status:
  latent — this axis has never received a signal (not in touched_roles)
  live   — observed, last efficiency was not a no-op
  dark   — observed, but the last reading carried η=0 / is absent now

Pole strings are opaque. Authored `AxisFace`s win; otherwise the emoji registry's
role_poles (still domain-registered) or the engine's neutral pair. Missing
evidence renders as absence — never a guessed label from geometry.
"""
from __future__ import annotations

from typing import Any

from umwelt.projection.emoji import role_poles
from umwelt.substrate.bloch import bloch_radius, qubit_purity

FORMAT = "nameplates/v1"


def _engine(host_or_engine: Any):
    return getattr(host_or_engine, "engine", host_or_engine)


def _face_for(node, role: str) -> Any:
    for f in getattr(node, "faces", ()) or ():
        r = getattr(f, "role", None)
        if r is None and isinstance(f, dict):
            r = f.get("role")
        if r == role:
            return f
    return None


def _poles(face, role: str) -> tuple[str, str]:
    if face is not None:
        p0 = getattr(face, "pole_0", None) or (face.get("pole_0") if isinstance(face, dict) else "")
        p1 = getattr(face, "pole_1", None) or (face.get("pole_1") if isinstance(face, dict) else "")
        if p0 or p1:
            return str(p0 or ""), str(p1 or "")
    return role_poles(role)


def _axis_id(face, node_name: str, role: str) -> str:
    if face is not None:
        aid = getattr(face, "axis_id", None) or (face.get("axis_id") if isinstance(face, dict) else "")
        if aid:
            return str(aid)
    return f"{node_name}.{role}"


def _status(engine, node_name: str, role: str) -> str:
    touched = getattr(getattr(engine, "sensor_bridge", None), "touched_roles", None) or set()
    if (node_name, role) not in touched:
        return "latent"
    # last observation confidence, if the cluster recorded it
    cluster = getattr(getattr(engine, "field", None), "clusters", {}).get(node_name)
    if cluster is not None:
        conf_map = getattr(cluster, "_obs_confidence", None) or {}
        idx = getattr(cluster, "role_index", {}).get(role)
        if idx is not None and idx in conf_map:
            try:
                if float(conf_map[idx]) <= 0.0:
                    return "dark"
            except (TypeError, ValueError):
                pass
    latest = None
    bridge = getattr(engine, "sensor_bridge", None)
    if bridge is not None and hasattr(bridge, "latest_value"):
        try:
            latest = bridge.latest_value(node_name, role)
        except Exception:
            latest = None
    if latest is None and (node_name, role) in touched:
        # touched historically but no raw cached (reload) — still live, not invented dark
        return "live"
    return "live"


def _bloch_row(engine, node_name: str, role: str) -> dict:
    cluster = getattr(getattr(engine, "field", None), "clusters", {}).get(node_name)
    if cluster is None:
        return {"z": None, "purity": None, "r_bloch": None}
    try:
        b = cluster.role_bloch(role)
        x, y, z = float(b[0]), float(b[1]), float(b[2])
    except Exception:
        return {"z": None, "purity": None, "r_bloch": None}
    r = bloch_radius(x, y, z)
    try:
        pur = float(qubit_purity((x, y, z)))
    except Exception:
        pur = None
    return {"z": round(z, 4), "purity": (round(pur, 4) if pur is not None else None),
            "r_bloch": round(float(r), 4)}


def nameplates(host_or_engine: Any) -> dict:
    """Read-only nameplate atlas. Degrades to an empty envelope on a broken engine."""
    eng = _engine(host_or_engine)
    graph = getattr(getattr(eng, "field", None), "graph", None) or getattr(eng, "graph", None)
    rows: list[dict] = []
    try:
        walk = list(graph.all_nodes()) if graph is not None else []
    except Exception:
        walk = []
    for node in walk:
        name = getattr(node, "name", "")
        if not name or str(name).startswith("_"):
            continue
        for role in list(getattr(node, "roles", []) or []):
            face = _face_for(node, role)
            poles = _poles(face, role)
            bloch = _bloch_row(eng, name, role)
            rows.append({
                "node": name,
                "role": role,
                "axis_id": _axis_id(face, name, role),
                "pole_0": poles[0],
                "pole_1": poles[1],
                "status": _status(eng, name, role),
                **bloch,
            })
    return {
        "schema": FORMAT,
        "n": len(rows),
        "axes": rows,
        "honesty": {"fog": True, "copies_rho": False},
    }
