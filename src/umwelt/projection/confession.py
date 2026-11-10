"""Confession join — sensed / inferred / predicted / decided / dispatched / echo.

A projection, not a second brain. For each nameplated axis it composes organs
that already exist and leaves missing pieces ABSENT (never guessed):

  observed  → last raw reading on a binding of this (node, role), with η if known
  believed  → live Bloch z + purity
  forecast  → shortest-horizon prediction + skill vs persistence (null if none)
  decided   → latest shadow/auto Action driven by this axis (null if none)
  dispatched→ False unless that Action left the process (shadow stays False)
  echo      → a later independent reading after a dispatch (null unless tagged)

Fog law: a dark/missing organ renders as null. Place-provenance cousin — a
label may only come from evidence, never from internal geometry.
"""
from __future__ import annotations

from typing import Any

from umwelt.projection.nameplates import nameplates

FORMAT = "confession/v1"


def _engine(host_or_engine: Any):
    return getattr(host_or_engine, "engine", host_or_engine)


def _forecast(eng, node: str, role: str) -> dict | None:
    fs = getattr(eng, "forecast_surface", None)
    if fs is None:
        return None
    try:
        preds = fs.predictions()
    except Exception:
        return None
    pairs = [(h, d) for (n, r, h), d in preds.items() if n == node and r == role]
    if not pairs:
        return None
    h, d = min(pairs, key=lambda hd: hd[0])
    skill_vs = d.get("skill_vs_persistence")
    return {
        "horizon_min": h,
        "z_pred": d.get("z_pred"),
        "skill": d.get("skill"),
        "skill_vs_persistence": skill_vs,
        "used": bool(skill_vs is not None and float(skill_vs) > 0.0),
    }


def _observed(eng, node: str, role: str) -> dict | None:
    bridge = getattr(eng, "sensor_bridge", None)
    if bridge is None:
        return None
    try:
        latest = bridge.latest_value(node, role)
    except Exception:
        latest = None
    if latest is None:
        return None
    raw, ts = latest
    # which binding contributed? pick the freshest matching sensor_id
    sid = None
    eta = None
    for b in getattr(bridge, "bindings", {}).values():
        if getattr(b, "node", None) != node or getattr(b, "qubit_role", None) != role:
            continue
        lr = None
        try:
            lr = bridge.latest_raw(b.sensor_id)
        except Exception:
            pass
        if lr is not None and abs(lr[1] - ts) < 1e-6:
            sid = b.sensor_id
            break
    cluster = getattr(getattr(eng, "field", None), "clusters", {}).get(node)
    if cluster is not None:
        conf_map = getattr(cluster, "_obs_confidence", None) or {}
        idx = getattr(cluster, "role_index", {}).get(role)
        if idx is not None and idx in conf_map:
            try:
                eta = float(conf_map[idx])
            except (TypeError, ValueError):
                eta = None
    return {"sensor_id": sid, "value": raw, "ts": ts, "eta": eta}


def _action_fields(rec) -> tuple:
    if isinstance(rec, dict):
        return rec.get("node"), rec.get("role"), rec.get("reason", ""), rec.get(
            "actuator_id"), rec.get("value")
    return (getattr(rec, "node", None), getattr(rec, "role", None),
            getattr(rec, "reason", ""), getattr(rec, "actuator_id", None),
            getattr(rec, "value", None))


def _decision(eng, node: str, role: str) -> dict | None:
    """Latest Action on this axis. `dispatched` is True only if OutputSurface
    actually handed it to a dispatcher (history live-flag). Shadow stays False."""
    surface = getattr(eng, "output_surface", None)
    if surface is None:
        return None
    hist = list(getattr(surface, "history", None) or ())
    for _ts, rec, live in reversed(hist):
        n, r, reason, aid, value = _action_fields(rec)
        if n == node and r == role:
            return {
                "actuator_id": aid, "reason": reason, "value": value,
                "shadow": not bool(live), "dispatched": bool(live),
            }
    recs = list(getattr(surface, "recommendations", None) or ())
    for rec in reversed(recs):
        n, r, reason, aid, value = _action_fields(rec)
        if n == node and r == role:
            return {
                "actuator_id": aid, "reason": reason, "value": value,
                "shadow": True, "dispatched": False,
            }
    return None


def recent(host_or_engine: Any, n: int = 32) -> dict:
    """Current confession join, most-recently-touched axes first. Caps at `n`."""
    eng = _engine(host_or_engine)
    atlas = nameplates(eng)
    items: list[dict] = []
    for axis in atlas.get("axes") or []:
        node, role = axis.get("node"), axis.get("role")
        if not node or not role:
            continue
        obs = _observed(eng, node, role)
        fc = _forecast(eng, node, role)
        dec = _decision(eng, node, role)
        items.append({
            "node": node,
            "role": role,
            "axis_id": axis.get("axis_id"),
            "status": axis.get("status"),
            "observed": obs,
            "believed": {"z": axis.get("z"), "purity": axis.get("purity")},
            "forecast": fc,
            "decided": dec,
            "dispatched": bool(dec and dec.get("dispatched")),
            "echo": None,
        })
    # live/dark (have evidence) before latent; otherwise keep spec order
    rank = {"live": 0, "dark": 1, "latent": 2}

    def _key(it):
        st = rank.get(it.get("status") or "latent", 9)
        ts = ((it.get("observed") or {}).get("ts")) or 0.0
        return (st, -float(ts))

    items.sort(key=_key)
    return {
        "schema": FORMAT,
        "n": min(len(items), int(n)),
        "items": items[: max(0, int(n))],
        "honesty": {"fog": True, "guesses_labels": False},
    }
