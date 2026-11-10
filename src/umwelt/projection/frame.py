"""Frame projection — `frame/1`, one document per subject, read-only on ρ.

A Frame (docs/FRAME_SCHEMA.md) is the nameplate atlas with one more fact per axis:
how old the number is. Every node carries `value`, `age_s` and a derived `state` ∈
current | stale | dark, so a renderer knows which numbers to trust and can never
promote one.

Mapping from the nameplate status (projection.nameplates):

  latent  → dark      the axis never received a signal; value is null, never zero
  dark    → dark      observed, but the last reading carried η=0 / is absent
  live    → current   age_s ≤ stale_after_s
          → stale     age_s >  stale_after_s

Age is sourced from the engine's own record of the last reading per (node, role):
SensorBridge.latest_value → `_last_raw_ts[sensor_id]`, the wall-clock time at which
the freshest bound signal was ingested. When an axis is live but carries no timestamp
of its own (hydrated reload, a driver-written raw), `age_s` is null and the node
is dark. An unrelated signal cannot refresh this axis's evidence.

Nothing here writes to the field: the only reads are nameplates() (O(1) Bloch rows)
and the bridge's timestamp map.
"""
from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Any

from umwelt.projection.nameplates import nameplates

FORMAT = "frame/1"
READOUT_FORMAT = "local-qubit-readout/1"
LAW = "a value nobody measured is null and renders dark, never zero"
PRODUCER = {"name": "umwelt", "kind": "engine"}
STATES = ("current", "stale", "dark")
PRODUCER_KINDS = ("engine", "log", "sample", "instrument")

_ID_RE = re.compile(r"[^a-z0-9]+")


def _engine(host_or_engine: Any):
    return getattr(host_or_engine, "engine", host_or_engine)


def _now_epoch(now: Any) -> float:
    if now is None:
        return time.time()
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.timestamp()
    return float(now)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _snake(axis_id: str) -> str:
    s = _ID_RE.sub("_", str(axis_id).lower()).strip("_")
    return s or "axis"


def _label(node: str, role: str) -> str:
    return f"{str(node).replace('_', ' ')}: {str(role).replace('_', ' ')}"


def _axis_ts(bridge, node: str, role: str) -> float | None:
    """The engine's own wall-clock time of the freshest reading bound to (node, role)."""
    if bridge is None or not hasattr(bridge, "latest_value"):
        return None
    try:
        latest = bridge.latest_value(node, role)
    except Exception:
        return None
    if latest is None:
        return None
    try:
        ts = float(latest[1])
    except (TypeError, ValueError, IndexError):
        return None
    return ts if math.isfinite(ts) else None


def _bridge_last_ts(bridge) -> float | None:
    """Newest ingest time the bridge recorded for ANY signal — the engine-wide fallback."""
    stamps = getattr(bridge, "_last_raw_ts", None) or {}
    best: float | None = None
    for ts in stamps.values():
        try:
            f = float(ts)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and (best is None or f > best):
            best = f
    return best


def _age(now: float, ts: float | None) -> int | None:
    if ts is None:
        return None
    return int(max(0.0, now - ts))


def derive_state(value: Any, age_s: Any, threshold: Any) -> str:
    """The schema's rule, verbatim: null value or null age → dark; within threshold →
    current; past it → stale."""
    if value is None or age_s is None:
        return "dark"
    return "current" if int(age_s) <= int(threshold) else "stale"


def _node_receipts(eng, node_name: str, role: str) -> list[dict]:
    """The measurement receipts (substrate.receipts) that touched (node, role), oldest first.
    A joint measurement appears on every node it names. Empty when nothing was measured."""
    from umwelt.substrate.receipts import receipts_of
    cluster = getattr(getattr(eng, "field", None), "clusters", {}).get(node_name)
    if cluster is None:
        return []
    return [r.to_dict() for r in receipts_of(cluster)
            if r.register == role or (r.joint and role in r.registers)]


def _applied_signals(receipts, now):
    """Historical/attempt-only/no-op/future receipts cannot establish freshness."""
    return [r for r in receipts if r.get('applied') is True
            and not _validate_receipts([r], 'signal')
            and r['strength'] > 0 and r['t'] <= now]


def _local_z(receipt):
    axis = receipt['axis']
    return (not receipt.get('joint') and abs(axis[0]) < 1e-9
            and abs(axis[1]) < 1e-9 and abs(abs(axis[2])-1) < 1e-9)


def frame(host_or_engine: Any, *, subject: str, stale_after_s: int = 3600,
          now: Any = None) -> dict:
    """Export one `frame/1` document for `subject` from a running engine. Read-only on ρ."""
    eng = _engine(host_or_engine)
    t_now = _now_epoch(now)
    threshold = int(stale_after_s)
    bridge = getattr(eng, "sensor_bridge", None)

    plates = nameplates(eng)
    nodes: list[dict] = []
    seen_ids: dict[str, int] = {}
    for ax in plates.get("axes", []):
        node_name, role = str(ax.get("node", "")), str(ax.get("role", ""))
        axis_id = str(ax.get("axis_id") or f"{node_name}.{role}")
        status = str(ax.get("status", "latent"))
        z = ax.get("z")
        note = ""
        rcpts = _node_receipts(eng, node_name, role)
        signals = _applied_signals(rcpts, t_now)
        local = [r for r in signals if _local_z(r)]

        nid = _snake(axis_id)
        k = seen_ids.get(nid, 0)
        seen_ids[nid] = k + 1
        if k:
            nid = f"{nid}_{k + 1}"

        bridge_ts = _axis_ts(bridge, node_name, role) if status == 'live' else None
        local_ts = max((r['t'] for r in local), default=None)
        stamps = [s for s in (bridge_ts, local_ts) if s is not None and s <= t_now]
        if stamps and z is not None:
            ts = max(stamps)
            age_s = _age(t_now, ts)
            value: Any = float(z) if age_s is not None else None
            if age_s is None:
                note = "observed, but no timestamp recorded for this axis"
        else:
            # latent (never signalled) or dark (η=0 / absent): no number, no age.
            value, age_s = None, None
            if status == "latent":
                note = "never signalled"
            elif status == "dark":
                note = "last reading carried no efficiency"
            if signals:
                latest = max(signals, key=lambda r: (r['t'], r['order_index']))
                note = ('joint observation only; no local z readout; see receipts'
                        if latest['joint'] else 'observed another axis; no local z readout; see receipts')

        node_doc = {
            "id": nid,
            "label": _label(node_name, role),
            "value": value,
            "unit": "bloch_z",
            "state": derive_state(value, age_s, threshold),
            "age_s": age_s,
            "source": f"umwelt nameplate {axis_id}",
            "note": note,
        }
        if value is not None:
            node_doc['model_readout'] = {
                'schema': READOUT_FORMAT,
                'purity': ax.get('purity'),
                'calibration': 'unmeasured',
            }
        if rcpts:
            node_doc["receipts"] = rcpts
        if signals:
            latest = max(signals, key=lambda r: (r['t'], r['order_index']))
            signal_age = _age(t_now, latest['t'])
            node_doc['signal'] = {
                'kind': 'joint' if latest['joint'] else 'axis',
                'age_s': signal_age, 'state': derive_state(True, signal_age, threshold),
                'receipt_order_index': latest['order_index'],
                'axis': latest['axis'],
                'registers': latest['registers'] or [latest['register']],
            }
        nodes.append(node_doc)

    return {
        "schema": FORMAT,
        "subject": str(subject),
        "as_of": _iso(t_now),
        "stale_after_s": threshold,
        "law": LAW,
        "producer": dict(PRODUCER),
        "nodes": nodes,
        "edges": [],
    }


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_model_readout(n: dict, where: str) -> list[str]:
    """Optional local model geometry never asserts empirical predictive reliability."""
    w = f"{where}.model_readout"
    readout = n['model_readout']
    if not isinstance(readout, dict):
        return [f"{w} must be an object"]
    out = []
    if readout.get('schema') != READOUT_FORMAT:
        out.append(f"{w}.schema must be {READOUT_FORMAT!r}")
    if readout.get('calibration') != 'unmeasured':
        out.append(f"{w}.calibration must be 'unmeasured'; geometry is not calibration")
    if n.get('unit') != 'bloch_z':
        out.append(f"{w} requires unit 'bloch_z'")
    value = n.get('value')
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                              or not -1 <= value <= 1):
        out.append(f"{w} requires a finite Bloch z value in [-1, 1] or null")
    if 'purity' not in readout:
        out.append(f"{w}.purity is required (null when unavailable)")
    purity = readout.get('purity')
    if purity is not None:
        if (not isinstance(purity, (int, float)) or isinstance(purity, bool)
                or not .5 <= purity <= 1):
            out.append(f"{w}.purity must be finite in [0.5, 1] or null")
        if value is None or n.get('state') == 'dark':
            out.append(f"{w}.purity must be null without a dated local scalar")
    return out


def _validate_receipts(rs: Any, where: str) -> list[str]:
    """`nodes[].receipts` is optional; when present each entry must be a well-formed receipt."""
    if not isinstance(rs, list):
        return [f"{where}.receipts must be an array"]
    out: list[str] = []
    for k, r in enumerate(rs):
        w = f"{where}.receipts[{k}]"
        if not isinstance(r, dict):
            out.append(f"{w} is not an object")
            continue
        if not isinstance(r.get("register"), str) or not r.get("register"):
            out.append(f"{w}.register must be a non-empty string")
        for key in ("axis", "axis2"):
            v = r.get(key)
            if key == "axis2" and v is None:
                continue
            if (not isinstance(v, (list, tuple)) or len(v) != 3
                    or not all(isinstance(c, (int, float)) and not isinstance(c, bool)
                               and math.isfinite(c) for c in v)):
                out.append(f"{w}.{key} must be a 3-vector of finite numbers")
        for key in ("record", "strength", "t"):
            v = r.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                out.append(f"{w}.{key} must be a finite number")
        if isinstance(r.get("strength"), (int, float)) and r["strength"] < 0:
            out.append(f"{w}.strength must be >= 0")
        confidence = r.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float))
                or isinstance(confidence, bool) or not math.isfinite(confidence)):
            out.append(f"{w}.confidence must be a finite number or null")
        if not _is_int(r.get("order_index")) or r.get("order_index") < 0:
            out.append(f"{w}.order_index must be a non-negative int")
        if "joint" in r and not isinstance(r["joint"], bool):
            out.append(f"{w}.joint must be a boolean")
        if 'applied' in r and not isinstance(r['applied'], bool):
            out.append(f'{w}.applied must be a boolean')
    return out


def validate(doc: dict) -> list[str]:
    """Re-derive every node's state per docs/FRAME_SCHEMA.md and list violations.
    Empty list means the document is a valid `frame/1`."""
    out: list[str] = []
    if not isinstance(doc, dict):
        return ["frame is not an object"]
    if doc.get("schema") != FORMAT:
        out.append(f"schema must be {FORMAT!r}, got {doc.get('schema')!r}")
    if not isinstance(doc.get("subject"), str) or not doc.get("subject"):
        out.append("subject must be a non-empty string")
    as_of = doc.get("as_of")
    if not isinstance(as_of, str):
        out.append("as_of must be an ISO-8601 string")
    else:
        try:
            datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError:
            out.append(f"as_of is not ISO-8601: {as_of!r}")
    default_th = doc.get("stale_after_s")
    if not _is_int(default_th) or default_th < 0:
        out.append("stale_after_s must be a non-negative int")
    if not isinstance(doc.get("law"), str) or not doc.get("law"):
        out.append("law must be a non-empty string")
    prod = doc.get("producer")
    if not isinstance(prod, dict):
        out.append("producer must be an object")
    else:
        if not isinstance(prod.get("name"), str) or not prod.get("name"):
            out.append("producer.name must be a non-empty string")
        if prod.get("kind") not in PRODUCER_KINDS:
            out.append(f"producer.kind must be one of {PRODUCER_KINDS}, got {prod.get('kind')!r}")

    nodes = doc.get("nodes")
    if not isinstance(nodes, list):
        out.append("nodes must be an array")
        nodes = []
    ids: set[str] = set()
    for i, n in enumerate(nodes):
        where = f"nodes[{i}]"
        if not isinstance(n, dict):
            out.append(f"{where} is not an object")
            continue
        nid = n.get("id")
        if not isinstance(nid, str) or not nid:
            out.append(f"{where}.id must be a non-empty string")
        else:
            if not re.fullmatch(r"[a-z0-9_]+", nid):
                out.append(f"{where}.id must be snake_case, got {nid!r}")
            if nid in ids:
                out.append(f"{where}.id duplicates {nid!r}")
            ids.add(nid)
            where = f"nodes[{nid}]"
        if not isinstance(n.get("label"), str) or not n.get("label"):
            out.append(f"{where}.label must be a non-empty string")
        value = n.get("value")
        if value is not None and not isinstance(value, (int, float, str, bool)):
            out.append(f"{where}.value must be number|string|boolean|null")
        if isinstance(value, float) and not math.isfinite(value):
            out.append(f"{where}.value must be finite")
        age = n.get("age_s")
        if age is not None and (not _is_int(age) or age < 0):
            out.append(f"{where}.age_s must be a non-negative int or null")
        th = n.get("stale_after_s", default_th)
        if "stale_after_s" in n and (not _is_int(n["stale_after_s"]) or n["stale_after_s"] < 0):
            out.append(f"{where}.stale_after_s must be a non-negative int")
        state = n.get("state")
        if state not in STATES:
            out.append(f"{where}.state must be one of {STATES}, got {state!r}")
        elif _is_int(th) and (age is None or _is_int(age)):
            expect = derive_state(value, age, th)
            if state != expect:
                out.append(f"{where}.state is {state!r} but derives to {expect!r} "
                           f"(value={value!r}, age_s={age!r}, threshold={th})")
        if value is None and state != "dark":
            out.append(f"{where}: null value must be dark")
        if 'model_readout' in n:
            out.extend(_validate_model_readout(n, where))
        for opt in ("unit", "source", "note"):
            if opt in n and not isinstance(n[opt], str):
                out.append(f"{where}.{opt} must be a string")
        if "receipts" in n:
            out.extend(_validate_receipts(n["receipts"], where))
        if 'signal' in n:
            sig = n['signal']
            if not isinstance(sig, dict):
                out.append(f'{where}.signal must be an object')
            else:
                receipts = n.get('receipts', [])
                matches = [r for r in (receipts if isinstance(receipts, list) else []) if isinstance(r, dict)
                           and r.get('order_index') == sig.get('receipt_order_index')]
                if (len(matches) != 1 or matches[0].get('applied') is not True
                        or not isinstance(matches[0].get('strength'), (int, float))
                        or not matches[0]['strength'] > 0):
                    out.append(f'{where}.signal requires one applied receipt')
                else:
                    r = matches[0]
                    if (sig.get('kind') != ('joint' if r.get('joint') else 'axis')
                            or sig.get('axis') != r.get('axis')
                            or sig.get('registers') != (r.get('registers') or [r.get('register')])):
                        out.append(f'{where}.signal differs from its receipt')
                age = sig.get('age_s')
                if not _is_int(age) or age < 0:
                    out.append(f'{where}.signal age must be non-negative integer')
                elif _is_int(th) and sig.get('state') != derive_state(True, age, th):
                    out.append(f'{where}.signal state differs from age')

    edges = doc.get("edges", [])
    if not isinstance(edges, list):
        out.append("edges must be an array")
    else:
        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                out.append(f"edges[{i}] is not an object")
                continue
            for key in ("from", "to", "rel"):
                if not isinstance(e.get(key), str) or not e.get(key):
                    out.append(f"edges[{i}].{key} must be a non-empty string")
            if ids and (e.get("from") not in ids or e.get("to") not in ids):
                out.append(f"edges[{i}] references an unknown node id")
    return out
