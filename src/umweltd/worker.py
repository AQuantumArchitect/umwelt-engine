"""The world worker — one engine, one process, one vocabulary, one HTTP surface.

Boot sequence (the event-sourcing contract):
    1. read world.json; import + call the vocabulary ref (registries are
       process-global — this process speaks exactly one domain's language)
    2. build the engine blank from the spec ref
    3. load snapshot + embedded row cursor (legacy timestamp cursor supported)
    4. replay logged append batches after the cursor; recovery_mode=log instead
       rebuilds from the complete tape with pinned RNGs (S3's explicit contract)
    5. bind an ephemeral port, write worker.port, serve

Endpoints (JSON in/out):
    GET  /health            {world, step, last_event_ts, seed_profile}
    POST /events            {"events": [[ts, sid, value, meta|null], ...],
                             "flush_secs": 30.0?} -> append to log, bucket via
                            replay_sensor_batches, ingest each batch. Batch
                            boundaries are per-request: a live pusher sends one
                            request per flush window.
    GET  /state             the canonical graph_state projection
    GET  /cognifold         the cognifold-trace export (Bloch registers + coupling edges)
    GET  /beliefs?node=&role=   one belief read (raw bloch; domains apply their own
                            read convention client-side)
    GET  /recommendations   the shadow layer — decisions that would have dispatched
    GET  /bindings          the declared signal vocabulary (what a client may ingest)
    GET  /nameplates        one face per node.role (poles, live/latent/dark, z)
    GET  /frame?subject=&stale_after_s=   the frame/1 document (docs/FRAME_SCHEMA.md)
    GET  /confession        sensed → believed → forecast → decided join
    GET  /glass_door        ≤5 glance tiles (fog when empty)
    POST /snapshot          save engine + cursor; returns {"field_canon_hash": ...}
    POST /shutdown          accept drain + snapshot + exit (202 is not completion)

SIGTERM snapshots before exit. Every engine touch is serialized by one lock.

world.json knobs: {"name", "spec": "module:ATTR", "vocabulary": "module:function"?,
"flush_secs": 30.0?, "pin_rngs": false?, "spec_path": "/abs/dir" | [dirs]?} —
pin_rngs seeds the process RNGs at boot (the parity proof's determinism switch; live
worlds leave it off). spec_path dirs are prepended to sys.path before the spec ref
imports, so a world authored OUTSIDE the installed packages (a forge workspace)
boots and event-source-recovers identically. Trust model: spec_path is arbitrary
code execution by design — exactly the trust already granted to the spec ref itself
(both import a module into this process).
"""
from __future__ import annotations

import argparse
from contextlib import closing
import importlib
import json
import pickle
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from umweltd.jsonutil import jsonable
from umweltd.worldstore import WorldDir
from umweltd.lifecycle import DrainingHTTPServer

logger = logging.getLogger("umweltd.worker")

FLUSH_SECS_DEFAULT = 30.0

# --- orphan reaping -------------------------------------------------------
# A worker outlives its supervisor whenever the supervisor is hard-killed:
# TerminateProcess (what Popen.terminate() does on Windows) skips the
# supervisor's `finally: sup.stop(name)` entirely, so nothing reaps the
# children. The old supervisor smoke test did exactly that and leaked workers
# against retained temporary worlds. Normal shutdown now uses HTTP to drain and
# snapshot owned workers, but the deleted-directory guard remains a last resort
# for a hard-killed parent.
#
# Watching the parent PID is the obvious guard and the wrong one here: the
# worker's parent is a launcher shim, not the supervisor. The world DIRECTORY
# is the honest liveness signal -- a worker whose world no longer exists has
# nothing left to serve, whoever killed it. Two consecutive misses, because
# these directories live under Syncthing: a peer's delete should be obeyed,
# a single stat blip should not.
ORPHAN_CHECK_S = 30.0
ORPHAN_STRIKES = 2


def _call_ref(ref: str) -> None:
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        # The supervisor's front-door gate refuses this shape with a 400 before a
        # worker ever spawns; this mirror keeps a directly-booted worker's death
        # note just as precise (the log is all a worker has).
        raise ValueError(
            f"vocabulary ref {ref!r} must be 'module:function' — a bare module "
            f"name gives the worker nothing to call")
    fn = getattr(importlib.import_module(module_name), attr)
    fn()


def _make_webhook_dispatch(url: str):
    """The app-owned transport, as a webhook: every AUTO (non-shadow) Action POSTs to
    `url` as JSON. Shadow stays the law — a spec output dispatches nothing until the
    domain flips it; this is merely where the flipped ones go. Failures are logged,
    never fatal (a dead sink must not kill the world)."""
    import urllib.request

    def _dispatch(action) -> None:
        try:
            req = urllib.request.Request(
                url, data=json.dumps(jsonable(action)).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            logger.warning("webhook dispatch failed: %r", exc)
    return _dispatch


class WorldHost:
    """Owns the engine + the world dir; every public method is lock-serialized."""

    def __init__(self, world_dir: Path):
        self.dir = WorldDir(Path(world_dir))
        self.manifest = self.dir.manifest()
        self.lock = threading.Lock()
        self.last_ts: str = ""
        self.flush_secs = float(self.manifest.get("flush_secs", FLUSH_SECS_DEFAULT))

        # spec_path first: the manifest is the event-sourced world identity, so a
        # generated spec module's home rides in world.json and survives every respawn
        # (watchdog, /start, supervisor restart) with no environment to reconstruct.
        import sys
        raw = self.manifest.get("spec_path")
        for p in ([raw] if isinstance(raw, str) else (raw or [])):
            resolved = str(Path(p).expanduser().resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

        if self.manifest.get("pin_rngs"):
            import random

            import numpy as np
            random.seed(1234)
            np.random.seed(1234)

        if self.manifest.get("vocabulary"):
            _call_ref(self.manifest["vocabulary"])

        from umwelt.boot import build_engine
        dispatch = (_make_webhook_dispatch(self.manifest["webhook_url"])
                    if self.manifest.get("webhook_url") else None)
        self.engine = build_engine(spec=self.manifest["spec"], population=False,
                                   dispatch=dispatch)

        full_log = self.manifest.get("recovery_mode") == "log"
        live_recovery = self.manifest.get("ingest_mode") == "live"
        if self.dir.snapshot_path.exists() and not full_log:
            self.engine.load(str(self.dir.snapshot_path))
        self.last_ts = "" if full_log else self.dir.cursor()
        self.last_row = self.dir.row_cursor(self.last_ts)
        self._needs_recovery = False
        if self.dir.snapshot_path.exists() and not full_log:
            with self.dir.snapshot_path.open("rb") as saved:
                cursor = pickle.load(saved).get("_umweltd_cursor")
            if cursor is not None:
                self.last_row, self.last_ts = cursor["rowid"], cursor["timestamp"]
        # Preserve the world's recorded physics during catch-up. Suppress its
        # output transport instead of changing the learning gauge: changing dt,
        # persist, or actuate during recovery changes the recovered field.
        from umwelt.boot import set_role
        from umwelt.learning.context import ContextState
        # Loaded snapshots already contain their imposed gauge. Hard-setting a
        # qubit-backed parameter again removes its evolved coherence even if the
        # scalar value is unchanged, so never reimpose it on a continuation.
        if not live_recovery:
            set_role(self.engine, ContextState.replay())
        elif not self.dir.snapshot_path.exists() or full_log:
            set_role(self.engine, ContextState.replay() if self.manifest.get("gauge") == "replay" else ContextState.live())
        surface = getattr(self.engine, "output_surface", None)
        transport = surface.dispatch if surface is not None else None
        if surface is not None:
            surface.dispatch = None
        try:
            replayed = self._replay_tail()
        finally:
            if surface is not None:
                surface.dispatch = transport
        if replayed:
            logger.info("replayed %d tail batches after cursor %r", replayed,
                       self.dir.cursor())
        if not live_recovery and self.manifest.get("gauge", "live") != "replay":
            set_role(self.engine, ContextState.live())

        # Pacing baseline: a world's decay rate is in STEP units, so turning it into
        # "how long may this world go unfed" needs the step<->wall mapping, which is
        # only knowable by watching this engine actually run. Captured after the
        # replay tail so catch-up substeps do not inflate the rate.
        self._pace_wall0 = time.monotonic()
        self._pace_step0 = int(getattr(self.engine, "_step", 0))

    @property
    def name(self) -> str:
        return self.manifest.get("name", self.dir.root.name)

    def pace(self) -> dict:
        """How long this world may go unfed before its beliefs stop being evidence.

        Derived from the world's OWN physics, not a configured interval:

          gamma_diss is the dissipative rate, per step, per role. It is not a spec
          constant -- calibration channel 5 nudges `gamma_diss_{role}` from live
          tracking error, so a world that discovers its beliefs decay faster than
          declared will ask to be fed sooner, with nobody editing anything.

          tau = 1/gamma is the decay time in STEPS; steps_per_s converts it to
          seconds using this engine's measured rate rather than an assumed tick.

        The FASTEST-decaying role sets the pace (max gamma -> min tau): a world is
        as stale as its most perishable axis, not its average one.

        Why not confidence: dissipation drives the qubit toward a PURE ground state,
        so |r| RISES as a belief goes stale. A confidence-floor pacemaker would read
        a fully-decayed world as maximally trustworthy -- the exact false-certainty
        failure yurt-mood shipped for weeks. Age is the honest signal; confidence is
        the compromised one.
        """
        now_wall = time.monotonic()
        with self.lock:
            step = int(getattr(self.engine, "_step", 0))
            gammas: dict[str, float] = {}
            graph = getattr(self.engine, "graph", None)
            root_name = getattr(getattr(graph, "root", None), "name", None)
            for node in (graph.nodes_with_roles() if graph is not None else []):
                bundle = getattr(node, "param_bundle", None)
                if bundle is None:
                    continue
                # The ROOT bundle is ENGINE DNA, not domain data — param_bundles.py
                # says so explicitly, and seeds gamma_diss at 5.0 with bounds
                # (0.5, 50.0) as a field-dynamics constant every world shares. A
                # spec's per-node gamma_diss is a belief DECAY RATE on (0.0, 1.0):
                # node-self's organs sit at 0.0003 while its root reads 5.0.
                # Different quantity, different scale, four orders apart — so max()
                # over both would let engine DNA set every world's heartbeat and
                # peg it to the floor. A world with only a root falls back to the
                # ceiling, which is honest; guessing from DNA would not be.
                if node.name == root_name:
                    continue
                for role in (node.roles or ()):
                    val = None
                    for key in (f"gamma_diss_{role}", "gamma_diss"):
                        if bundle.get_param(key) is not None:
                            val = bundle.get(key)
                            break
                    if val:
                        gammas[f"{node.name}.{role}"] = float(val)

        elapsed = max(1e-9, now_wall - self._pace_wall0)
        steps = step - self._pace_step0
        steps_per_s = (steps / elapsed) if steps > 0 else None

        hottest = max(gammas.items(), key=lambda kv: kv[1], default=(None, None))
        gamma_max = hottest[1]
        tau_steps = (1.0 / gamma_max) if gamma_max else None

        # A step's WALL meaning is ingest_hold_s, and it is the only place wall time
        # enters at all — engine.ingest() divides the gap since the last batch by
        # ingest_hold_s and advances that many unit substeps, capped at
        # _wall_catchup_max. Its own comment insists this is "cadence plumbing at
        # the ingest membrane, NOT a time model", which is exactly why the honest
        # conversion has to come from here rather than from a measured step rate:
        # steps only happen on ingest, so steps-per-second measures HOW OFTEN WE
        # FEED IT, not how fast it runs. Pacing on that is circular — a world fed
        # rarely looks slow, so it gets fed even more rarely, forever.
        hold = float(getattr(self.engine, "ingest_hold_s", 0) or 0)
        cap = int(getattr(self.engine, "_wall_catchup_max", 32) or 32)

        # Two independent deadlines; the world is due at whichever comes first.
        #
        # RESOLUTION — past hold*(cap+1) seconds the catch-up saturates and the
        # excess wall time is simply dropped. The world does not decay more for
        # waiting longer; it stops being able to represent the gap at all. This is
        # a hard limit of the membrane, not a preference.
        resolution_s = (hold * (cap + 1)) if hold else None
        # DRIFT — each step relaxes the belief by about gamma toward the ground
        # state, so a caller willing to tolerate `frac` of drift may skip frac/gamma
        # steps, i.e. frac*tau_steps, i.e. that many holds of wall time.
        drift_s = (tau_steps * hold) if (tau_steps and hold) else None

        tau_s = None
        for cand in (resolution_s, drift_s):
            if cand and (tau_s is None or cand < tau_s):
                tau_s = cand

        age_s = None
        if self.last_ts:
            try:
                last = datetime.fromisoformat(self.last_ts.replace("Z", "+00:00"))
                age_s = (datetime.now(timezone.utc) - last).total_seconds()
            except ValueError:
                age_s = None

        return {"world": self.name, "step": step,
                "last_event_ts": self.last_ts or None, "age_s": age_s,
                "steps_per_s": steps_per_s, "observed_steps": steps,
                "hottest_axis": hottest[0], "gamma_max": gamma_max,
                "tau_steps": tau_steps, "tau_s": tau_s,
                "ingest_hold_s": hold or None, "wall_catchup_max": cap,
                "resolution_s": resolution_s, "drift_s": drift_s,
                "bound": ("resolution" if (resolution_s and tau_s == resolution_s)
                          else ("drift" if drift_s and tau_s == drift_s else None)),
                "gammas": gammas}

    # ── the ingest paths (both go through umwelt.events bucketing) ──────────────────
    def _ingest_rows(self, rows) -> dict:
        from umwelt.events import replay_sensor_batches
        counts = {"batches": 0, "actions": 0}
        batches = replay_sensor_batches(rows, flush_secs=self.flush_secs)
        if self.manifest.get("ingest_mode") == "live":
            for bt, readings, conf, _last in batches:
                result = self.engine.ingest(sensor_readings=readings, now=bt, **({"confidence": conf} if conf else {}))
                counts["batches"] += 1
                counts["actions"] += len((result or {}).get("actions") or ())
        else:
            # Existing worlds retain the established BrainRunner contract.
            from umwelt.learning.runner import BrainRunner
            def on_batch(n, item, result):
                counts["batches"] = n
                counts["actions"] += len((result or {}).get("actions") or ())
            BrainRunner(self.engine).replay(((readings, bt, conf) for bt, readings, conf, _last in batches), on_batch=on_batch)
        if rows:
            self.last_ts = max(self.last_ts, max(r[0] for r in rows))
        return counts

    def _replay_tail(self) -> int:
        count = 0
        with closing(self.dir.iter_tail(self.last_row)) as batches:
            for last_row, rows in batches:
                count += self._ingest_rows(rows)["batches"]
                self.last_row = last_row
        return count

    # ── the API surface ──────────────────────────────────────────────────────────────
    def post_events(self, body: dict) -> dict:
        rows = [tuple(r) + (None,) * (4 - len(r)) for r in body.get("events", ())]
        for r in rows:
            datetime.fromisoformat(r[0])            # fail loudly before logging
        with self.lock:
            if self._needs_recovery:
                raise RuntimeError("interrupted ingest: restart worker to replay its durable log")
            appended = self.dir.append_events(rows, event_id=body.get("event_id"))
            counts = {"batches": 0, "actions": 0}
            try:
                # Includes a prior append whose response or processing was interrupted.
                with closing(self.dir.iter_tail(self.last_row)) as batches:
                    for last_row, pending in batches:
                        result = self._ingest_rows(pending)
                        counts = {key: counts[key] + result[key] for key in counts}
                        self.last_row = last_row
            except BaseException:
                self._needs_recovery = True
                raise
        return {"appended": appended, "duplicate": bool(body.get("event_id") is not None and not appended), **counts}

    def snapshot(self) -> dict:
        with self.lock:
            if self._needs_recovery:
                raise RuntimeError("cannot snapshot partially processed state; restart worker")
            tmp = self.dir.snapshot_path.with_suffix(".pkl.tmp")
            self.engine.save(str(tmp))
            with tmp.open("rb") as saved:
                data = pickle.load(saved)
            data["_umweltd_cursor"] = {"rowid": self.last_row, "timestamp": self.last_ts}
            with tmp.open("wb") as saved:
                pickle.dump(data, saved)
            tmp.replace(self.dir.snapshot_path)
            self.dir.write_cursor(self.last_ts)
            return {"field_canon_hash": self.engine.field_canon_hash(),
                    "cursor": self.last_ts}

    def state(self) -> dict:
        from umwelt.projection.graph_state import graph_state
        with self.lock:
            return jsonable(graph_state(self.engine))

    def anchor(self) -> dict:
        """Atomic, read-only canonical field identity and process coordinates."""
        from umwelt.projection.cognifold_trace import cognifold_trace
        with self.lock:
            trace = cognifold_trace(self.engine, world=self.name)
            value = {"status": "available", "state_hash": self.engine.field_canon_hash(),
                     "cursor": self.last_ts, "event_rowid": self.last_row, "source": "umweltd-canonical-field",
                     "berry": sum(r.get("berry_phase") or 0 for r in trace["registers"]),
                     "local": trace["registers"], "edges": trace["edges"]}
            return jsonable(value)

    def cognifold(self) -> dict:
        """The cognifold-trace export — belief registers (Bloch spherical geometry + unified
        gauge) + coupling edges, for a per-register field renderer. Cheap read, same discipline
        as /state (never engine.context())."""
        from umwelt.projection.cognifold_trace import cognifold_trace
        with self.lock:
            return jsonable(cognifold_trace(self.engine, world=self.name))

    def belief(self, node: str, role: str) -> dict:
        from umwelt.host.api import _forecast_skill
        with self.lock:
            cluster = self.engine.field.clusters.get(node)
            if cluster is None:
                raise KeyError(f"unknown node {node!r}")
            bloch = cluster.role_bloch(role)
            value, confidence = cluster.role_gauge(role)
            # learned reliability (observation trust) — O(1) direct read, None if unseen
            reliability = None
            ot = getattr(self.engine, "_obs_trust", None)
            if ot is not None:
                e = getattr(ot, "innov", {}).get((node, role))
                if e is not None:
                    reliability = round(ot._alpha_from_innov(e), 4)
            forecast_skill = _forecast_skill(self.engine, node, role)
        # `bloch`/`z` kept verbatim for backward-compat (foresight_boot reads ["z"]);
        # the gauge coordinates are additive keys that flow up to every consumer.
        return {"node": node, "role": role, "bloch": jsonable(bloch),
                "z": float(bloch[2]), "value": float(value),
                "confidence": float(confidence), "reliability": reliability,
                "forecast_skill": forecast_skill}

    def recommendations(self) -> list:
        surface = getattr(self.engine, "output_surface", None)
        recs = getattr(surface, "recommendations", None) or ()
        return jsonable(list(recs))

    def bindings(self) -> list:
        """The world's declared signal vocabulary — what a client may ingest.
        Drives the playground's push-readings panel."""
        with self.lock:
            return jsonable(self.engine.sensor_bridge.list_bindings())

    def nameplates(self) -> dict:
        from umwelt.projection.nameplates import nameplates
        with self.lock:
            return jsonable(nameplates(self.engine))

    def frame(self, subject: str | None = None, stale_after_s: int = 3600) -> dict:
        from umwelt.projection.frame import frame
        with self.lock:
            return jsonable(frame(self.engine, subject=subject or self.name,
                                  stale_after_s=int(stale_after_s)))

    def confession(self) -> dict:
        from umwelt.projection.confession import recent
        with self.lock:
            return jsonable(recent(self.engine))

    def glass_door(self) -> dict:
        from umwelt.projection.glass_door import glass_door
        with self.lock:
            return jsonable(glass_door(self.engine))

    def health(self) -> dict:
        def _size(p: Path) -> int:
            return p.stat().st_size if p.exists() else 0
        return {"world": self.name,
                "step": int(getattr(self.engine, "_step", -1)),
                "last_event_ts": self.last_ts,
                "seed_profile": getattr(self.engine, "seed_profile", None),
                "events_db_bytes": _size(self.dir.events_db),
                "snapshot_bytes": _size(self.dir.snapshot_path)}


class _Handler(BaseHTTPRequestHandler):
    host: WorldHost = None          # injected by serve()

    def log_message(self, fmt, *args):  # replaced by the access log below
        pass

    def _send(self, code: int, payload) -> None:
        self._last_status = code
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _fail(self, exc: Exception) -> None:
        if isinstance(exc, (ValueError, KeyError)):
            self._send(400, {"error": str(exc)})
            return
        # Unexpected — degrade, never crash the world, but don't leak internals
        # (paths, tracebacks) into the response; the real exception goes to the log.
        logger.error("unhandled error on %s: %r", self.path, exc)
        self._send(500, {"error": "internal error"})

    def _access_log(self, method: str, t0: float) -> None:
        logger.info(json.dumps({
            "event": "access", "world": self.host.name, "method": method,
            "path": self.path, "status": getattr(self, "_last_status", None),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }))

    def do_GET(self):
        t0 = time.time()
        self._last_status = None
        url = urlparse(self.path)
        try:
            if url.path == "/health":
                self._send(200, self.host.health())
            elif url.path == "/state":
                self._send(200, self.host.state())
            elif url.path == "/anchor":
                self._send(200, self.host.anchor())
            elif url.path == "/cognifold":
                self._send(200, self.host.cognifold())
            elif url.path == "/recommendations":
                self._send(200, self.host.recommendations())
            elif url.path == "/bindings":
                self._send(200, self.host.bindings())
            elif url.path == "/nameplates":
                self._send(200, self.host.nameplates())
            elif url.path == "/frame":
                q = parse_qs(url.query)
                self._send(200, self.host.frame(
                    subject=(q.get("subject") or [None])[0],
                    stale_after_s=int((q.get("stale_after_s") or ["3600"])[0])))
            elif url.path == "/confession":
                self._send(200, self.host.confession())
            elif url.path == "/glass_door":
                self._send(200, self.host.glass_door())
            elif url.path == "/pace":
                self._send(200, self.host.pace())
            elif url.path == "/beliefs":
                q = parse_qs(url.query)
                self._send(200, self.host.belief(q["node"][0], q["role"][0]))
            else:
                self._send(404, {"error": f"no route {url.path}"})
        except Exception as exc:                     # degrade, never crash the world
            self._fail(exc)
        finally:
            self._access_log("GET", t0)

    def do_POST(self):
        t0 = time.time()
        self._last_status = None
        url = urlparse(self.path)
        try:
            if url.path == "/shutdown":
                try:
                    self._send(202, {"status": "shutdown requested"})
                finally:
                    self.server.request_shutdown()
            elif self.server.stopping.is_set():
                self._send(503, {"error": "worker is shutting down"})
            elif url.path == "/events":
                self._send(200, self.host.post_events(self._body()))
            elif url.path == "/snapshot":
                self._send(200, self.host.snapshot())
            else:
                self._send(404, {"error": f"no route {url.path}"})
        except Exception as exc:
            self._fail(exc)
        finally:
            self._access_log("POST", t0)


def _orphan_watch(world_dir: Path, server, name: str,
                  every: float = ORPHAN_CHECK_S,
                  strikes: int = ORPHAN_STRIKES) -> None:
    """Exit when the world directory goes away. See ORPHAN_CHECK_S.

    Deliberately does NOT snapshot on the way out: the directory is gone, and
    writing one would recreate the very world that was deleted. The loud line
    is the point -- a silent exit here reads identically to a crash.
    """
    missing = 0
    while True:
        time.sleep(every)
        if world_dir.exists():
            missing = 0
            continue
        missing += 1
        logger.warning("world %r directory is gone (%d/%d) -- %s", name,
                       missing, strikes, world_dir)
        if missing >= strikes:
            logger.error("world %r ORPHANED: directory deleted, exiting without "
                         "snapshot (nothing left to serve)", name)
            server.snapshot_on_exit = False
            server.shutdown()
            return


def _publish_port(path: Path, port: int) -> None:
    # Readers use existence as readiness. Never expose write_text's empty-file
    # window; the rename is atomic because both paths share a directory.
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(str(port), encoding="ascii")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def serve(world_dir: Path, port: int = 0) -> None:
    host = WorldHost(world_dir)
    _Handler.host = host
    server = DrainingHTTPServer(("127.0.0.1", port), _Handler)
    server.snapshot_on_exit = True
    published = serving = False
    try:
        actual_port = server.server_address[1]
        _publish_port(host.dir.port_path, actual_port)
        published = True
        logger.info("world %r serving on 127.0.0.1:%d", host.name, actual_port)

        def _shutdown(signum, frame):
            threading.Thread(target=server.request_shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)
        threading.Thread(target=_orphan_watch, args=(world_dir, server, host.name),
                         daemon=True).start()
        serving = True
        server.serve_forever()
    finally:
        try:
            server.server_close()       # all accepted ingests finish first
            if serving and server.snapshot_on_exit:
                if not world_dir.exists():
                    raise RuntimeError("world directory disappeared before snapshot")
                host.snapshot()         # failure must produce a nonzero exit
        finally:
            if published:
                host.dir.port_path.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=os.environ.get("UMWELTD_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="umweltd world worker")
    ap.add_argument("--dir", required=True, help="the world directory")
    ap.add_argument("--port", type=int, default=0, help="TCP port (0 = ephemeral)")
    args = ap.parse_args()
    serve(Path(args.dir), args.port)


if __name__ == "__main__":
    main()
