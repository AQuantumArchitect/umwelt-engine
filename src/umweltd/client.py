"""A thin stdlib client for umweltd — what a harness imports to talk to the daemon."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
import urllib.request


class UmweltClient:
    """Point at a supervisor (`/worlds/<name>/...` routes) or directly at one worker
    (base_url with world=None and worker-local paths)."""

    def __init__(self, base_url: str, world: str | None = None, timeout: float = 120.0,
                 api_key: str | None = None) -> None:
        """Configure a client without connecting; timeout applies to each request.

        Omit world for a direct worker; name an admitted world for a supervisor.
        api_key, when supplied, is sent as X-API-Key. HTTP/URL errors propagate.
        """
        self.base = base_url.rstrip("/")
        self.world = world
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _url(self, path: str) -> str:
        prefix = f"/worlds/{self.world}" if self.world else ""
        return f"{self.base}{prefix}/{path.lstrip('/')}"

    def _call(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method,
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read() or b"null")

    # ── supervisor-level ────────────────────────────────────────────────────────────
    def _admin_call(self, method: str, path: str, payload: dict | None = None):
        """Supervisor-level calls address `{base}/...` directly (not the
        `/worlds/<world>/` prefix `_call` uses), so they work regardless of what
        `self.world` is set to."""
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f"{self.base}/{path.lstrip('/')}", data=data,
                                     method=method, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read() or b"null")

    def create_world(self, name: str, spec: str, vocabulary: str | None = None, **knobs: Any) -> dict:
        """Ask the supervisor to admit a world; spec and vocabulary are module refs.

        Additional manifest knobs are checked by the server, not by this client.
        """
        payload = {"name": name, "spec": spec, "vocabulary": vocabulary, **knobs}
        return self._admin_call("POST", "worlds", payload)

    def list_worlds(self) -> list:
        """Read the supervisor's world catalog, regardless of the selected world."""
        return self._admin_call("GET", "worlds")

    def stop_world(self, name: str) -> dict:
        """Ask the supervisor to drain and stop the named world."""
        return self._admin_call("POST", f"worlds/{name}/stop")

    def start_world(self, name: str) -> dict:
        """Ask the supervisor to start an already admitted world."""
        return self._admin_call("POST", f"worlds/{name}/start")

    # ── world-level ─────────────────────────────────────────────────────────────────
    def ingest(self, events: list[tuple], flush_secs: float | None = None, *,
               event_id: str | None = None) -> dict:
        """Append timestamp/sensor/value/metadata rows and ingest one batch request.

        event_id is keyword-only and is included only when supplied. The same id
        with the same rows is a named exact replay; a different payload is a
        conflicting identity. The client sends one request and does not retry.
        """
        payload: dict = {"events": [list(e) for e in events]}
        if flush_secs is not None:
            payload["flush_secs"] = flush_secs
        if event_id is not None:
            payload["event_id"] = event_id
        return self._call("POST", "events", payload)

    def health(self) -> dict:
        """Read worker health; this is not a forecast-calibration measurement."""
        return self._call("GET", "health")

    def state(self) -> dict:
        """Read the graph-state projection; nested diagnostic fields may expand."""
        return self._call("GET", "state")

    def cognifold(self) -> dict:
        """Read the worker's cognifold projection."""
        return self._call("GET", "cognifold")

    def belief(self, node: str, role: str) -> dict:
        """Read one named raw belief; its z component is not a calibrated probability."""
        return self._call("GET", "beliefs?" + urlencode({"node": node, "role": role}))

    def recommendations(self) -> list:
        """Read shadow recommendations without dispatching them."""
        return self._call("GET", "recommendations")

    def bindings(self) -> list:
        """Read the world's declared input vocabulary."""
        return self._call("GET", "bindings")

    def anchor(self) -> dict:
        """Return the current raw anchor observation unchanged.

        Raw geometry is separate from calibrated confidence. It carries no
        promise of full continuation across a snapshot. Durable field hash
        and cursor are the restart record.
        """
        return self._call("GET", "anchor")

    def snapshot(self) -> dict:
        """Persist the worker's engine snapshot and return its field canon hash."""
        return self._call("POST", "snapshot", {})
