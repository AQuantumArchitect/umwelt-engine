"""Shutdown is complete only after accepted work drains and owned children exit."""
from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from umweltd import supervisor, worker
from umweltd.lifecycle import DrainingHTTPServer
from umweltd.worldstore import WorldDir


def post(url, key=None):
    headers = {"X-API-Key": key} if key else {}
    req = urllib.request.Request(url, data=b"{}", method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status, json.load(response)


@pytest.mark.parametrize("snapshot_fails", [False, True])
def test_worker_shutdown_drains_accepted_ingest_before_snapshot(tmp_path, monkeypatch, snapshot_fails):
    entered, release = threading.Event(), threading.Event()
    finished, saved, errors = [], [], []

    class Host:
        name = "probe"
        dir = WorldDir(tmp_path)

        def __init__(self, path):
            assert path == tmp_path

        def post_events(self, body):
            entered.set()
            assert release.wait(5), "test did not release accepted ingest"
            finished.append("ingest")
            return {"ok": True}

        def snapshot(self):
            assert finished == ["ingest"]
            if snapshot_fails:
                raise RuntimeError("injected snapshot failure")
            saved.append("snapshot")

    monkeypatch.setattr(worker, "WorldHost", Host)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    # Keep this test's orphan watcher finite; the real guard has separate tests.
    monkeypatch.setattr(worker, "_orphan_watch", lambda *args: None)
    servers = []

    def make_server(*args, **kwargs):
        server = DrainingHTTPServer(*args, **kwargs)
        servers.append(server)
        return server

    monkeypatch.setattr(worker, "DrainingHTTPServer", make_server)

    def serve():
        try:
            worker.serve(tmp_path)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        port_file = tmp_path / "worker.port"
        deadline = time.monotonic() + 5
        while not port_file.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert port_file.exists()
        base = f"http://127.0.0.1:{port_file.read_text()}"
        with ThreadPoolExecutor(1) as clients:
            ingest = clients.submit(post, base + "/events")
            try:
                assert entered.wait(5)
                assert post(base + "/shutdown")[0] == 202
                assert thread.is_alive()       # acceptance is not completion
                assert saved == []
            finally:
                release.set()
            assert ingest.result(timeout=5)[0] == 200
        thread.join(5)
        assert not thread.is_alive()
        assert not port_file.exists()
        assert saved == ([] if snapshot_fails else ["snapshot"])
        assert [str(exc) for exc in errors] == (["injected snapshot failure"] if snapshot_fails else [])
    finally:
        release.set()
        if thread.is_alive():
            for server in servers:
                server.request_shutdown()
            thread.join(5)
        assert not thread.is_alive(), "owned worker did not stop during cleanup"


def test_supervisor_shutdown_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("UMWELTD_HOME", str(tmp_path))
    sup = supervisor.Supervisor()
    monkeypatch.setattr(supervisor._Handler, "sup", sup)
    monkeypatch.setattr(supervisor._Handler, "api_key", "secret")
    server = DrainingHTTPServer(("127.0.0.1", 0), supervisor._Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as denied:
            post(base + "/shutdown")
        assert denied.value.code == 401
        assert not sup.stopping.is_set()
        assert post(base + "/shutdown", "secret")[0] == 202
        thread.join(5)
        assert not thread.is_alive()
        assert sup.stopping.is_set()
    finally:
        server.request_shutdown()
        thread.join(5)
        server.server_close()


class Proc:
    returncode = None

    def __init__(self, *, exitcode=0, hang=False):
        self.exitcode, self.hang = exitcode, hang
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if self.hang and not self.terminated:
            raise subprocess.TimeoutExpired("owned worker", timeout)
        self.returncode = -15 if self.terminated else self.exitcode
        return self.returncode

    def terminate(self):
        self.terminated = True


@pytest.mark.parametrize("accepted,exitcode,hang,expected", [
    (True, 0, False, "graceful"), (True, 1, False, "failed"),
    (True, 0, True, "forced"), (False, 0, False, "forced"),
])
def test_stop_reports_forced_and_failed_snapshot_honestly(monkeypatch, accepted, exitcode, hang, expected):
    sup = supervisor.Supervisor()
    proc = Proc(exitcode=exitcode, hang=hang)
    sup.procs["probe"] = proc
    sup.desired.add("probe")
    monkeypatch.setattr(sup, "proxy", lambda *args, **kwargs: (202 if accepted else 404, b"{}"))
    result = sup.stop("probe")
    assert result["shutdown"] == expected
    assert result["running"] is False and proc.poll() is not None
    assert "probe" not in sup.desired


def test_stop_all_attempts_remaining_workers_after_a_failure(monkeypatch):
    sup = supervisor.Supervisor()
    sup.procs = {"bad": Proc(), "good": Proc()}
    visited = []

    def stop(name):
        visited.append(name)
        if name == "bad":
            raise OSError("cannot terminate owned process")
        return {"shutdown": "graceful"}

    monkeypatch.setattr(sup, "stop", stop)
    with pytest.raises(RuntimeError, match="unclean worker shutdown"):
        sup.stop_all()
    assert visited == ["bad", "good"]


def test_shutdown_refuses_new_spawn_create_start_and_watchdog(monkeypatch):
    sup = supervisor.Supervisor()
    sup.procs["probe"] = Proc()
    sup.procs["probe"].returncode = 1
    sup.desired.add("probe")
    sup.begin_shutdown()
    for call in (lambda: sup.spawn("probe"), lambda: sup.start("probe"),
                 lambda: sup.create({"name": "new", "spec": "x:Y"})):
        with pytest.raises(RuntimeError, match="shutting down"):
            call()
    sup.watchdog_tick()
    assert sup.pacemaker_tick() == []
    assert sup._pulse("probe", {}) is False
    assert sup.procs["probe"].poll() == 1


def test_shutdown_waits_for_in_progress_spawn_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("UMWELTD_HOME", str(tmp_path))
    sup = supervisor.Supervisor()
    world = WorldDir(sup.worlds_root() / "probe")
    world.root.mkdir()
    entered, release = threading.Event(), threading.Event()
    proc = Proc()

    def popen(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        world.port_path.write_text("12345")
        return proc

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(sup, "proxy", lambda *args, **kwargs: (202, b"{}"))
    with ThreadPoolExecutor(2) as calls:
        spawn = calls.submit(sup.spawn, "probe")
        try:
            assert entered.wait(5)
            sup.begin_shutdown()
            cleanup = calls.submit(sup.stop_all)
        finally:
            release.set()
        assert spawn.result(timeout=5) == 12345
        cleanup.result(timeout=5)
    assert proc.poll() == 0
