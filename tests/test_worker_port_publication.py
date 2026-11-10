"""Discovery must never observe an empty/partial listening-port announcement."""
from pathlib import Path
import threading
import time

import pytest

from umweltd import worker
from umweltd.lifecycle import DrainingHTTPServer
from umweltd.worldstore import WorldDir


@pytest.mark.parametrize("write_fails", [False, True])
def test_port_is_published_only_after_complete_write(tmp_path, monkeypatch, write_fails):
    opened, release, serving = threading.Event(), threading.Event(), threading.Event()
    servers, errors, snapshots, advertised = [], [], [], []
    original_write = Path.write_text

    class Host:
        name = "publication-probe"
        dir = WorldDir(tmp_path)

        def __init__(self, path):
            assert path == tmp_path

        def snapshot(self):
            snapshots.append(True)

    def pause_write(path, data, *args, **kwargs):
        if path.parent == tmp_path and path.name.startswith("worker.port"):
            original_write(path, "", *args, **kwargs)
            opened.set()
            assert release.wait(5), "test did not release port publication"
            if write_fails:
                raise OSError("injected port write failure")
        return original_write(path, data, *args, **kwargs)

    def make_server(*args, **kwargs):
        server = DrainingHTTPServer(*args, **kwargs)
        original_serve = server.serve_forever

        def serve_forever():
            advertised.append(Host.dir.port_path.read_text())
            serving.set()
            return original_serve()

        server.serve_forever = serve_forever
        servers.append(server)
        return server

    def serve():
        try:
            worker.serve(tmp_path)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(worker, "WorldHost", Host)
    monkeypatch.setattr(worker, "DrainingHTTPServer", make_server)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "_orphan_watch", lambda *args: None)
    monkeypatch.setattr(Path, "write_text", pause_write)
    thread = threading.Thread(target=serve)
    thread.start()
    try:
        assert opened.wait(5)
        assert not Host.dir.port_path.exists(), "an empty port was advertised"
    finally:
        release.set()
        deadline = time.monotonic() + 5
        while thread.is_alive() and not serving.is_set() and time.monotonic() < deadline:
            time.sleep(.001)
        if serving.is_set():
            for server in servers:
                server.request_shutdown()
        thread.join(5)
        # Also close explicitly on the unfixed startup-failure path so this
        # regression itself cannot leak sockets into the rest of the suite.
        closed = [server.socket.fileno() == -1 for server in servers]
        for server in servers:
            server.server_close()
        assert not thread.is_alive()
    assert closed == [True]
    assert not list(tmp_path.glob("worker.port*"))
    assert snapshots == ([] if write_fails else [True])
    assert advertised == ([] if write_fails else [str(servers[0].server_address[1])])
    assert [str(exc) for exc in errors] == (["injected port write failure"] if write_fails else [])


def test_failed_replace_preserves_previous_announcement(tmp_path, monkeypatch):
    port = tmp_path / "worker.port"
    port.write_text("54321")

    def fail_replace(*args):
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="atomic replace failure"):
        worker._publish_port(port, 12345)
    assert port.read_text() == "54321"
    assert list(tmp_path.iterdir()) == [port]
