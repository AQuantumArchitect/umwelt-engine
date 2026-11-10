"""HTTP shutdown that drains accepted handlers before the owner saves state."""
from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer


class DrainingHTTPServer(ThreadingHTTPServer):
    # ThreadingHTTPServer defaults to daemon handlers: server_close would not
    # wait for an ingest already accepted by the worker.
    daemon_threads = False
    request_timeout_s = 30.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopping = threading.Event()
        self._stop_lock = threading.Lock()

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(self.request_timeout_s)
        return sock, address

    def request_shutdown(self):
        # shutdown() waits for serve_forever, so it must run off that thread
        # even when requested by its signal handler. Repeated requests coalesce.
        with self._stop_lock:
            if not self.stopping.is_set():
                self.stopping.set()
                threading.Thread(target=self.shutdown, daemon=True).start()
