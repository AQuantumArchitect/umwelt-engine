"""A consumer's node/role identity must survive the actual HTTP query boundary."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import parse_qs, urlsplit

import pytest

from umweltd.client import UmweltClient


@pytest.fixture
def endpoint():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            row = {"path": url.path, "query": parse_qs(url.query, keep_blank_values=True)}
            seen.append(row)
            raw = json.dumps(row).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("world", [None, "fixture-world"])
@pytest.mark.parametrize("node,role", [
    ("plain-node", "plain-role"), ("room alpha", "quality score"),
    ("node&role=other", "role&node=other"), ("node#fragment", "g#fragment"),
    ("node+literal", "g+literal"), ("node%2F", "g%26"), ("🐜 room", "σz / 温度"),
])
def test_belief_query_preserves_identity_over_http(endpoint, world, node, role):
    base, seen = endpoint
    result = UmweltClient(base, world=world).belief(node, role)
    expected = {"path": (f"/worlds/{world}" if world else "") + "/beliefs",
                "query": {"node": [node], "role": [role]}}
    assert result == expected
    assert seen == [expected]
