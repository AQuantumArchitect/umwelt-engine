"""umweltd — the belief-field engine as a local service.

The engine stays a library; umweltd wraps it: a supervisor process manages a catalog
of WORLDS, each world runs in its OWN worker process (vocabulary registries are
process-global, so one world = one process = one vocabulary), each with an events.db
write-ahead log and periodic snapshots. Restart = load snapshot + replay the log tail
through the production ingest path — the same event-sourcing loop the offline
cassettes already prove.

The founding claim (pinned by tests/test_daemon_parity.py): the daemon adds nothing
and loses nothing — a world driven over the wire ends at the same field canon hash as
the same stream replayed library-direct.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from umweltd.client import UmweltClient

# The module exports remain for compatibility; only UmweltClient and the
# documented CLI/wire contract are stable (docs/DAEMON-API.md).
__all__ = ["worker", "supervisor", "client", "UmweltClient"]


def __getattr__(name: str):
    if name == "UmweltClient":
        from umweltd.client import UmweltClient
        globals()[name] = UmweltClient
        return UmweltClient
    raise AttributeError(f"module 'umweltd' has no attribute {name!r}")
