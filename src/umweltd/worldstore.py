"""The world directory — one world's durable state on disk.

    <home>/worlds/<name>/
        world.json      the world's manifest: spec ref, vocabulary ref, knobs
        events.db       the write-ahead log (umwelt.events schema, appended forever)
        snapshot.pkl    the engine state at the last snapshot (canonical save)
        cursor.txt      the last event timestamp INCLUDED in snapshot.pkl
        worker.port     the live worker's TCP port (written at bind, removed at exit)

Recovery uses an additive row cursor embedded in the snapshot, with timestamp-cursor
migration for older snapshots. Event batch identities and boundaries live in the same
database transaction as their rows. S3 worlds explicitly replay the full log: engine
snapshots do not yet preserve every ingress cache needed for exact continuation.
"""
from __future__ import annotations

import json
import sqlite3
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DDL = """CREATE TABLE IF NOT EXISTS events (
    timestamp TEXT NOT NULL,
    source_device TEXT NOT NULL,
    value TEXT NOT NULL,
    metadata TEXT
)"""


@dataclass
class WorldDir:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "world.json"

    @property
    def events_db(self) -> Path:
        return self.root / "events.db"

    @property
    def snapshot_path(self) -> Path:
        return self.root / "snapshot.pkl"

    @property
    def cursor_path(self) -> Path:
        return self.root / "cursor.txt"

    @property
    def port_path(self) -> Path:
        return self.root / "worker.port"

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def write_manifest(self, manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    def append_events(self, rows: list[tuple], event_id: str | None = None) -> int:
        """Append (ts_iso, sensor_id, value_str, metadata_json|None) rows to the log.
        The append happens BEFORE ingest — the log is the write-ahead truth."""
        con = sqlite3.connect(str(self.events_db))
        try:
            con.execute(DDL)
            con.execute("CREATE TABLE IF NOT EXISTS event_batches(id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, first_row INTEGER NOT NULL, last_row INTEGER NOT NULL)")
            con.execute("BEGIN IMMEDIATE")
            batch_id = "id:" + str(event_id) if event_id is not None else "anonymous:" + uuid.uuid4().hex
            digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            old = con.execute("SELECT content_hash FROM event_batches WHERE id=?", (batch_id,)).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError("event_id refers to a different event batch")
                return 0
            first = con.execute("SELECT COALESCE(MAX(rowid),0)+1 FROM events").fetchone()[0]
            con.executemany("INSERT INTO events VALUES (?,?,?,?)", rows)
            con.execute("INSERT INTO event_batches VALUES(?,?,?,?)", (batch_id, digest, first, first + len(rows) - 1))
            con.commit()
            return len(rows)
        finally:
            con.close()

    def row_cursor(self, timestamp=""):
        if not self.events_db.exists():
            return 0
        con = sqlite3.connect(str(self.events_db))
        try:
            return con.execute("SELECT COALESCE(MAX(rowid),0) FROM events WHERE timestamp<=?", (timestamp,)).fetchone()[0]
        finally:
            con.close()

    def tail(self, after_row):
        """Compatibility list view of ordered append batches."""
        from contextlib import closing
        with closing(self.iter_tail(after_row)) as batches:
            return list(batches)

    def iter_tail(self, after_row):
        """Yield one recorded batch at a time from a consistent read transaction.

        Close the iterator when stopping early. Memory is proportional to the
        largest batch/legacy gap, preserving the existing replay boundaries.
        """
        if not self.events_db.exists():
            return
        con = sqlite3.connect(str(self.events_db))
        try:
            con.execute("BEGIN")
            has_batches = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_batches'").fetchone()
            spans = con.execute("SELECT first_row,last_row FROM event_batches WHERE last_row>? ORDER BY first_row", (after_row,)) if has_batches else ()
            cursor = after_row
            for first, last in spans:
                if first > cursor + 1:
                    rows = con.execute("SELECT timestamp,source_device,value,metadata FROM events WHERE rowid>? AND rowid<? ORDER BY rowid", (cursor, first)).fetchall()
                    if rows:
                        yield first - 1, rows
                rows = con.execute("SELECT timestamp,source_device,value,metadata FROM events WHERE rowid>? AND rowid<=? ORDER BY rowid", (max(cursor, first - 1), last)).fetchall()
                if rows:
                    yield last, rows
                cursor = max(cursor, last)
            rows = con.execute("SELECT rowid,timestamp,source_device,value,metadata FROM events WHERE rowid>? ORDER BY rowid", (cursor,)).fetchall()
            if rows:
                yield rows[-1][0], [r[1:] for r in rows]
        finally:
            con.close()

    def cursor(self) -> str:
        return self.cursor_path.read_text().strip() if self.cursor_path.exists() else ""

    def write_cursor(self, ts_iso: str) -> None:
        self.cursor_path.write_text(ts_iso)
