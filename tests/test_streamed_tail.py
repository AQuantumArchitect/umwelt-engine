"""Stream payloads without changing recorded replay boundaries or cursor meaning."""
from contextlib import closing
import sqlite3

import pytest

from umweltd.worldstore import DDL, WorldDir


def rows(value, count=1):
    return [("2026-11-09T00:00:00", "signal", str(value), None)] * count


def store(tmp_path):
    wd = WorldDir(tmp_path)
    wd.append_events(rows(1, 2), "a")
    wd.append_events(rows(2), "b")
    wd.append_events(rows(3, 2), "c")
    return wd


@pytest.mark.parametrize("cursor,expected", [
    (0, [(2, rows(1, 2)), (3, rows(2)), (5, rows(3, 2))]),
    (1, [(2, rows(1)), (3, rows(2)), (5, rows(3, 2))]),
    (2, [(3, rows(2)), (5, rows(3, 2))]),
    (5, []), (99, []),
])
def test_stream_and_list_preserve_batch_and_partial_cursor_contract(tmp_path, cursor, expected):
    wd = store(tmp_path)
    with closing(wd.iter_tail(cursor)) as batches:
        assert list(batches) == expected
    assert wd.tail(cursor) == expected


def test_legacy_gaps_keep_their_original_boundaries(tmp_path):
    wd = WorldDir(tmp_path)
    with closing(sqlite3.connect(wd.events_db)) as con:
        con.execute(DDL)
        con.executemany("INSERT INTO events VALUES(?,?,?,?)", rows(0, 2))
        con.commit()
    assert wd.tail(0) == [(2, rows(0, 2))]
    wd.append_events(rows(1), "new")
    with closing(sqlite3.connect(wd.events_db)) as con:
        con.executemany("INSERT INTO events VALUES(?,?,?,?)", rows(2, 2))
        con.commit()
    assert wd.tail(0) == [(2, rows(0, 2)), (3, rows(1)), (5, rows(2, 2))]


def test_absent_log_has_no_side_effect(tmp_path):
    wd = WorldDir(tmp_path)
    assert list(wd.iter_tail(0)) == [] and wd.tail(0) == []
    assert not wd.events_db.exists()


def test_iterator_reads_a_consistent_snapshot_while_new_rows_arrive(tmp_path):
    wd = store(tmp_path)
    with closing(sqlite3.connect(wd.events_db)) as con:
        con.execute("PRAGMA journal_mode=WAL")
    with closing(wd.iter_tail(0)) as batches:
        assert next(batches) == (2, rows(1, 2))
        wd.append_events(rows(4), "later")
        assert list(batches) == [(3, rows(2)), (5, rows(3, 2))]
    assert wd.tail(5) == [(6, rows(4))]


@pytest.mark.parametrize("early", [False, True])
def test_connection_closes_after_exhaustion_or_explicit_early_close(tmp_path, monkeypatch, early):
    wd = store(tmp_path)
    connection_factory = sqlite3.connect
    connections = []
    def capture(*args, **kwargs):
        con = connection_factory(*args, **kwargs)
        connections.append(con)
        return con
    monkeypatch.setattr(sqlite3, "connect", capture)
    with closing(wd.iter_tail(0)) as batches:
        next(batches)
        if not early: list(batches)
    assert connections
    for con in connections:
        with pytest.raises(sqlite3.ProgrammingError): con.execute("SELECT 1")


def test_first_batch_is_yielded_before_later_payload_queries(tmp_path, monkeypatch):
    wd = store(tmp_path)
    connection_factory = sqlite3.connect
    queries = []
    def capture(*args, **kwargs):
        con = connection_factory(*args, **kwargs)
        con.set_trace_callback(queries.append)
        return con
    monkeypatch.setattr(sqlite3, "connect", capture)
    with closing(wd.iter_tail(0)) as batches:
        assert next(batches) == (2, rows(1, 2))
        payloads = [q for q in queries if "source_device,value,metadata FROM events" in q]
        assert len(payloads) == 1
