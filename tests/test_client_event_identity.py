"""Frozen consumer contract: S3 needs anchors and durable named event batches.

The builder may change client/docs and the existing API inventory, never this
file or the worker/store. A restart uses the same real durable world directory.
"""
from unittest.mock import patch
from urllib.error import HTTPError
import json
import sqlite3

import pytest

from umweltd import UmweltClient
from test_daemon_parity import world
from examples.gridworld.world import agent_walk, gridworld_spec, synthesize_rows


def test_anchor_is_a_transparent_read():
    client = UmweltClient('http://127.0.0.1:1', world='s3')
    observed = {'status': 'available', 'state_hash': 'real-hash', 'unknown': None}
    with patch.object(client, '_call', return_value=observed) as transport:
        assert client.anchor() is observed
        transport.assert_called_once_with('GET', 'anchor')


@pytest.mark.parametrize('flush', [None, 0, 30.0])
def test_named_ingest_preserves_payload_and_does_not_mutate_input(flush):
    client = UmweltClient('http://127.0.0.1:1')
    rows = [('2026-11-09T00:00:00', 'sensor', 'value', None)]
    before = list(rows)
    reply = {'appended': 1, 'duplicate': False}
    with patch.object(client, '_call', return_value=reply) as transport:
        assert client.ingest(rows, flush, event_id='receipt:one') is reply
        expected = {'events': [list(rows[0])], 'event_id': 'receipt:one'}
        if flush is not None:
            expected['flush_secs'] = flush
        transport.assert_called_once_with('POST', 'events', expected)
    assert rows == before


def test_anonymous_calls_remain_compatible():
    client = UmweltClient('http://127.0.0.1:1')
    with patch.object(client, '_call', return_value={}) as transport:
        client.ingest([], 12)
        transport.assert_called_once_with('POST', 'events', {'events': [], 'flush_secs': 12})


def test_identity_is_keyword_only():
    with pytest.raises(TypeError):
        UmweltClient('http://127.0.0.1:1').ingest([], None, 'ambiguous')


def test_transport_failure_is_not_retried_or_translated():
    client = UmweltClient('http://127.0.0.1:1')
    failure = HTTPError('http://127.0.0.1:1/events', 400, 'conflicting batch', None, None)
    with patch.object(client, '_call', side_effect=failure) as transport:
        with pytest.raises(HTTPError) as caught:
            client.ingest([], event_id='conflict')
        assert caught.value is failure
        assert transport.call_count == 1


def test_real_worker_named_batch_survives_restart_without_new_evidence(world):
    client, directory, process, spawn, env = world
    rows = synthesize_rows(gridworld_spec(), agent_walk(seed=7, days=.003), seed=7)
    assert rows
    before = client.anchor()
    first = client.ingest(rows, event_id='s3-outcome:prospective-one')
    assert first['appended'] == len(rows) and first['duplicate'] is False
    after = client.anchor()
    assert after['state_hash'] != before['state_hash']
    assert after['event_rowid'] > before['event_rowid']
    duplicate = client.ingest(rows, event_id='s3-outcome:prospective-one')
    assert duplicate['appended'] == 0 and duplicate['duplicate'] is True
    assert client.anchor() == after
    conflict = list(rows)
    conflict[0] = (*rows[0][:3], json.dumps({'conflicting': True}))
    with pytest.raises(HTTPError):
        client.ingest(conflict, event_id='s3-outcome:prospective-one')
    assert client.anchor() == after
    client.snapshot()
    process.terminate()
    process.wait(timeout=15)
    restarted, client = spawn()
    try:
        restored = client.anchor()
        # Anchor geometry is a current observation, not a snapshot-equivalence
        # promise. Only the persisted field identity/cursor must survive boot.
        for key in ('status', 'state_hash', 'cursor', 'event_rowid', 'source'):
            assert restored[key] == after[key]
        duplicate = client.ingest(rows, event_id='s3-outcome:prospective-one')
        assert duplicate['appended'] == 0 and duplicate['duplicate'] is True
        assert client.anchor() == restored
        with sqlite3.connect(directory/'events.db') as db:
            assert db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == len(rows)
            assert db.execute('SELECT COUNT(*) FROM event_batches').fetchone()[0] == 1
    finally:
        restarted.terminate()
        restarted.wait(timeout=15)
