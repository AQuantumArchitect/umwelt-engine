import json
import uuid

import pytest

from umweltd.worker import WorldHost
from umweltd.worldstore import WorldDir


# A neutral in-repo spec: one root, one component, one observed role. The tests only need a
# single binding whose sensor id they can post to.
_SPEC_MODULE = """\
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

SPEC = DomainSpec(
    name="idempotency_probe",
    nodes=(
        NodeSpec("root", parent=None, kind="root", roles=("skill",)),
        NodeSpec("unit", parent="root", kind="component", roles=("readiness",),
                 role_modes={"readiness": "dissipative"}),
    ),
    bindings=(
        BindingSpec("unit_readiness", "unit", "readiness",
                    normalizer={"type": "range", "lo": 0, "hi": 1}, force_observe=True),
    ),
)
"""


@pytest.fixture
def host(tmp_path):
    module = "idempotency_world_" + uuid.uuid4().hex
    (tmp_path / (module + ".py")).write_text(_SPEC_MODULE)
    directory = WorldDir(tmp_path / "world")
    directory.write_manifest({"name": "test", "spec": module + ":SPEC", "spec_path": [str(tmp_path)],
                              "pin_rngs": True, "ingest_mode": "live", "recovery_mode": "log"})
    return WorldHost(directory.root)


def rows(host, timestamp="2026-10-11T12:00:00+00:00", value=.8):
    sensor = host.bindings()[0]["sensor_id"]
    return [[timestamp, sensor, value, json.dumps({"bridge": "forecast", "confidence": .98})]]


def test_duplicate_batch_does_not_change_field_and_conflict_rejected(host):
    events = rows(host)
    assert host.post_events({"event_id": "input", "events": events})["appended"] == 1
    before = host.engine.field_canon_hash()
    repeat = host.post_events({"event_id": "input", "events": events})
    assert repeat["duplicate"] and repeat["batches"] == 0
    assert before == host.engine.field_canon_hash()
    with pytest.raises(ValueError, match="different event batch"):
        host.post_events({"event_id": "input", "events": rows(host, value=.2)})
    assert before == host.engine.field_canon_hash()


def test_snapshot_restart_and_same_timestamp_new_batch(host):
    first = rows(host)
    host.post_events({"event_id": "one", "events": first})
    host.snapshot()
    # A distinct batch at the same timestamp must survive the old timestamp cursor.
    second = rows(host, value=.1)
    host.post_events({"event_id": "two", "events": second})
    expected = host.engine.field_canon_hash()
    reopened = WorldHost(host.dir.root)
    assert reopened.engine.field_canon_hash() == expected
    assert reopened.post_events({"event_id": "two", "events": second})["batches"] == 0
    assert reopened.engine.field_canon_hash() == expected
    reopened.snapshot()
    restarted = WorldHost(host.dir.root)
    assert restarted.post_events({"event_id": "one", "events": first})["batches"] == 0
    assert restarted.engine.field_canon_hash() == expected


def test_interrupted_append_tail_is_applied_once(host):
    events = rows(host)
    host.dir.append_events([tuple(row) for row in events], event_id="interrupted")
    result = host.post_events({"event_id": "interrupted", "events": events})
    assert result["duplicate"] and result["batches"] == 1
    before = host.engine.field_canon_hash()
    assert host.post_events({"event_id": "interrupted", "events": events})["batches"] == 0
    assert host.engine.field_canon_hash() == before


def test_partial_ingest_refuses_more_mutation_until_restart(host, monkeypatch):
    events = rows(host)
    monkeypatch.setattr(host, "_ingest_rows", lambda batch: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError):
        host.post_events({"event_id": "partial", "events": events})
    with pytest.raises(RuntimeError, match="restart worker"):
        host.post_events({"event_id": "partial", "events": events})
    with pytest.raises(RuntimeError, match="partially processed"):
        host.snapshot()
    restarted = WorldHost(host.dir.root)
    assert restarted.post_events({"event_id": "partial", "events": events})["batches"] == 0


def test_legacy_snapshot_duplicate_never_reapplies(host):
    manifest = host.dir.manifest()
    manifest.pop("recovery_mode")
    manifest.pop("ingest_mode")
    host.dir.write_manifest(manifest)
    legacy = WorldHost(host.dir.root)
    events = rows(legacy)
    legacy.post_events({"event_id": "legacy", "events": events})
    legacy.snapshot()
    restarted = WorldHost(host.dir.root)
    before = restarted.engine.field_canon_hash()
    assert restarted.post_events({"event_id": "legacy", "events": events})["batches"] == 0
    assert before == restarted.engine.field_canon_hash()


def test_worker_restart_and_append_do_not_materialize_the_list_tail(host, monkeypatch):
    def forbidden(*args):
        raise AssertionError("worker must consume streamed batches")
    monkeypatch.setattr(WorldDir, "tail", forbidden)
    events = rows(host)
    host.post_events({"event_id": "streamed", "events": events})
    expected = host.engine.field_canon_hash()
    restarted = WorldHost(host.dir.root)
    assert restarted.engine.field_canon_hash() == expected
    assert restarted.post_events({"event_id": "streamed", "events": events})["batches"] == 0


def test_failed_worker_ingest_closes_the_stream_before_refusing_retry(host, monkeypatch):
    original = host.dir.iter_tail
    closed = []
    def watched(cursor):
        try:
            yield from original(cursor)
        finally:
            closed.append(True)
    monkeypatch.setattr(host.dir, "iter_tail", watched)
    def fail(rows): raise RuntimeError("interrupted")
    monkeypatch.setattr(host, "_ingest_rows", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        host.post_events({"event_id": "failed-stream", "events": rows(host)})
    assert closed == [True]
    with pytest.raises(RuntimeError, match="restart worker"):
        host.post_events({"event_id": "failed-stream", "events": rows(host)})
