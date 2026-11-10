# Supported service client and worker

`umweltd` ships in `umwelt-engine`. The client and worker surface below follows
[STABILITY.md](STABILITY.md). Internal engine values are not a promise of
calibration, nor a license to dispatch an actuator.

## Client

`UmweltClient(base_url, world=None, timeout=120.0, api_key=None)` constructs a
client without making a request. With `world=None` it addresses a direct worker;
with a world name it addresses that world's supervisor routes. `timeout` is a
per-request transport deadline, not an execution allowance. A supplied API key
goes in `X-API-Key`. HTTP and connection errors propagate from the standard
library; the client does not retry a possibly applied write.

```python
from umweltd import UmweltClient

direct = UmweltClient("http://127.0.0.1:9000", timeout=5.0)
selected = UmweltClient("http://127.0.0.1:9001", world="example", timeout=5.0)
```

This example only constructs clients. The addresses are examples, not discovered
services. A caller discovers the actual worker port or supplies its supervisor
address; it should not probe arbitrary ports or infer success from construction.

| Method | Result and effect |
|---|---|
| `create_world(name, spec, vocabulary=None, **knobs)` | Supervisor admission/start request; returns a JSON object. `spec` is `module:attribute`, vocabulary is `module:function`. Server validates extra manifest knobs. |
| `list_worlds()` | Supervisor catalog as a JSON list. |
| `start_world(name)` / `stop_world(name)` | Supervisor start or drain/stop request; returns a JSON object. |
| `ingest(events, flush_secs=None, *, event_id=None)` | Durable append/ingest request; rows are `(timestamp, sensor, value, metadata)` and return includes `appended` and `batches`. Keyword-only `event_id` is sent only when supplied. The same id and rows are a named exact replay; a different payload for that id is a conflicting identity. |
| `health()` | Worker health object, including `world`, `step`, `last_event_ts`, `seed_profile`. |
| `state()` / `cognifold()` | Read-only JSON object projections; nested diagnostics are extensible. |
| `belief(node, role)` | Raw belief object. Node and role are query identities, not URL fragments. `z` is not a calibrated probability. |
| `recommendations()` | JSON list of shadow recommendations; reading it dispatches nothing. |
| `bindings()` | JSON list of declared input bindings. |
| `anchor()` | Current raw anchor observation, returned unchanged. Raw geometry is separate from calibrated confidence, with no promise of full continuation across a snapshot. |
| `snapshot()` | Persist engine/cursor; returns `field_canon_hash`. |

Keyword-only `event_id` on `ingest` is included only when the caller supplies
it. The same id with the same rows is a named exact replay (`duplicate` true,
`appended` 0). The same id with a different payload is a conflicting identity:
the worker's HTTP error propagates from that single request. `anchor()` returns
the current anchor object unchanged. Raw geometry in that object is separate
from calibrated confidence and carries no promise of full continuation across
`snapshot()`. The restart record is the durable field hash and cursor.

Supervisor methods address the supervisor regardless of the client's selected
world. World names obey supervisor admission rules; arbitrary node/role strings
are URL-encoded by the client. Private attributes and `_call` helpers are not
supported consumer extension points. The old `umweltd.client.UmweltClient`
import remains an alias to this exact class.

## Worker CLI and discovery

The installed `umweltd-worker` entry point and
`python -m umweltd.worker --dir WORLD_DIR [--port PORT]` are equivalent. `--dir`
is required; omitted `--port` or `0` binds an available loopback port. `--help`
prints usage and exits without creating a world.

The directory contains `world.json` with `name` and `spec`, optionally
`vocabulary`, `spec_path`, `flush_secs` and `pin_rngs`. Spec/vocabulary references
execute trusted Python in the worker. This CLI is not the declarative-only
admission gate and is not a sandbox for arbitrary code. Use the supervisor's
admission path when importing generated worlds.

After loading/replaying the world and binding its listener, the worker publishes
`worker.port` atomically as a complete decimal port. A supervised caller must
associate the directory with its own live process and wait for a successful
health response. A retained file by itself does not prove that process is alive.
The worker removes its announcement on ordinary shutdown. Shutdown is a drain:
HTTP `202` acknowledges the request, not completion; wait for the owned process
to exit before restarting it. `SIGTERM` snapshots before exit where supported.

World directories, event logs and engine snapshots are owned by the worker.
Consumers should use the HTTP surface instead of editing or depending on their
internal schemas. One worker process owns one world and vocabulary. No new
scheduler, service installation or production activation is introduced here.
