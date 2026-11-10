# Breaking changes in 0.2.0

Compared with `v0.1.1` (`a047f59`). See [CHANGELOG.md](../CHANGELOG.md) for the full list.

## Breaking

- **`umwelt.consumers` is gone.** The application source that used to live under
  `umwelt.consumers` left this package (layering law). Nothing under
  `umwelt.consumers` imports any more.
- **The spec gate refuses dangling references.** `validate_spec` now fails unknown
  node/bridge kinds, unknown `reduce`, binding zone/role mismatches, faces/role_modes/
  projection on undeclared roles, `forecast_leaves` at a missing node/role, bad
  `forecast_horizons_min` (non-positive or non-increasing), `efficiency` outside
  [0, 1], a bad `rest_window`, and an exact `ignored` id that is also bound. Specs that
  were soft-accepted before may now fail; run `validate_spec` on yours.
- **Forecast confidence is tied to evidence:** `max(0, skill_vs_persistence) × purity`.
  With no baseline it is `None` (DARK) and `consume_targets()` omits the leaf.
  `DissolvedForecastSurface` no longer asserts `skill=1.0`; its confidence decays with
  horizon. Do not rely on a non-`None` confidence.
- **RK4 stability guard.** Steps whose stiffest rate × dt leaves RK4's stable region
  are substepped, and a step that still goes non-finite is refused. In-bounds defaults
  are unchanged, but dense trajectories can differ when the system is stiff. A
  non-finite Bloch vector in `collapse.py` is now DARK (no commit).

## Supported replacements

```python
from umwelt import register_role_mode, register_observe_role, register_role_emoji, register_node_icon
from umweltd import UmweltClient   # `umweltd.client` alias kept
```

Additive, non-breaking:

- #139: `UmweltClient.anchor()` returns the raw anchor observation, not calibrated
  confidence. `UmweltClient.ingest(..., event_id=...)` is keyword-only and sends a named
  exact-replay batch; a conflicting payload is an identity conflict, and the client never
  retries.
- #120: frame/1 nodes may carry an optional `model_readout` (`schema:
  local-qubit-readout/1`, local `purity`, `calibration: 'unmeasured'`), checked in
  `frame.validate`. Old frames still validate ([FRAME_SCHEMA.md](FRAME_SCHEMA.md)).
- #142: `ci-public/` stages hosted-runner CI and release workflows for the public
  export (no secrets).
- Newly supported Penduline deep imports ([STABILITY.md](STABILITY.md)):
  `umwelt.spec.schema.match_ignored`, `umwelt.projection.confession.recent`,
  `umwelt.projection.nameplates.nameplates`, `umwelt.projection.emoji.node_icon` and
  `role_poles`, `umwelt.events.replay_sensor_batches`,
  `umwelt.learning.runner.BrainRunner`.

The worker CLI and discovery contract are in [DAEMON-API.md](DAEMON-API.md). Other deep
imports still work but are internal ([STABILITY.md](STABILITY.md)).

## Not API breaks

- #127: belief query encoding fix.
- #129: atomic `worker.port` write.

## Pinning

Stay on `v0.1.1` until you are ready to take these 0.2.0 breaks, then pin a
public release tag. The current release is `v0.2.1` (license exception and
documentation fixes; no code changes relative to `v0.2.0`):

```
umwelt-engine @ git+https://github.com/AQuantumArchitect/umwelt-engine@v0.2.1
```

See [CONSUMER_PIN.md](CONSUMER_PIN.md).
