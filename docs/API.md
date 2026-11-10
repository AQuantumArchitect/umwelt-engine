# The public API

Everything a consumer should import comes from the top-level package:

```python
import umwelt
from umwelt import DomainSpec, NodeSpec, BindingSpec, GameHost
```

The names below are the whole stable surface (`umwelt.__all__`). They follow the
SemVer and deprecation rules in [STABILITY.md](STABILITY.md). Anything reached by a
deep path (`umwelt.substrate.…`, `umwelt.host.api.…`) is internal and can change in
any release, even though it still imports.

| group | names |
|---|---|
| the world as data | `DomainSpec`, `NodeSpec`, `BridgeSpec`, `BindingSpec`, `OutputSpec`, `DriverSpec`, `AxisFace` |
| the gate | `validate_spec`, `ValidationReport` |
| the engine | `build_engine`, `BeliefEngine` |
| the host face | `GameHost`, `WorldSession`, `Observation`, `Intent`, `Decision`, `Belief` |
| the wire format | `Event` |
| read-outs | `field_gauge`, `model_snapshot` |
| domain vocabulary | `register_role_mode`, `register_observe_role`, `register_role_emoji`, `register_node_icon` |
| metadata | `__version__` |

The package ships `py.typed`, so type checkers use the inline annotations.

The separate service client is `from umweltd import UmweltClient`. Its supported
methods and the worker CLI are described in [DAEMON-API.md](DAEMON-API.md).

**The examples are tested.** Each `python` block on this page runs in order, in one
namespace, under `tests/test_api_docs.py`. A block can use names defined by the
blocks above it. If you edit this page, run that test.

---

## The world as data

### `DomainSpec`, `NodeSpec`, `BridgeSpec`, `BindingSpec`, `OutputSpec`, `DriverSpec`, `AxisFace`

Frozen dataclasses that describe a world. A `NodeSpec` is a node in a
parent-referenced tree, with the roles (belief axes) it carries. A `BridgeSpec` is a
lateral edge between nodes. A `BindingSpec` maps a signal onto `(node, role)` through
a declarative normalizer. An `OutputSpec` declares how a decision leaves the field;
it is shadow by default. A `DriverSpec` is a periodic clock. An `AxisFace` gives an
axis human-readable pole names. [SPEC.md](SPEC.md) covers every field.

```python
from umwelt import (AxisFace, BindingSpec, BridgeSpec, DomainSpec, DriverSpec,
                    NodeSpec, OutputSpec)

spec = DomainSpec(
    name="tiny-world",
    nodes=(
        NodeSpec("hall", parent=None, kind="root", roles=("occupied",)),
        NodeSpec("den", parent="hall", roles=("occupied", "warmth"),
                 role_modes={"occupied": "unitary", "warmth": "dissipative"},
                 faces=(AxisFace("warmth", pole_0="cold", pole_1="warm"),)),
    ),
    bridges=(BridgeSpec("hall", "den", shared_roles=("occupied",), kind="open"),),
    bindings=(
        BindingSpec("den_motion", zone="den", role="occupied", normalizer="binary",
                    force_observe=True),
        BindingSpec("den_temp", zone="den", role="warmth",
                    normalizer={"type": "regime", "center": 21, "width": 4}),
    ),
    outputs=(OutputSpec("den_light", node="den", role="occupied"),),
    drivers=(DriverSpec("day", period_s=86400.0),),
)
assert spec.outputs[0].shadow is True     # outputs decide visibly, dispatch nothing
```

## The gate

### `validate_spec(spec_or_ref, *, require_shadow=True, exercise_rounds=3, resolve_only=False, declarative_only=False) -> ValidationReport`

Runs every gate check against a `DomainSpec` or a `"module:ATTR"` reference. It
resolves the spec, checks the schema, builds the topology, registers bindings
strictly, boots blank, drives every binding on synthetic input, and round-trips
save/load. `resolve_only=True` stops after the cheap checks.

### `ValidationReport`

The result. `ok` is the verdict, `failures()` and `warnings()` list checks,
`summary()` gives a readable report, and `to_dict()` gives JSON.

```python
from umwelt import ValidationReport, validate_spec

report = validate_spec(spec)
assert isinstance(report, ValidationReport)
assert report.ok, report.summary()
assert report.to_dict()["ok"] is True
assert validate_spec(spec, resolve_only=True).ok
```

## The engine

### `build_engine(*, spec, seed=None, population=True, dispatch=None, ...) -> BeliefEngine`

Constructs and wires a `BeliefEngine` from a spec. `spec` can be a `DomainSpec`, a
`"module:ATTR"` string, or omitted to read `UMWELT_SPEC`; with no spec at all it
raises `ValueError`. The engine boots blank: maximum entropy, located nowhere.
`seed` makes a world a function of its stream. With `dispatch=None`, nothing leaves
the process.

### `BeliefEngine`

Exported for type hints and `isinstance`. Its stable members are `ingest(...)`,
`save(path)`, `load(path)`, `field_canon_hash()` and the `field` attribute (the
belief field, read-only by convention). Other attributes are internal.

```python
from datetime import datetime, timedelta, timezone
from umwelt import BeliefEngine, build_engine

engine = build_engine(spec=spec, seed=7)
assert isinstance(engine, BeliefEngine)

t = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
for i in range(5):
    engine.ingest(sensor_readings={"den_motion": 1.0, "den_temp": 18.2},
                  now=t + timedelta(seconds=10 * i))

h0 = engine.field_canon_hash()
engine.save(str(tmp / "engine.pkl"))          # `tmp` is a scratch directory
twin = build_engine(spec=spec, seed=7)
twin.load(str(tmp / "engine.pkl"))
assert twin.field_canon_hash() == h0          # save/load keeps the field byte-for-byte
```

## The host face

### `GameHost`

One world, one mind, plain types at the boundary. `register_world(spec, ...)`
returns the host. After that you have `observe(observer_id, channel, value,
confidence=1.0)`, `observe_many(...)`, `intend(actor_id, intent)`, `beliefs(...)`,
`belief_value(node, role)`, `step()`, `step_turn(n)`, `save` / `load`,
`field_canon_hash()` and `snapshot()`. A confidence (η) of 0 is a no-op.

### `Observation`, `Intent`, `Decision`, `Belief`

Frozen dataclasses for the boundary types. An `Observation` is one reading for one
mind. An `Intent` asks to act through a declared output and is shadow by default.
A `Decision` is the engine's answer; `dispatched` is True only when a live command
left the process. A `Belief` is a calibrated `value` with a `confidence`.

```python
from umwelt import Belief, Decision, GameHost, Intent

host = GameHost().register_world(spec)
for _ in range(5):
    host.observe("me", "den_motion", 1.0, confidence=1.0)
    host.step()

b = host.belief_value("den", "occupied")
assert isinstance(b, Belief) and 0.0 <= b.value <= 1.0

d = host.intend("me", Intent(actor_id="me", name="den_light"))
assert isinstance(d, Decision)
assert d.mode == "shadow" and d.dispatched is False
assert host.live_dispatches == []
```

### `WorldSession`

N private minds over one shared ground. Each mind is its own blank `GameHost`,
which can have a channel mask. `apply_ground_event` pushes one event to the minds
that can sense it.

```python
from umwelt import Observation, WorldSession

session = WorldSession().register_world(spec)
session.add_mind("scout")
session.add_mind("blind", channel_mask={"den_temp"})   # cannot sense motion

session.observe(Observation("scout", "den_motion", 1.0))
assert session.observe_raw("blind", "den_motion", 1.0)["accepted"] is False
session.step()
assert "den.occupied" in session.beliefs("scout")
```

## The wire format

### `Event`

One timestamped reading from a named source. Every ingress adapter produces this
format, and the daemon's event log stores it.

```python
from umwelt import Event

e = Event(timestamp=t, source_device="den_motion", event_type="motion", value=1.0,
          metadata={"confidence": 0.8})
assert e.synthetic is False
```

## Read-outs

### `field_gauge(field, *, tape=None) -> dict`

The gauge state of every cluster, keyed and sorted, so that two snapshots diff
cleanly. See [GAUGE.md](GAUGE.md).

### `model_snapshot(engine) -> dict`

The legible model: learned params, clusters, and the coupling web. It only does
cheap reads. See [TRANSPARENCY.md](TRANSPARENCY.md).

```python
from umwelt import field_gauge, model_snapshot

gauge = field_gauge(engine.field)
assert "den" in gauge and list(gauge) == sorted(gauge)

snap = model_snapshot(engine)
assert {"params", "clusters", "summary"} <= set(snap)
```

## Domain vocabulary

### `register_role_mode(role, mode, *, analog=False) -> None`

Declare `unitary` or `dissipative` input for a role; an invalid mode raises
`ValueError`. Unregistered ordinary roles default to dissipative. `analog=True`
also declares continuous-value collapse. `analog=False` does not undo an earlier
analog registration. `NodeSpec.role_modes` supplies the mode declaratively when
building a spec; the function also supports vocabulary modules.

### `register_observe_role(role) -> None`

Declare out-of-band observation input instead of continuous drive. Such roles
use a unitary input channel even if a spec also declares a dissipative mode.

### `register_role_emoji(role, mapping) -> None`

Register domain-owned glyphs. `pos`, `zero` and `neg` are required; `coherent` is
optional. Missing required keys raise `ValueError`. The mapping is copied.
Glyph registration changes presentation, never confidence or an observation.

### `register_node_icon(node, icon) -> None`

Register the icon for a named node. Re-registration replaces that node's icon.

These registrations are **process-global**, not per `DomainSpec`. Perform them
before building the world's engine. A worker owns one world and one vocabulary;
use separate worker processes for incompatible domain vocabularies. They do not
expose the internal mutable registries as a supported API.

```python
from umwelt import (
    register_role_mode, register_observe_role,
    register_role_emoji, register_node_icon,
)

register_role_mode("example_level", "dissipative", analog=True)
register_observe_role("example_seen")
register_role_emoji("example_level", {"pos": "+", "zero": "=", "neg": "-"})
register_node_icon("example_node", "*")
```

## Metadata

```python
import umwelt

assert umwelt.__version__.count(".") == 2
assert "GameHost" in umwelt.__all__
```
