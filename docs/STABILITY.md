# Stability: versioning and deprecation

This page says what downstream consumers can rely on across releases of
`umwelt-engine`, and what they cannot.

## What is public

The engine's public API is exactly `umwelt.__all__`. [API.md](API.md) documents each name.
`tests/test_public_api.py` pins the list, so adding or removing a name fails CI until
the snapshot, API.md and CHANGELOG.md are updated together.

For each public name, the contract covers:

- its import path, `from umwelt import X`;
- the names, order and defaults of its parameters, and its return type;
- for dataclasses, the field names, their order and their defaults;
- for `GameHost`, `WorldSession` and `ValidationReport`, every public method and
  property;
- for `BeliefEngine`, only `ingest`, `save`, `load`, `field_canon_hash` and `field`.

Everything else is **internal**. That includes any deep import path
(`umwelt.substrate.…`, `umwelt.host.api.GameHost`, `umwelt.spec.schema.DomainSpec`),
any `_`-prefixed name, and any engine attribute not listed above. Internal code can
change in any release. Deep paths keep working because the facade re-exports the
defining object rather than a copy, but they carry no promise. New code should
import from `umwelt`.

The numbers the engine produces (a `Belief.value`, a field hash) are a function of
the spec, the stream and the seed. A release can change those numbers when a
physics or learning fix lands. Such a change is a behaviour change, not an API
break, and it is always listed in CHANGELOG.md. Consumers that replay against a
recorded hash should pin an exact version ([CONSUMER_PIN.md](CONSUMER_PIN.md)).

## Versioning (SemVer)

`umwelt-engine` follows [SemVer 2.0](https://semver.org/).

**While the version is 0.x (now):**

- A **minor** bump (`0.1 → 0.2`) may remove or change a public name, but only after
  that name has spent at least one minor release deprecated (see below).
- A **patch** bump (`0.1.1 → 0.1.2`) never removes or breaks a public name. It may
  add names, fix bugs and change internals.
- Pin an exact version or SHA, as [CONSUMER_PIN.md](CONSUMER_PIN.md) describes.

**From 1.0:**

- A **major** bump is the only release that may remove or break a public name.
- A **minor** bump may add names and deprecate existing ones.
- A **patch** bump fixes bugs only.

Version bumps land in their own release PR, never in a feature PR.

## Deprecation

When a public name is renamed or moved:

1. The new name is added to `umwelt.__all__`.
2. The old name goes into `umwelt._MOVED` as `old → (new, removal_version)`.
   Accessing `umwelt.<old>` then emits a `DeprecationWarning` that names the
   replacement and the release that removes it, and returns the new object.
   `tests/test_public_api.py` tests this mechanism.
3. The CHANGELOG lists the deprecation under the release that introduces it.
4. The old name is removed no earlier than the next minor release on 0.x, or the
   next major release from 1.0. The removal is listed in the CHANGELOG too.

A changed signature works the same way. The old form keeps working with a
`DeprecationWarning` for the same period. For example, a renamed keyword argument
keeps accepting the old keyword.

Internal names are not deprecated. They change without notice. If an internal name
turns out to be load-bearing for a consumer, it should be promoted into the facade on
purpose (with docs, types and a snapshot update) rather than depended on quietly.

## Deep imports consumers use today

The #66 survey looked at the consumer repos visible to the builder. Deep imports
that now have a public name:

| deep import | use instead |
|---|---|
| `umwelt.spec.schema.{DomainSpec, NodeSpec, BridgeSpec, BindingSpec, OutputSpec, DriverSpec, AxisFace}` | `umwelt.<same>` |
| `umwelt.spec.validate.validate_spec` | `umwelt.validate_spec` |
| `umwelt.boot.build_engine` | `umwelt.build_engine` |
| `umwelt.host.api.GameHost`, `umwelt.host.session.WorldSession`, `umwelt.host.api.Intent` | `umwelt.<same>` |
| `umwelt.projection.gauge.field_gauge`, `umwelt.projection.transparency.model_snapshot` | `umwelt.<same>` |
| `umwelt.spec.roles.register_role_mode`, `register_observe_role` | `umwelt.<same>` |
| `umwelt.projection.emoji.register_role_emoji`, `register_node_icon` | `umwelt.<same>` |

Deep imports that were seen but are **not** public yet: `umwelt.substrate.*`
(`CumulantCluster`, `Chart`, `BlochGeometricPhase`, `berry_holonomy`),
`umwelt.clocks.*` (`BerryTape`, `score_breathing`), `umwelt.host.agency_loop`
(`AgencyLoop`, `SubRoutine`), the other names in `umwelt.spec.roles` and
`umwelt.projection.emoji` (other than `node_icon` and `role_poles`, below), and `umwelt.foresight.encounter`.
They work today but are internal. Each one is a candidate for a later, deliberate
promotion.

The service is a separate Python package in the same distribution. Its supported
client is `from umweltd import UmweltClient`; the existing
`from umweltd.client import UmweltClient` path remains a supported alias to the
same class. The constructor and public method names, parameter order/defaults,
and return types documented in [DAEMON-API.md](DAEMON-API.md) follow this same
SemVer/deprecation policy. The documented worker CLI and discovery protocol are
also covered. Additional JSON diagnostic fields may appear; internal daemon
classes, the historical module exports, and private client members are not public
contracts. `umweltforge` remains outside this policy.

The client stays in `umweltd`, not in the engine's declarative-spec import
allowlist. Supporting a transport interface does not authorize network access
from a declarative-only world definition.

## Penduline consumer surfaces (stable from 0.2.0)

Penduline imports these six deep paths today. From 0.2.0 they are **supported**: they
follow the SemVer and deprecation policy above, and `tests/test_penduline_surface.py`
pins that each one imports and is callable.

| deep import | kind |
|---|---|
| `umwelt.spec.schema.match_ignored` | function |
| `umwelt.projection.confession.recent` | function |
| `umwelt.projection.nameplates.nameplates` | function |
| `umwelt.projection.emoji.node_icon`, `umwelt.projection.emoji.role_poles` | functions |
| `umwelt.events.replay_sensor_batches` | function |
| `umwelt.learning.runner.BrainRunner` | class |

The rest of those modules stays internal.
