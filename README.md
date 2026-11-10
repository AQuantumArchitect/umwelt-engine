# umwelt

Umwelt is a Python library that keeps a live, readable model of a system, built from
that system's own event log, and keeps honest books about what it has *not* measured.
You describe a world as data (what exists, what can be sensed), feed it readings, and
ask what it currently believes, how sure it is, and how old each belief is. It is for
people building monitors, agents, games, or analyses who need to trust the difference
between "the number is zero" and "nobody has looked." Its stance in one sentence:
**dark is not zero.** A value nobody measured is `None`, with an age, never a quiet 0.

> New to this and not an engineer? Read the plain-terms explainer in
> [docs/overview/](docs/overview/) first. Want the long technical tour that used to
> live here? It is in [docs/LONGFORM.md](docs/LONGFORM.md).

## What it has been measured to do, and what it has not

[CLAIMS.md](CLAIMS.md) is the ledger of every claim, sorted by how well it survives a
challenge. **If this README and the ledger disagree, the ledger wins.** A short slice:

| Measured (and where) | Denied or corrected by its own experiment |
|---|---|
| A blank engine boots a world it has never seen, replays a synthetic day, and tracks the ground truth; the learned state survives save/load. In the gate. ([blank-slate theorem](CLAIMS.md)) | **A fancier filter lost.** The full Belavkin filter scored 0.3034 error against 0.1349 for a two-line blend and 0.1346 for bare persistence on the origin's real data. It ships off. ([DENIED](CLAIMS.md)) |
| A confidence of zero changes nothing: the field, learned weights and trust are byte-identical to silence. In the gate. ([η = 0 row](CLAIMS.md)) | **The cheap closure is blind to synergy.** The approximation that makes the engine fast reads a pure-synergy (XOR) state as featureless. Synergy detection ships dense-only. ([DENIED](CLAIMS.md)) |
| Driven over the wire, the daemon ends at exactly the state of a direct library replay, and a killed worker recovers it. In the gate. ([daemon parity](CLAIMS.md)) | **A speedup did not survive re-measurement.** The origin's 3.4x merge speedup re-measured at about 1.1x and was corrected. We keep it as the example. ([DENIED / CORRECTED](CLAIMS.md)) |
| On real data from the origin deployment, a learner that ignores its own actions credited them at 10.8x their true strength; tagging them cut that bias 79%. Origin-measured, data not available. ([CLAIMS §2](CLAIMS.md)) | |

Tiers you will see in the ledger: PINNED (a test or proof runs it every time),
ORIGIN-MEASURED and FOREIGN-MEASURED (real data we cannot share), EVALUATION OWED
(designed, not yet measured), and DENIED (measured and lost). Read the DENIED tier
first; it is the reason to believe the rest.

## Install and run it

Python 3.10 or newer; numpy is the only required dependency.

```bash
git clone https://github.com/AQuantumArchitect/umwelt-engine && cd umwelt-engine
python3 -m venv .venv && . .venv/bin/activate
pip install -e .          # the distribution is named umwelt-engine; not on PyPI yet
python examples/quickstart.py
```

To depend on it from your own app, pin a release tag from git (not `main`):

```bash
pip install "umwelt-engine @ git+https://github.com/AQuantumArchitect/umwelt-engine@v0.2.1"
```

The distribution is `umwelt-engine`, but the import name is `umwelt`. The unrelated
PyPI project `umwelt` also installs an `umwelt` package, so do not install both in one
environment. Details and upgrade steps: [docs/CONSUMER_PIN.md](docs/CONSUMER_PIN.md).

It runs offline in about a second. You will see (trimmed):

```
hall_occupied  value=None  state=dark  age_s=None
den_occupied   value=0.95  state=current  age_s=0
den_warmth     value=None  state=dark  age_s=None
DARK: ['hall_occupied', 'den_warmth']
frame/1: frame/1 violations: none
purity (how sharp each belief is): {'hall.occupied': 0.9549, 'den.occupied': 0.9513, 'den.warmth': 0.6839}
CONTINUATION APPROXIMATE: the snapshot's continuation digest (...) does not match ...
save/load hash-equal: True
```

What happened: [examples/quickstart.py](examples/quickstart.py) (about 50 lines) boots a
blank two-room world from a spec, feeds only the motion sensor, and prints beliefs. The
temperature was never read, so `den_warmth` is `None` and dark, not 0. It exports a
`frame/1` document, saves the engine, loads it into a twin, and checks the two have the
same field hash. The `CONTINUATION APPROXIMATE` line is a warning, not a failure: the
belief field round-trips exactly, but some hidden bookkeeping does not yet, and the
engine says so out loud ([CLAIMS.md](CLAIMS.md), continuation rows).

Next steps: [docs/API.md](docs/API.md) (the stable public names),
[docs/NEW_DOMAIN.md](docs/NEW_DOMAIN.md) (your own world), and
[examples/gridworld/](examples/gridworld/) (a fuller template). Depending on umwelt from
another repo? See [docs/CONSUMER_PIN.md](docs/CONSUMER_PIN.md). Building a consumer
right now? [docs/KERNEL-1.md](docs/KERNEL-1.md) is the contract: what is frozen, what may
still move, and what a build bot may do before the `kernel/1` tag. A private
downstream application depends on Umwelt from its own repository; the engine imports no
application code.

## How it is shaped

The event log is the truth; the belief is a view derived from it. Replaying the same
stream into the same spec gives the same state, and the belief a person reads is the
same object the engine computes with. Each reading carries a confidence and a time, so
every belief carries its age too.

- **world** and **spec**: a world is described as data, a `DomainSpec` (nodes, roles,
  bridges, bindings, drivers). The engine ships no domain vocabulary and runs any spec
  ([docs/SPEC.md](docs/SPEC.md)).
- **face**: the human-readable pole labels and status of one belief axis, such as
  "occupied: yes or no, live, last read 10 s ago."
- **dark**: an axis that was never signalled, or whose last reading carried no
  confidence. Its value is `None`, never zero.
- **nameplate**: the one-row summary of a face (`nameplates/v1`): status, a number, and
  how sharp the belief is ([docs/TRANSPARENCY.md](docs/TRANSPARENCY.md)).
- **frame**: a `frame/1` document, one operation's model with every number marked
  current, stale, or dark by its age ([docs/FRAME_SCHEMA.md](docs/FRAME_SCHEMA.md)).
- **confession**: the engine's own record of what it forecast, observed, and decided,
  readable after the fact (`umwelt.projection.confession`).
- **glass door**: a read-only view into the engine's insides that never writes to it
  (`umwelt.projection.glass_door`).
- **η** (eta): confidence on a reading, from 0 to 1. η = 0 is a proven no-op; no input
  can move a belief faster than its admitted confidence allows.

The belief state is a density matrix over a graph of qubit clusters, computed on an
ordinary CPU. It is a quantum-formalism model on classical hardware, not a claim to run
on a quantum computer ([docs/THEORY.md](docs/THEORY.md)). It is not an ML framework, not
an LLM, and not a drop-in Kalman filter. The API is 0.x ([docs/STABILITY.md](docs/STABILITY.md)).

## Where the proofs are

- `proofs/`: standalone scripts, each named for what it proves (`blank_slate.py`,
  `frame_export.py`, `ladder_walk.py`, `deconfound_smoke.py`), seeded and offline.
- `tests/`: the unit and pin tests, including `tests/test_claims_witness.py`, which
  checks that every path and link cited in this README and the ledger resolves.
- Run the gate: `pip install -e ".[dev]"` then `python -m pytest -q`. There are no
  application tests in this tree; consumers test themselves against the pinned engine
  ([docs/CONSUMER_PIN.md](docs/CONSUMER_PIN.md)).

House laws:

1. Dark is not zero. A number nobody measured is `null`, with an age.
2. Nothing secret-shaped is committed: no tokens, hosts, mail, or household data.
3. CLAIMS.md holds only claims with a proof or a cited measurement behind them.

## Lineage and privacy

Umwelt was extracted from a home-comprehension system that ran in one real apartment
for about 18 months, and results from that deployment are reported honestly as such.
Every one is tiered ORIGIN-MEASURED in [CLAIMS.md](CLAIMS.md): measured on that
deployment's own sensor data (the dataset is described in the ledger row: duration,
bins, provenance), not witnessed by this repository's gate, and with the data not
available, because it is a household's. A second set of results comes from a 13-day
smart-home export from a house nobody on the project had instrumented, tiered
FOREIGN-MEASURED, data also private ([docs/FIELD_NOTES.md](docs/FIELD_NOTES.md) keeps the
transferable lessons). What was extracted is the method; the data stays where it is.

## License, contributing, and the notebook

Copyright 2026 Luke Spooner. Licensed under AGPL-3.0-or-later ([LICENSE](LICENSE),
[NOTICE](NOTICE)). An additional permission under GNU AGPL version 3, section 7
for games and applications whose source repository is owned by the
AQuantumArchitect GitHub account is in [LICENSE-EXCEPTION.md](LICENSE-EXCEPTION.md).

Pull requests are welcome. CI runs the gate (`python -m pytest -q`). A new row in
CLAIMS.md needs a proof or a test behind it, and a number you could not measure goes in
as dark. Changes to the version history are in [CHANGELOG.md](CHANGELOG.md).
Contribution process and the contributor licence agreement (draft) are in
[CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).

The public notebook for this work, negatives included, is called The Broadcast. It
renders `frame/1` documents like the one the quickstart prints.
