# `frame/1` — a Frame of one operation

A Frame is a live model of one operation — a company, a house, a game world — drawn from
that operation's own ledger, with every number marked **current**, **stale**, or **dark**.
This file is the schema three producers write and one renderer reads:

- **Umwelt** exports one from a running world (`umwelt.projection.frame`, read-only on ρ).
- **The Broadcast** renders any `frame/1` and shows its own company's as the Engine screen.
- **Cerebral Frame** ships a worked sample and sells the real thing.

## Law

1. A value nobody measured is `null` and renders **dark**, never zero.
2. Every value carries the age of the face it came from. Age decides current vs stale; the
   producer states the threshold; the renderer never invents one.
3. A renderer may reshape; it may never promote. It cannot turn stale into current or dark
   into a number.

## Shape

### Direct Protocol observations

Successful positive-strength `measure_axis` and `measure_pair` calls now append a
receipt with `applied: true` after the update completes. Failed calls do not append
an applied receipt; zero-strength calls retain a non-applied receipt. Historical
receipts without this field remain unchanged and cannot establish freshness.
This is producer completion evidence, not authentication of an outside event.

The Umwelt projector uses these receipts without private SensorBridge timestamp
writes. A local z observation can date the local z readout. A joint or other-axis
observation alone leaves that scalar null/dark, with a receipt pointer and an
explanatory note. It cannot refresh an older local z observation. Existing genuine
bridge observations remain supported; consumers must remove synthetic bridge writes.
The rendered scalar is a conditional model readout, not a calibrated probability.

An optional node `signal` describes the newest confirmed observation separately:
`kind` (`axis` or `joint`), `axis`, `registers`, `receipt_order_index`, `age_s`, and
`state`. The index binds exactly one positive-strength applied receipt. Its age
uses the receipt timestamp and the node's existing stale threshold. A current
joint signal may coexist with a dark scalar. Future timestamps do not signal.
Projection does not update the underlying state or receipt journal.

### Local model readout (optional)

Umwelt nodes also carry `model_readout: {schema: "local-qubit-readout/1",
purity: number|null, calibration: "unmeasured"}`. This is local **model geometry**,
not an additional observation, receipt confidence, or empirical probability.
It accompanies an already dated local Bloch-z value; the projector omits the
diagnostic for null/dark scalars, including pair-only and off-axis-only observations.
If a producer includes the diagnostic there, purity must be null. Stale scalars retain
their existing age and stale state. Missing diagnostics on older frames stay
unknown. The validator never requires them of other frame producers.

`purity` is the local single-qubit `Tr(rho_i^2) = (1 + |r_i|^2) / 2`, in [0.5, 1],
rounded to the nameplate's existing four decimal places. The projector reuses
that nameplate value and never constructs a joint density matrix. It is **not**
the joint state's purity: a pure entangled pair can have maximally mixed local
states. A pure coherent state can also have z=0. Neither purity=1 nor z=0 tells a
consumer whether its external forecast is calibrated. Cumulant/classical values
describe those backends' represented local states, not unrepresented joint truth.
The diagnostic describes the current model at frame `as_of`; the node age dates
the supporting local observation, not a stored snapshot of geometry at that time.

`calibration: "unmeasured"` states that this readout has no measured mapping from
its geometry to the consumer's outside outcomes. Other calibration claims are
invalid under this version. A consumer may preregister a later comparison using
purity as an input; exposing it does not justify automatic confidence weighting.
The definition follows the density-operator convention in
[IBM Quantum's quantum information reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info).

```json
{
  "schema": "frame/1",
  "subject": "Harbor Tool & Supply",
  "as_of": "2026-10-16T19:00:00Z",
  "stale_after_s": 259200,
  "law": "a value nobody measured is null and renders dark, never zero",
  "producer": {"name": "cerebral-frame-site", "kind": "sample"},
  "nodes": [
    {
      "id": "orders_week",
      "label": "Orders this week",
      "value": 41,
      "unit": "orders",
      "state": "current",
      "age_s": 3600,
      "source": "point-of-sale export",
      "note": ""
    }
  ],
  "edges": [
    {"from": "orders_week", "to": "cash_in", "rel": "feeds"}
  ]
}
```

### Fields

| field | type | rule |
|---|---|---|
| `schema` | string | exactly `"frame/1"` |
| `subject` | string | whose Frame this is |
| `as_of` | ISO-8601 | when the producer wrote it |
| `stale_after_s` | int | default threshold; a node may carry its own |
| `law` | string | the sentence above, verbatim or in the producer's words |
| `producer` | object | `name`, `kind` ∈ `engine` \| `log` \| `sample` \| `instrument` |
| `nodes[]` | array | one per measured-or-not thing |
| `nodes[].id` | string | stable, snake_case |
| `nodes[].label` | string | human words |
| `nodes[].value` | number \| string \| boolean \| null | `null` ⇒ `state` must be `dark` |
| `nodes[].unit` | string | optional |
| `nodes[].state` | `current` \| `stale` \| `dark` | derived (below) |
| `nodes[].age_s` | int \| null | seconds since the value's own timestamp at `as_of`; `null` ⇒ dark |
| `nodes[].stale_after_s` | int | optional per-node override |
| `nodes[].source` | string | where the value came from, in words |
| `nodes[].note` | string | optional, one line |
| `nodes[].receipts[]` | array | optional; what the value was measured from, oldest first (below). Absent ⇒ no measurement receipts were recorded — it says nothing about the value. |
| `edges[]` | array | optional; `from`, `to`, `rel` (free vocabulary; `feeds`, `blocks`, `part_of` are conventional) |

#### `nodes[].receipts[]` (optional)

Written by the umwelt engine when `measure_axis` / `measure_pair` (docs/QUANTUM_KALMAN.md) measured
this node's register. A receipt never changes `value`, `state` or `age_s`; it records how the
register was interrogated, so two non-commuting measurements (z then x) are distinguishable from
one repeated measurement. Fields: `register` (role name), `axis` (unit 3-vector σ·n̂; for a joint
measurement the first register's), `record` (the observed outcome in [−1,1]), `strength` (≥ 0),
`confidence` (number|null), `order_index` (int, monotone per cluster), `t` (wall-clock epoch
seconds), and for a pair measurement `joint: true`, `registers` (both role names) and `axis2`
(the second register's axis). A joint receipt appears on every node it names. The shape maps onto
the glyph module's `Receipt` through `umwelt.substrate.receipts.to_glyph_receipt`.

### Deriving `state`

```
if value is null or age_s is null:  dark
elif age_s <= (node.stale_after_s or frame.stale_after_s):  current
else:  stale
```

A producer must write `state` and a validator must be able to re-derive it. Disagreement is a
producer bug, not a renderer choice.

## Mappings

| frame/1 | Umwelt nameplate | The Broadcast pulse | Cerebral Frame copy |
|---|---|---|---|
| current | `live`, age within threshold | LIVE | current |
| stale | `live`, age past threshold | STALE / DOWN | stale |
| dark | `latent` (never signalled) or `dark` (η=0 / absent) | DARK | dark |

## What a Frame is not

Not a dashboard (a dashboard shows numbers; a Frame shows which numbers to trust). Not a
score — the count of dark nodes is information, not failure. Not a write path: nothing in a
Frame changes the ledger it was drawn from.
