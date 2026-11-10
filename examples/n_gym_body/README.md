# n_gym_body — umwelt leaf for N-gym capacity

**Why this exists:** “How much free CPU is enough?” is a **body belief**, not a
BAR constant and not a one-off coral organ. This DomainSpec grows umwelt leaves
so capacity / portfolio / stream-class sit in the field.

## Map

| Node | Role |
|------|------|
| `body` | free_cpu, free_mem, busy_load, enough |
| `portfolio` | rank / lease / pack (shadow decisions) |
| `stream_heavy` | exclusive gym class (BAR/SW…) |
| `stream_light` | cheap probes (registry, spine smoke) |

Bindings are **report-shaped** (`force_observe=True`) so organic rest/work claims
hold between sessions (hive_relay lesson).

## Coral bridge (file dialect when umweltd dark)

| Belief | Face |
|--------|------|
| capacity.* | `commons/vitals/organism_body_params.json` → `groups.capacity` |
| leases / pack | `commons/vitals/muscle_portfolio.json` |
| streams | `commons/vitals/workstream_registry.json` |

`muscle_cpu_learned_params` is a **compat shim** → body capacity.

## Run

```bash
# inspect spec
python -c "from examples.n_gym_body.world import n_gym_body_spec; s=n_gym_body_spec(); print(s.name, len(s.nodes))"

# from repo root with package path
PYTHONPATH=src python -c "from examples.n_gym_body.world import n_gym_body_spec; print(n_gym_body_spec().name)"
```

When umweltd is live, post probe rows onto these sensor_ids; when dark, coral
continues with organism_body_params.capacity.* only — field layer is additive.
