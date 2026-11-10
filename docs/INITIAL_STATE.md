# Declared local initial states

A new consumer can declare its initial belief geometry instead of preparing
registers after boot or relying on the process-global role registry:

```python
NodeSpec(
    'sample', roles=('a', 'b'), cluster_backend='cumulant',
    initial_bloch={'a': (0., 0., 0.), 'b': (0., .6, .8)},
)
```

`initial_bloch` must cover exactly every role on that node. Each vector contains
three finite real coordinates with Euclidean norm at most one. Zero is a
maximally mixed prior; a unit vector is a pure local prior. These are authored
initial conditions, not observations or calibrated confidence. The initial
vectors define independent local states, not an entangled/correlated joint prior.
The spec does not claim these priors are empirically optimal.

Both resolve-only validation and direct build reject invalid/incomplete maps
before constructing an engine. Folded nodes have no independent cluster and
cannot declare this preparation. Graph/spec reconstruction preserves the map
without aliasing caller-owned vector lists. Preparation uses the shipped
substrates' existing `observe_qubit(..., alpha=1)` state operation after the
legacy seed profile, before attaching the forecast organ. It creates no
measurement receipts, sensor timestamps or touched roles. A frame remains DARK
until actual evidence arrives. A saved state restores learned geometry rather
than reapplying the boot prior after load.

An explicit `cluster_filter` may omit a declared node from a scoped engine;
that node stays absent. Without such a filter, a missing preparation target or
changed role geometry is an error, not a silently ignored declaration.

`None` retains historical behavior. In particular, the legacy `blank` profile
neutralizes selected parameter priors and mixes analog dissipative beliefs;
other registers can still start at a committed ground pole. Analog registration
is currently process-global. This change does not relabel historical specs as
unbiased, migrate all of them, or isolate their role registries. New consumers
requiring a repeatable prior must declare it explicitly. The declaration wins
over either blank/magic seed profile for that node, with the original profile
name retained as provenance for the rest of the engine.

This addresses the explicit preparation path requested by the first kernel
consumer (issue100 ask10). Broad legacy-prior migration and generalized joint
initial states remain separate work. Runtime pins are unchanged by repository
integration.
