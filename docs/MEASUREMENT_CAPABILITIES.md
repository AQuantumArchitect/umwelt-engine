# Measurement capabilities

Ask what a backend implements without changing state or creating a receipt:

```python
from umwelt.substrate.backend import measurement_capabilities

capabilities = measurement_capabilities(cluster)
if capabilities.pair is False:
    # Select an explicitly labelled single-axis path or refuse the operation.
    # Two single observations do not record the joint correlation.
    ...
```

The frozen `MeasurementCapabilities` record has contract
`measurement-capabilities/1`, `axis`, `pair`, and `description`. Flags mean a
positive-strength operation is implemented, subject to valid indices, axes and
other inputs. `None` is undeclared, distinct from `False` (unsupported). A method
named `measure_pair` or an absent backend flag cannot establish support: product
and classical backends implement that method to refuse it. Unknown backends
remain unknown; malformed declarations raise a visible `TypeError`.

| Backend | Axis | Pair | Qualification |
|---|---|---|---|
| Dense | yes | yes | Conditioned Kraus update on its full joint density matrix |
| Cumulant | yes | yes | Second-order closure; nonlocal back-action is approximate |
| Product | yes | no | Independent single-qubit states cannot store joint measurement correlation |
| Classical | yes | no | Axis updates retain only z, discarding coherence |

These are code-level capability declarations, not measurements, learned
confidence, calibration, or a claim that quantum geometry improves a task.
They do not change the measurement methods or select a backend. Input validation
and each method's documented approximation remain in force. The optional query
does not add requirements to the existing structural `SubstrateBackend` protocol.
Third-party backends may expose a `measurement_capabilities` attribute containing
this immutable type. Legacy consumers may keep exception fallback when the query
is unavailable; a declared-supported operation that refuses a valid call is a
contract disagreement, not evidence that a fallback silently succeeded.
