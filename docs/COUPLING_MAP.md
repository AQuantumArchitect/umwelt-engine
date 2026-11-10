# Coupling map — where the three QAGIS maps already live in this engine

Status: observation-grade on the code paths named; recommendation-grade (η 0.25) on the
vocabulary binding. Nothing here adds a module. It names what exists.

QAGIS is the umbrella for the quantum strand of Luke's fleet; Umwelt is its core. The
QAGIS fiber-bundle framework names three coupling maps between three layers — physical,
quantum, cognitive. This engine has had all three for some time under plainer names.
This file is the dictionary, so that the lexicon can stop being flavor and start being a
set of pointers into `src/umwelt/`.

| QAGIS map | Direction | Where it is in this engine | Public symbol | Default law |
|---|---|---|---|---|
| **κ** (kappa) | physical → quantum | ingest: a classical event moves a qubit | `engine.ingest()` → `membranes.ingress.SensorBridge.process()` → `substrate.cluster.QubitCluster.observe_qubit()` | L0 α-blend ships; L4 `substrate.belavkin.measure_bloch` measured, lost on real data, ships OFF. η=0 is a no-op on the field. |
| **ε** (epsilon) | quantum → cognitive | projection: the field becomes something readable without being mutated | `projection.nameplates`, `.brief`, `.gauge`, `.glass_door`, `.confession`, `.cognifold_trace`, `.transparency`; and `substrate.collapse.CollapseEngine.collapse_node` onto `substrate.ground` | read-only on ρ; an unknown tile renders `dark`, never 0 |
| **χ** (chi) | cognitive → physical | egress: a decision becomes an action | `membranes.egress.SpecTendril.step()` → `OutputSurface.route()`; cutover plane `learning.actuator_autonomy` | shadow-first: `OutputSpec.shadow=True`; only `*_auto` on a non-shadow tendril reaches `dispatch` |

The **classical ledger** under all three is `events.db` (`umweltd/worldstore.py`:
`timestamp, source_device, value, metadata`). The law is already written in the daemon:
*the log is the truth, the snapshot is a cache*. Boot = snapshot + replay of the tail
through κ. This is the "classical ledger over a quantum-field expression of information"
that the whole fleet is an exploration of — not a plan, the running shape.

## What the first sentence measured

`proofs/first_sentence.py` walks a blank engine through one deterministic synthetic day
on the production κ path and reads the result through ε, with χ held in shadow. Receipt:
`proofs/receipts/first_sentence.json`. The sentence:

> A blank engine replayed one synthetic day (605 ingest batches, 2396 events). Its 10
> agent_near beliefs began as one pure pole and ended as 9 distinct points spanning 1.95
> of the 2.00 Bloch range; its learnable fiber lost 9.39 bits of entropy over 87 qubits
> (it learned) while its world beliefs gained 11.97 bits over 21 qubits (it gave back a
> certainty it never earned); 18 of 19 declared axes went live; 0 actions were dispatched.

Three corrections this makes to the lexicon's intuitions, each with a number behind it:

1. **Observation does not lower entropy here; it differentiates.** The blank floor is a
   pure pole with entropy 0 — false certainty, not maximal ignorance. Comprehension shows
   up as the cells' `agent_near` qubits coming apart (1 point → 9), not as any one qubit
   purifying. If a Quogniton-style term wants to mean "belief sharpened", it has to be
   defined on *distinguishability across the field*, not on single-qubit purity.
2. **Learning lives in the fiber.** The `_param_*` qubits are the learnable fiber of the
   bundle (the docstrings in `substrate/cluster.py`, `params.py`, `continuous_geometry.py`
   already say "fiber bundle"). Over the day they lost 9.39 bits: priors became learned
   values. That is the measurable meaning of "the fiber drifted off its priors" in
   `proofs/blank_slate.py`.
3. **Honest uncertainty costs bits.** The dissipative `resource` role relaxed toward
   mixed because offline replay has no live tick and the engine does not pretend
   otherwise. Total entropy rose. A theory that treated total entropy as a score would
   punish honesty; this engine's gauge (`projection/nameplates`: latent / live / dark)
   does not, and the lexicon should follow the gauge, not the other way round.

## Vocabulary binding (recommendation, η 0.25 — to be earned by further proofs)

| Lexicon term | Candidate engine referent | Status |
|---|---|---|
| Quogniton | one role-qubit's Bloch vector in a cluster (`QubitCluster.role_bloch`) | pointer; needs no new object |
| Cognifold / Cognafold | the Bloch atlas export already called `cognifold_trace` (`projection/cognifold_trace.py`, `docs/TRANSPARENCY.md`) | **already a name in the tree**; spell it the engine's way |
| Cognoflux | the per-batch change in the field under κ (`BrainRunner.replay` on_batch deltas) | fog until a proof prints it |
| Clastoform / collapse | `substrate.collapse.CollapseEngine.collapse_node` → `ground.py` | pointer |
| Memoquanta | rows of `events.db`; the classical ledger | pointer |
| Fitness Clastoforms | `substrate.manifold_fitness`, `structure_selection` | fog until read |
| i_fields spun independently | `substrate.batched_evolve`, `fractal_stack` — unverified | fog |

Terms with no row here stay flavor. A row moves from fog to pointer only when a proof
under `proofs/` prints a number from the named symbol.

## What this file is not

Not a product claim, not a Broadcast node, not LIVE anything. It raises no η. It is a
dictionary between a vocabulary that lives in a Claude Project and an engine that lives
here, so the two stop talking past each other.
