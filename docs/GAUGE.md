# GAUGE — what the gauge/git ledger actually certifies

The tracking system's theory foundation. Every statement here has a CLAIMS.md row or a
named test; where a convention is a choice rather than physics, this document says so.
(Docs may cite domains freely; engine source stays domain-free.)

The one-line thesis, imported from the sibling substrate that forced this audit:
**the numbers you write are bookkeeping; the numbers that survive every re-bookkeeping
belong to the world.**

## 1. The three legs

```
live field ──pack()──► canonical state ──canonicalize()──► gauge-fixed state
                          │                                   │
                          ├─ canon_hash (v1: as-built frame)  ├─ canon_hash v2
                          │                                   └─ invariant_sheet
                          ▼
              field DAG node (substrate/field_gauge + shelf: index.jsonl,
              content-addressed pickle, parent links, both hashes + sheet)
                          │
                          ▼
              git commit of the archive — strobed at the Poincaré section
              (maybe_record_field_node; UMWELT_FIELD_GIT opt-in)
```

- `substrate/field_unify.py` — the canonical state: `{partition, modes, diss_rates,
  e1[N,3], e2[N,N,3,3], h[N,3], zz, xy}`, one form that unpacks into the clustered
  forebrain or the 1-matrix manifold.
- `substrate/field_gauge.py` — the append-only learning DAG. A node is written only
  when the hash moved; every node carries v1 + v2 + the invariant sheet.
- `substrate/link_field.py` — the lattice-gauge ledger over the coupling graph
  (Wilson loops, β₁, gauge fixing). Bookkeeping ABOUT the graph; never feeds dynamics.

## 2. The gauge group of the representation

Transformations that change the stored bytes while changing nothing physical:

| freedom | where it lives | status |
|---|---|---|
| cluster dict-insertion order | `pack()` global qubit index | **quotiented** by `canonicalize` (canon v2) |
| role declaration order within a cluster | cluster construction | **quotiented** by `canonicalize` (canon v2) |
| incidental cluster grouping (clustered vs manifold) | `partition` | **quotiented** by v1 already (partition excluded from the hash) |
| `Bridge(source, target)` ordering | `graph.py` | **proven physicless** (`test_gauge_invariance`); load-bearing only for param naming |
| Bloch azimuth about a param's value axis | the +x confidence convention, 4 sites: `qubit_param.kalman_update`, `qubit_ema._surface_point`, `qubit_trust_web`, `universal_learner.observe_raw` | **recorded convention**, invariance pinned; NOT quotiented by any hash (a residual gauge — see §3) |
| global X/Y frame flip of the H tower | `fractal_stack.phase_align` | **certified gauge move** (sheet-invariant); fires only on a domain-declared reference label |
| local U(1)/Z₂ convention at a coupling-graph node | `link_field.gauge_transform` | **quotiented** for loop content (Wilson loops survive; `gauge_fix_tree` is the canonical choice) |

The engine's remaining frozen frame conventions (axis order x,y,z; big-endian kron;
`i<j` edge orientation) are recorded here as conventions: they are consistent
everywhere, tested nowhere individually, and any change to them is a format break, not
a refactor.

## 3. What each certificate certifies

- **v1 hash unchanged** — *no change in the as-built coordinates.* The strongest and
  narrowest claim; what existing referees (daemon parity, packet verify) compare. A
  pure relabeling breaks it — that is the documented defect, kept as a passing
  assertion in `test_canon_gauge`.
- **v2 hash unchanged** — *no change modulo the relabeling gauge group.* Two builds of
  the same brain with different bookkeeping agree.
- **invariant sheet unchanged** — *no physical motion.* β₁ + per-cycle Z₂ Wilson signs
  (through `link_field`), per-cluster sorted Bloch radii, |h| magnitudes, link
  coupling-magnitude triples, at the hash's own 10dp.
- **The diff-decomposition rule**: v2 moved + sheet identical → gauge motion only
  (someone re-bookkeeped; nothing learned). Sheet moved → physical motion, and which
  row moved says what kind (a radius = a belief; a link triple = a coupling; a Wilson
  sign = loop content — topology-grade news).

Honest residuals, stated rather than claimed away:
- The **Bloch azimuth** convention is respected by the sheet (it stores radii) but not
  quotiented by the hashes — rotating a param qubit about its value axis changes v1
  AND v2 while changing nothing the value getter reads. The invariance is pinned at
  the param layer instead (`test_gauge_invariance`).
- The certificate covers the **belief field**, not the whole process: `engine.save` is
  a curated learned-state cache, and the continuation gap (CLAIMS.md REFUTED row,
  2235 slots) is exactly the state the hash does not see. An empty diff means "the
  brain's canonical state did not change", never "the process has no hidden state."

## 4. Link variables and loops

Where the Z₂ structure already lives: `sign(J)` on intra-cluster zz couplings, and
exchange `(J, J)` vs antiexchange `(J, −J)` on xy — the only two forms
`coupling_learn` mints (θ ∈ {0, π}; there is no σxσy slot, so no continuous U(1) yet).
`link_field.from_couplings` reads exactly that structure into the honest starting
ledger; `from_canon` lifts a whole canonical state.

The executable lessons (all pinned in `test_link_field`):
- a local gauge transform changes every incident link and no Wilson loop;
- β₁ = E − V + C counts the places gauge-invariant information can live;
- `gauge_fix_tree` combs every erasable number to zero and leaves exactly β₁
  residues — the fundamental cycles' phases;
- **closure honesty**: `wilson_phase` on an open or broken walk is NaN, `wilson_sign`
  None, and an absent edge reads NaN, never 0. A refusal, not an exception.

The *bridge* sector was loop-trivial until this pass (the found defect, now repaired,
default ON): `calibration.py` mapped correlation ∈ [−1, 1] to (corr+1)/2 before
learning, bridge params were minted with `lo=0.0`, and reconciliation nudged both ends
toward the same averaged RDM — identity parallel transport, U = 1 pinned. Now the
coupling IS the signed correlation, and a negative link transports the partner through
the link before reconciling: U = σ_x, the π-rotation that flips the value axis (what
"anti-correlated" means for beliefs riding z) while preserving the +x confidence
meridian — the choice among equatorial π-axes is itself residual gauge, and σ_x is the
one consistent with the recorded azimuth convention. Consequences (pinned in
`test_signed_bridges`): an honestly anti-aligned pair is a fixed point of its −1 link
instead of being averaged away; the learned web seeds, keeps, and decays edges
sign-aware; a frustrated bridge triangle reads W = −1 via `from_bridges`, and no local
gauge flip can clear it. The bridge sector's loop content is a LIVE read
(`from_bridges(graph.bridges)`) — bridge couplings ride params, not the canonical
state, so the invariant sheet covers the intra-cluster/manifold web and the bridge
loops are read beside it, not inside it.

## 5. Closure honesty for the Berry machinery

Winding and holonomy are invariants of CLOSED curves. `continuous_geometry` refuses
open walks (NaN winding; `net_loops` keeps the honest open-path net displacement;
`berry_holonomy` requires closure by default). The berry tape is the only door to a
loop's samples: `BerryTape.loop_invariants(ret)` is None without a genuine
`GeometricReturn` — the return detector's Berry-adjacency is the closure certificate
and its `phase_delta` rides along as the closure defect. `field_gauge(field, tape=...)`
emits the `_loops` block only at certified returns; absence of the key IS the refusal.

The speed-invariance the old gauge.py header asserted is now earned: the same closed
loop traversed at three speeds reports identical winding/holonomy/directedness to 1e-9
(`test_closure_honesty`).

## 6. The commit contract

`maybe_record_field_node(reservoir_or_field, ...)`:
- **strobe**: with a clock cluster present, record only in the rest window — the
  Poincaré section, so consecutive committed snapshots are same-phase comparable. No
  clock → record unconditionally.
- **write trigger**: canon-hash inequality vs the last field node (v1 for now; v2 is
  recorded on every node and becomes the trigger only via a coordinated migration).
- **git**: `UMWELT_FIELD_GIT=1` (or `git_commit=True`) commits the node into the
  archive repo. One training event = one commit; a frozen field leaves `git log` and
  the worktree diff empty — README claim #4, pinned in `test_field_gauge_node`.
- **failure honesty**: a non-git archive still writes the DAG node, silently.

## 7. Roadmap — designed, not claimed

- **Signed bridges on a live deployment** — the mechanism shipped (§4, default ON);
  the owed artifact is a real-deployment before/after replay, since every existing
  learned bridge value was learned under the old sign-stripping law and will re-learn.
- **U(1) links** — needs the σxσy − σyσx (DM) slot in `_H_op`, the basis, and a canon
  v3; only worth feeding once Z₂ shows signal.
- **Berry-tape persistence** in `engine.save` — belongs to the continuation-gap repair
  track (the REFUTED row), not this one.
- **Canon-v2 as the referee/write trigger** — a coordinated flag-day across the packet
  contracts after v2 has coexisted in records.
