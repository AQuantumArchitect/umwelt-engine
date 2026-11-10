# kernel/1 — the contract a build bot may build against

Status: **proposed** by the head seat, 2026-11-08 (late), from Luke's ruling that everything downstream stays paused "while we mess with the kernel." This page separates the two things that were paused together. Numbers follow GLYPH-LANGUAGE §25: straight = measured, `~` = forecast, `?` = dark.

## Two gates, not one

| gate | what it freezes | who it releases | when |
|---|---|---|---|
| **API freeze** | signatures, schemas, module surfaces (§1) | downstream **build bots** — code against the contract, run against fixtures | when the merge train lands on `main` and `frame/1` carries the additions below: `📍 2026-11-10~` |
| **kernel/1 tag** | the numbers under the API (§2) | **production**: real household/audience records may be ingested and real belief recorded | when §2 is green: `📍 2026-11-13~` |

A build bot released at the first gate **may not** ingest real data, publish a belief, or write a tape row that anything downstream reads, until the second gate. It builds, tests on fixtures, and waits. This keeps the pause where it matters (the belief) and lifts it where it does not (the code).

## 1. Frozen at the API freeze

Once on `main`, these do not change shape without a `kernel/2` discussion. Numerics beneath them may still move (§2).

**Measurement Protocol** (`umwelt.substrate.backend.SubstrateBackend`, PR #90):
- `measure_qubit(qubit_idx, record, strength, confidence=None)` — unchanged; alias of `measure_axis(axis=(0,0,1))`.
- `measure_axis(qubit_idx, axis, record, strength, confidence=None)` — `axis` is `'x' | 'y' | 'z'` or a non-zero 3-vector.
- `measure_pair(i, j, record, strength, basis=("z","z"), confidence=None)` — raises `NotImplementedError` on backends that cannot hold a correlation when `strength > 0`; `strength ≤ 0` is a no-op everywhere.
- Backend selection by flag; `"gaussian"` (PR #91) exists and is **opt-in only**; nothing selects it by default.

**Receipts** (`umwelt.substrate.receipts.MeasurementReceipt`, PR #90): per-cluster bounded log, created on first measurement; `to_glyph_receipt()` maps `strength → eta = 1 − exp(−strength)`.

**`frame/1`** (`docs/FRAME_SCHEMA.md`): schema string stays `"frame/1"`; additions are optional fields, so an old renderer still validates:
- `nodes[].receipts` — list; a joint receipt appears on every node it names (#90).
- `"forecast"` beside any numeric value (`bool`; `[bool,bool,bool]` on a Bloch vector) (#93). A dark node flagged forecast is invalid.
- `validate()` accepts frames with and without these fields.

**Glyph module** (`umwelt.glyph`, PRs #87, #93): `Num` (value + provenance ∈ {measured, forecast, dark}); `parse / bind / render / metrics / receipt.grade` public names; `bind(..., engine_origin=True)` is the only way a straight number is earned without a receipt. The thirteen + twelve grammar decisions in `docs/GLYPH.md` are **proposed** and may be re-sat by Luke; a build bot treats them as the current grammar, not as law.

**CLAIMS.md tiers**: PINNED / MEASURED / ORIGIN-MEASURED / FOREIGN-MEASURED / DENIED / DARK, plus the proposed **VOUCHED** (the-broadcast law amendment; GLYPH-LANGUAGE §29). A build bot never writes a tier.

## 2. Must be true before the kernel/1 tag

Each line is a test or a measured pin; the tag is cut when all are green on `main`.

1. Dense ↔ cumulant parity on the measured register for `measure_axis` and `measure_pair`: ≤ 1e-6 (measured ~5e-8 / ~6e-8, #90).
2. Off-axis peer means on the cumulant backend vs dense: ≤ 5e-7 on mixed correlated states (measured ~1e-7, #92); pure-state residual ≤ 2e-2 and attributed to the pole cap, not the closure.
3. **Post-measurement correlation block e2[i,j] exact from stored moments** — answered: Codex draft #97 (exact local Kraus moments incl. z, 2e-12 vs dense on 2/3/4-qubit witnesses; g/Z may exceed 1, so "always coherence damping" is struck). Tolerance to pin: `≤ 1e-9`.
4. e2[j,k] between peers and the third qubit under `measure_pair`: the triple moment is **not identifiable** from second-order moments (umwelt #95 turn 2: ρ± = (I ± ZZZ)/8 share every first and second moment and give opposite peer correlations ±tanh(0.5) after one measurement — an executable counterexample). So the stated closure is a **positive, unit-trace maximum-entropy completion** consistent with the measured moments, then the actual Kraus update and moment projection; infeasible constraints are *reported*, never silently repaired; the discrepancy vs dense is pinned and the non-identifiability documented. Zero cumulants plus clipping is not a closure. Not a tag blocker once stated and pinned.
5. The z-axis peer update switched to the full form in the same change as item 3 — **one break**, not two; the old bit-for-bit test is retired with a dated note, not deleted silently.
6. Order-dependence pins unchanged on the measured register (z→x ≠ x→z; z→z order-free to 1e-9).
7. Belavkin filter stays OFF (CLAIMS.md L4 DENIED). Gaussian backend stays opt-in; its two-level-limit pins (3e-3 step, 1.5e-2 measurement) hold.
8. `proofs/` and the full gate green; CLAIMS.md rows for #90–#93 moved from EVALUATION OWED to MEASURED where the number exists, left owed where it does not.

## 3. What a build bot may do before the tag

- Build against §1. Import, call, serialize, render.
- Run against **fixtures**: synthetic tapes, the `examples/` worlds, recorded frames with `forecast` flags.
- Write its own tests that pin *its* behaviour against the frozen surface.
- File issues against this page when §1 does not hold on `main` — that is a kernel bug, and a build bot's report of it is welcome.

## 4. What a build bot may not do before the tag

- Ingest real household or audience records into a live field.
- Publish a belief, a tier, or a straight number anywhere the Broadcast or S3 reads.
- Depend on numerics in §2 (e.g. pin a test to the current e2 value — it will change).
- Select the Gaussian backend.
- Work around a `NotImplementedError` from `measure_pair` on a product or classical backend by computing a correlation itself.

## 5. After the tag

Consumer rollout order belongs to the downstream coordinators; the historical named order is retained in [the decoupling issue](https://github.com/AQuantumArchitect/umwelt/issues/81). This contract does not itself resume a consumer or replace its owner's controls. Umwelt's public release (a dam, Luke's) is independent of this tag and may precede it; a public repo with an honest CLAIMS.md and a stated kernel/1 checklist is more honest than a delayed one.

## 6. Done when

1. This page is on `main` and linked from `README.md` and `CLAIMS.md`.
2. `frame/1` on `main` validates frames with and without `receipts` / `forecast`.
3. The eight lines of §2 are each a named test or a pinned row; the tag `kernel/1` is annotated with their status at the moment of tagging.
