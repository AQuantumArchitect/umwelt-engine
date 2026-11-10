# Changelog

Notable changes, loosely following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
See [CLAIMS.md](CLAIMS.md) for the full evidence ledger behind any claim made here —
if the two ever disagree, CLAIMS.md wins.

## [Unreleased]

### License
- Relicensed from Apache-2.0 to AGPL-3.0-or-later (copyright 2026 Luke Spooner): full AGPL-3.0 text in `LICENSE`, copyright in `NOTICE`. v0.2.0 and earlier were Apache-2.0.
- Docs: the downstream application is referred to generically; the withdrawn real-household figure is no longer listed as owed.

## [0.2.0] - 2026-11-09

### Supported consumer vocabulary and service surface (#80)
- Promote `register_role_mode`, `register_observe_role`, `register_role_emoji` and
  `register_node_icon` through the lazy `umwelt` facade. Existing deep imports
  still work; process-global vocabulary behavior is unchanged.
- Support `from umweltd import UmweltClient` and its existing client-module alias,
  with typed/documented methods and the worker CLI/discovery contract in
  `docs/DAEMON-API.md`. The transport remains outside declarative-spec imports.
  Breaking-change notes for consumers: [docs/BREAKING_0.2.0.md](docs/BREAKING_0.2.0.md).

### Layering — S3 left the package (2026-11-08)
- A private downstream application's source moved to its own repository (layering law, VISION §II); this engine no longer ships a consumer for it.
  `src/umwelt/consumers/` is gone, and so are that application's tests, docs and examples (189 files). The vocabulary lint now has no
  exemption and `test_consumer_boundary.py` guards all of `src/`. CI no longer needs any private-repo credential.

### Engine — RK4 stability guard inside declared bounds (review issue #43, item 6)
- Every integrator substeps a step whose stiffest rate × dt exceeds RK4's stable region
  (`density_matrix.rk4_substeps`), and refuses a step that still goes non-finite. In-bounds
  defaults are a single step, byte-identical to before.
- `collapse.py`: a non-finite Bloch vector is DARK (no commit), not a −1 pole at NaN confidence.

### Spec gate — dangling references refused (review issue #43, item 9; #58)
- `validate_spec` `schema_sanity` (so also `--resolve-only`, umweltd's front door) now
  cross-checks binding zone/role and refuses unknown node/bridge `kind`, unknown `reduce`,
  faces/role_modes/projection on undeclared roles, `forecast_leaves` at a nonexistent
  node/role, non-positive or non-increasing `forecast_horizons_min`, `efficiency` outside
  [0, 1], an inverted/out-of-range `rest_window`, and an exact `ignored` id that is also bound.
- New advisory `ignored_overlap` warning for a `prefix_*` that also covers bound ids
  (binding wins; e.g. `manifold_game`'s `wallet_*`). All shipped example specs still validate.

### Changed
- Forecast confidence is tied to evidence (`max(0, skill_vs_persistence) × purity`); no baseline → `None` (DARK) and `consume_targets()` omits the leaf. `DissolvedForecastSurface` no longer asserts `skill=1.0` and decays confidence with horizon. Prediction rows carry `issued_at` and `age_s` (#57).

### Added
- `examples/n_gym_body`: the N-gym body world ported from the rescue branch, with its example test; station lanes were already on main (#54).
- `UmweltClient.anchor()` returns the raw anchor observation; it is not calibrated
  confidence. `UmweltClient.ingest(..., event_id=...)` (keyword-only) sends a named
  exact-replay batch: a conflicting payload under the same id is an identity conflict,
  and the client never retries. Documented in `docs/DAEMON-API.md` (#139).
- frame/1 nodes can carry an optional `model_readout` (`schema: local-qubit-readout/1`,
  local `purity`, `calibration: 'unmeasured'`), with validator rules in `frame.validate`.
  Old frames still validate. See `docs/FRAME_SCHEMA.md` (#120).
- CI and release workflows for the public export (hosted runners, no secrets); staged
  under `ci-public/` (#142).
- Six Penduline deep imports are now supported under the SemVer and deprecation policy
  (`docs/STABILITY.md`): `umwelt.spec.schema.match_ignored`,
  `umwelt.projection.confession.recent`, `umwelt.projection.nameplates.nameplates`,
  `umwelt.projection.emoji.node_icon` and `role_poles`,
  `umwelt.events.replay_sensor_batches`, `umwelt.learning.runner.BrainRunner`.
  Pinned by `tests/test_penduline_surface.py`.


### Consumer kit (#65)
- **`.github/actions/install-umwelt`** (new): composite action for dependent repos
  (since removed: consumers pin a release tag with a plain `pip install`; see
  `docs/CONSUMER_PIN.md`).

### Public API facade (#66)
- **`umwelt` is now the stable import surface**: `from umwelt import DomainSpec, NodeSpec,
  BridgeSpec, BindingSpec, OutputSpec, DriverSpec, AxisFace, validate_spec,
  ValidationReport, build_engine, BeliefEngine, GameHost, WorldSession, Observation,
  Intent, Decision, Belief, Event, field_gauge, model_snapshot`. The list comes from what
  the consumer repos actually import. Names load lazily, so `import umwelt` still pulls
  in neither numpy nor the engine.
- **[docs/API.md](docs/API.md)** documents every public name with an example.
  `tests/test_api_docs.py` runs every example on the page.
- **[docs/STABILITY.md](docs/STABILITY.md)** sets out the SemVer and deprecation policy.
  Deep paths still import but are internal. A moved public name keeps a shim in
  `umwelt._MOVED` that raises `DeprecationWarning` for at least one minor release.
- `tests/test_public_api.py` pins `__all__`, and checks that every public name has a
  docstring and type hints and that `import umwelt` stays lazy. `py.typed` now ships
  in the wheel.
- Docstrings and type hints added to the public host face (`GameHost`, `WorldSession`,
  the boundary dataclasses), `ValidationReport`, `Event`, `validate_spec`, `field_gauge`,
  `build_engine`'s `spec`/`dispatch`, and `BeliefEngine.save`/`load`. Behaviour is
  unchanged.

## [0.1.1] - 2026-11-02

Pin target for consumers (`git tag v0.1.1`) instead of a raw SHA. Headlines, from the #43
review: CLAIMS.md witnesses itself (#44), ZZ Hebbian learning works for underscore roles and
gamma 0.0 survives repacking (#45), η = 0 is a no-op at `engine.ingest` (#46), a world is a
function of its stream (#47), and the forge gate audits a spec before importing it (#48).
Everything below it, back to 0.1.0, ships in this release.

### Forge gate — the judge no longer executes the defendant (review issue #43, item 3)
- **`umwelt.spec.declarative`** (new): a static, source-text audit of a spec module and
  everything its import would execute (package `__init__`s, relative / same-package
  imports): allowlisted imports only, no dunders, no `eval`/`open`/`getattr`/…, no
  `with`/`try`/`while`, no bare module-level call. `validate_spec(..., declarative_only=True)`
  / `--declarative-only` runs it BEFORE resolve, so a refused module never imports.
  Verified before the fix: a four-line module patching `ValidationReport.ok` at import
  time registered `SPEC = "not a DomainSpec"` as OK.
- **`validate.gate_command` / `gate_env`**: the one way to run the gate in a subprocess —
  `python -I`, parent import path first and untrusted dirs LAST, environment scrubbed of
  `*KEY*`/`*TOKEN*`/`*SECRET*`/`*PASSWORD*`/`*CREDENTIAL*` while keeping `UMWELT*` dials.
  `umweltforge.pipeline.run_validation` (always strict) and `umweltd.supervisor._gate_spec`
  (manifest knob `declarative_only`, default off) both use it.
- CLAIMS row re-scoped from "a lying claim" to "a lying claim or a lying module", with the
  boundary stated (audit + isolation at the gate; not a sandbox for an admitted module).
### Membranes — η = 0 is a no-op at `engine.ingest` (review issue #43, item 2)
- `SensorBridge.process` / `observe_targets` check confidence BEFORE any write: an η=0
  report no longer lands in the raw cache (it read as live), the Hebbian input
  (`hebbian_weight_update` learned the signal weight from it: fog `w_scout_place_0`
  1.0 → 1.0000007), the event-dedup value, or `touched_roles`; observe bindings emit no
  alpha=0 target (it fed the observation-trust innovation EMA).
- `SensorBridge.admitted(readings, confidence)` is the set of readings that exist past
  the membrane; `engine.ingest` routes every downstream consumer of `sensor_readings`
  (calibration's range/tracking channels, the Hamiltonian re-bridge, the Hebbian credit)
  through it.
- `TrainingSession._apply_record` replays a recorded event at its recorded `confidence`
  (payload key) instead of η=1.
- Pinned by `tests/test_eta_zero_membrane.py` (4): a 40-tick η=0 stream leaves field hash,
  weights, trust, touched roles and liveness identical to silence, while η=1 does not.
### Engine — determinism by default (review issue #43, item 1)
- `build_engine(..., seed=None)` now threads `seed` to the engine; `BeliefEngine` takes
  `population_seed` and, when no seed is given, derives it from the topology
  (`_topology_seed`) instead of letting `Population` draw `np.random.default_rng(None)`
  from OS entropy. Two blank boots over one stream now share a field hash with the
  population on (they did not before). Snapshots record `population_seed`.
  **Behavior note**: a world's replay hash with the population enabled changes from
  "random every run" to one stable value; nothing could have depended on the old ones.
- Not yet: unseeded cross-process identity (Thompson sampling uses the process-global
  `random`; #43 S8). Pass `seed=` or pin the process RNGs, as `umweltd` does.
### Substrate — two silent defects (review issue #43, items 4–5)
- **ZZ Hebbian learning was dead for roles with an underscore** (`fractal_stack.hebbian_update`):
  the gradient loop recovered a ZZ term's roles by splitting the label on `_`
  ("ZZ_agent_near_safe" → ("agent", "near_safe")), so on fledgeling_fog and every folded
  manifold role the coupling never moved. Terms now resolve through the basis's
  `coupling_map`. Pinned: an underscore role learns the same ZZ value as a plain one.
  **Behavior note**: worlds with such roles now learn couplings they silently never did.
- **`field_unify._state_of` repacked gamma 0.0 as 0.05** (`or GAMMA_PRIOR`); a unitary world
  or meta-scale round-tripped with wrong dynamics and `canon_hash` (gamma-blind) never
  noticed. `is None` now. Pinned by a pack → to_clusters round-trip.

### Ledger hygiene (Opus clear-out, 2026-11-01)
- **CLAIMS.md witnesses itself** (`tests/test_claims_witness.py`): evidence paths must exist,
  cited test node ids must be defined, relative doc links must resolve; foreign-repo rows
  are exempt by marker with the count printed. Caught a fourth stale pointer on its first
  run (FL-core Phase 5 row → `examples/kits/`). Review issue #43, item 10.
- CLAIMS.md: every evidence pointer re-checked against main (111 paths, 11 pinned test
  node ids all collect). Three rows corrected: the gate row now says what the S3 half
  needs and that the CI consumer job is DARK-skipped until its secret exists; the facet-kit
  row pointed at `src/umwelt/kits/`, which moved to `examples/kits/` in July; the Butler
  "PAID" half of the domain-examples row is marked **DARK here** — its only witness is a
  test in the yurt repo that this gate does not run.
- Branch deletion was not possible from the clear-out session (the GitHub proxy refuses
  ref deletes); the 30 fully-merged remote branches are listed in the summary issue with
  the one-line command to remove them.

### S3 consumer
- **Context budget bound made visible (issue #35)**: `worklist._compact_context` stamps
  `projection.budget_compaction` with `hard_bound_bytes` (64000), `target_bytes` (56000,
  now `CONTEXT_SOFT_TARGET_BYTES`), `soft_target_met`, `hard_headroom_bytes` and
  `protected_residue_bytes` per protected section. A context whose owner intent + latest
  evidence exceed the soft target settles between the two numbers and says so on the
  receipt, instead of leaving the bound to the next overflow's exception. The bound and
  the measured residue are documented in `docs/S3_CONTEXT_BUDGET.md`; the fail-closed
  behaviour is unchanged.
- **PR #36 — scoped languages, ant homes and receipt-backed contribution accounts.**
  Versioned hierarchical emoji dictionaries and Bloch-state expressions over existing S3
  semantic records (`scoped_language.py`, `spacewheat_lexicon.py`, `scoped_examples.py`):
  full state, joint references, frames, unknowns and provenance survive readout; reading
  neither executes nor learns; an unknown `p` carries an `unknown_reason`, never 0. Native
  planning takes bounded scoped context (`scoped_planning.py`, `scoped_workspace.py`);
  independent reviews project exact check outcomes and admit persistent ant-authored
  homes (`scoped_review.py`, `ant_home.py`). Opt-in ant economy (`ant_economy.py`,
  `ant_child_admission.py`, docs/S3_ANT_ECONOMY.md) keeps three accounts apart —
  reviewed contribution credit (independent non-baseline review, exact tested bytes),
  forecast accuracy (precommitted questions, owner-attested observations inside the
  window), and currency (owner-attested realized transactions only; seconds, clicks,
  passing tests and predictions cannot mint money; largest-remainder allocation conserves
  every minor unit). Captures under `examples/s3_scoped_language/`.
- Review fixes on merge: the scoped-context overflow message is one constant
  (`scoped_planning.SCOPED_CONTEXT_OVERFLOW`) shared by `resident_context_recovery`,
  which previously matched a string planning never raised (overflow recovery could never
  fire); `ideas.capture` takes an `actor` and ant child invitations are stamped
  `ant-child-invitation`, not `owner-intake`; `ant_economy._route` requires a named idea
  project; a home without review provenance raises `ValueError`, not `KeyError`.
- **PR #34 — review evidence into native repairs and lane continuation.** Worklist routes
  can opt into `candidate_review`: an independently approved read-only referee station
  reviews the exact candidate revision after the builder's pinned checks; rejection
  enters planning context and bounded focused repairs get lane priority
  (`review_followup.py`, `resident_lanes.py`, `resident_growth.py`,
  docs/S3_REVIEW_LOOP.md). The GitHub poller publishes enrolled research findings with
  native task / revision / referee provenance (`research_findings.py`); settled results
  return original receipts through the Desk (`worklist_publication.py`); failed ideas may
  be reconsidered once after recorded remediation (`resident_reconsideration.py`);
  identical duplicated context fields are aliased losslessly before any lossy
  compaction (`worklist._compact_references`); stations may enrol the downstream
  exact-replacements edit protocol and fail before model execution when the loaded
  adapter lacks it (`station_executor._require_edit_protocol`).
- **The engine/application seam is now pinned** (`tests/test_consumer_boundary.py`): no
  engine module may import `umwelt.consumers` or the downstream application's package; `consumers/` is exempt from
  the domain-vocabulary lint (CLAIMS row updated to say so). CI splits into an `engine`
  job (3.10 + 3.12, no S3) and an `s3-consumer` job against the pinned private downstream
  checkout (`docs/CONSUMER_TESTING.md`).
- Review fixes on merge: the `s3-consumer` CI job is **skipped and annotated DARK** when
  the deploy key is unset instead of failing red (a red from a missing
  credential hides engine regressions; a green skip would be false coverage); the
  cap-bridge protocol tests now fake the downstream `fill_edits` in the same package as `fill`
  (they were red on the branch since `_require_edit_protocol` landed); the live-adapter
  protocol test skips with its reason on checkouts predating the downstream edit-protocol change; a cached
  review lookup no longer swallows `KeyError`s from the completion/observe path;
  `_negative_child_evidence` raises `ValueError` (not `KeyError`) for a request outside its
  batch run; a shadowed `policy` local in `resident_growth` renamed.
- **Noise is not learning** (the downstream consumer's manager): an SPSA replay whose step is
  floating-point residue (two identical alternatives give `plus_loss == minus_loss` to
  ~1e-14, a ~1e-15 step) no longer mints a policy version or a `trained` record; moves
  below `POLICY_CHANGE_TOLERANCE` (1e-9) are `no-policy-change`. `test_s3_runtime` was
  passing only when that residue happened to be nonzero; it now trains on a genuinely
  different alternative and a new test pins the symmetric case.

### Notation faces (Penduline-facing, domain-free)
- **Nameplates** (`projection/nameplates.py`): one face per `node.role` — authored
  `AxisFace` poles, live/latent/dark from evidence, Bloch z + purity. No ρ.
- **Confession** (`projection/confession.py`): sensed → believed → forecast →
  decided join; missing organs are null; forecast *used* only if skill-vs-persistence > 0.
  `dispatched` is the OutputSurface live-flag (shadow recommendations stay False).
- **Glass door** (`projection/glass_door.py`): ≤5 tiles; authored `DomainSpec.tiles`
  win; unknown pair is dark.
- **Latent codebook**: `learning/embedder.py` no longer mints `concept_N`; confident
  unlabeled cells stay `label=null` until `ground`.
- Daemon routes: `GET /nameplates`, `/confession`, `/glass_door`. Playground shows
  tiles + one confession sentence first. Docs: `docs/TRANSPARENCY.md`.

### Gauge-theory pass (proper theory under the gauge/git ledger)
- **Docstring honesty**: the `projection/gauge.py` snapshot is documented as a
  coordinate chart (diff-stable by sorted keys + rounding), not an invariant;
  `canon_hash` v1 documented as coordinate-sensitive; README claim #4 downgraded
  to mechanism-shipped/claim-owed until the ledger carries its test.
- **`substrate/link_field.py`** (new): pure-math lattice gauge ledger — U(1)/Z₂
  link phases, gauge transforms, Wilson loops with closure honesty (open walks
  refuse: NaN/None), β₁, spanning-tree gauge fixing, adapters from the engine's
  signed zz/xy structure (`tests/test_link_field.py`).
- **Canon v2 + invariant sheet** (`field_unify.canonicalize` /
  `canon_hash(version=2)` / `invariant_sheet`): the relabeling gauge freedom
  (cluster insertion order, role declaration order) is quotiented by a true
  gauge fixing; field DAG nodes record both hashes + the gauge-invariant sheet,
  so a ledger diff decomposes into gauge motion vs physical motion. v1 stays
  byte-identical and regression-pinned (`tests/test_canon_gauge.py`).
- **Invariance suite** (`tests/test_gauge_invariance.py`): value-axis rotation
  invariance of learned params (+x azimuth convention named at its four sites),
  bridge source/target order proven physicless, the global X/Y frame flip proven
  sheet-invariant, frozen-subsystem rows proven byte-stable under a neighbor's
  learning.
- **BEHAVIOR NOTE — `phase_align`**: the hard-coded `"Y_solar"` sign heuristic is
  gone from engine source (vocabulary law). The flip now fires only on a
  domain-declared `TrainingConfig.phase_reference_label` (e.g. `"Y_solar"`);
  with no label declared, the learned frame is kept. Origin deployments that
  relied on the old default must set the label.
- **Closure honesty** (`continuous_geometry`, `berry_tape`, `projection/gauge`):
  winding/holonomy are closed-curve-only (NaN on open walks; `net_loops` keeps the
  open-path net displacement); the berry tape yields loop invariants only at a
  certified `GeometricReturn`; `field_gauge(field, tape=...)` gains the return-gated
  `_loops` block (`tests/test_closure_honesty.py`).
- **The git leg made real** (`field_gauge.maybe_record_field_node`): Poincaré-strobed
  field-node recording (rest-window gate when a clock cluster exists), git commit
  opt-in via `UMWELT_FIELD_GIT=1`; README #4's empty-git-diff certificate is now a
  pinned test. Also fixed: the growth service's bare-field caller silently never
  recorded its learning event (`tests/test_field_gauge_node.py`; theory:
  `docs/GAUGE.md`).
- **BEHAVIOR CHANGE (default ON) — signed bridges**: bridge calibration learns the
  SIGNED Bloch correlation (was (corr+1)/2 ∈ [0,1]); bridge coupling params are
  minted lo=−1.5 (was 0.0); a negative coupling is a Z₂ link — bridge reconciliation
  transports the partner through the link (σ_x conjugation) instead of identity
  averaging, so an anti-aligned pair is a fixed point; the learned web grows
  sign-seeded edges and keeps healthy anti-correlated ones. **Deployed consumers
  (meerkat, umwelt-market): positive bridges will settle to slightly different
  values (corr, not (corr+1)/2) and genuinely anti-correlated pairs will re-learn
  from weak-positive to negative.** Frustrated bridge triangles now read W = −1
  (`link_field.from_bridges`; `tests/test_signed_bridges.py`).

### Fledgeling core (FL-core Phases 1–5)
- **Host API** (`umwelt.host`): thin game face — `GameHost.register_world` /
  `observe` / `intend` / `beliefs` / `step` / `step_turn`; calibrated
  `Belief(value, confidence)` on the default face (no substrate vectors); η=0 no-op;
  shadow intents leave no live world side effects (`tests/test_host_api.py`).
- **Multi-mind** (`WorldSession`): shared classical ground + N private engines,
  per-observer channel masks, actor-keyed intent logging on the confounding
  surface (`tests/test_multimind_privacy.py`).
- **Agency loop** (`umwelt.host.agency_loop`): sub-routines, attention budget,
  FF pause on surprise/rest, shadow auto-intend only after
  `PromotionGate.min_successes`, explicit live promotion
  (`tests/test_agency_loop.py`, `examples/fledgeling_fog/agency_demo.py`).
- **Fog corridor domain** (`examples/fledgeling_fog/`): public synthetic place
  graph, tick driver (not solar), host-API demo + freeze bake-off + blank-slate
  proof (`proofs/fledgeling_fog_blank.py`).
- **Facet kits** (`umwelt.kits.{fog,attention,market,dream}`): optional cassette
  + baseline + README honesty; dream path never actuates
  (`tests/test_facet_kits.py`).
- Roadmap status: [docs/FLEDGELING_CORE.md](docs/FLEDGELING_CORE.md) (Phase 6
  host-repo integration still open). Ledger rows in CLAIMS.md.

### Engine
- **`foresight/encounter.py` (new, PR #37)**: the encounter field — who a mind is likely to
  meet next and where, from a typed relation graph (ally / adversary polarities), traits ×
  place features, haunts, the observer's ground stance, mood and attention. Seeded
  Gumbel-top-k sampling; every result carries reasons. `relation_bridges` couples a
  dissipative `draw` role along the graph so the field itself spreads word of someone;
  `draw_from_engine` reads it back. Docs: `docs/ENCOUNTER.md`; pinned in
  `tests/test_encounter.py`. Review fixes on merge: the `presence` name was a banned
  engine word (vocabulary lint) and is `arrival` throughout (`EncounterField.arrival`,
  `Weights.arrival_*`, survey key `arrival`) — the Verdance consumer
  (`tools/verdance/company/encounters.py`) must follow the rename; and a node/role the
  field does not hold now reads `None` (dark) instead of 0.5, with `arrival()` giving a
  dark draw no pull.
- `umwelt.spec.validate` — the deterministic spec gate, one reusable command
  (`python -m umwelt.spec.validate module:ATTR`): topology, strict binding
  registration (surfacing exactly what the boot path's membrane guard swallows),
  blank boot, a synthetic exercise proving every binding drives the field (driver-
  role bindings exempt — the ingest path routes them out of band), and a save/load
  round-trip. The shadow law is enforced (`shadow=False` fails unless waived).
- Actor-keyed confounding helpers (`record_actor_intent`, `actor_confounded_now`)
  extend the graph-derived surface for multi-mind hygiene — they do not replace it.

### umweltd
- `spec_path` manifest knob: worlds whose spec module lives outside the installed
  packages (a forge workspace) boot and event-source-recover through the manifest
  alone; the worker prepends it to `sys.path` before the spec ref imports.
- **The playground** (`/ui`): a self-contained browser dashboard served by the
  supervisor — live per-role belief bars, a push-readings panel over the new
  `GET /worlds/<n>/bindings` endpoint, the shadow-decision feed, raw state. Loads
  without auth (static, no world data); its API calls carry the visitor's key.
- **The docs site** (`/docs`): project docs rendered server-side from the checkout
  (plain-terms overview first); `python -m umweltd.docsite --export DIR` writes the
  same pages as a standalone static site. `UMWELTD_UI=off` disables both surfaces.
- `umweltctl bindings --world <w>` and `UmweltClient.bindings()`; `umweltctl
  create --spec-path` exposes the manifest knob from the operator CLI.
- SERVICE.md gained a "Sharing on your LAN" recipe (API key, WSL2 port-forward
  note, trust model).

### umweltforge — the embedded compiler + warden (EXPERIMENTAL)
- `umwelt-forge new <name> --rant "..."`: an embedded coding agent (Claude Agent
  SDK, optional `forge` extra) authors the `DomainSpec` module in a scoped
  workspace; the pipeline independently re-runs the deterministic gate in a fresh
  subprocess and registers the world only on green — a lying agent provably cannot
  register a broken world (test-pinned, offline, no API key in the repo gate).
- `umwelt-forge warden tick <name>`: cron-able one-shot inspection of a running
  world under **earned autonomy** — per-world, per-change-type dials, everything
  defaulting to propose-only; `topology_change` can never auto-apply; the warden
  cannot promote itself (policy hash-checked around the session); every proposal,
  apply, and failure lands on an append-only competence ledger.
- Authoring quality on real rants is unmeasured — evaluation owed (see CLAIMS.md).

### Examples
- `examples/mirror/` — the daemon's self-portrait: a world whose five sensors
  are umweltd's own `/health` telemetry (including the byte-size of its own
  event log, which grows because of the readings that measure it), fed back in
  through `POST /events`. The first sitting witnessed a sibling world's
  operator-initiated outage as a single `gh_alive = 0` reading and recovered;
  it doubles as an end-to-end exercise of the whole service API.

## [0.1.0] - 2026-07-12

First tagged release.

### Engine
- The belief-field engine: `DomainSpec`-driven worlds, blank-slate boot, weak-
  measurement observation (confidence = η, η=0 provably a no-op), forecasting (trust
  web, free-run rollouts), and shadow-first output tendrils with causal self-tagging.
- Membrane cadence (`ingest_hold_s`) for sparse-feed worlds, kept structurally
  separate from compute time and from a domain's own in-universe clocks
  ([docs/TIME.md](docs/TIME.md)).
- Blank boot mixes analog-dissipative beliefs to max entropy — no boot transient.
- Leave-one-out trust-web learning isolates a corrupted source when a referee exists.
- Spec binding validation: a `BindingSpec` targeting a nonexistent node or an
  undeclared role now raises (and is always logged), instead of failing silently or
  warn-only.

### umweltd — the engine as a service
- Supervisor + per-world worker daemon: event-sourced boot (snapshot + log-tail
  replay), API-key auth (constant-time compare), TLS, webhook dispatch for non-shadow
  outputs, gauge discipline (REPLAY on recovery, LIVE once caught up).
- `umweltctl` operator CLI and a `Dockerfile`/`docker-compose.yml` for one-command
  startup.
- Structured logging plus a per-request JSON access log.
- A crash watchdog auto-restarts a world that dies unexpectedly, backing off after
  repeated crashes in a window rather than tight-looping; `UMWELTD_MAX_WORLDS` caps
  world count; `/health` reports per-world disk usage.

### Docs & onboarding
- [docs/NEW_DOMAIN.md](docs/NEW_DOMAIN.md): the one-page checklist for a new domain,
  indexing what used to be scattered across several docs (the dissipative-role law,
  the `gamma_diss` knob, `ingest_hold_s`, the vocabulary lint gate).
- `examples/gridworld/` promoted to a complete, standalone new-domain template —
  also the proof gate's own fixture, so the two can never drift apart.
- README leads with Install + a working Quickstart before the evidence/claims
  framing.

See [CLAIMS.md](CLAIMS.md) for what's measured-and-pinned versus designed-but-owed
versus denied.
