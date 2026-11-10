# Epistemics act-or-abstain harness

**PREEXISTING: built 2026-11-08 before the sprint; contains no sprint result.**
This harness is the instrument, not the result. Anything it prints on synthetic data is a
pipeline demonstration, labelled "synthetic pilot - not the sprint result". The sprint judges
only work done during the sprint; this directory is disclosed as preexisting work.

## The question

> The sprint question is "does honest, calibrated abstention beat self-stated confidence," not "does the engine beat persistence."

An operator must decide, case by case, whether an agent should **act** on its own forecast or
**abstain**. Does a confidence signal *derived from evidence* improve the cost-weighted quality
of those decisions, compared with (A) no signal at all and (B) persistence with a
self-stated confidence?

The signal under test is the engine's `ForecastSurface` confidence: `max(0, skill_vs_persistence)
x purity`, and `None` ("DARK", no evidence) when the belief is frozen or unscored. Substrate
(see `CLAIMS.md`): "Forecast confidence comes from evidence, not absolute skill"; "A world that
declares `forecast_leaves` foresees ITSELF".

## The four arms (same cases, same data, no paid model calls)

| arm | forecast | acts when |
|---|---|---|
| A `always_act` | the engine's forecast | always |
| B `persistence_selfconf` | persistence (predict the last value) | its self-stated confidence (absolute-skill style, the mirage) exceeds a fixed cut |
| C `evidence_conf` | the engine's forecast | the evidence-derived confidence is not `None` and exceeds the pre-registered cut |
| C' `evidence_conformal` | the engine's forecast | the belief is not frozen (else abstain, reason `dark`), the leaf's window holds at least `k` resolved cases (else `warming`), and `1 - q_hat >= cut` (else `lowconf`) |

Arm B's cut is tuned on the pilot just like C's and C''s, so the persistence baseline is as strong as C is allowed to be.
Every abstain records its reason (`dark` / `warming` / `lowconf`).

### Why C' exists

In the first synthetic pilot arm C lost to B. C's confidence is `max(0, skill_vs_persistence) x purity`:
it is *relative*, so it runs high exactly where persistence is bad *and* the engine's own forecast is also
bad (it beats a bad baseline, which is not the same as being right within `tol`). C' replaces it with an
*absolute* signal: how wrong has the engine actually been, on this leaf, recently?

Per (seed, leaf): keep the last `W` absolute residuals `|engine forecast - outcome|` of cases that have
**resolved by now** (`resolve_t <= t`; a forecast issued at `t` is never in its own window, nor are
residuals whose outcome is still in the future). Scale each residual to `[0, 1]` by dividing by the
outcome range 2.0 (outcomes are belief-z in `[-1, 1]`, every source is normalised to that, so `|error| <= 2`).
`q_hat` is the empirical `(1 - alpha)` quantile, taken as the conformal order statistic of rank
`ceil((n + 1)(1 - alpha))` capped at `n`. Confidence is `1 - q_hat`. C's frozen-belief rule is **kept**: a
belief the surface reports as frozen has confidence `None` (CLAIMS: "Forecast confidence comes from
evidence, not absolute skill") and C' abstains with reason `dark`, ahead of any window logic.

### Why W=50, k=10, alpha=0.2 (fixed before any pilot; not tuned on pilot results)

These were set a priori from the structure of the problem and written into `arms.py` and `prereg.json`
before the pilot was run. The pilot chooses only C''s *cut*, exactly as it does C's.

- **alpha = 0.2**: the operator question is "what error would I be surprised to exceed?" - a one-in-five
  surprise rate is the loosest level that still reads as a promise (80%) and, at `W=50`, keeps `q_hat`
  off the sample maximum (rank 41 of 50, not 50). `alpha = 0.05` at `W=50` would be the maximum of the window:
  one outlier would set the confidence.
- **W = 50**: the 80% quantile then rests on about ten tail residuals (sampling noise on the quantile level
  about +-0.06), while the window is only about two cycles of the 24-tick periodic leaf and two to three of the
  ~20-tick parent/dependent flips, so it still follows a change of regime. A leaf yields several hundred
  scored cases per seed, far more than one window.
- **k = 10**: the conformal rank is non-trivial from `n >= 1/alpha - 1 = 4`; 10 is a round number about
  twice that and a fifth of `W`, so the quantile is not read off a handful of points, yet warming costs only the
  first ~10 resolved cases per (seed, leaf) after burn-in.

All three are round numbers chosen for these reasons. They are also a limitation: no sensitivity sweep has
been run, and none will be tuned on the pilot or held-out result.

## Scoring (`score.py`)

- A wrong act costs `c_wrong`; a right act earns 0; an abstain costs `c_abstain` (`c_abstain < c_wrong`).
  "Right" means the forecast is within `tol` of the realized outcome. Loss is the mean over **all** cases.
- Coverage = fraction of cases acted on.
- **Coverage floor**: an arm with coverage below the floor is marked `FAIL_COVERAGE_FLOOR`
  regardless of loss, so winning by abstaining on everything does not count.
- Also reported: hit rate and mean squared error of acted forecasts, and a Brier score of stated
  confidence against hit/miss. Every number carries `n`.
- The numeric values of `c_wrong`, `c_abstain`, `tol` and the floor live in `score.py` and are copied into `prereg.json`; this README quotes none of them.

## Pre-registration (`preregister.py`)

1. A **pilot on synthetic data only**, on pilot seeds disjoint from the held-out seeds.
2. Cuts for B and C: among grid cuts meeting the coverage floor on the pilot, the one with lowest pilot loss (ties to the lowest cut).
3. Margin (minimum loss improvement of C over A *and* over B): the largest of a fixed floor, a shrunk fraction of the pilot improvement, and twice the pilot standard error. The rule is in the script docstring.
4. `prereg.json` records timestamp, git sha, a hash of `arms.py + score.py + world.py`, pilot vs held-out seeds, cuts, margin and its inputs, coverage floor, cost weights, the held-out data source (for a csv: its sha256), and the failure criterion.
5. C' is pre-registered exactly like C: cut by rule 2 (acts iff `1 - q_hat >= cut`), margin by rule 3 with C' in
   place of C (`margin_cprime`). One cut grid (0.00 to 0.99, step 0.01) serves B, C and C': C''s confidence is
   `1 - q_hat / 2`, which compresses toward 1, and a coarser grid would leave it three cuts above the
   tolerance-equivalent 0.875.
6. **Warm-up seeds** (optional, declared): see below. They are recorded in `prereg.json` and refused if they
   overlap pilot or held-out seeds.
7. **Freeze** = the file exists. `run.py` refuses to run without it, or if the code hash differs, or if the seeds overlap. `preregister.py` refuses to overwrite it.

## Fair warm-up (declared)

C's confidence is a learned quantity (an error EMA against persistence), so a cold engine unfairly handicaps
it. `--warmup-seeds` (on `preregister.py`; `run.py` accepts it only as a consistency check against the frozen
list) streams the engine through those seeds **before every scored seed**, pilot and held-out alike: online,
causal, time kept continuous. Warm-up seeds are disjoint from pilot and held-out seeds (`preregister.py`
refuses otherwise, and `run.py` re-checks the frozen file), and no warm-up case is ever scored. C and C' read
the same warmed ledger, so they are warmed identically. B (persistence) has no learned state and needs no
warm-up. A also acts on the engine's forecast, so it benefits too, which is conservative toward the baseline. C''s
residual window is *not* pre-filled from warm-up: it is per-(seed, leaf) state, built only from resolved scored cases.

## Dark rises error (pre-registered check)

When the engine says DARK (abstain by rule), is the realized error of the engine's forecast higher than on the
cases where the arm acts? For every arm that can abstain (B, C, C'; A never does) the held-out cases are split
into **acted / abstained-dark / abstained-warming / abstained-lowconf**, and `RESULTS.md` reports for each bucket
`n` and the mean `|error|` of the **engine's** forecast (not of the arm's own decision forecast; for B those differ).

Pre-registered check, arms C and C' (the arms with a dark bucket): mean `|error|` on abstained-dark exceeds mean
`|error|` on acted by at least a margin set on the pilot by the same rule as the main margin
(`max(floor, shrink x pilot effect, 2 x SE_pilot)`, SE across pilot seeds). A verdict needs at least 30 cases in
both buckets, else it is inconclusive. For C' there is also a calibration row: among acted cases, the fraction
whose engine `|error|` is within the predicted quantile (about `1 - alpha` if calibrated), and the same fraction over
every case that carried a quantile, which separates quantile calibration from the selection of tight windows.

The synthetic source includes the constant `frozen` leaf in both pilot and held-out, because without it no leaf
is ever DARK and the table would be empty. That choice was made from the dark count alone (0 without it). Note
what a constant leaf implies: DARK there means "nothing is moving", and the engine's forecast of a constant is
close to perfect, so this synthetic world is a poor place for dark to *rise* error. Whatever the table says is
reported as is.

## Honest failure criterion

> If arm C does not beat BOTH always-act (A) and persistence with self-stated confidence (B) by at least the pre-registered margin in cost-weighted loss on the held-out cases, or achieves its loss only with coverage below the pre-registered coverage floor, the premise is falsified (or, below the floor, inconclusive and counted as a failure to meet the criterion) and is reported as a negative result.

The same sentence, with C' in place of C, is `failure_criterion_cprime` in `prereg.json`.
`run.py` computes both verdicts mechanically from `prereg.json`: `premise supported`,
`premise falsified`, or `inconclusive (below coverage floor)`. On synthetic data any verdict is
**"synthetic pilot - not the sprint result"**.

## Data sources (`world.py`)

- `synthetic` (default): deterministic by seed; leaves of differing predictability - `steady`
  (slow, persistence is strong), `periodic`, `noisy` (near-white), `parent` (jittered flips),
  `dependent` (flips a fixed lag after `parent`), plus an optional constant `frozen` leaf used to
  pin the DARK behaviour (included in the pilot and held-out source, see the dark-rises-error section).
- `csv` stub: rows of `(timestamp, leaf, value)`, values already in `[-1, 1]`, a regular grid;
  segment `seed` is the `seed`-th block of `ticks` timestamps. Declare it at pre-registration time with
  `preregister.py --heldout-csv PATH`; its sha256 is recorded before any held-out case is generated.
  No network data is fetched.

Each seed gets a fresh engine that learns online and causally (labels used only after they resolve);
there is no training on held-out seeds and no state shared across seeds, except the declared warm-up seeds
(above), which are disjoint from pilot and held-out. The outcome of a case is the
leaf's realized belief-z at the resolution tick (with `force_observe`, a fixed monotone readback of
the observation).

## What is DARK

- **The real dataset**: UNKNOWN until the sprint. It must be chosen and recorded (csv sha256 in
  `prereg.json`) before any held-out case is seen.
- **The numeric threshold** (cuts and margin): set by `preregister.py` from a pilot on synthetic
  data only, then frozen. No number is justified by real data.
- The sprint track, repo, and any model-run budget: UNKNOWN (see the packet).
- Any claim that the signal helps on real data: owed at the sprint.

## Limits

Held-out seeds 201-205 are fresh: seeds 101-105 had been scored once (A/B/C only) before C' existed and C' was
designed in response, so they are not unseen for C'. All results here stay **SYNTHETIC PILOT - NOT THE SPRINT RESULT**.

Synthetic leaves are a mechanism check, not evidence about any real domain. Persistence-relative
skill rewards beating a bad baseline (e.g. on near-white noise), which need not mean the forecast is
*right* within `tol`; this harness exists to find out whether that matters for decisions.
The synergy / O-information case (CLAIMS, DENIED, dense-only) is out of scope.

## Run

```
python examples/epistemics_act_or_abstain/preregister.py --warmup-seeds 8801 8802 8803   # pilot -> prereg.json
python examples/epistemics_act_or_abstain/run.py             # held-out -> results.json, RESULTS.md
python -m pytest -q -p no:cacheprovider tests/test_epistemics_example.py
```
