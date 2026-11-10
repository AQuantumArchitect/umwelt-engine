# Forecasting — the future as first-class beliefs

## A sensor is a forecast with horizon 0

The engine's central foresight identity: a physical signal, a forecasting model, and an
upstream engine's published belief are the same KIND of thing — an estimate of a leaf's
value with a confidence attached. So there is ONE fusion operator for all three problems:

- **sensor health** — a flaky signal's learned reliability reroutes weight to its peers;
- **forecast ensembling** — model predictions fuse with live readings at their own η;
- **engine chaining** — an upstream engine's beliefs enter downstream as one more leaf.

That operator is the **trust web** (`umwelt.foresight.trust_web`; optional denser
variant in `qubit_trust_web`): one learned per-leaf fuser, prior-initialized so day-1
behavior is
provably the confidence-weighted average — turning it on changes nothing until it has
evidence to learn from. Opt-in at the ingress membrane (`UMWELT_TRUST_WEB`); webs-off is
byte-equivalent to last-wins.

## Confidence is learned, not asserted

`umwelt.learning.observation_trust` estimates each source's η online from its innovation
EMA — a source whose surprises systematically exceed its admitted confidence prunes
itself out of the fusion. Downstream, the confidence contract holds everywhere: η=0 is a
provable no-op (docs/THEORY.md).

A forecast row's confidence follows the same rule. `ForecastSurface` publishes
`max(0, skill_vs_persistence) × purity` — foresight over predict-last-value, not the absolute
`skill`, which a frozen belief inflates to ~1. With no persistence baseline to grade against
(a frozen belief, or no scored update yet) the confidence is `None` (DARK), and
`consume_targets()` omits that leaf: unknown pulls nothing. `DissolvedForecastSurface` scores
nothing, so it asserts no skill (`None`); its confidence is purity × exp(−H/τ), decaying with
horizon. Every prediction row carries `issued_at` and `age_s` (seconds since issue).

Both surfaces refresh geometry from the current observation. A missing or
unreadable leaf, invalid declared purity, or invalid trained-path Bloch-z makes
confidence DARK and removes the downstream target. The previously issued value
and its original timestamps remain inspectable; they do not renew themselves
when an observation disappears. A later valid observation can produce a new
consumable forecast. Endpoint roundoff is limited to eight float32 ulps, matching
the stored moment precision. Valid existing backend weighting is unchanged:
joint purity, local radius and empirical calibration remain distinct quantities.
This geometry weighting is a declared heuristic, not measured calibration.

### Issued forecasts earn credit; fitting does not

Leaf rows now name `scoring_contract: issued-leaf-forecast/1`. Each pending
forecast retains its emitted prediction (including binary output clipping),
persistence baseline, due time and allowed label lateness. Its error is measured
against that frozen prediction, even if learning has changed the weights before
the outcome arrives. Gradient updates cannot improve an already issued score.
Other `OnlineRegressor` callers retain their existing post-fit diagnostics; those
diagnostics alone are not prospective forecast evidence.
Published rows freeze `horizon_min`, `label_lateness_minutes` and `label_deadline`
at issuance too; changing future priors cannot rewrite an existing row's contract.

`label_lateness_minutes` is an explicit engineering prior, initially **0**:
the observation must match the requested time exactly. A consumer with a declared
sampling window may supply a finite, nonnegative allowance to `LeafForecaster`
or `ForecastSurface`; it is frozen for each pending forecast. The declaration
is also available through `DomainSpec.forecast_label_lateness_min`.
The two-minute `shaman_self` stream explicitly declares one minute of lateness
for its odd-minute horizons: [55,56] and [89,90], not unobserved exact55/89 labels.
Its historical scores used an undeclared late-label rule and are not a calibration
result for this explicit windowed contract. Changing the
allowance cannot rescue already expired predictions. Choose and compare future
windows against the consumer's actual observation cadence and forecast objective;
do not enlarge a window retrospectively to improve a score. No interpolation or
last-value substitution is implicit. Irregular feeds can therefore produce more
censoring and less confidence than before, which is honest missing evidence.

The leaf snapshot distinguishes training updates, prospectively scored outcomes,
pending labels and censored labels. Missing initial predictions and offline
warm-start epochs may train weights, but cannot earn forecast confidence.
Predict-only evaluation neither trains nor scores. The existing demonstration
that a six-step cyclic tape outranked a seven-step tape reversed under the new
issued-prediction score (0.5071 versus0.6588 on that fixture); the old post-fit
advantage is withdrawn, not retuned into existence.

Independent trials must call `start_sequence()` before reusing a clock origin.
It retains trained weights and previously recorded scores, but censors unresolved
labels and clears held values, issued predictions and transient surprise context.
A backwards clock without that explicit boundary is refused. The contraption
and synergy harnesses now begin a fresh field and forecast sequence per seed;
the earlier harness retained pending labels across rewound clocks, so its stated
multi-seed training did not actually consume all those labels as intended.
Historical results remain historical; corrected protocol results must be reported
separately against the same acceptance thresholds.
The corrected default CLI (six training seeds, four held-out seeds,1200 ticks)
also returns PARTIAL: control advantage0.0748, dependent-leaf advantage0.1815,
hard-case advantage−0.1028, with4768 matched predictions per leaf and7152 training
updates per leaf/model. This is a different cohort from the smaller CI gate;
the dependent-leaf improvement there does not rescue the failed control or
retroactively turn the CI gate's0.0975 into a pass.

For the field surface the outcome is the leaf's observed Bloch-z at the declared
time. This validates attribution of that internal forecast; it does not establish
independent physical truth, useful routing gain, calibration, or AGI. Consumer
acceptance and comparison on independent future outcomes remain separate gates.

## Restart continuation

Engine snapshots carry `forecast-continuation/1` for the learned forecast surface:
full-precision weights and quality history, unresolved original predictions with
their frozen windows and persistence anchors, current predictions, and cross-leaf
context. The rounded `snapshot()` display is not a persistence record. Loading
validates the whole block before changing any engine state, then restores existing
surface/leaf/regressor objects so references remain valid. Changed surface or
feature geometry, malformed versions, invalid numbers and inconsistent clocks are
refused. A bad live model cannot overwrite the prior snapshot through this path.

The registry guard records feature-builder names, order and callable identity;
it does not prove unchanged implementation of arbitrary user driver code. Use a
matching source/configuration revision. Missing legacy blocks retain historical
load behavior and earn no new continuation claim. Dissolved rollouts do not carry
a learned bank. The existing overall continuation verdict stays in force: exact
forecast restoration does not establish full engine restoration. This block uses
the existing trusted pickle container, not a new untrusted-input format.
### Refused observations preserve the learner

The shared online regressor admits only finite, correctly shaped observations,
baselines and features. Learning rates and L2 penalties must be finite and
nonnegative; EMA coefficients lie in [0,1]. These are arithmetic conditions, not
learned performance optima. It computes the candidate weights and diagnostics
before committing either, so invalid input or numerical overflow raises without
changing weights, coefficients, counters or quality history. A subsequent valid
observation can still learn. Existing nonfinite weights are refused and require
an explicit reset; rejection does not silently erase historical corruption.

## Free-run rollouts

The engine passes its effective learning gate to the attached forecast surface.
Frozen forebrain/test contexts can publish predictions but do not fit, score, or
enqueue training labels. Existing explicit stream-learning ticks still train.
Returning to live learning censors expired pending labels; frozen observations
are not retrospectively inserted into the training queue. The dissolved rollout
accepts the same interface and has no online regressor to train in either mode.

`umwelt.foresight.forecast_rollout` runs the field's own dynamics forward — snapshot,
evolve without input, read the future beliefs, restore. The forecast is the model
imagining, not a separate regressor bolted on; where a deterministic future is KNOWN (a
periodic driver's phase), the dissolved form serves the exact label instead of an
approximation of it. Forecasts the engine consumes re-enter through ordinary bindings
(`forecast_zflip` normalizer) at their recorded confidence — and REPLAY preserves that
confidence, so training equals deployment gauge-for-gauge (`umwelt.events`).

## Bounded context construction

With a positive `max_context_features`, the forecast surface samples combination
ranks before constructing monomials. It retains the historical deterministic
feature identities and ordering wherever the old NumPy sampler supported the
population; learned coefficients keep their coordinates. Larger integer spaces
use bounded sampling without enumerating the population. The existing preference
for all first-order atoms when they fit is unchanged. `max_context_features=0`
still explicitly requests the uncapped basis and may be expensive.

The capped path allocates selected features, not every discarded candidate. This
is an allocation repair, not evidence that a higher interaction order predicts
better; that choice still needs prospective comparison.

## Dreaming

At rest, `umwelt.foresight.dreaming` mutates shadow copies against replayed evidence and
`dream_topology` proposes coupling growth — discover-and-record first, graft only when
proven. Region/stream vocabularies are registries; the engine ships them empty.

## What's owed

Trust-web A/B effect sizes and any domain's forecast-skill claims are OWED until measured
(see CLAIMS.md). The one measured verdict shipped with the library is a negative one, and
it stays: the fanciest estimator rung lost to an α-blend on the origin deployment's real
data (docs/THEORY.md).
