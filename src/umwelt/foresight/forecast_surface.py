"""
ForecastSurface — the forecast brain's readout: every comprehended leaf learns
to anticipate its OWN future Bloch-z, at fractal (φ-separated) timescales.

This is the third brain in the braid (forecast → live → hind). It runs the
`LeafForecaster` machinery as a *bank* over a curated forecast allowlist
(`DEFAULT_FORECAST_LEAVES`, empty by default — a domain supplies its leaves) × a
ladder of Fibonacci/golden-ratio horizons. Each leaf reads its current Bloch-z
from the field and predicts that same z forward H minutes; the "fractal
timescales" are simply the same leaf run at several H at once.

The product is a *post-comprehended forecast stream*: per leaf, a predicted
future z and a `confidence = forecast skill × belief purity`. That stream flows
downstream into the live brain as a confidence-gated observation (the shipped
confidence contract: `observe_qubit(target, alpha × confidence)`), so a forecast with no
foresight over persistence is a provable no-op and a bad forecast can never hijack the
live belief — it can only gently nudge. The evidence is `skill_vs_persistence` (signed,
graded against predict-last-value), NOT the absolute `skill`, which a frozen belief
inflates to ~1 for free. With no persistence baseline to grade against (a frozen belief,
or too few scored updates) the confidence is DARK (`None`), never a number.

The comprehension core never references this surface; it attaches as
`engine.forecast_surface`.

Forecast emission is a foresight concern (distinct from the action-router egress
surface): the allowlist of forecast leaves lives HERE, self-contained.
"""
from __future__ import annotations
from umwelt._util import clamp01

import logging
import math
from datetime import datetime
from numbers import Real

from umwelt.foresight.leaf_forecast import LeafForecaster
from umwelt.foresight.forecast_rollout import _GEOMETRY_ROUNDOFF, _unit_geometry

logger = logging.getLogger(__name__)

# The curated forecast allowlist — the high-utility (node, role) leaves scored
# against observed Bloch-z at expiry. Empty by default (domain-free); a domain (or
# the app) passes its own leaves in, or the origin deployment's example registers
# them. Kept HERE so forecast emission stays self-contained — it does NOT belong on
# the action-router egress surface.
DEFAULT_FORECAST_LEAVES: tuple[tuple[str, str], ...] = ()

# Fractal timescales: Fibonacci minutes (consecutive ratios → φ). The same leaf
# anticipated at several horizons at once is the "fractal" forecast — near
# horizons are actionable anticipation, far ones carry slow structure.
DEFAULT_HORIZONS_MIN: tuple[float, ...] = (13.0, 21.0, 34.0, 55.0)

# The leaf whose near-horizon prediction is published for the live brain to
# consume is the SHORTEST horizon (the most actionable anticipation). The full
# ladder rides the snapshot for observability + any future deeper consumer.


class ForecastSurface:
    """A bank of per-leaf forward models over the field's comprehension leaves.

    bank[(node, role, horizon)] = LeafForecaster predicting that leaf's Bloch-z
    H minutes ahead. Leaves come from the curated `DEFAULT_FORECAST_LEAVES`
    allowlist (the compute/memory throttle — all-leaves × all-horizons is the cliff).
    """

    def __init__(
        self,
        leaves: tuple[tuple[str, str], ...] | None = None,
        horizons_min: tuple[float, ...] = DEFAULT_HORIZONS_MIN,
        lr: float = 0.03,
        l2: float = 0.005,
        n_context: int = 0,
        surprise_decay: float = 0.85,
        interaction_order: int = 1,
        max_context_features: int = 64,
        label_lateness_minutes: float = 0.0,
    ):
        self.leaves = tuple(leaves) if leaves is not None else DEFAULT_FORECAST_LEAVES
        self.horizons = tuple(float(h) for h in horizons_min)
        # n_context > 0 turns on the SELF-DISCOVERING HIGHER-ORDER CONTEXT. The atom pool is, per
        # leaf, TWO signals: its current state z, and its decaying collapse-surprise trace (peak-hold
        # of |Δz| — "how recently did it collapse"). The context is the set of MONOMIALS (products) of
        # those atoms up to `interaction_order`: order 1 = the raw atoms (first-order), order 2 adds
        # every pairwise product z_i·z_j / z_i·trace_j / trace_i·trace_j, etc. Every leaf sees the
        # same monomial vector and its linear layer (with L2) SELECTS which monomials predict its own
        # future — a hallucinated higher-order layer whose readout checks what holds. This is what
        # forecasts genuine SYNERGY (an XOR/parity target where no first-order feature carries signal
        # but a product does), which the earlier trace-only order-1 context could not. When the full
        # monomial set exceeds max_context_features, a DETERMINISTIC random subset is kept (all
        # first-order atoms + a hallucinated sample of the higher-order products). Default n_context=0
        # → channel empty, construction/stepping byte-identical to a context-free surface.
        self.surprise_decay = float(surprise_decay)
        self.interaction_order = max(1, int(interaction_order))
        self._monomials: tuple[tuple[int, ...], ...] = ()
        if int(n_context):
            from umwelt.foresight.context_features import select_monomials
            n_atoms = 2 * len(self.leaves)                 # per leaf: z (even index), trace (odd)
            self._monomials = select_monomials(n_atoms, self.interaction_order,
                                               int(max_context_features))
        self.n_context = len(self._monomials)
        self.bank: dict[tuple[str, str, float], LeafForecaster] = {}
        for node, role in self.leaves:
            for h in self.horizons:
                # Forecast in raw Bloch-z space: continuous, centred at 0, unit
                # scale (z ∈ [-1, +1]). _denorm returns the value directly, so the
                # prediction IS a Bloch-z we can publish and re-observe.
                self.bank[(node, role, h)] = LeafForecaster(
                    leaf_id=f"{node}_{role}@{int(h)}m",
                    sensor_id=f"forecast_{node}_{role}",
                    node=node, role=role, binary=False,
                    center=0.0, scale=1.0, horizon_minutes=h, lr=lr, l2=l2,
                    n_context=self.n_context,
                    label_lateness_minutes=label_lateness_minutes,
                )
        # Last belief purity read per (node, role) — the OUTPUT-side confidence,
        # multiplied with each leaf's forecast skill (INPUT-side confidence).
        self._purity: dict[tuple[str, str], float] = {}
        # The `now` of the last step() — the reference clock for a prediction row's age.
        self._now: datetime | None = None
        # Last Bloch-z per leaf → the realized displacement that feeds the context channel.
        self._last_z: dict[tuple[str, str], float] = {}
        # Decaying collapse-surprise trace per leaf (leaky integral of |Δz|).
        self._trace: dict[tuple[str, str], float] = {}

    # -- live step: read each leaf's Bloch-z, predict it forward ----------------
    def start_sequence(self) -> None:
        """Retain learned maps, clear transient context across independent trials."""
        for fc in self.bank.values():
            fc.start_sequence()
        self._now = None
        self._last_z.clear()
        self._trace.clear()
        self._purity.clear()

    def step(self, now: datetime, field,
             surprise: dict[tuple[str, str], float] | None = None,
             train: bool = True) -> None:
        """Read each comprehension leaf's current Bloch-z from the field and feed it to
        every horizon's forecaster (online delayed-label learning).

        When n_context>0, build ONE stable-order cross-leaf context vector — the per-leaf
        realized displacement |Δz| this tick ("which beliefs just moved / collapsed"), or an
        external `surprise` override if supplied — and hand it to every leaf, so a leaf whose
        future depends on another belief's collapse can learn to anticipate it. n_context==0
        (the default) → context stays None → byte-identical to the context-free surface."""
        if self._now is not None and now < self._now:
            raise ValueError("forecast clock moved backward; start_sequence for an independent timeline")
        self._now = now
        # Confidence must describe this observation, not the last successful
        # sample of a leaf that is now missing or unreadable. Forecasts themselves
        # retain their original identity/times for inspection.
        self._purity.clear()
        # Pass 1: read z + purity, and the realized displacement since last tick.
        cur: dict[tuple[str, str], float] = {}
        disp: dict[tuple[str, str], float] = {}
        for node, role in self.leaves:
            cluster = field.clusters.get(node)
            if cluster is None:
                continue
            idx = cluster.role_index.get(role)
            if idx is None:
                continue
            try:
                raw_z = cluster.qubit_bloch(idx)[2]
                purity = _unit_geometry(cluster.purity)
                if (purity is None or isinstance(raw_z, bool) or not isinstance(raw_z, Real)
                        or not -1 - _GEOMETRY_ROUNDOFF <= raw_z <= 1 + _GEOMETRY_ROUNDOFF):
                    continue
                z = float(raw_z)
                if not math.isfinite(z):
                    continue
                z = max(-1., min(1., z))
                self._purity[(node, role)] = purity
            except Exception:
                continue
            cur[(node, role)] = z
            disp[(node, role)] = abs(z - self._last_z.get((node, role), z))
            self._last_z[(node, role)] = z
        # Advance the decaying collapse-surprise trace (recency of |Δz|), so a collapse a few
        # ticks ago is still visible when a dependent leaf needs it (an impulse would already be 0).
        if self.n_context:
            for key in self.leaves:
                d = float(surprise.get(key, 0.0)) if surprise is not None else disp.get(key, 0.0)
                # decaying PEAK-HOLD: jump to the full collapse magnitude |Δz|, then decay by
                # `surprise_decay` each tick. Preserves the full spike (an EMA would attenuate it by
                # (1−decay) and starve the signal) AND is bounded by max|Δz|≈2 for ANY decay, so a
                # long trace timescale (decay→1, to span temporally-separated parent collapses) is
                # numerically safe — no unbounded leaky-sum overflow.
                self._trace[key] = max(self.surprise_decay * self._trace.get(key, 0.0), d)
        # Build the shared monomial context once: atom pool = [z, trace] per leaf, then every
        # monomial's product. Every leaf sees the same vector; each leaf's linear layer selects the
        # monomials that predict ITS own future (a hallucinated higher-order layer + a checking readout).
        ctx = None
        if self.n_context:
            atoms: list[float] = []
            for key in self.leaves:
                atoms.append(cur.get(key, 0.0))            # z
                atoms.append(self._trace.get(key, 0.0))    # decaying collapse trace
            ctx = []
            for mono in self._monomials:
                p = 1.0
                for j in mono:
                    p *= atoms[j]
                ctx.append(p)
        # Pass 2: step every present leaf forward with the shared context.
        for node, role in self.leaves:
            if (node, role) not in cur:
                continue
            z = cur[(node, role)]
            for h in self.horizons:
                self.bank[(node, role, h)].step(now, z, context=ctx, train=train)

    def _confidence(self, fc: LeafForecaster) -> float | None:
        """confidence = evidence × belief purity, in [0, 1]; None (DARK) without evidence.

        Evidence is `max(0, skill_vs_persistence)`: foresight OVER predict-last-value. A model that
        merely ties or loses to persistence earns 0 (a provable no-op downstream). When there is no
        persistence baseline to grade against — no scored update yet, or persistence is ~perfect
        (a frozen belief, svp reported 0.0 by construction) — the evidence is UNKNOWN, so the
        confidence is None, not a default number. Unknown purity is dark too."""
        reg = fc.fc
        if reg.error_ema is None or reg.baseline_error_ema is None or reg.baseline_error_ema < 1e-9:
            return None
        purity = _unit_geometry(self._purity.get((fc.node, fc.role)))
        if purity is None:
            return None
        return clamp01(max(0.0, float(reg.skill_vs_persistence)) * purity)

    def _age_s(self, fc: LeafForecaster, now: datetime | None) -> float | None:
        ref = now if now is not None else self._now
        if ref is None or fc.issued_at is None:
            return None
        return max(0.0, (ref - fc.issued_at).total_seconds())

    # -- the full fractal ladder (observability) --------------------------------
    def predictions(self, now: datetime | None = None) -> dict[tuple[str, str, float], dict]:
        """Every (node, role, horizon) with a live prediction → {z_pred, confidence (None = DARK),
        skill, horizon_min, prediction_for, issued_at, age_s}. `age_s` is seconds since the
        prediction was issued, against `now` (default: the last step's clock)."""
        out: dict[tuple[str, str, float], dict] = {}
        for key, fc in self.bank.items():
            if fc.prediction is None:
                continue
            conf = self._confidence(fc)
            out[key] = {
                "scoring_contract": "issued-leaf-forecast/1",
                "label_lateness_minutes": fc.issued_lateness_min,
                "label_deadline": fc.label_deadline.isoformat() if fc.label_deadline else None,
                "z_pred": float(fc.prediction),
                "confidence": None if conf is None else round(conf, 5),
                "skill": round(float(fc.fc.skill), 5),
                "skill_vs_persistence": round(float(fc.fc.skill_vs_persistence), 5),
                "horizon_min": fc.issued_horizon_min,
                "prediction_for": fc.prediction_for.isoformat() if fc.prediction_for else None,
                "issued_at": fc.issued_at.isoformat() if fc.issued_at else None,
                "age_s": self._age_s(fc, now),
            }
        return out

    # -- the consumable near-horizon stream (the braid feed) --------------------
    def consume_targets(self, now: datetime | None = None) -> dict[tuple[str, str], dict]:
        """Per (node, role), the SHORTEST-horizon prediction — the actionable
        anticipation the live brain consumes. One value per leaf so the downstream
        consume binding stays a single sensor_id per leaf. A DARK leaf (no evidence) is
        omitted: an unknown confidence pulls nothing, it is not coerced to a number."""
        h0 = min(self.horizons) if self.horizons else None
        out: dict[tuple[str, str], dict] = {}
        if h0 is None:
            return out
        for node, role in self.leaves:
            fc = self.bank.get((node, role, h0))
            if fc is None or fc.prediction is None:
                continue
            conf = self._confidence(fc)
            if conf is None:
                continue
            out[(node, role)] = {
                "scoring_contract": "issued-leaf-forecast/1",
                "label_lateness_minutes": fc.issued_lateness_min,
                "label_deadline": fc.label_deadline.isoformat() if fc.label_deadline else None,
                "z_pred": float(fc.prediction),
                "confidence": round(conf, 5),
                "horizon_min": fc.issued_horizon_min,
                "issued_at": fc.issued_at.isoformat() if fc.issued_at else None,
                "age_s": self._age_s(fc, now),
            }
        return out

    # -- offline warm-start (optional) ------------------------------------------
    def pre_train(self, z_series_by_leaf: dict[tuple[str, str], list[tuple[datetime, float]]],
                  epochs: int = 2) -> int:
        """Warm-start from per-leaf Bloch-z series `{(node, role): [(dt, z)]}`. Each
        horizon's forecaster pre-trains on the same series at its own H. Returns the
        total (feature → label) pairs fit. Optional — the online step() learns too."""
        total = 0
        for (node, role), series in z_series_by_leaf.items():
            for h in self.horizons:
                fc = self.bank.get((node, role, h))
                if fc is None:
                    continue
                total += fc.pre_train(series, epochs=epochs)
        return total

    def snapshot(self) -> dict:
        return {
            "leaves": len(self.leaves),
            "horizons_min": [round(h) for h in self.horizons],
            "predictions": {
                f"{n}.{r}@{int(h)}m": p
                for (n, r, h), p in self.predictions().items()
            },
        }
