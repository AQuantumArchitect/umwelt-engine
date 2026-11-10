"""World + data sources for the act-or-abstain harness.

PREEXISTING: built 2026-11-08 before the Apart AI x Epistemics sprint; contains no sprint result.

One DomainSpec: a root plus one node per leaf, each with a single role ("value"), each
force-observed (the belief-z is a fixed monotone readback of the observation, so a leaf's
realized future z is an objectively resolvable outcome). The spec DECLARES `forecast_leaves`,
so `build_engine` attaches the ForecastSurface and `engine.ingest` steps it - nothing is
bolted on (CLAIMS row: "A world that declares `forecast_leaves` foresees ITSELF").

Two pluggable data sources, same interface (`Source`):
  synthetic  deterministic by seed; leaves of differing predictability (the shaman_self /
             contraption pattern): steady, periodic, noisy, parent, dependent (+ optional
             `frozen`, a constant leaf, used to pin the DARK/None behaviour).
  csv        a stub adapter over (timestamp, leaf, value) rows so the sprint can drop in a
             real dataset without code changes. Values must already lie in [-1, 1].
             No network data is ever fetched.

A "seed" is a stream identity. Synthetic: the RNG seed. CSV: the index of a contiguous
segment of `ticks` rows-per-leaf (so disjoint pilot / held-out segments are disjoint seeds).
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec

ROLE = "value"
H_TICKS = 4                  # forecast horizon in ticks (forecast issued at t resolves at t + H_TICKS)
DT_MIN = 1.0                 # minutes per tick (synthetic)
OBS_NOISE = 0.03             # observation noise on every non-frozen synthetic leaf
LAG = 2                      # ticks between a parent flip and the dependent flip

ALL_KINDS = ("frozen", "steady", "periodic", "noisy", "parent", "dependent")
DEFAULT_LEAVES = ("steady", "periodic", "noisy", "parent", "dependent")
_T0 = datetime(2026, 1, 1, 0, 0, 0)
_NAME = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class Source:
    """A data source: which leaves, how fast, and where the rows come from."""
    kind: str = "synthetic"                  # "synthetic" | "csv"
    leaves: tuple = DEFAULT_LEAVES
    dt_min: float = DT_MIN
    horizon_ticks: int = H_TICKS
    csv_path: str | None = None

    @property
    def horizon_min(self) -> float:
        return float(self.horizon_ticks) * float(self.dt_min)

    def sensor(self, leaf: str) -> str:
        return f"{leaf}_{ROLE}"

    def describe(self) -> dict:
        d = {"kind": self.kind, "leaves": list(self.leaves), "dt_min": self.dt_min,
             "horizon_ticks": self.horizon_ticks}
        if self.kind == "csv":
            d["csv_path"] = self.csv_path
            d["csv_sha256"] = file_sha256(self.csv_path)
        return d


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_source(leaves: tuple = DEFAULT_LEAVES) -> Source:
    bad = [k for k in leaves if k not in ALL_KINDS]
    if bad:
        raise ValueError(f"unknown synthetic leaf kinds: {bad}; choose from {ALL_KINDS}")
    return Source(kind="synthetic", leaves=tuple(leaves))


def source_from_description(desc: dict) -> Source:
    """Rebuild a Source from `Source.describe()` (as stored in prereg.json)."""
    if desc["kind"] == "synthetic":
        return Source(kind="synthetic", leaves=tuple(desc["leaves"]), dt_min=desc["dt_min"],
                      horizon_ticks=desc["horizon_ticks"])
    if desc["kind"] == "csv":
        if file_sha256(desc["csv_path"]) != desc["csv_sha256"]:
            raise RuntimeError("csv data file changed since pre-registration (sha256 mismatch)")
        return csv_source(desc["csv_path"], horizon_ticks=desc["horizon_ticks"])
    raise ValueError(f"unknown source kind {desc['kind']!r}")


# --------------------------------------------------------------------------- spec
def make_spec(source: Source) -> DomainSpec:
    for leaf in source.leaves:
        if not _NAME.match(leaf):
            raise ValueError(f"leaf name {leaf!r} must match [A-Za-z0-9_]+")
    return DomainSpec(
        name="epistemics_act_or_abstain",
        nodes=(
            NodeSpec("world", parent=None, kind="root", roles=()),
            *(NodeSpec(leaf, parent="world", roles=(ROLE,)) for leaf in source.leaves),
        ),
        bindings=tuple(
            BindingSpec(source.sensor(leaf), zone=leaf, role=ROLE,
                        normalizer={"type": "range", "lo": -1.0, "hi": 1.0},
                        force_observe=True)
            for leaf in source.leaves
        ),
        drivers=(DriverSpec("day", period_s=86400.0),),
        forecast_leaves=tuple((leaf, ROLE) for leaf in source.leaves),
        forecast_horizons_min=(source.horizon_min,),
        forecast_context_surprise=True,        # each leaf sees the others' collapse-surprise
    )


SPEC = make_spec(synthetic_source())


# --------------------------------------------------------------------------- synthetic
def _synthetic_series(seed: int, ticks: int) -> dict:
    """All synthetic kinds for one seed, generated in a fixed order so the subset a source
    selects never changes any leaf's values (deterministic by seed)."""
    rng = np.random.default_rng(int(seed))
    k = np.arange(ticks)
    out: dict = {}
    out["frozen"] = np.full(ticks, 0.5)
    # steady: slow mean-reverting AR(1) - persistence is a strong honest baseline
    x, ar = 0.0, np.empty(ticks)
    for i in range(ticks):
        x = 0.97 * x + 0.04 * rng.normal()
        ar[i] = x
    out["steady"] = ar
    # periodic: a clean cycle with a seed-dependent phase
    out["periodic"] = 0.8 * np.sin(2 * np.pi * k / 24.0 + rng.uniform(0, 2 * np.pi))
    # noisy: near-white - nothing to foresee beyond the mean
    out["noisy"] = np.clip(rng.normal(0.0, 0.5, ticks), -0.9, 0.9)
    # parent: a held +-0.8 level that flips on a jittered schedule (jitter the clock cannot foresee)
    par, level, nxt = np.empty(ticks), 0.8, 20 + int(rng.integers(-5, 6))
    flips: list[int] = []
    for i in range(ticks):
        if i >= nxt:
            level = -level
            flips.append(i)
            nxt = i + 20 + int(rng.integers(-5, 6))
        par[i] = level
    out["parent"] = par
    # dependent: flips exactly LAG ticks after each parent flip (single-parent dependency)
    dep, level = np.empty(ticks), 0.8
    due = {f + LAG for f in flips}
    for i in range(ticks):
        if i in due:
            level = -level
        dep[i] = level
    out["dependent"] = dep
    for name in ALL_KINDS:
        if name != "frozen":
            out[name] = out[name] + rng.normal(0.0, OBS_NOISE, ticks)
    return {n: np.clip(v, -1.0, 1.0) for n, v in out.items()}


def synthetic_stream(source: Source, seed: int, ticks: int):
    series = _synthetic_series(seed, ticks)
    for i in range(ticks):
        now = _T0 + timedelta(minutes=source.dt_min * i)
        yield now, {source.sensor(l): float(series[l][i]) for l in source.leaves}


# --------------------------------------------------------------------------- csv stub
@lru_cache(maxsize=4)
def _read_csv(path: str):
    """(timestamp, leaf, value) rows -> (sorted timestamps, {ts: {leaf: value}}, leaves, dt_min)."""
    by_ts: dict = {}
    leaves: set = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            ts_s, leaf, val_s = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                val = float(val_s)
            except ValueError:
                continue                      # header or junk line
            ts = datetime.fromisoformat(ts_s)
            if not -1.0 <= val <= 1.0:
                raise ValueError(f"csv value {val} outside [-1, 1]; pre-normalize the dataset")
            by_ts.setdefault(ts, {})[leaf] = val
            leaves.add(leaf)
    stamps = sorted(by_ts)
    if len(stamps) < 3:
        raise ValueError("csv needs at least 3 timestamps")
    gaps = sorted((b - a).total_seconds() for a, b in zip(stamps, stamps[1:]))
    return tuple(stamps), by_ts, tuple(sorted(leaves)), gaps[len(gaps) // 2] / 60.0


def csv_source(path: str, horizon_ticks: int = H_TICKS) -> Source:
    _, _, leaves, dt_min = _read_csv(path)
    return Source(kind="csv", leaves=leaves, dt_min=dt_min, horizon_ticks=horizon_ticks, csv_path=path)


def csv_stream(source: Source, seed: int, ticks: int):
    """Segment `seed` = rows [seed*ticks, (seed+1)*ticks) of the timestamp-sorted table.
    Assumes a regular grid (spacing = the median gap); a leaf missing at a timestamp is skipped."""
    stamps, by_ts, _, _ = _read_csv(source.csv_path)
    lo, hi = int(seed) * ticks, (int(seed) + 1) * ticks
    if hi > len(stamps):
        raise ValueError(f"csv segment {seed} needs timestamps up to {hi}; file has {len(stamps)}")
    for ts in stamps[lo:hi]:
        yield ts, {source.sensor(l): v for l, v in by_ts[ts].items() if l in source.leaves}


def stream(source: Source, seed: int, ticks: int):
    if source.kind == "synthetic":
        return synthetic_stream(source, seed, ticks)
    if source.kind == "csv":
        return csv_stream(source, seed, ticks)
    raise ValueError(f"unknown source kind {source.kind!r}")
