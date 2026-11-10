"""EncounterField — who a mind is likely to meet next, and where.

A world is full of people the observer has not met. Some are drawn toward it and some are not;
some will turn up on its own ground and some will wait where the ground is disputed. This module
turns three things the domain already has into that forecast:

  • a relation graph between people, typed by polarity (attract, bond, rivalry, tension, ...);
  • each person's and each place's character on the domain's own trait axes;
  • the observer's beliefs: who is with it now, how each place stands (ours / disputed / theirs),
    its own mood on the same axes, and a per-person DRAW belief held in the engine field.

The draw is the engine's part. Give every person node a dissipative `draw` role and couple bonded
people with `relation_bridges(...)`: an observation that someone is near (word of them, a sighting,
a bonded companion joining) then spreads along the relation graph through the field itself, and
fades toward the open centre when nobody hears of them. `draw_from_engine(...)` reads it back.

Polarity decides HOW someone turns up, not just whether:
  ally polarities (attract, bond, sibling, mentor, mirror) pull a person toward ground the
  observer holds; adversary polarities (rivalry, tension, friction) draw them just as surely but
  they wait on disputed ground. Both kinds raise the chance of meeting at all.

Everything is deterministic for a seed (Gumbel-top-k over the scores), and every placement carries
its reasons, so a host can show the player WHY someone is where the mind believes they are.
Pure functions over plain data: no engine import is needed except in the two field helpers.

See docs/ENCOUNTER.md.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

ALLY = "ally"
ADVERSARY = "adversary"

# polarity -> (side, strength). Strength scales the pull; side decides where they appear.
POLARITY: dict[str, tuple[str, float]] = {
    "bond": (ALLY, 1.0),
    "attract": (ALLY, 1.0),
    "sibling": (ALLY, 0.9),
    "mentor": (ALLY, 0.8),
    "mirror": (ALLY, 0.6),
    "rivalry": (ADVERSARY, 0.9),
    "tension": (ADVERSARY, 0.7),
    "friction": (ADVERSARY, 0.5),
}


@dataclass(frozen=True)
class Entity:
    """Someone the observer may meet. traits: axis -> [-1, 1]. haunts: place or region id ->
    weight (a soft prior: where they have been known to be). anchored_at: a place id they never
    leave (an encounter there is a fixture, not a draw)."""
    id: str
    traits: Mapping[str, float] = field(default_factory=dict)
    haunts: Mapping[str, float] = field(default_factory=dict)
    anchored_at: str | None = None
    label: str = ""


@dataclass(frozen=True)
class Place:
    """Somewhere they could be. features: axis -> [-1, 1] on the SAME axes as the traits.
    stance: the observer's belief about the ground, +1 ours ... 0 nobody's ... -1 theirs.
    contest: [0, 1], how disputed it is believed to be. known: [0, 1], how confidently the
    observer believes any of this (a dark place pulls no one by stance)."""
    id: str
    region: str
    features: Mapping[str, float] = field(default_factory=dict)
    stance: float = 0.0
    contest: float = 0.0
    known: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class Relation:
    a: str
    b: str
    polarity: str
    weight: float = 1.0          # bond depth, 0..1+ (e.g. points earned together)


@dataclass(frozen=True)
class Weights:
    haunt: float = 1.2           # the soft prior of known haunts
    fit: float = 0.9             # trait x place-feature agreement
    ally_ground: float = 1.1     # allies drift onto the observer's ground
    adversary_ground: float = 1.3  # adversaries wait on disputed ground
    partner_region: float = 0.8  # a related person already placed in a region pulls the other there
    arrival_relation: float = 1.0  # each relation to someone present raises the chance of meeting
    # Per side, when the host wants allies and adversaries to turn up at different rates (None = arrival_relation).
    # A host can make the people bound to the observer rarer than the ones at odds with it, or the reverse.
    arrival_ally: float | None = None
    arrival_adversary: float | None = None
    arrival_draw: float = 2.0   # the engine's draw belief
    arrival_mood: float = 0.5   # the observer's mood agreeing with someone's traits
    attention: float = 3.0       # places the observer is deliberately looking at
    temperature: float = 1.0


@dataclass(frozen=True)
class Encounter:
    entity: str
    place: str
    region: str
    score: float                 # the sampled log-score (comparable within one draw)
    p_place: float               # placement probability of this place for this entity
    anchored: bool
    reasons: tuple[str, ...]


def _cos(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    na = math.sqrt(sum(float(a.get(k, 0.0)) ** 2 for k in keys))
    nb = math.sqrt(sum(float(b.get(k, 0.0)) ** 2 for k in keys))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _uniform(*parts: Any) -> float:
    """A deterministic uniform in (0, 1) from its parts (stable across runs and platforms)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return (n + 0.5) / float(1 << 64)


def _gumbel(*parts: Any) -> float:
    return -math.log(-math.log(_uniform(*parts)))


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    t = max(1e-6, float(temperature))
    m = max(scores.values())
    ex = {k: math.exp((v - m) / t) for k, v in scores.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


class EncounterField:
    """The forecast of meetings. Build once per world; ask per moment."""

    def __init__(self, entities: Iterable[Entity], places: Iterable[Place], relations: Iterable[Relation] = (),
                 weights: Weights | None = None):
        self.entities: dict[str, Entity] = {e.id: e for e in entities}
        self.places: dict[str, Place] = {p.id: p for p in places}
        self.w = weights or Weights()
        self.links: dict[str, list[tuple[str, str, float, str]]] = {}   # id -> [(other, side, strength, polarity)]
        for r in relations:
            side, k = POLARITY.get(str(r.polarity), (ALLY, 0.5))
            strength = k * max(0.0, float(r.weight))
            for x, y in ((r.a, r.b), (r.b, r.a)):
                self.links.setdefault(x, []).append((y, side, strength, str(r.polarity)))

    # -- who: how strongly someone is drawn toward the observer at all
    def arrival(self, entity_id: str, *, present: Iterable[str] = (), draw: Mapping[str, float] | None = None,
                 mood: Mapping[str, float] | None = None) -> tuple[float, list[str]]:
        e = self.entities[entity_id]
        here = set(present)
        score, why = 0.0, []
        for other, side, strength, pol in self.links.get(entity_id, ()):
            if other in here:
                per_side = self.w.arrival_ally if side == ALLY else self.w.arrival_adversary
                score += (self.w.arrival_relation if per_side is None else per_side) * strength
                why.append(f"drawn by {other} ({pol})")
        if draw and entity_id in draw:
            if draw[entity_id] is None:              # DARK: the field holds no draw for them — not 0.5, no pull
                why.append("no word of them (draw dark)")
            else:
                d = float(draw[entity_id]) - 0.5      # 0.5 is the open centre: heard of, nothing either way
                score += self.w.arrival_draw * 2.0 * d
                if abs(d) >= 0.05:
                    why.append(f"word has {'got around' if d > 0 else 'gone quiet'} (draw {float(draw[entity_id]):.2f})")
        if mood:
            c = _cos(e.traits, mood)
            score += self.w.arrival_mood * c
            if abs(c) >= 0.35:
                why.append(f"the mood {'calls for' if c > 0 else 'runs against'} them ({c:+.2f})")
        return score, why

    # -- where: a distribution over places for one person
    def placement(self, entity_id: str, *, present: Iterable[str] = (), placed: Mapping[str, str] | None = None,
                  attention: Iterable[str] = (), places: Iterable[str] | None = None) -> tuple[dict[str, float], dict[str, list[str]]]:
        e = self.entities[entity_id]
        pool = [self.places[p] for p in (places if places is not None else self.places) if p in self.places]
        if e.anchored_at:
            if e.anchored_at in {p.id for p in pool}:
                return {e.anchored_at: 1.0}, {e.anchored_at: ["anchored here"]}
            return {}, {}
        here = set(present)
        placed = dict(placed or {})
        look = set(attention)
        ally = sum(s for o, side, s, _ in self.links.get(entity_id, ()) if side == ALLY and o in here)
        adv = sum(s for o, side, s, _ in self.links.get(entity_id, ()) if side == ADVERSARY and o in here)
        partner_regions: dict[str, list[tuple[str, str, float]]] = {}
        for o, side, s, pol in self.links.get(entity_id, ()):
            at = placed.get(o)
            if at and at in self.places:
                partner_regions.setdefault(self.places[at].region, []).append((o, pol, s))
        scores, why = {}, {}
        for p in pool:
            s, r = 0.0, []
            # a haunted place counts on top of its haunted country
            h = float(e.haunts.get(p.id, 0.0)) + 0.5 * float(e.haunts.get(p.region, 0.0))
            if h:
                s += self.w.haunt * h
                r.append("a known haunt" if p.id in e.haunts else "haunts this country")
            fit = _cos(e.traits, p.features)
            s += self.w.fit * fit
            if fit >= 0.35:
                r.append(f"suits the ground ({fit:+.2f})")
            k = max(0.0, min(1.0, float(p.known)))
            if ally and p.stance > 0:
                s += self.w.ally_ground * ally * p.stance * k
                r.append("drifts toward our ground")
            if adv:
                disputed = max(float(p.contest), max(0.0, -float(p.stance)) * 0.6)
                if disputed > 0:
                    s += self.w.adversary_ground * adv * disputed * k
                    r.append("waits on disputed ground")
            for o, pol, st in partner_regions.get(p.region, ()):
                s += self.w.partner_region * st
                r.append(f"{o} is out here too ({pol})")
            if p.id in look:
                s += self.w.attention
                r.append("we are looking here")
            scores[p.id], why[p.id] = s, r
        return _softmax(scores, self.w.temperature), why

    # -- the full survey (for a ledger / a hub readout)
    def survey(self, *, present: Iterable[str] = (), draw: Mapping[str, float] | None = None,
               mood: Mapping[str, float] | None = None, placed: Mapping[str, str] | None = None,
               attention: Iterable[str] = (), candidates: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        here = set(present)
        out = {}
        for eid in (candidates if candidates is not None else self.entities):
            if eid not in self.entities or eid in here:
                continue
            pres, pwhy = self.arrival(eid, present=here, draw=draw, mood=mood)
            dist, why = self.placement(eid, present=here, placed=placed, attention=attention)
            if not dist:
                continue
            best = max(dist, key=dist.get)
            regions: dict[str, float] = {}
            for pid, p in dist.items():
                regions[self.places[pid].region] = regions.get(self.places[pid].region, 0.0) + p
            out[eid] = {"arrival": round(pres, 4), "where": best, "p": round(dist[best], 4),
                        "regions": {k: round(v, 4) for k, v in sorted(regions.items(), key=lambda kv: -kv[1])},
                        "anchored": bool(self.entities[eid].anchored_at),
                        "reasons": tuple(pwhy + why.get(best, []))}
        return out

    # -- the draw: who turns up in one region this time
    def sample(self, region: str, k: int, *, seed: Any, present: Iterable[str] = (), draw: Mapping[str, float] | None = None,
               mood: Mapping[str, float] | None = None, placed: Mapping[str, str] | None = None,
               attention: Iterable[str] = (), candidates: Iterable[str] | None = None) -> list[Encounter]:
        """Up to k people met in `region`, Gumbel-top-k over arrival + log(placement mass in the
        region); each is then placed inside the region by a second seeded draw. Anchored people are
        candidates only when their anchor is in the region."""
        here = set(present)
        local = [pid for pid, p in self.places.items() if p.region == region]
        if not local or k <= 0:
            return []
        rows = []
        for eid in (candidates if candidates is not None else self.entities):
            if eid not in self.entities or eid in here:
                continue
            dist, why = self.placement(eid, present=here, placed=placed, attention=attention, places=self.places)
            mass = sum(dist.get(pid, 0.0) for pid in local)
            if mass <= 0:
                continue
            pres, pwhy = self.arrival(eid, present=here, draw=draw, mood=mood)
            score = pres + math.log(mass) + _gumbel(seed, region, eid, "who")
            rows.append((score, eid, dist, why, pwhy))
        rows.sort(key=lambda r: (-r[0], r[1]))
        out = []
        for score, eid, dist, why, pwhy in rows[:k]:
            local_dist = {pid: dist[pid] for pid in local if dist.get(pid, 0.0) > 0}
            z = sum(local_dist.values())
            pick = max(local_dist, key=lambda pid: math.log(local_dist[pid] / z) + _gumbel(seed, region, eid, pid, "where"))
            out.append(Encounter(entity=eid, place=pick, region=region, score=round(score, 4),
                                 p_place=round(local_dist[pick] / z, 4), anchored=bool(self.entities[eid].anchored_at),
                                 reasons=tuple(pwhy + why.get(pick, []))))
        return out


# ----------------------------------------------------------------------------- engine helpers

def relation_bridges(relations: Iterable[Relation], *, node: Any = "{}", role: str = "draw",
                     polarities: Sequence[str] | None = None) -> tuple:
    """BridgeSpecs that couple the `role` of related person nodes, so the engine itself spreads a
    draw observation along the relation graph. `node` formats an id into a node name ("hero:{}")
    or is a callable. One bridge per unordered pair."""
    from umwelt.spec.schema import BridgeSpec
    fmt = node if callable(node) else (lambda i: str(node).format(i))
    seen, out = set(), []
    for r in relations:
        if polarities is not None and r.polarity not in polarities:
            continue
        key = tuple(sorted((r.a, r.b)))
        if key in seen or r.a == r.b:
            continue
        seen.add(key)
        out.append(BridgeSpec(fmt(key[0]), fmt(key[1]), shared_roles=(role,), kind="open"))
    return tuple(out)


def draw_from_engine(engine: Any, ids: Iterable[str], *, node: Any = "{}", role: str = "draw") -> dict[str, float | None]:
    """{id: draw in [0, 1], or None} from the field ((z + 1) / 2; 0.5 = heard of, nothing either way).
    Accepts a BeliefEngine or anything with `.engine` (a GameHost). A node or role the field does
    not hold reads None — DARK, never the open centre: an older field that never carried the role
    has not "heard nothing", it has no belief at all, and `arrival()` gives a dark draw no pull."""
    eng = getattr(engine, "engine", engine)
    clusters = getattr(getattr(eng, "field", None), "clusters", {}) or {}
    fmt = node if callable(node) else (lambda i: str(node).format(i))
    out: dict[str, float | None] = {}
    for i in ids:
        c = clusters.get(fmt(i))
        if c is None:
            out[i] = None
            continue
        try:
            z = float(c.role_bloch(role)[2])
        except Exception:      # the role is absent on this node: dark
            out[i] = None
            continue
        out[i] = round((z + 1.0) / 2.0, 4)
    return out


__all__ = ["ALLY", "ADVERSARY", "POLARITY", "Entity", "Place", "Relation", "Weights", "Encounter",
           "EncounterField", "relation_bridges", "draw_from_engine"]
