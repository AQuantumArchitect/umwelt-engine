# Encounter field

`umwelt.foresight.encounter` forecasts **who a mind is likely to meet next, and where**. It is
for domains with a population the observer has not met yet: wandering heroes in a campaign,
contacts in a network, specimens in a survey.

It reads four things the domain already has:

- **A relation graph** between people, typed by polarity:
  - `bond`, `attract`, `sibling`, `mentor` and `mirror` are *allies*;
  - `rivalry`, `tension` and `friction` are *adversaries*.
- **Traits and features** on the domain's own axes: each person's character and each place's
  character, on the same axes.
- **The observer's beliefs:**
  - who is with it now;
  - how each place stands, from ours (`stance` +1) through nobody's (0) to theirs (−1), plus
    how disputed it is (`contest`) and how confidently all this is known (`known`);
  - the observer's own mood, on the trait axes.
- **A `draw` belief per person**, held in the engine field.

## The draw lives in the field

Give every person node a dissipative `draw` role. Then couple related people with
`relation_bridges(relations, node="hero:{}")`. An observation that someone is near, such as word
of them, a sighting, or a bonded companion joining, then **spreads along the relation graph through
the field itself**. It fades back toward the open centre (0.5) when nobody hears of them.
`draw_from_engine(host, ids, node="hero:{}")` reads it back as `[0, 1]`. A node or role the field
does not hold reads `None` — **dark, not 0.5**: an older field that never carried the role has no
belief about them, which is not the same as having heard nothing. `arrival()` gives a dark draw no
pull and says so in its reasons.

## Polarity decides how someone turns up

| Side | Chance of meeting | Where they appear |
|---|---|---|
| ally | rises with each relation to someone present | drifts toward ground the observer holds (`stance > 0`) |
| adversary | rises too | waits on disputed ground (`contest`, or the other side's ground) |

The rest of the placement score is made of:
- a person's **haunts**, a soft prior by place or by region;
- **trait × feature agreement**, by cosine;
- **partners already placed** in the same region;
- the observer's **attention**, the places it is deliberately looking at.

Placement is a softmax over places at `Weights.temperature`.

By default every relation to someone present raises the chance of meeting by
`Weights.arrival_relation`, whatever its side. `arrival_ally` and `arrival_adversary` set the two
sides apart (None falls back to `arrival_relation`): a host can make the people bound to the observer
turn up more rarely than the ones at odds with it, or the reverse.

## API

```python
from umwelt.foresight.encounter import Entity, Place, Relation, EncounterField

f = EncounterField(entities, places, relations)
f.arrival(eid, present=..., draw=..., mood=...)    # (score, reasons): how drawn toward us at all
f.placement(eid, present=..., placed=..., attention=...)   # ({place: p}, {place: reasons})
f.survey(present=..., draw=..., mood=...)          # per person: arrival, best place, region mass, reasons
f.sample(region, k, seed=..., present=..., draw=...)   # who turns up in one region: Gumbel-top-k, seeded
```

- `sample` is deterministic for a seed. It picks who appears by arrival + log(placement mass in
  the region), then places each person inside the region by a second seeded draw.
- **Anchored people** (`Entity.anchored_at`) never move. They are candidates only where they are
  anchored.
- **Every result carries its reasons**, so a host can show why someone is believed to be where they
  are.

Tests: `tests/test_encounter.py`, including the engine spreading draw along `relation_bridges`.
First consumer: Verdance's Ashes, where wild heroes roam by the bond web
(`tools/verdance/company/encounters.py`).
