"""EncounterField: who a mind meets next, and where (docs/ENCOUNTER.md)."""
from datetime import datetime, timezone

from umwelt.foresight.encounter import (Entity, EncounterField, Place, Relation, Weights, draw_from_engine,
                                        relation_bridges)

AX = ("reach", "water", "guard")


def _world():
    people = [
        Entity("ash", traits={"guard": 1.0}),
        Entity("brin", traits={"reach": 1.0}, haunts={"north": 1.0}),
        Entity("cole", traits={"water": 1.0}),
        Entity("dov", traits={"guard": 0.5, "reach": 0.5}),
        Entity("eel", traits={"water": 0.6}, anchored_at="ford"),
    ]
    places = [
        Place("keep", "north", features={"guard": 1.0}, stance=1.0, contest=0.0),
        Place("ridge", "north", features={"reach": 1.0}, stance=0.0, contest=0.9),
        Place("ford", "south", features={"water": 1.0}, stance=-0.6, contest=0.4),
        Place("field", "south", features={"reach": 0.3, "guard": 0.3}, stance=0.0, contest=0.1),
    ]
    rels = [Relation("ash", "dov", "bond"), Relation("ash", "brin", "rivalry"), Relation("cole", "brin", "mirror")]
    return EncounterField(people, places, rels)


def test_a_relation_to_someone_present_draws_them():
    f = _world()
    alone, _ = f.arrival("dov")
    with_ash, why = f.arrival("dov", present={"ash"})
    assert with_ash > alone and any("drawn by ash (bond)" in w for w in why)


def test_allies_and_adversaries_can_turn_up_at_different_rates():
    base = _world()
    rare_allies = EncounterField(list(base.entities.values()), list(base.places.values()),
                                 [Relation("ash", "dov", "bond"), Relation("ash", "brin", "rivalry"),
                                  Relation("cole", "brin", "mirror")],
                                 weights=Weights(arrival_ally=0.25, arrival_adversary=1.5))
    bonded_before, _ = base.arrival("dov", present={"ash"})
    bonded_after, _ = rare_allies.arrival("dov", present={"ash"})
    rival_before, _ = base.arrival("brin", present={"ash"})
    rival_after, _ = rare_allies.arrival("brin", present={"ash"})
    assert bonded_after < bonded_before and rival_after > rival_before


def test_allies_drift_onto_our_ground_and_rivals_wait_on_disputed_ground():
    f = _world()
    ally, _ = f.placement("dov", present={"ash"})
    stranger, _ = f.placement("dov")
    assert ally["keep"] > stranger["keep"]              # bonded to someone with us: toward our ground
    rival, why = f.placement("brin", present={"ash"})
    calm, _ = f.placement("brin")
    assert rival["ridge"] > calm["ridge"]               # rival of someone with us: the disputed ridge
    assert "waits on disputed ground" in why["ridge"]


def test_traits_and_haunts_shape_the_place():
    f = _world()
    dist, _ = f.placement("cole")
    assert max(dist, key=dist.get) == "ford"            # water person, water place
    dist, why = f.placement("brin")
    assert dist["ridge"] > dist["field"] and "a known haunt" not in why["ridge"] and "haunts this country" in why["ridge"]


def test_anchored_people_never_move_and_only_appear_at_home():
    f = _world()
    dist, _ = f.placement("eel")
    assert dist == {"ford": 1.0}
    assert all(e.entity != "eel" for e in f.sample("north", 5, seed=1))
    assert any(e.entity == "eel" and e.anchored for e in f.sample("south", 5, seed=1))


def test_sampling_is_seeded_and_respects_k_and_the_present():
    f = _world()
    a = f.sample("north", 2, seed="m3", present={"ash"})
    b = f.sample("north", 2, seed="m3", present={"ash"})
    assert a == b and len(a) == 2 and all(e.entity != "ash" for e in a)
    seeds = {tuple(e.entity for e in f.sample("north", 2, seed=s)) for s in range(40)}
    assert len(seeds) > 1                               # the draw is a draw, not a list


def test_attention_collapses_onto_the_place_we_look_at():
    f = _world()
    dist, why = f.placement("cole", attention={"field"})
    assert max(dist, key=dist.get) == "field" and "we are looking here" in why["field"]


def test_draw_belief_raises_arrival_and_the_survey_explains():
    f = _world()
    low, _ = f.arrival("cole", draw={"cole": 0.3})
    high, why = f.arrival("cole", draw={"cole": 0.8})
    assert high > low and any("got around" in w for w in why)
    s = f.survey(present={"ash"}, draw={"cole": 0.8})
    assert "ash" not in s and s["eel"]["anchored"] and s["eel"]["where"] == "ford"
    assert abs(sum(s["dov"]["regions"].values()) - 1.0) < 1e-3


def test_the_engine_spreads_draw_along_the_relation_graph():
    from umwelt.host.api import GameHost
    from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec
    ids = ("a", "b", "c", "d")
    nodes = [NodeSpec("root", parent=None, kind="root", roles=("fit",)),
             NodeSpec("people", parent="root", roles=("fit",), reduce="mean")]
    nodes += [NodeSpec(f"p:{i}", parent="people", kind="entity", roles=("draw",), role_modes={"draw": "dissipative"},
                       params={"gamma_diss": (0.01, 0.005, 0.0005, 0.1)}) for i in ids]
    rels = [Relation("a", "b", "bond"), Relation("b", "c", "rivalry"), Relation("b", "a", "bond")]
    bridges = relation_bridges(rels, node="p:{}")
    assert len(bridges) == 2                            # one bridge per unordered pair
    norm = {"type": "range", "lo": 0.0, "hi": 1.0}
    binds = tuple(BindingSpec(f"d_{i}", zone=f"p:{i}", role="draw", normalizer=norm, force_observe=True, efficiency=1.0, strength=0.8)
                  for i in ids)
    spec = DomainSpec(name="enc", nodes=tuple(nodes), bridges=bridges, bindings=binds, drivers=(DriverSpec("day", period_s=86400.0),))
    host = GameHost().register_world(spec, population=False, start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    for _ in range(4):
        host.observe_many("prime", {f"d_{i}": 0.5 for i in ids}, {f"d_{i}": 1.0 for i in ids}, t=host.now)
    for _ in range(3):
        host.observe_many("mind", {"d_a": 1.0}, {"d_a": 0.9}, t=host.now)
    for _ in range(10):
        host.step(dt_s=1800)
    d = draw_from_engine(host, ids, node="p:{}")
    assert d["a"] > 0.55 and d["b"] > 0.55 and d["c"] > 0.55   # word travels a -> b -> c
    assert abs(d["d"] - 0.5) < 0.02                              # nobody related to d heard anything
    # dark is not zero (and not the open centre): an absent role / node reads None, and a dark draw pulls nothing
    assert draw_from_engine(host, ["a"], node="p:{}", role="absent") == {"a": None}
    assert draw_from_engine(host, ["zz"], node="p:{}") == {"zz": None}


def test_a_dark_draw_pulls_nothing_and_says_so():
    f = _world()
    alone, _ = f.arrival("cole")
    dark, why = f.arrival("cole", draw={"cole": None})
    assert dark == alone and "no word of them (draw dark)" in why
    centre, why = f.arrival("cole", draw={"cole": 0.5})
    assert centre == alone and not any("dark" in w for w in why)
