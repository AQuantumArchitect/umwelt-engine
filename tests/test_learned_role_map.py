"""Step 1: synthetic fit evidence, not a claim of live consumer improvement."""
from dataclasses import replace

import numpy as np
import pytest

from umwelt.spec.build import build_graph_from_spec, extract_spec
from umwelt.spec.schema import BridgeSpec, DomainSpec, NodeSpec
from umwelt.spec.validate import validate_spec
from umwelt.substrate.graph import Bridge
from umwelt.substrate.role_map_learner import LearnedRoleMap


def declaration(**overrides):
    return dict(map_id="grade", family="linear", inputs=["f1", "f2"],
                target="g", ridge=1e-3) | overrides


def spec(bridges=None):
    return DomainSpec(name="learn-test", nodes=(
        NodeSpec("root", kind="root"),
        NodeSpec("source", parent="root", roles=("f1", "f2")),
        NodeSpec("target", parent="root", roles=("g",)),
    ), bridges=bridges if bridges is not None else (
        BridgeSpec("source", "target", kind="open", learn=declaration()),))


def features(x):
    return dict(zip(("f1", "f2"), x))


def test_recovers_weights_and_predicts_held_out_row():
    rng = np.random.default_rng(88)
    learner = LearnedRoleMap(["f1", "f2"], "g", ridge=1e-3)
    xs = rng.uniform(-1, 1, (400, 2))
    ys = xs @ [0.6, -0.3] + 0.1 + rng.normal(0, 0.02, len(xs))
    for x, y in zip(xs, ys):
        learner.observe(features(x), y)
    readout = learner.readout()
    assert list(readout["weights"].values()) == pytest.approx([0.6, -0.3], abs=0.05)
    assert readout["bias"] == pytest.approx(0.1, abs=0.05)
    assert learner.predict({"f1": 0.4, "f2": -0.7}) == pytest.approx(0.55, abs=0.05)
    assert 0 < readout["residual"] < 0.03
    assert (readout["n"], readout["eta_total"], readout["rank"]) == (400, 400, 2)


def test_eta_downweights_corrupted_block_against_equal_weight_baseline():
    rng = np.random.default_rng(61)
    xs = rng.uniform(-1, 1, (160, 2))
    ys = xs @ [0.6, -0.3] + 0.1
    weighted = LearnedRoleMap(["f1", "f2"], "g", 1e-3)
    equal = LearnedRoleMap(["f1", "f2"], "g", 1e-3)
    for x, y in zip(xs, ys):
        weighted.observe(features(x), y)
        equal.observe(features(x), y)
    for x, y in zip(xs[:80], ys[:80] + 2):
        weighted.observe(features(x), y, eta=0.05)
        equal.observe(features(x), y)
    held = rng.uniform(-1, 1, (30, 2))
    truth = held @ [0.6, -0.3] + 0.1
    errors = [np.mean((np.array([m.predict(features(x)) for x in held]) - truth)**2)
              for m in (weighted, equal)]
    assert errors[0] < errors[1] / 20
    assert weighted.readout()["eta_total"] == pytest.approx(164)


def test_matches_independent_weighted_batch_ridge_and_training_rms():
    rng = np.random.default_rng(17)
    xs = rng.normal(size=(60, 2)) + [1000, -2000]
    ys = xs @ [0.6, -0.3] + 0.1 + rng.normal(0, 0.1, len(xs))
    etas = rng.uniform(0.01, 1, len(xs))
    ridge = 0.25
    learner = LearnedRoleMap(["f1", "f2"], "g", ridge)
    for x, y, eta in zip(xs, ys, etas):
        learner.observe(features(x), y, eta)
    # Independent augmented-design least squares penalizes only slopes.
    design = np.column_stack((xs, np.ones(len(xs))))
    penalty = np.column_stack((np.sqrt(ridge) * np.eye(2), np.zeros(2)))
    beta = np.linalg.lstsq(np.vstack((design * np.sqrt(etas[:, None]), penalty)),
                           np.r_[ys * np.sqrt(etas), [0, 0]], rcond=None)[0]
    got = learner.readout()
    assert list(got["weights"].values()) == pytest.approx(beta[:2], abs=1e-8)
    assert got["bias"] == pytest.approx(beta[2], abs=1e-5)
    expected = np.sqrt(np.average((ys - design @ beta)**2, weights=etas))
    assert got["residual"] == pytest.approx(expected, abs=1e-8)


def test_centered_large_offset_and_rank_deficient_unregularized_fit():
    learner = LearnedRoleMap(["f1", "f2"], "g", ridge=0)
    for t in range(20):
        learner.observe({"f1": 1e9 + t, "f2": 2e9 + 2*t}, 5 + 0.5*t)
    assert learner.predict({"f1": 1e9 + 30, "f2": 2e9 + 60}) == pytest.approx(20)
    assert learner.readout()["rank"] == 1  # coefficients not separately identified
    assert learner.readout()["residual"] < 1e-6


def test_empty_evidence_is_dark_zero_eta_does_not_count():
    learner = LearnedRoleMap(["f1", "f2"], "g", 1e-3)
    original = learner.readout()
    assert original == dict(weights=None, bias=None, residual=None, n=0, eta_total=0, rank=0)
    learner.observe(features([1, 2]), 9, eta=0)
    assert learner.readout() == original
    with pytest.raises(ValueError, match="DARK"):
        learner.predict(features([1, 2]))
    learner.observe(features([1, 2]), 9, eta=0.5)
    assert learner.readout()["rank"] == 0
    assert learner.predict(features([3, 4])) == 9  # constant conditional estimate


@pytest.mark.parametrize("x,y,eta", [
    ({"f1": 1}, 2, 1), ({"f1": 1, "f2": None}, 2, 1),
    (features([float("nan"), 1]), 2, 1), (features([1, float("inf")]), 2, 1),
    (features(["1", 2]), 2, 1), (features([np.bool_(True), 2]), 2, 1),
    (features([1, 2]), float("nan"), 1), (features([1, 2]), True, 1),
    (features([1, 2]), 3, -0.1), (features([1, 2]), 3, 1.1),
    (features([1, 2]), 3, float("inf")), (features([1e308, 2]), 3, 1),
])
def test_rejected_observation_preserves_accumulated_evidence(x, y, eta):
    learner = LearnedRoleMap(["f1", "f2"], "g", 1e-3)
    for row in range(3):
        learner.observe(features([row, row*2]), row + 1)
    before = learner.readout()
    with pytest.raises(ValueError):
        learner.observe(x, y, eta)
    assert learner.readout() == before


def test_storage_does_not_grow_with_stream_length():
    learner = LearnedRoleMap(["f1", "f2"], "g", 0.01)
    def size():
        return {k: v.nbytes for k, v in vars(learner).items() if isinstance(v, np.ndarray)}
    before = size()
    for t in range(2000):
        learner.observe(features([t % 13, t % 17]), t % 3)
    assert size() == before
    assert learner.readout()["n"] == 2000
    assert not any(isinstance(v, list) for v in vars(learner).values())


@pytest.mark.parametrize("change,match", [
    ({"family": "density"}, "family must be linear"),
    ({"inputs": ["missing"]}, "absent on source"),
    ({"inputs": ["g"]}, "outside inputs"),
    ({"target": "missing"}, "absent on target"),
    ({"inputs": []}, "distinct nonempty"),
    ({"inputs": ["f1", "f1"]}, "distinct nonempty"),
    ({"target": "f1"}, "outside inputs"),
    ({"ridge": -1}, "nonnegative"),
    ({"ridge": float("nan")}, "finite real"),
    ({"ridge": True}, "finite real"),
    ({"map_id": ""}, "map_id must be nonempty"),
    ({"extra": 1}, "requires map_id"),
])
def test_validation_rejects_invalid_declarations(change, match):
    bridge = BridgeSpec("source", "target", learn=declaration(**change))
    report = validate_spec(spec((bridge,)), resolve_only=True)
    assert not report.ok and match in report.summary()


@pytest.mark.parametrize("extra", [{"kind": "tendril"}, {"role_map": {"f1": "g"}}])
def test_learn_cannot_change_tendril_semantics(extra):
    bridge = BridgeSpec("source", "target", learn=declaration(), **extra)
    report = validate_spec(spec((bridge,)), resolve_only=True)
    assert not report.ok and "not allowed on a tendril" in report.summary()


def test_roles_must_be_on_correct_ends_and_shared_map_must_agree():
    bridge = BridgeSpec("target", "source", learn=declaration())
    report = validate_spec(spec((bridge,)), resolve_only=True)
    assert "absent on source" in report.summary() and "absent on target" in report.summary()
    original = spec().bridges[0]
    assert validate_spec(spec((original, original)), resolve_only=True).ok
    for changed in (declaration(ridge=0.1), declaration(inputs=["f2", "f1"])):
        report = validate_spec(spec((original, replace(original, learn=changed))), resolve_only=True)
        assert not report.ok and "conflicting declaration" in report.summary()


def test_build_extract_carries_owned_declaration_without_activating_learner():
    world = spec()
    assert validate_spec(world, resolve_only=True).ok
    graph = build_graph_from_spec(world)
    built = graph.bridges[0]
    assert not built.is_tendril and built.coupling_param is None
    assert built.learn == world.bridges[0].learn
    roundtrip = extract_spec(graph)
    built.learn["inputs"].append("mutated")
    assert world.bridges[0].learn == roundtrip.bridges[0].learn == declaration()
    rebuilt = build_graph_from_spec(roundtrip)
    assert rebuilt.bridges[0].learn == declaration()


def test_no_learn_build_preserves_legacy_bridge_and_extraction():
    bridges = (BridgeSpec("source", "target", ("f1",), "door"),
               BridgeSpec("source", "target", role_map={"f1": "g"}))
    graph = build_graph_from_spec(spec(bridges))
    assert graph.bridges[0] == Bridge("source", "target", ["f1"], "gated")
    assert graph.bridges[0].coupling_weight == 0.5  # unchanged canonical gated fallback
    assert graph.bridges[1].is_tendril
    assert all(b.learn is None for b in graph.bridges)
    extracted = extract_spec(graph)
    assert extracted.bridges[1] == bridges[1]
    assert build_graph_from_spec(extracted).bridges == graph.bridges
