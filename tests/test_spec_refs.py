"""The spec gate refuses dangling references (#43 item 9, #58).

Every case below used to pass validate_spec and then fail SILENTLY downstream: an
unknown kind/reducer, a face/role_mode/projection on an undeclared role, a forecast leaf
at a nonexistent node (the organ `continue`s forever), nonsense horizons, efficiency>1,
an inverted rest window (never entered), an exact id both bound and ignored. Each is a
failing spec that must now come back not-OK from schema_sanity with a named reason —
and in `--resolve-only` form, because that is umweltd's front door.
"""
from __future__ import annotations

import importlib

import pytest

from umwelt.spec.schema import (AxisFace, BindingSpec, BridgeSpec, DomainSpec,
                                DriverSpec, NodeSpec)
from umwelt.spec.validate import validate_spec


def _spec(*, top=None, area=None, **overrides) -> DomainSpec:
    """Root + one region with two roles + one binary signal — valid as given."""
    base = dict(
        name="refs",
        nodes=(
            NodeSpec("top", parent=None, kind="root", roles=("level",),
                     **(top or {})),
            NodeSpec("area_a", parent="top", roles=("level", "busy"),
                     **(area or {})),
        ),
        bindings=(BindingSpec("sig_a", zone="area_a", role="level",
                              normalizer="binary"),),
    )
    base.update(overrides)
    return DomainSpec(**base)


CASES = {
    "unknown_kind": (_spec(area={"kind": "spaceship"}),
                     "unknown kind 'spaceship'"),
    "unknown_bridge_kind": (_spec(bridges=(BridgeSpec("area_a", "top", kind="portal"),)),
                            "unknown kind 'portal'"),
    "unknown_reduce": (_spec(top={"reduce": "median"}), "unknown reduce 'median'"),
    "face_undeclared_role": (_spec(area={"faces": (AxisFace(role="ghost"),)}),
                             "face for undeclared role 'ghost'"),
    "role_modes_undeclared_role": (_spec(area={"role_modes": {"ghost": "unitary"}}),
                                   "role_modes names undeclared role 'ghost'"),
    "role_modes_bad_mode": (_spec(area={"role_modes": {"level": "sideways"}}),
                            "expected 'unitary' or 'dissipative'"),
    "projection_undeclared_role": (_spec(area={"projection": {"ghost": "level"}}),
                                   "projection from undeclared role 'ghost'"),
    "projection_into_undeclared_parent_role": (
        _spec(area={"projection": {"busy": "ghost"}}),
        "parent 'top' does not declare"),
    "forecast_leaf_unknown_node": (_spec(forecast_leaves=(("nowhere", "level"),)),
                                   "forecast_leaves names unknown node 'nowhere'"),
    "forecast_leaf_undeclared_role": (_spec(forecast_leaves=(("area_a", "ghost"),)),
                                      "forecast_leaves names role 'ghost'"),
    "horizons_nonpositive": (_spec(forecast_horizons_min=(0, -5, 13, 13)),
                             "finite number > 0"),
    "horizons_duplicate": (_spec(forecast_horizons_min=(13, 13, 21)),
                           "strictly increasing"),
    "efficiency_above_one": (_spec(bindings=(BindingSpec(
        "sig_a", zone="area_a", role="level", strength=0.5, efficiency=1.7),)),
        "efficiency 1.7 outside [0, 1]"),
    "rest_window_inverted": (_spec(drivers=(DriverSpec(
        "day", period_s=86400.0, rest_window=(2, -1)),)),
        "rest_window (2, -1)"),
    "ignored_and_bound": (_spec(ignored=(("sig_a", "retired"),)),
                          "ignored pattern 'sig_a' is also a bound signal"),
    "binding_unknown_node": (_spec(bindings=(BindingSpec(
        "sig_x", zone="ghost_zone", role="level"),)),
        "binding 'sig_x' targets unknown node 'ghost_zone'"),
    "binding_undeclared_role": (_spec(bindings=(BindingSpec(
        "sig_x", zone="area_a", role="ghost_role"),)),
        "binding 'sig_x' targets role 'ghost_role'"),
}


def test_base_spec_is_clean():
    report = validate_spec(_spec())
    assert report.ok, report.summary()


@pytest.mark.parametrize("case", sorted(CASES))
def test_dangling_reference_fails_resolve_only_with_named_reason(case):
    spec, reason = CASES[case]
    report = validate_spec(spec, resolve_only=True)
    assert not report.ok, report.summary()
    sanity = next(c for c in report.checks if c.name == "schema_sanity")
    assert not sanity.ok
    assert reason in sanity.detail, sanity.detail


def test_wildcard_ignored_over_bound_ids_is_advisory_not_a_failure():
    """`prefix_*` over bound ids is the 'rest of this family' idiom: ingest consults
    `ignored` only for UNMATCHED ids, so the binding wins. Taught, never gated."""
    spec = _spec(ignored=(("sig_*", "the rest of the sig family"),))
    report = validate_spec(spec, resolve_only=True)
    assert report.ok, report.summary()
    warn = next(c for c in report.checks if c.name == "ignored_overlap")
    assert warn.warning and "'sig_*' covers bound ['sig_a']" in warn.detail


def test_valid_horizons_and_rest_window_pass():
    spec = _spec(forecast_horizons_min=(5.0, 8.0, 13.0),
                 drivers=(DriverSpec("day", period_s=86400.0,
                                     rest_window=(0.45, 0.65)),))
    report = validate_spec(spec, resolve_only=True)
    assert report.ok, report.summary()


SHIPPED = [
    ("examples.contraption.world", "CONTRAPTION_SPEC"),
    ("examples.fledgeling_fog.world", "FOG_SPEC"),
    ("examples.gridworld.world", "gridworld_spec"),
    ("examples.manifold_game.world", "SPEC"),
    ("examples.mirror.world_mirror", "SPEC"),
    ("examples.n_gym_body.world", "n_gym_body_spec"),
    ("examples.shaman_self.world", "SHAMAN_SELF_SPEC"),
    ("examples.synergy.world", "SYNERGY_SPEC"),
    ("examples.law_world.spec", "law_spec"),
    ("examples.kits.attention.cassette", "ATTENTION_SPEC"),
    ("examples.kits.market.cassette", "MARKET_SPEC"),
]


@pytest.mark.parametrize("module,attr", SHIPPED)
def test_shipped_specs_pass_the_reference_checks(module, attr):
    obj = getattr(importlib.import_module(module), attr)
    spec = obj() if callable(obj) else obj
    report = validate_spec(spec, resolve_only=True, require_shadow=False)
    assert report.ok, report.summary()
