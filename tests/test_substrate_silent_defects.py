"""Two silent substrate defects found by the 2026-11-01 review (issue #43, items 4 and 5).

1. ZZ Hebbian learning was dead for any role carrying an underscore: the gradient loop
   recovered the two roles of a ZZ term by splitting the LABEL ("ZZ_agent_near_safe" →
   ("agent", "near_safe")) instead of reading the basis's coupling_map, so neither index
   resolved and the coupling never moved. fledgeling_fog (`agent_near`) and every folded
   manifold role ({node}_{role}) were affected.
2. `field_unify._state_of` repacked a legitimate gamma of 0.0 as the prior (0.05) via
   `float(... or GAMMA_PRIOR)`; `canon_hash` ignores gamma, so nothing noticed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.boot import build_engine
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec


def _spec(roles):
    a, b = roles
    return DomainSpec(
        name="zz-" + a,
        nodes=(NodeSpec("root", parent=None, kind="root", roles=(a,)),
               NodeSpec("den", parent="root", roles=(a, b), role_modes={a: "dissipative", b: "dissipative"})),
        bindings=(BindingSpec("s_a", zone="den", role=a, normalizer={"type": "range", "lo": 0.0, "hi": 1.0}),
                  BindingSpec("s_b", zone="den", role=b, normalizer={"type": "range", "lo": 0.0, "hi": 1.0})),
    )


def _zz_param_after(roles, ticks=80):
    import random
    random.seed(3); np.random.seed(3)
    engine = build_engine(spec=_spec(roles), population=False)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(ticks):                                 # anti-correlated, switching inputs → a real residual
        phase = (i // 7) % 2
        engine.ingest(sensor_readings={"s_a": float(phase), "s_b": float(1 - phase)},
                      now=t0 + timedelta(seconds=10 * i))
    bundle = engine.fractal_stack.scales[0]._h_bundles["den"]
    label = f"ZZ_{roles[0]}_{roles[1]}"
    assert label in bundle.params, bundle.params.keys()
    return bundle.get_param(label).value


def test_zz_coupling_learns_for_roles_with_underscores_exactly_as_for_plain_roles():
    plain = _zz_param_after(("near", "safe"))
    scored = _zz_param_after(("agent_near", "safe"))
    assert abs(plain) > 1e-6, "control: a plain-role ZZ term should learn under switching inputs"
    assert abs(scored) > 1e-6, "an underscore in a role name silenced ZZ learning"
    assert np.isclose(plain, scored), "the role name must not change the physics"


def test_gamma_zero_repacks_as_zero():
    from umwelt.substrate.cumulant_cluster import CumulantCluster
    from umwelt.substrate import field_unify
    c = CumulantCluster("g", ["p", "q"], gamma=0.0, dt=0.02)
    canon = field_unify.pack({"g": c})
    (_name, _roles, gamma, dt), = canon["partition"]
    assert gamma == 0.0 and dt == 0.02
    rebuilt = field_unify.to_clusters(canon)["g"]
    assert float(rebuilt.gamma) == 0.0
    assert float(rebuilt.dt) == 0.02
