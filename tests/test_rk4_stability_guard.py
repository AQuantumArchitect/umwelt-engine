"""Issue #55 (#43 item 6): a learnable dial inside its declared bounds must not drive the
field to NaN, and a NaN never commits a pole.

gamma_diss <= 50, dt <= 0.1, dt_scale <= 8 put gamma_diss*dt*dt_scale at 40, far outside RK4's
stability region: dense rho went NaN and the cumulant z reached ~7e44 inside 60 steps, and a NaN z
then committed -1 with confidence NaN at collapse.py. The integrators now substep, and collapse
treats a non-finite Bloch vector as DARK.
"""
from __future__ import annotations

import numpy as np
import pytest

from umwelt.boot import build_engine
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
from umwelt.substrate.collapse import CollapseReason
from umwelt.substrate.density_matrix import RK4_STABLE_LIMIT, rk4_substeps


def _spec():
    roles = ("near", "safe")
    return DomainSpec(
        name="rk4-guard",
        nodes=(NodeSpec("root", parent=None, kind="root", roles=("near",)),
               NodeSpec("den", parent="root", roles=roles,
                        role_modes={r: "dissipative" for r in roles})),
        bindings=tuple(BindingSpec(f"s_{r}", zone="den", role=r,
                                   normalizer={"type": "range", "lo": 0.0, "hi": 1.0})
                       for r in roles),
    )


def _corner_field(monkeypatch, cumulant: bool):
    if cumulant:
        monkeypatch.setenv("UMWELT_CUMULANT", "1")
    field = build_engine(spec=_spec(), population=False, seed=1).field
    for node in field._node_map.values():                     # the declared-bounds corner
        b = node.param_bundle
        if b is not None and "dt" in b.params:
            b.get_param("gamma_diss").value = 50.0
            b.get_param("dt").value = 0.1
    return field


def _finite(cluster) -> bool:
    if getattr(cluster, "is_product", False):
        return True
    arrs = (cluster.e1, cluster.e2) if getattr(cluster, "is_cumulant", False) else (cluster.rho,)
    return all(np.all(np.isfinite(a)) for a in arrs)


@pytest.mark.parametrize("cumulant", [False, True], ids=["dense", "cumulant"])
def test_declared_bounds_corner_stays_finite_and_physical(monkeypatch, cumulant):
    field = _corner_field(monkeypatch, cumulant)
    for _ in range(60):
        field.step({}, dt_scale=8.0)
    for name, cluster in field.clusters.items():
        assert _finite(cluster), f"{name} went non-finite inside declared bounds"
        if getattr(cluster, "is_product", False):
            continue
        for q in range(cluster.n_qubits):
            assert np.linalg.norm(cluster.qubit_bloch(q)) < 1.5, (name, q)


def test_substep_count_is_one_below_the_limit_and_grows_above_it():
    assert rk4_substeps(0.0) == 1
    assert rk4_substeps(RK4_STABLE_LIMIT) == 1
    assert rk4_substeps(RK4_STABLE_LIMIT * 3.5) == 4
    assert rk4_substeps(float("nan")) == 1                    # unfixable by splitting; caller refuses


def test_in_bounds_default_step_is_unsplit():
    """dt_scale=1 at the shipped defaults is a single RK4 step (byte-identity preserved)."""
    from umwelt.substrate.cumulant_cluster import CumulantCluster
    c = CumulantCluster("z", ["a", "b"], role_modes={"a": "dissipative", "b": "unitary"})
    S = c._scalar_vector(None)
    assert rk4_substeps(c._stiffness(S) * c.dt * 8.0) == 1


def test_nan_bloch_is_dark_not_a_pole():
    """A NaN z must leave the world model without a commit (not -1 at confidence NaN)."""
    engine = build_engine(spec=_spec(), population=False, seed=1)
    cluster = engine.field.clusters["den"]
    cluster.qubit_bloch = lambda idx: np.array([np.nan, np.nan, np.nan])
    before = engine.world.get_value("den", "near")
    engine.collapse_engine.collapse_node(engine.field, engine.world, "den", CollapseReason.QUERY)
    after = engine.world.get_value("den", "near")
    assert after == before, "a NaN Bloch vector committed a state"
