"""A declared initial state is preparation, never a dated outside observation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from umwelt.boot import build_engine
from umwelt.projection.frame import frame, validate
from umwelt.spec.schema import DomainSpec, NodeSpec
from umwelt.spec.build import build_graph_from_spec, extract_spec
from umwelt.spec.validate import validate_spec
from umwelt.substrate.receipts import receipts_of

ROLES = ('declared_boot_a', 'declared_boot_b')


def spec(priors, backend='qubit'):
    return DomainSpec(name='declared-boot', nodes=(NodeSpec('world', roles=ROLES,
        cluster_backend=backend, role_modes=dict.fromkeys(ROLES, 'dissipative'),
        initial_bloch=priors),))


def boot(s, **kwargs):
    return build_engine(spec=s, seed=0, population=False, fractal=False,
                        calibration=False, **kwargs)


@pytest.mark.parametrize('backend', ['qubit', 'cumulant'])
@pytest.mark.parametrize('profile', ['blank', 'magic'])
def test_declared_geometry_wins_over_legacy_profile_and_global_registry(backend, profile, monkeypatch):
    from umwelt.spec import roles
    monkeypatch.setattr(roles, '_ANALOG_ROLES', set(ROLES))
    declared = {ROLES[0]: (0., .6, .8), ROLES[1]: (0., 0., 0.)}
    e = boot(spec(declared, backend), seed_profile=profile)
    for role, target in declared.items():
        np.testing.assert_allclose(e.field.clusters['world'].role_bloch(role), target, atol=1e-7)
    assert receipts_of(e.field.clusters['world']) == []
    assert e.sensor_bridge.touched_roles == set()
    doc = frame(e, subject='prior', now=100.)
    assert validate(doc) == []
    assert all(n['state'] == 'dark' and n['value'] is None for n in doc['nodes'])


def test_preparation_is_not_reapplied_after_first_evidence_or_restart(tmp_path):
    s = spec(dict.fromkeys(ROLES, (0., 0., 0.)), 'cumulant')
    e = boot(s)
    c = e.field.clusters['world']
    c.receipt_clock = lambda: 100.
    c.measure_axis(0, 'z', .8, .6)
    row = next(n for n in frame(e, subject='prior', now=110.)['nodes']
               if n['source'].endswith('world.' + ROLES[0]))
    assert row['state'] == 'current' and row['value'] > 0
    before = e.field_canon_hash()
    path = tmp_path / 'state.pkl'
    e.save(str(path))
    restored = boot(s)
    restored.load(str(path))
    assert restored.field_canon_hash() == before
    np.testing.assert_array_equal(restored.field.clusters['world'].role_bloch(ROLES[0]),
                                  c.role_bloch(ROLES[0]))


def test_declared_prior_survives_graph_spec_reconstruction_without_aliasing():
    data = {ROLES[0]: [0., .6, .8], ROLES[1]: [0., 0., 0.]}
    graph = build_graph_from_spec(spec(data))
    data[ROLES[0]][0] = .5
    rebuilt = extract_spec(graph).nodes[0]
    assert rebuilt.initial_bloch[ROLES[0]] == (0., .6, .8)
    assert build_graph_from_spec(extract_spec(graph)).root.initial_bloch == rebuilt.initial_bloch


@pytest.mark.parametrize('bad', [[], 'mixed', {}, {ROLES[0]: (0., 0., 0.)},
    {**dict.fromkeys(ROLES, (0., 0., 0.)), 'unknown': (0., 0., 0.)},
    *[{ROLES[0]: value, ROLES[1]: (0., 0., 0.)} for value in
       [(True, 0., 0.), (float('nan'), 0., 0.), (float('inf'), 0., 0.),
        (10**400, 0., 0.), (1., 1., 0.), (.1, .2), '000']]])
def test_invalid_prior_fails_resolve_and_direct_build_before_construction(bad, monkeypatch):
    s = spec(bad)
    report = validate_spec(s, resolve_only=True)
    assert not report.ok
    assert any('initial_bloch' in c.detail for c in report.checks if not c.ok)
    def unexpected(*args, **kwargs):
        raise AssertionError('invalid prior reached engine construction')
    monkeypatch.setattr('umwelt.engine.BeliefEngine', unexpected)
    with pytest.raises(ValueError, match='initial_bloch'):
        boot(s)


def test_folded_prior_is_refused_instead_of_silently_discarded():
    s = spec(dict.fromkeys(ROLES, (0., 0., 0.)))
    s = replace(s, nodes=(replace(s.nodes[0], folded=True),))
    assert not validate_spec(s, resolve_only=True).ok
    with pytest.raises(ValueError, match='folded'):
        boot(s)


def test_legacy_unspecified_start_is_preserved(monkeypatch):
    from umwelt.spec import roles
    monkeypatch.setattr(roles, '_ANALOG_ROLES', set())
    e = boot(spec(None))
    assert e.field.clusters['world'].role_bloch(ROLES[0])[2] == 1.


def test_explicitly_scoped_out_prior_is_not_a_hidden_missing_target():
    from umwelt.spec.initial_state import apply_initial_bloch
    e = SimpleNamespace(field=SimpleNamespace(clusters={}))
    priors = {'world': {ROLES[0]: (0., 0., 0.)}}
    with pytest.raises(ValueError, match='no cluster'):
        apply_initial_bloch(e, priors)
    apply_initial_bloch(e, priors, scoped=True)


def test_dense_preparation_is_product_state_not_just_matching_local_means():
    declared = {ROLES[0]: (0., .6, .8), ROLES[1]: (0., 0., -.4)}
    e = boot(spec(declared))
    a = np.array([[.9, -.3j], [.3j, .1]])
    b = np.diag([.3, .7])
    np.testing.assert_allclose(e.field.clusters['world'].rho, np.kron(a, b), atol=1e-7)
