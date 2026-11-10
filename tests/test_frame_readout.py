"""Local model geometry is inspectable; it cannot establish calibration or freshness."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from umwelt.projection.frame import frame, validate
from umwelt.substrate.cluster import QubitCluster
from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.product_cluster import ProductQubitCluster
from umwelt.substrate.classical_reservoir import ClassicalReservoirCluster
from umwelt.substrate.continuation import walk


def world(cls=QubitCluster, *, observed=False):
    cluster = cls('sample', ['a', 'b'])
    cluster.receipt_clock = lambda: 100.
    node = SimpleNamespace(name='sample', roles=['a', 'b'], faces=())
    eng = SimpleNamespace(field=SimpleNamespace(clusters={'sample': cluster},
        graph=SimpleNamespace(all_nodes=lambda: [node])))
    if observed:
        eng.sensor_bridge = SimpleNamespace(touched_roles={('sample', 'a'), ('sample', 'b')},
            latest_value=lambda *_: (0., 100.))
    return cluster, eng


def project(eng, now=110):
    doc = frame(eng, subject='sample', now=now, stale_after_s=30)
    assert validate(doc) == []
    return doc


@pytest.mark.parametrize('cls', [QubitCluster, CumulantCluster, ProductQubitCluster,
                               ClassicalReservoirCluster])
@pytest.mark.parametrize('z,purity', [(0., .5), (.6, .68), (1., 1.)])
def test_local_purity_accompanies_observed_value_on_shipped_backends(cls, z, purity):
    cluster, eng = world(cls, observed=True)
    cluster.observe_qubit(0, (0., 0., z), 1.)
    node = project(eng)['nodes'][0]
    assert node['value'] == pytest.approx(z)
    assert node['model_readout'] == {
        'schema': 'local-qubit-readout/1', 'purity': purity, 'calibration': 'unmeasured'}
    stale = project(eng, now=200)['nodes'][0]
    assert stale['state'] == 'stale' and stale['age_s'] == 100
    assert stale['model_readout'] == node['model_readout']


def test_coherent_pure_state_can_have_zero_z_without_being_mixed():
    cluster, eng = world(observed=True)
    cluster.observe_qubit(0, (1., 0., 0.), 1.)
    row = project(eng)['nodes'][0]
    assert row['value'] == 0.
    assert row['model_readout']['purity'] == 1.
    assert row['model_readout']['calibration'] == 'unmeasured'


def test_pure_entangled_joint_state_has_mixed_local_readout():
    cluster, eng = world(observed=True)
    bell = np.array([1., 0., 0., 1.]) / np.sqrt(2.)
    cluster.rho = np.outer(bell, bell)
    assert np.trace(cluster.rho @ cluster.rho).real == pytest.approx(1.)
    assert all(n['model_readout']['purity'] == .5 for n in project(eng)['nodes'])


@pytest.mark.parametrize('operation', ['none', 'zero', 'x', 'pair'])
def test_unmeasured_local_geometry_never_lights_dark_value(operation):
    c, eng = world()
    if operation == 'zero':
        c.measure_axis(0, 'z', 1., 0.)
    elif operation == 'x':
        c.measure_axis(0, 'x', 1., .5)
    elif operation == 'pair':
        c.measure_pair(0, 1, 1., .5)
    for n in project(eng)['nodes']:
        assert n['value'] is None and n['state'] == 'dark'
        assert 'model_readout' not in n  # do not repeat an empty diagnostic on every dark row


def test_direct_receipt_readout_is_read_only_on_state_and_evidence():
    cluster, eng = world()
    cluster.observe_qubit(0, (0., 0., 0.), 1.)
    cluster.measure_axis(0, 'z', .8, .6)
    before = walk(cluster).digest()
    node = project(eng)['nodes'][0]
    assert .5 < node['model_readout']['purity'] <= 1.
    project(eng, now=200)
    assert walk(cluster).digest() == before


@pytest.mark.parametrize('bad', [None, [], {},
    {'schema': 'local-qubit-readout/2', 'purity': .5, 'calibration': 'unmeasured'},
    {'schema': 'local-qubit-readout/1', 'purity': .5, 'calibration': 'calibrated'},
    *[{'schema': 'local-qubit-readout/1', 'purity': p, 'calibration': 'unmeasured'}
      for p in [True, '0.5', .49, 1.01, float('nan'), float('inf'), 10**400]]])
def test_invalid_geometry_or_unearned_calibration_is_rejected(bad):
    _, eng = world(observed=True)
    doc = project(eng)
    doc['nodes'][0]['model_readout'] = bad
    assert any('model_readout' in err for err in validate(doc))


def test_validator_rejects_readout_on_wrong_unit_or_dark_scalar():
    _, eng = world(observed=True)
    doc = project(eng)
    wrong_unit = deepcopy(doc)
    wrong_unit['nodes'][0]['unit'] = 'orders'
    assert any('model_readout' in err for err in validate(wrong_unit))
    doc['nodes'][0].update(value=None, age_s=None, state='dark')
    assert any('model_readout' in err for err in validate(doc))


def test_legacy_frames_without_diagnostics_remain_valid():
    _, eng = world(observed=True)
    doc = project(eng)
    for n in doc['nodes']:
        n.pop('model_readout', None)
    assert validate(doc) == []


@pytest.mark.parametrize('bad', [True, '0', 10**400, float('nan'), 1.1])
def test_model_readout_rejects_non_bloch_value_without_throwing(bad):
    _, eng = world(observed=True)
    doc = project(eng)
    doc['nodes'][0]['value'] = bad
    assert any('model_readout' in err for err in validate(doc))
