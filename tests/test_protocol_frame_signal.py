"""Consumer #100: public Protocol -> public frame, no bridge bookkeeping writes."""
from types import SimpleNamespace
import numpy as np
import pytest
from umwelt.substrate.cluster import QubitCluster
from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.product_cluster import ProductQubitCluster
from umwelt.substrate.classical_reservoir import ClassicalReservoirCluster
from umwelt.projection.frame import frame, validate


def setup(cls=QubitCluster):
    cluster = cls('sample', ['a', 'b', 'other'])
    for i in range(3): cluster.observe_qubit(i, (0,0,0), 1)
    cluster.receipt_clock = lambda: 100.
    node = SimpleNamespace(name='sample', roles=['a','b','other'], faces=())
    engine = SimpleNamespace(field=SimpleNamespace(clusters={'sample':cluster},
                             graph=SimpleNamespace(all_nodes=lambda:[node])))
    return cluster, engine


def snapshot(engine, now=110):
    doc=frame(engine, subject='sample', now=now, stale_after_s=30)
    assert validate(doc)==[]
    return {n['id']:n for n in doc['nodes']}


@pytest.mark.parametrize('cls',[QubitCluster,CumulantCluster])
def test_direct_z_signal_dates_only_its_register_without_bridge(cls):
    c,e=setup(cls)
    c.measure_axis(0,'z',.8,.6)
    rows=snapshot(e)
    assert rows['sample_a']['state']=='current'
    assert rows['sample_a']['age_s']==10
    assert rows['sample_a']['value']>0
    assert rows['sample_b']['value'] is None
    assert snapshot(e,200)['sample_a']['state']=='stale'


def test_pair_only_signal_has_no_fake_marginal_zero():
    c,e=setup()
    c.measure_pair(0,1,1,.5)
    for key in ('sample_a','sample_b'):
        row=snapshot(e)[key]
        assert row['value'] is None and row['state']=='dark'
        assert row['signal']['kind']=='joint'
        assert row['signal']['age_s']==10
        assert 'joint' in row['note'] and row['receipts'][0]['joint']


def test_offaxis_signal_is_not_a_measurement_of_z():
    c,e=setup(); c.measure_axis(0,'x',1,.5)
    row=snapshot(e)['sample_a']
    assert row['value'] is None
    assert row['signal']['kind']=='axis' and row['signal']['axis']==[1.,0.,0.]
    assert 'never signalled' not in row['note']


def test_zero_strength_does_not_freshen_existing_signal():
    c,e=setup(); c.measure_axis(0,'z',1,.5)
    c.receipt_clock=lambda:199.
    c.measure_axis(0,'z',-1,0)
    row=snapshot(e,200)['sample_a']
    assert row['state']=='stale' and row['age_s']==100


def test_failed_measurement_is_not_signal():
    c,e=setup()
    with pytest.raises(ValueError): c.measure_pair(0,0,1,.5)
    assert all(n['value'] is None and 'signal' not in n for n in snapshot(e).values())


def test_future_or_historical_unconfirmed_receipt_cannot_signal():
    from umwelt.substrate.receipts import record_axis
    c,e=setup(); record_axis(c,0,'z',1,.5,None)
    assert snapshot(e)['sample_a']['value'] is None
    c.receipt_clock=lambda:200.
    c.measure_axis(0,'z',1,.5)
    assert snapshot(e,110)['sample_a']['value'] is None


def test_frame_reads_do_not_mutate_quantum_state_or_receipts():
    from umwelt.substrate.receipts import receipts_of
    c,e=setup(); c.measure_axis(0,'z',1,.5)
    state, receipts=c.rho.copy(),receipts_of(c)
    snapshot(e); snapshot(e,200)
    np.testing.assert_array_equal(c.rho,state)
    assert receipts_of(c)==receipts


def test_pair_signal_does_not_refresh_older_local_readout():
    c,e=setup(); c.measure_axis(0,'z',1,.5)
    c.receipt_clock=lambda:199.
    c.measure_pair(0,1,1,.5)
    row=snapshot(e,200)['sample_a']
    assert row['state']=='stale' and row['age_s']==100
    assert row['signal']['kind']=='joint' and row['signal']['age_s']==1


@pytest.mark.parametrize('index,record,strength', [(-1,1,.5),(3,1,.5),(0,float('nan'),.5),(0,1,float('inf'))])
def test_invalid_call_preserves_state_and_receipts(index,record,strength):
    from umwelt.substrate.receipts import receipts_of
    c,e=setup(); before=c.rho.copy()
    with pytest.raises(ValueError): c.measure_axis(index,'z',record,strength)
    np.testing.assert_array_equal(c.rho,before)
    assert receipts_of(c)==[]


def test_signal_validation_rejects_missing_or_conflicting_receipt():
    c,e=setup(); c.measure_axis(0,'z',1,.5)
    doc=frame(e,subject='sample',now=110)
    doc['nodes'][0]['signal']['kind']='joint'
    assert any('differs from its receipt' in err for err in validate(doc))
    doc['nodes'][0]['receipts']=None
    assert any('requires one applied receipt' in err for err in validate(doc))


@pytest.mark.parametrize('cls',[QubitCluster,CumulantCluster,ProductQubitCluster,ClassicalReservoirCluster])
def test_supported_backends_mark_only_positive_success_applied(cls):
    from umwelt.substrate.receipts import receipts_of
    c=cls('sample',['a','b'])
    c.measure_axis(0,'z',.3,.5)
    c.measure_axis(0,'z',.3,0)
    assert [r.applied for r in receipts_of(c)]==[True,False]


def test_backend_exception_does_not_append_success_receipt(monkeypatch):
    from umwelt.substrate.receipts import receipts_of
    def fail(*args): raise RuntimeError('failed update')
    monkeypatch.setattr('umwelt.substrate.belavkin.measure_rho_axis',fail)
    c,e=setup()
    with pytest.raises(RuntimeError): c.measure_axis(0,'z',.3,.5)
    assert receipts_of(c)==[]
