"""Dense QubitCluster refuses above DENSE_MAX_QUBITS; Field falls back to the cumulant backend."""
import pytest

from umwelt.substrate.cluster import DENSE_MAX_QUBITS, QubitCluster
from umwelt.substrate.field import QuantumField
from umwelt.substrate.graph import WorldGraph, WorldNode
import umwelt.substrate.cluster as dense


def _field(n_roles, backend=None):
    root = WorldNode(name="root", roles=[f"r{i}" for i in range(n_roles)])
    root.cluster_backend = backend
    return QuantumField(WorldGraph(root=root))


def test_dense_cluster_refuses_above_limit():
    n = DENSE_MAX_QUBITS + 1
    with pytest.raises(ValueError, match="cumulant"):
        QubitCluster("big", [f"r{i}" for i in range(n)])


def test_field_falls_back_to_cumulant_above_limit():
    f = _field(22)
    c = f.clusters["root"]
    assert type(c).__name__ == "CumulantCluster" and c.n_qubits == 22


def test_field_explicit_qubit_above_limit_refuses():
    with pytest.raises(ValueError):
        _field(DENSE_MAX_QUBITS + 1, backend="qubit")


def test_small_node_stays_dense():
    assert type(_field(3).clusters["root"]).__name__ == "QubitCluster"


def test_fallback_is_recorded():
    assert "root" in _field(22).backend_fallbacks
    assert _field(3).backend_fallbacks == {}


def test_byte_budget_refuses_before_any_dense_array_allocation(monkeypatch):
    monkeypatch.setattr(dense, 'DENSE_MAX_BYTES', dense.dense_construction_bytes(3) - 1)
    def forbidden(*args, **kwargs):
        pytest.fail('dense allocation occurred before resource refusal')
    monkeypatch.setattr(dense.np, 'zeros', forbidden)
    with pytest.raises(ValueError, match='DENSE_MAX_BYTES'):
        QubitCluster('bounded', ['a', 'b', 'c'])


def test_byte_fallback_is_explicit_and_a_pinned_dense_node_refuses(monkeypatch, caplog):
    monkeypatch.setattr(dense, 'DENSE_MAX_BYTES', dense.dense_construction_bytes(3) - 1)
    field = _field(3)
    assert type(field.clusters['root']).__name__ == 'CumulantCluster'
    assert 'DENSE_MAX_BYTES' in field.backend_fallbacks['root']
    assert '3 roles' in caplog.text and 'DENSE_MAX_BYTES' in caplog.text
    with pytest.raises(ValueError, match='DENSE_MAX_BYTES'):
        _field(3, backend='qubit')


def test_byte_boundary_and_precision_are_accounted_for(monkeypatch):
    required = dense.dense_construction_bytes(3)
    monkeypatch.setattr(dense, 'DENSE_MAX_BYTES', required)
    assert type(_field(3).clusters['root']).__name__ == 'QubitCluster'
    original = dense.np.dtype(dense.EVOLVE_DTYPE).itemsize
    monkeypatch.setattr(dense, 'EVOLVE_DTYPE', dense.np.complex128)
    assert dense.dense_construction_bytes(3) == required * (16 // original)


def test_default_admission_does_not_construct_a_twelve_role_dense_node():
    # This checks admission only: never allocate the multi-GiB counterexample.
    assert dense.dense_construction_bytes(12) > dense.DENSE_MAX_BYTES
    assert 'DENSE_MAX_BYTES' in dense.dense_limit_reason(12)
