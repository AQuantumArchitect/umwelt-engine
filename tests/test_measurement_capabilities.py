"""Capabilities must describe actual operations without trying them or minting evidence."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from umwelt.substrate.cluster import QubitCluster
from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.classical_reservoir import ClassicalReservoirCluster
from umwelt.substrate.product_cluster import ProductQubitCluster
from umwelt.substrate.backend import MeasurementCapabilities, measurement_capabilities
from umwelt.substrate.receipts import receipts_of
from umwelt.substrate.continuation import walk


@pytest.mark.parametrize("backend,pair", [
    (QubitCluster, True), (CumulantCluster, True),
    (ProductQubitCluster, False), (ClassicalReservoirCluster, False),
])
def test_declaration_matches_actual_calls_without_mutating_query(backend, pair):
    cluster = backend("capability", ["a", "b"])
    before = walk(cluster).digest()
    declared = measurement_capabilities(cluster)
    assert declared.contract == "measurement-capabilities/1"
    assert declared.axis is True and declared.pair is pair
    assert walk(cluster).digest() == before
    assert not receipts_of(cluster)
    with pytest.raises(FrozenInstanceError):
        declared.pair = not pair
    cluster.measure_axis(0, "z", .5, .2)
    assert receipts_of(cluster)
    if pair:
        cluster.measure_pair(0, 1, .5, .2)
        assert receipts_of(cluster)[-1].joint is True
    else:
        before = walk(cluster).digest()
        with pytest.raises(NotImplementedError):
            cluster.measure_pair(0, 1, .5, .2)
        assert walk(cluster).digest() == before


def test_method_names_and_backend_flags_cannot_invent_support():
    def forbidden(*args): raise AssertionError("query attempted measurement")
    cluster = SimpleNamespace(measure_axis=forbidden, measure_pair=forbidden,
                              is_cumulant=True)
    assert measurement_capabilities(cluster) == MeasurementCapabilities()
    assert measurement_capabilities(object()).pair is None


@pytest.mark.parametrize("bad", [{"pair": True}, True, "yes"])
def test_invalid_declaration_is_visible(bad):
    with pytest.raises(TypeError):
        measurement_capabilities(SimpleNamespace(measurement_capabilities=bad))


def test_classical_and_cumulant_approximations_are_disclosed():
    assert "only z" in measurement_capabilities(ClassicalReservoirCluster("x", ["a"])).description
    assert "approximate" in measurement_capabilities(CumulantCluster("x", ["a"])).description


@pytest.mark.parametrize("bad", [1, "true", [], object()])
def test_support_flags_cannot_be_truthy_substitutes(bad):
    with pytest.raises(TypeError): MeasurementCapabilities(pair=bad)
