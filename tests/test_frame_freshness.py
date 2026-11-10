from types import SimpleNamespace
from unittest.mock import patch

from umwelt.projection.frame import frame


def test_unrelated_recent_signal_cannot_refresh_an_axis_without_a_timestamp():
    bridge = SimpleNamespace(latest_value=lambda *_: None, _last_raw_ts={'other': 999})
    engine = SimpleNamespace(sensor_bridge=bridge)
    plates = {'axes': [{'node': 'house', 'role': 'occupied', 'status': 'live', 'z': 0.9}]}
    with patch('umwelt.projection.frame.nameplates', return_value=plates):
        result = frame(engine, subject='house', now=1000)
    node = result['nodes'][0]
    assert node['state'] == 'dark'
    assert node['age_s'] is None
    assert node['value'] is None


def test_old_axis_keeps_its_age_even_with_recent_unrelated_ingestion():
    bridge = SimpleNamespace(latest_value=lambda *_: (0.9, 100), _last_raw_ts={'other': 999})
    engine = SimpleNamespace(sensor_bridge=bridge)
    plates = {'axes': [{'node': 'house', 'role': 'occupied', 'status': 'live', 'z': 0.9}]}
    with patch('umwelt.projection.frame.nameplates', return_value=plates):
        result = frame(engine, subject='house', stale_after_s=60, now=1000)
    assert result['nodes'][0]['state'] == 'stale'
    assert result['nodes'][0]['age_s'] == 900
