"""Pins the Penduline consumer surfaces promised stable in docs/STABILITY.md."""
import importlib
import inspect

import pytest

SURFACES = [
    ("umwelt.spec.schema", "match_ignored"),
    ("umwelt.projection.confession", "recent"),
    ("umwelt.projection.nameplates", "nameplates"),
    ("umwelt.projection.emoji", "node_icon"),
    ("umwelt.projection.emoji", "role_poles"),
    ("umwelt.events", "replay_sensor_batches"),
    ("umwelt.learning.runner", "BrainRunner"),
]


@pytest.mark.parametrize("module,name", SURFACES)
def test_penduline_surface_importable_and_callable(module, name):
    obj = getattr(importlib.import_module(module), name)
    assert callable(obj)


def test_brain_runner_is_a_class():
    from umwelt.learning.runner import BrainRunner

    assert inspect.isclass(BrainRunner)
