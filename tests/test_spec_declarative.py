"""umwelt.spec.declarative — a spec module may describe a world, never act (issue #43, item 3)."""
from __future__ import annotations

import textwrap

import pytest

from umwelt.spec.declarative import audit_module, audit_source, locate_sources

GOOD = """\
'''A world as data.'''
from __future__ import annotations
import math
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

__all__ = ["SPEC"]
N = 3
CELLS = tuple(NodeSpec(f"cell_{i}", parent="top", roles=("level",)) for i in range(N))
for _i in range(N):
    pass

def _bind(i):
    return BindingSpec(f"s_{i}", zone=f"cell_{i}", role="level",
                       normalizer={"type": "regime", "center": math.pi, "width": 1.0})

SPEC = DomainSpec(name="good", nodes=(NodeSpec("top", parent=None, kind="root", roles=()),) + CELLS,
                  bindings=tuple(_bind(i) for i in range(N)))
"""


def _write(tmp_path, name, text, pkg=None):
    d = tmp_path if pkg is None else tmp_path / pkg
    d.mkdir(parents=True, exist_ok=True)
    (d / (name + ".py")).write_text(textwrap.dedent(text))
    return d


def test_a_data_shaped_module_passes(tmp_path):
    _write(tmp_path, "good", GOOD)
    assert audit_module("good", [str(tmp_path)]) == []


@pytest.mark.parametrize("snippet, needle", [
    ("import sys\nSPEC = 1\n", "import of 'sys'"),
    ("import os\nSPEC = 1\n", "import of 'os'"),
    ("from importlib import import_module\nSPEC = 1\n", "import of 'importlib'"),
    ("import numpy as np\nSPEC = np.zeros(1)\n", "import of 'numpy'"),
    ("SPEC = eval('1')\n", "use of 'eval'"),
    ("SPEC = open('x').read()\n", "use of 'open'"),
    ("SPEC = ().__class__\n", "dunder attribute .__class__"),
    ("SPEC = __builtins__\n", "dunder name '__builtins__'"),
    ("x = getattr(object, 'x', None)\nSPEC = 1\n", "use of 'getattr'"),
    ("with open('x') as f:\n    SPEC = 1\n", "not declarative"),
    ("try:\n    SPEC = 1\nexcept Exception:\n    pass\n", "not declarative"),
    ("while True:\n    break\nSPEC = 1\n", "not declarative"),
    ("print('hello')\nSPEC = 1\n", "bare expression at module level"),
    ("raise RuntimeError('bomb')\n", "Raise"),
    ("def f():\n    import subprocess\nSPEC = f\n", "import of 'subprocess'"),   # inside a function too
    ("import umwelt.spec.schema as s\nSPEC = s.__dict__\n", "dunder attribute .__dict__"),
])
def test_non_declarative_constructs_are_refused(tmp_path, snippet, needle):
    _write(tmp_path, "bad", snippet)
    problems = audit_module("bad", [str(tmp_path)])
    assert problems and any(needle in p for p in problems), problems


def test_the_judge_patching_module_is_refused_on_source_alone(tmp_path):
    _write(tmp_path, "evil", '''
        import sys
        _m = sys.modules["__main__"]
        _m.ValidationReport.ok = property(lambda s: True)
        SPEC = "not a DomainSpec"
    ''')
    problems = audit_module("evil", [str(tmp_path)])
    assert any("import of 'sys'" in p for p in problems)
    assert any(".modules" in p for p in problems)
    assert any("'property'" in p for p in problems)


def test_package_initializers_and_siblings_are_audited_too(tmp_path):
    pkg = _write(tmp_path, "__init__", "import os\n", pkg="worldpkg")          # payload in __init__
    _write(tmp_path, "spec", "from umwelt.spec.schema import DomainSpec\nSPEC = DomainSpec(name='x')\n", pkg="worldpkg")
    problems = audit_module("worldpkg.spec", [str(tmp_path)])
    assert any("import of 'os'" in p for p in problems), problems           # the __init__ payload is seen
    # a relative sibling import is followed
    _write(tmp_path, "__init__", "", pkg="pkg2")
    _write(tmp_path, "helper", "import subprocess\nX = 1\n", pkg="pkg2")
    _write(tmp_path, "spec", "from .helper import X\nfrom umwelt.spec.schema import DomainSpec\nSPEC = DomainSpec(name='x')\n", pkg="pkg2")
    problems = audit_module("pkg2.spec", [str(tmp_path)])
    assert any("pkg2.helper" in p and "subprocess" in p for p in problems), problems


def test_namespace_packages_and_missing_modules(tmp_path):
    (tmp_path / "ns").mkdir()                                                  # no __init__.py: PEP 420
    _write(tmp_path / "ns", "spec", "from umwelt.spec.schema import DomainSpec\nSPEC = DomainSpec(name='x')\n")
    assert audit_module("ns.spec", [str(tmp_path)]) == []
    assert locate_sources("ns.spec", [str(tmp_path)]) == [tmp_path / "ns" / "spec.py"]
    assert audit_module("nope.spec", [str(tmp_path)]) == ["nope.spec: module not found on the gate's import path"]


def test_validate_spec_refuses_before_importing(tmp_path, monkeypatch):
    """The gate's check runs BEFORE resolve: a refused module leaves no trace in sys.modules."""
    import sys
    from umwelt.spec.validate import validate_spec
    _write(tmp_path, "sneaky", "import sys\nsys.modules['PROOF_OF_IMPORT'] = object()\nSPEC = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    report = validate_spec("sneaky:SPEC", declarative_only=True, resolve_only=True)
    assert not report.ok and report.checks[0].name == "declarative_only"
    assert [c.name for c in report.checks] == ["declarative_only"]
    assert "PROOF_OF_IMPORT" not in sys.modules and "sneaky" not in sys.modules
    _write(tmp_path, "fine", GOOD)
    report = validate_spec("fine:SPEC", declarative_only=True, resolve_only=True)
    assert report.ok and report.checks[0].name == "declarative_only" and report.checks[0].ok
