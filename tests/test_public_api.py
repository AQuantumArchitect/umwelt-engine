"""The public surface of `umwelt` is pinned (#66, docs/STABILITY.md).

Adding or removing a top-level name is an API change: update `PUBLIC` here, document
it in docs/API.md and record it in CHANGELOG.md. A change you did not mean to make
fails this test instead of reaching a consumer.
"""
from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import umwelt

PUBLIC = [
    "AxisFace",
    "BeliefEngine",
    "Belief",
    "BindingSpec",
    "BridgeSpec",
    "Decision",
    "DomainSpec",
    "DriverSpec",
    "Event",
    "GameHost",
    "Intent",
    "NodeSpec",
    "Observation",
    "OutputSpec",
    "ValidationReport",
    "WorldSession",
    "__version__",
    "build_engine",
    "field_gauge",
    "model_snapshot",
    "register_role_mode",
    "register_observe_role",
    "register_role_emoji",
    "register_node_icon",
    "validate_spec",
]

# Classes whose public methods are part of the contract (all must be typed + documented).
FULL_CLASSES = {"GameHost", "WorldSession", "ValidationReport"}
# BeliefEngine is large; only these members are promised (docs/API.md).
ENGINE_STABLE = ("ingest", "save", "load", "field_canon_hash")


def _symbols():
    return [n for n in umwelt.__all__ if n != "__version__"]


def test_all_snapshot():
    assert sorted(umwelt.__all__) == sorted(PUBLIC)


@pytest.mark.parametrize("name", _symbols())
def test_every_public_name_resolves_and_is_the_defining_object(name):
    obj = getattr(umwelt, name)
    module = umwelt._EXPORTS[name]
    assert obj is getattr(__import__(module, fromlist=[name]), name)
    assert name in dir(umwelt)


def _is_autodoc(obj) -> bool:
    # dataclasses synthesize "Name(field: type, ...)" when no docstring was written
    return (obj.__doc__ or "").startswith(f"{obj.__name__}(")


def _unannotated(fn) -> list[str]:
    sig = inspect.signature(fn)
    out = [p for p, v in sig.parameters.items()
           if p not in ("self", "cls")
           and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)
           and v.annotation is inspect.Parameter.empty]
    if sig.return_annotation is inspect.Signature.empty:
        out.append("return")
    return out


def _check_callable(label: str, fn, problems: list[str]) -> None:
    if not (inspect.getdoc(fn) or "").strip():
        problems.append(f"{label}: no docstring")
    missing = _unannotated(fn)
    if missing:
        problems.append(f"{label}: unannotated {missing}")


def test_every_public_symbol_is_typed_and_documented():
    problems: list[str] = []
    for name in _symbols():
        obj = getattr(umwelt, name)
        if inspect.isclass(obj):
            if not (obj.__doc__ or "").strip() or _is_autodoc(obj):
                problems.append(f"{name}: no class docstring")
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    if f.type is inspect.Parameter.empty:
                        problems.append(f"{name}.{f.name}: untyped field")
            members = (ENGINE_STABLE if name == "BeliefEngine"
                       else [m for m in vars(obj) if not m.startswith("_")]
                       if name in FULL_CLASSES else [])
            for m in members:
                attr = vars(obj)[m]
                fn = attr.fget if isinstance(attr, property) else attr
                if callable(fn):
                    _check_callable(f"{name}.{m}", fn, problems)
            if name in FULL_CLASSES and not dataclasses.is_dataclass(obj):
                _check_callable(f"{name}.__init__", obj.__init__, problems)
        else:
            _check_callable(name, obj, problems)
    assert not problems, "\n".join(problems)


def test_py_typed_ships():
    assert (Path(umwelt.__file__).parent / "py.typed").is_file()


def test_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError):
        umwelt.no_such_name  # noqa: B018


def test_moved_name_warns_and_still_resolves(monkeypatch):
    monkeypatch.setitem(umwelt._MOVED, "OldHost", ("GameHost", "0.3.0"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = umwelt.OldHost
    assert obj is umwelt.GameHost
    assert any(issubclass(w.category, DeprecationWarning)
               and "umwelt.GameHost" in str(w.message) for w in caught)


def test_moved_names_never_shadow_live_ones():
    assert not set(umwelt._MOVED) & set(umwelt.__all__)


def test_import_umwelt_stays_lazy():
    # the facade must not boot the engine stack on a bare import
    code = ("import sys, umwelt; "
            "print(any(m.startswith(('umwelt.engine', 'umwelt.host', 'numpy')) "
            "for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "False"
