"""umwelt — a belief-field engine.

Describe a world as data (a DomainSpec); feed it observations as weak measurements;
it holds a live, honest, uncertain comprehension of that world — and forecasts from it.

This module is the public API. Every name in ``__all__`` is covered by the SemVer
and deprecation policy in docs/STABILITY.md and documented, with a runnable example,
in docs/API.md. Deep imports (``umwelt.substrate.…``, ``umwelt.host.api.…``) still
work, but they are internal: they can change in any release.

    the world as data   DomainSpec, NodeSpec, BridgeSpec, BindingSpec, OutputSpec,
                        DriverSpec, AxisFace
    the gate            validate_spec, ValidationReport
    the engine          build_engine, BeliefEngine
    the host face       GameHost, WorldSession, Observation, Intent, Decision, Belief
    the wire format     Event
    read-outs           field_gauge, model_snapshot

Names load lazily on first access, so ``import umwelt`` stays cheap.
"""
from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

# public name → defining module. The single source of truth for the facade.
_EXPORTS: dict[str, str] = {
    "DomainSpec": "umwelt.spec.schema",
    "NodeSpec": "umwelt.spec.schema",
    "BridgeSpec": "umwelt.spec.schema",
    "BindingSpec": "umwelt.spec.schema",
    "OutputSpec": "umwelt.spec.schema",
    "DriverSpec": "umwelt.spec.schema",
    "AxisFace": "umwelt.spec.schema",
    "validate_spec": "umwelt.spec.validate",
    "ValidationReport": "umwelt.spec.validate",
    "build_engine": "umwelt.boot",
    "BeliefEngine": "umwelt.engine",
    "GameHost": "umwelt.host.api",
    "WorldSession": "umwelt.host.session",
    "Observation": "umwelt.host.api",
    "Intent": "umwelt.host.api",
    "Decision": "umwelt.host.api",
    "Belief": "umwelt.host.api",
    "Event": "umwelt.events",
    "field_gauge": "umwelt.projection.gauge",
    "model_snapshot": "umwelt.projection.transparency",
    "register_role_mode": "umwelt.spec.roles",
    "register_observe_role": "umwelt.spec.roles",
    "register_role_emoji": "umwelt.projection.emoji",
    "register_node_icon": "umwelt.projection.emoji",
}

# Deprecated top-level names: old name → (new public name, version it goes away in).
# Accessing one warns DeprecationWarning and returns the new object (docs/STABILITY.md).
# Empty today: nothing has moved since the facade was introduced.
_MOVED: dict[str, tuple[str, str]] = {}

__all__ = sorted([*_EXPORTS, "__version__"])

if TYPE_CHECKING:  # static view of the lazy names, for type checkers and IDEs
    from umwelt.boot import build_engine
    from umwelt.engine import BeliefEngine
    from umwelt.events import Event
    from umwelt.host.api import Belief, Decision, GameHost, Intent, Observation
    from umwelt.host.session import WorldSession
    from umwelt.projection.gauge import field_gauge
    from umwelt.projection.transparency import model_snapshot
    from umwelt.projection.emoji import register_node_icon, register_role_emoji
    from umwelt.spec.roles import register_observe_role, register_role_mode
    from umwelt.spec.schema import (
        AxisFace,
        BindingSpec,
        BridgeSpec,
        DomainSpec,
        DriverSpec,
        NodeSpec,
        OutputSpec,
    )
    from umwelt.spec.validate import ValidationReport, validate_spec


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    if name in _MOVED:
        new, removal = _MOVED[name]
        warnings.warn(
            f"umwelt.{name} is deprecated; use umwelt.{new} "
            f"(removed in {removal})",
            DeprecationWarning, stacklevel=2,
        )
        return __getattr__(new)
    raise AttributeError(f"module 'umwelt' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
