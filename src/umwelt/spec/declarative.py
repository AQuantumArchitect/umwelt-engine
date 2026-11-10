"""The declarative-only audit — a spec module may describe a world, never act.

Why this exists. The forge gate (umweltforge.pipeline.run_validation) and the daemon's
front door (umweltd.supervisor._gate_spec) judge a spec by IMPORTING its module inside
the judge's own interpreter. Import executes code. Until 2026-11-01 a four-line module
could therefore turn the gate green from the inside:

    import sys
    _m = sys.modules["__main__"]
    _m.ValidationReport.ok = property(lambda s: True)
    SPEC = "not a DomainSpec"

— and register a non-world. The ledger's "a lying agent cannot register a broken world"
was true of a lying CLAIM and false of a lying MODULE.

This audit runs BEFORE the import, on the source text only (``ast``), and refuses any
module that is not shaped like data:

  * imports only from an allowlist (the engine's own ``umwelt.*``, ``math``, ``datetime``,
    ``dataclasses``, ``typing``, ``enum``, ``itertools``, ``functools``, ``__future__``) —
    no ``sys``, ``os``, ``importlib``, ``builtins``, ``subprocess``, ``pathlib`` …;
  * no dangerous names anywhere (``exec``, ``eval``, ``compile``, ``open``,
    ``__import__``, ``getattr``/``setattr``/``delattr``, ``globals``/``locals``/``vars``,
    ``breakpoint``, ``input``, ``type`` as a call) and no dunder attribute or name at all
    (``__class__``/``__subclasses__``/``__builtins__`` are the sandbox-escape alphabet);
  * only statement kinds that build data: assignments, function/class definitions,
    ``for``/``if`` (a grid of nodes is still data), ``pass``, ``assert``, a docstring —
    never ``with``, ``try``, ``while``, ``global``, ``nonlocal``, ``del``;
  * every package ``__init__.py`` on the module's dotted path and every relative or
    same-package import is audited too — a package initializer executes at import just
    as the module does.

It is a static gate, not a sandbox: process isolation (``-I``, a scrubbed environment)
stays in the callers, and the two together are the boundary. What a spec cannot do
under this audit — read a tape from disk, call the clock, reach the network — a spec
should not do: "the spec never expresses an algorithm" (docs/SPEC.md). Worlds that
genuinely need to load data at import stay valid for the library and the CLI without
``--declarative-only``; the FORGE pipeline, which judges agent-authored modules, is
always strict.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ALLOWED_IMPORT_ROOTS = frozenset({
    "umwelt", "math", "datetime", "dataclasses", "typing", "enum", "itertools",
    "functools", "__future__", "collections", "fractions", "decimal", "statistics", "hashlib",
})

# Dunder NAMES that are declarations, not interpreter reach-ins.
_BENIGN_DUNDERS = frozenset({"__all__", "__doc__", "__version__"})

FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "breakpoint", "input", "type", "super", "memoryview",
    "classmethod", "staticmethod", "property", "object",
})

ALLOWED_STATEMENTS = (
    ast.Assign, ast.AnnAssign, ast.AugAssign, ast.FunctionDef, ast.ClassDef, ast.Return,
    ast.For, ast.If, ast.Pass, ast.Assert, ast.Expr, ast.Import, ast.ImportFrom,
    ast.Break, ast.Continue,
)


def locate_sources(module_name: str, search_path: "list[str] | None" = None) -> "list[Path]":
    """Every source file that import of ``module_name`` would execute, WITHOUT importing:
    each package ``__init__.py`` along the dotted path, then the module itself (or its
    own ``__init__.py``). Resolved against ``search_path`` (default ``sys.path``) the way
    the import system would. Empty list ⇒ not found (the resolve check will say so)."""
    parts = module_name.split(".")
    for entry in (sys.path if search_path is None else search_path):
        base = Path(entry or ".")
        found: list[Path] = []
        cur = base
        ok = True
        for i, part in enumerate(parts):
            cur = cur / part
            last = i == len(parts) - 1
            if cur.is_dir() and (cur / "__init__.py").is_file():
                found.append(cur / "__init__.py")
                if last:
                    break
            elif last and (cur.parent / (part + ".py")).is_file():
                found.append(cur.parent / (part + ".py"))
            elif cur.is_dir() and not last:
                continue                                      # PEP 420 namespace package: nothing executes
            else:
                ok = False
                break
        if ok and found:
            return found
    return []


def _import_root(node: ast.AST, package_parts: list[str]) -> "list[tuple[str, str | None]]":
    """(root, same_package_module_or_None) for each imported name."""
    out: list[tuple[str, str | None]] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            out.append((alias.name.split(".")[0], None))
    elif isinstance(node, ast.ImportFrom):
        if node.level:                                        # relative: same package
            up = package_parts[: len(package_parts) - (node.level - 1)] if package_parts else []
            target = ".".join([*up, node.module] if node.module else up)
            out.append(("", target or None))
        else:
            out.append(((node.module or "").split(".")[0], None))
    return out


def audit_source(path: Path, *, module_name: str = "", allow_roots: "frozenset[str] | set[str]" = frozenset()) -> "list[str]":
    """Problems with ONE source file (empty list ⇒ declarative). Does not follow imports;
    ``allow_roots`` names same-package roots whose modules the caller audits itself."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError) as exc:
        return [f"{path.name}: cannot parse: {type(exc).__name__}: {exc}"]
    problems: list[str] = []
    label = module_name or path.name
    for i, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.Expr):
            if not (i == 0 and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                problems.append(f"{label}:{stmt.lineno}: bare expression at module level "
                                f"(a spec module declares; it does not call)")
        elif not isinstance(stmt, ALLOWED_STATEMENTS):
            problems.append(f"{label}:{stmt.lineno}: statement kind {type(stmt).__name__} is not "
                            f"declarative (no with/try/while/global/del in a spec module)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for root, same_pkg in _import_root(node, []):
                if same_pkg is not None:
                    continue                                  # followed by audit_module
                if root not in ALLOWED_IMPORT_ROOTS and root not in allow_roots:
                    problems.append(f"{label}:{node.lineno}: import of {root!r} is outside the "
                                    f"declarative allowlist {sorted(ALLOWED_IMPORT_ROOTS)}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                problems.append(f"{label}:{node.lineno}: use of {node.id!r}")
            elif node.id.startswith("__") and node.id.endswith("__") and node.id not in _BENIGN_DUNDERS:
                problems.append(f"{label}:{node.lineno}: dunder name {node.id!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                problems.append(f"{label}:{node.lineno}: dunder attribute .{node.attr}")
            elif node.attr in ("modules", "path", "environ", "argv"):
                problems.append(f"{label}:{node.lineno}: attribute .{node.attr} (interpreter/"
                                f"process state is not world data)")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.Delete, ast.With, ast.AsyncWith,
                               ast.Try, ast.While, ast.Await, ast.Yield, ast.YieldFrom)):
            problems.append(f"{label}:{node.lineno}: {type(node).__name__} is not declarative")
    return problems


def audit_module(module_name: str, search_path: "list[str] | None" = None) -> "list[str]":
    """Problems with everything importing ``module_name`` would execute: the module, its
    package initializers, and (transitively) its relative / same-top-package imports.
    Empty list ⇒ the import is declarative and may proceed."""
    seen: set[str] = set()
    problems: list[str] = []
    top = module_name.split(".")[0]
    queue = [module_name]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        sources = locate_sources(name, search_path)
        if not sources:
            problems.append(f"{name}: module not found on the gate's import path")
            continue
        for src in sources:
            problems.extend(audit_source(src, module_name=name, allow_roots={top}))
            # follow same-package imports so a sibling module cannot carry the payload
            try:
                tree = ast.parse(src.read_text(encoding="utf-8"))
            except Exception:
                continue
            pkg_parts = name.split(".")[:-1] if src.name != "__init__.py" else name.split(".")
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for root, same_pkg in _import_root(node, pkg_parts):
                        if same_pkg:
                            queue.append(same_pkg)
                        elif root == top and root not in ALLOWED_IMPORT_ROOTS:
                            if isinstance(node, ast.ImportFrom) and node.module:
                                queue.append(node.module)
                            else:
                                for alias in node.names:
                                    queue.append(alias.name)
    return problems


__all__ = ["ALLOWED_IMPORT_ROOTS", "FORBIDDEN_NAMES", "locate_sources", "audit_source", "audit_module"]
