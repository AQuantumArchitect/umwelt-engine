"""Applications depend on the engine; the engine cannot depend on them.

A private downstream application's source used to live in this tree (the umwelt.consumers namespace) and now
lives in its own repository. This test stays as the regression guard against any application
re-entering the engine's source trees (umwelt, umweltd, umweltforge)."""
import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def application_imports(source, package):
    """Resolve static absolute, relative and literal dynamic imports uniformly."""
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = resolve_name("." * node.level + name, package)
            names.append(name)
            names.extend(name + "." + alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            dynamic = (isinstance(function, ast.Name) and function.id in ("__import__", "import_module")) or (
                isinstance(function, ast.Attribute) and function.attr == "import_module")
            argument = node.args[0]
            if dynamic and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                name = argument.value
                names.append(resolve_name(name, package) if name.startswith(".") else name)
    return [name for name in names if name == "umwelt.consumers" or name.startswith("umwelt.consumers.")
            or name == "downstream_app" or name.startswith("downstream_app.")]


def test_engine_has_no_application_dependencies():
    leaks = []
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC)
        package = ".".join(relative.parts[:-1])
        leaks.extend(f"{relative.as_posix()}: {name}" for name in
                     application_imports(path.read_text(encoding="utf-8"), package))
    assert not leaks, "Engine imports application adapters:\n" + "\n".join(leaks)


@pytest.mark.parametrize("source", [
    "import umwelt.consumers.s3", "from umwelt import consumers",
    "from ..consumers.s3 import ledger", "from .. import consumers",
    "from downstream_app import fill", "import downstream_app.hive",
    "importlib.import_module('umwelt.consumers.s3')",
    "__import__('downstream_app')",
])
def test_boundary_detects_application_dependency(source):
    assert application_imports(source, "umwelt.learning")


def test_boundary_allows_reusable_engine_dependencies():
    assert application_imports("from ..learning import regressor\nimport numpy", "umwelt.projection") == []
