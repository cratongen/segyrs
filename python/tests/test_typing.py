"""The package is typed: ``py.typed`` ships, and ``_segyrs.pyi`` matches the extension."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import segyrs
from segyrs import _segyrs

PACKAGE = Path(segyrs.__file__).parent
STUB = ast.parse((PACKAGE / "_segyrs.pyi").read_text(encoding="utf-8"))


def _stub_defs(body):
    return {n.name: n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


STUB_DEFS = _stub_defs(STUB.body)
RUNTIME = {n: getattr(_segyrs, n) for n in dir(_segyrs) if not n.startswith("__")}


def _stub_params(fn: ast.FunctionDef) -> list[tuple[str, str]]:
    a = fn.args
    out = [(p.arg, "positional") for p in a.posonlyargs + a.args]
    out += [(p.arg, "keyword") for p in a.kwonlyargs]
    return [p for p in out if p[0] not in ("self", "cls")]


def _runtime_params(obj) -> list[tuple[str, str]]:
    text = obj.__text_signature__.replace("$self", "self").replace("$type", "cls")
    return _stub_params(ast.parse(f"def f{text}: ...").body[0])


def test_the_package_ships_py_typed_and_the_native_stub():
    assert (PACKAGE / "py.typed").is_file()
    assert (PACKAGE / "_segyrs.pyi").is_file()


def test_the_stub_names_exactly_what_the_extension_exports():
    assert set(STUB_DEFS) == set(RUNTIME)


@pytest.mark.parametrize("name", sorted(n for n, o in RUNTIME.items() if callable(o)))
def test_stub_signatures_match_the_extension(name):
    obj, stub = RUNTIME[name], STUB_DEFS[name]
    if inspect.isclass(obj) and issubclass(obj, BaseException):
        bases = {b.id for b in stub.bases if isinstance(b, ast.Name)}
        assert bases == {b.__name__ for b in obj.__bases__}
        return
    if not inspect.isclass(obj):
        assert _stub_params(stub) == _runtime_params(obj)
        return
    members = _stub_defs(stub.body)
    assert obj.__text_signature__ is not None, f"{name} has no text signature"
    assert _stub_params(members["__new__"]) == _runtime_params(obj)
    for attr, value in vars(obj).items():
        if attr.startswith("__"):
            continue
        assert attr in members, f"{name}.{attr} is missing from the stub"
        decorators = {d.id for d in members[attr].decorator_list if isinstance(d, ast.Name)}
        if inspect.isgetsetdescriptor(value):
            assert "property" in decorators, f"{name}.{attr} is a property"
        else:
            params = _stub_params(members[attr])
            assert params == _runtime_params(getattr(obj, attr)), f"{name}.{attr}"
    extra = set(members) - set(vars(obj)) - {"__new__"}
    assert not extra, f"the stub has {extra} that {name} does not"
