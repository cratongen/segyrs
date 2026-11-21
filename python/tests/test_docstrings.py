"""Public docstrings: NumPy sections that match the signatures, and examples that run.

Every example runs as a doctest in a fresh directory holding ``survey.sgy`` (a copy of
``test-data/small.sgy``) and ``new.sgy`` (six zero traces written by ``segyrs.create``).
"""

from __future__ import annotations

import doctest
import inspect
import re
import shutil

import numpy as np
import pytest
from conftest import DATA

import segyrs

SECTIONS = ("Parameters", "Returns", "Raises", "Examples")
LEDGER_ID = re.compile(r"\b[SX]-\d{2}\b|\bM[1-5]\b")
NO_SECTIONS = {"__enter__", "__exit__", "close"}


def _public():
    for name in segyrs.__all__:
        obj = getattr(segyrs, name)
        yield name, obj
        if inspect.isclass(obj) and not issubclass(obj, BaseException):
            for attr, value in vars(obj).items():
                if attr.startswith("_") and attr not in ("__init__", "__enter__", "__exit__"):
                    continue
                if isinstance(value, property) or callable(value):
                    yield f"{name}.{attr}", value


PUBLIC = list(_public())


def _sections(doc: str) -> set[str]:
    return {s for s in SECTIONS if re.search(rf"^{s}\n-{{{len(s)}}}$", doc, re.M)}


def _documented(doc: str) -> set[str]:
    params = re.search(r"^Parameters\n-+\n(.*?)(?:^\S[^\n]*\n-+$|\Z)", doc, re.S | re.M)
    names: set[str] = set()
    for line in (params.group(1) if params else "").splitlines():
        if line and not line.startswith(" "):
            head = line.split(" : ")[0]
            names.update(n.strip().lstrip("*") for n in head.split(","))
    return names


@pytest.mark.parametrize("qual, obj", PUBLIC, ids=[q for q, _ in PUBLIC])
def test_public_docstring_is_numpy_style(qual, obj):
    target = obj.fget if isinstance(obj, property) else obj
    doc = target.__doc__
    assert doc and doc.strip(), f"{qual} has no docstring"
    doc = inspect.cleandoc(doc)
    assert not doc.startswith("Initialize self"), f"{qual} shows the generic __init__ text"
    assert not LEDGER_ID.search(doc), f"{qual} cites an internal ledger id"
    assert "—" not in doc, f"{qual} has an em dash"
    if inspect.isclass(obj) and issubclass(obj, BaseException):
        return
    leaf = qual.rsplit(".", 1)[-1]
    if leaf in NO_SECTIONS:
        return
    found = _sections(doc)
    if isinstance(obj, property) or inspect.isclass(obj):
        assert "Examples" in found, f"{qual} has no Examples section"
        return
    if leaf == "__init__":
        return
    params = [
        p for p in inspect.signature(target).parameters.values() if p.name not in ("self", "cls")
    ]
    want = {"Raises", "Examples"} | ({"Parameters"} if params else set())
    if inspect.signature(target).return_annotation not in ("None", None):
        want.add("Returns")
    assert want <= found, f"{qual} lacks {sorted(want - found)}"
    assert {p.name for p in params} <= _documented(doc), (
        f"{qual} does not document {sorted({p.name for p in params} - _documented(doc))}"
    )


def _examples():
    finder = doctest.DocTestFinder(recurse=False)
    seen = set()
    for qual, obj in PUBLIC:
        target = obj.fget if isinstance(obj, property) else obj
        for test in finder.find(target, qual, globs={}):
            if test.examples and id(target) not in seen:
                seen.add(id(target))
                yield pytest.param(test, id=qual)


@pytest.mark.parametrize("test", list(_examples()))
def test_docstring_examples_run(test, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    shutil.copy(DATA / "small.sgy", tmp_path / "survey.sgy")
    segyrs.create(
        tmp_path / "new.sgy",
        np.zeros((2, 3, 4), np.float32),
        sample_format="ieee_float32",
        sample_interval_us=2000,
    )
    test.globs = {"segyrs": segyrs, "np": np}
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    out: list[str] = []
    result = runner.run(test, out=out.append)
    assert result.failed == 0, "".join(out)


def test_every_public_callable_has_an_example():
    names = {p.id for p in _examples()}
    missing = [
        q
        for q, obj in PUBLIC
        if not (inspect.isclass(obj) and issubclass(obj, BaseException))
        and q.rsplit(".", 1)[-1] not in NO_SECTIONS | {"__init__"}
        and q not in names
    ]
    assert not missing, missing


def test_user_guide_lists_the_open_signature_in_order():
    page = (DATA.parent / "docs" / "user-guide.md").read_text(encoding="utf-8")
    block = re.search(r"^segyrs\.open\(\n(.*?)^\)", page, re.S | re.M).group(1)
    listed = [m.group(1) for m in re.finditer(r"^\s+(\w+)\s*[,=]", block, re.M)]
    assert listed == list(inspect.signature(segyrs.open).parameters)


def test_the_public_namespace_is_exactly_all():
    public = {n for n in dir(segyrs) if not n.startswith("_")}
    assert public == set(segyrs.__all__), sorted(public ^ set(segyrs.__all__))
