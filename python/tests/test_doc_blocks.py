"""Run every ```python block of the README and docs/ literally, page by page.

Each page runs in a fresh empty directory with one namespace, so a block sees the names
the blocks above it defined. Nothing is copied in: a page either writes its own files or
starts from the quick-start block of docs/index.md, which writes `survey.sgy`. An HTML
comment on the line before a fence changes how the block runs:

- `<!-- doc-block: setup -->` writes the files the other pages use. README.md and
  docs/index.md carry the same one.
- `<!-- doc-block: needs view -->` runs only when the `segyrs-view` extension is
  installed (with SEGYRS_REQUIRE_EXTRAS=1 a missing one fails).
- `<!-- doc-block: no-run, <reason> -->` is only compiled: it needs the network,
  credentials or a display.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
FENCE = re.compile(
    r"(?:^<!-- doc-block: (?P<marker>[^>]*?) -->\n)?^```python\n(?P<code>.*?)^```", re.S | re.M
)
EXTRAS = {"view": "_segyrs_view"}


def _blocks(page: Path) -> list[tuple[int, str, str]]:
    text = page.read_text(encoding="utf-8")
    return [
        (text[: m.start("code")].count("\n") + 1, m.group("marker") or "", m.group("code"))
        for m in FENCE.finditer(text)
    ]


def _setup(page: Path) -> str | None:
    found = [code for _, marker, code in _blocks(page) if marker == "setup"]
    assert len(found) <= 1, f"{page.name} has more than one setup block"
    return found[0] if found else None


def _has_extra(name: str) -> bool:
    try:
        importlib.import_module(EXTRAS[name])
    except ImportError:
        if os.environ.get("SEGYRS_REQUIRE_EXTRAS") == "1":
            raise
        return False
    return True


PAGES_WITH_BLOCKS = [p for p in PAGES if _blocks(p)]


@pytest.mark.parametrize("page", PAGES_WITH_BLOCKS, ids=lambda p: p.name)
def test_documented_python_blocks_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    namespace: dict = {}
    if _setup(page) is None:
        exec(compile(_setup(ROOT / "docs" / "index.md"), "index.md:setup", "exec"), namespace)
    failures = []
    for line, marker, code in _blocks(page):
        where = f"{page.name}:{line}"
        try:
            compiled = compile(code, where, "exec")
            if marker.startswith("no-run"):
                continue
            if marker.startswith("needs ") and not _has_extra(marker.removeprefix("needs ")):
                continue
            exec(compiled, namespace)
        except Exception as exc:
            failures.append(f"{where}: {type(exc).__name__}: {exc}")
    assert not failures, "\n".join(failures)


def test_readme_and_docs_home_share_one_quick_start_setup():
    readme, home = _setup(ROOT / "README.md"), _setup(ROOT / "docs" / "index.md")
    assert home is not None, "docs/index.md needs the <!-- doc-block: setup --> quick start"
    assert readme == home, "README.md and docs/index.md must write survey.sgy the same way"


def test_every_block_marker_is_known_and_no_run_says_why():
    for page in PAGES_WITH_BLOCKS:
        for line, marker, _ in _blocks(page):
            ok = (
                marker in ("", "setup")
                or marker.removeprefix("needs ") in EXTRAS
                or re.fullmatch(r"no-run, \S.*", marker)
            )
            assert ok, f"{page.name}:{line}: unknown doc-block marker {marker!r}"
