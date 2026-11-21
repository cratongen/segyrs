"""The worked examples in docs/examples/ cannot drift from the scripts in examples/.

Each page names its script. Every ```python block of the page is a verbatim part of the
script, every ```text block is a verbatim part of what the script prints, and every figure
the page shows is one the script draws. The script asserts its own results.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = {
    "explore-3d-volume.md": "explore_3d_volume.py",
    "prestack-and-2d.md": "prestack_and_2d.py",
    "write-and-edit.md": "write_and_edit.py",
}

pytestmark = pytest.mark.skipif(
    not (ROOT / "examples").is_dir(), reason="needs the source checkout"
)


def _fences(text: str, lang: str) -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```", text, re.S | re.M)


@pytest.mark.parametrize("page", sorted(PAGES))
def test_example_page_matches_its_script(page, tmp_path):
    text = (ROOT / "docs" / "examples" / page).read_text(encoding="utf-8")
    script = ROOT / "examples" / PAGES[page]
    source = script.read_text(encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(script), "--figures", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-4000:]

    code = _fences(text, "python")
    assert code, f"{page} has no python block"
    missing = [block for block in code if block not in source]
    assert not missing, f"not in {script.name}:\n" + "\n---\n".join(missing)

    results = _fences(text, "text")
    assert results, f"{page} has no result block"
    stale = [block for block in results if block not in run.stdout]
    assert not stale, "not printed by the script:\n" + "\n---\n".join(stale)

    shown = re.findall(r"\]\(img/([\w.-]+\.png)\)", text)
    assert shown, f"{page} shows no figure"
    for name in shown:
        assert (tmp_path / name).is_file(), f"{script.name} does not draw {name}"
        assert (ROOT / "docs" / "examples" / "img" / name).is_file(), f"{name} is not committed"
