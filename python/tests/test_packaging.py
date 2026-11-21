"""Release metadata checks on the source checkout: URLs, links and the docs deploy."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SITE = "https://cratongen.github.io/segyrs/"
GITHUB = "https://github.com/cratongen/segyrs"

pytestmark = pytest.mark.skipif(
    not (REPO / "pyproject.toml").is_file() or not (REPO / "mkdocs.yml").is_file(),
    reason="needs the source checkout",
)


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def _text_files() -> list[Path]:
    files = [
        REPO / n
        for n in (
            "pyproject.toml",
            "mkdocs.yml",
            "README.md",
            "Cargo.toml",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
        )
    ]
    files += sorted((REPO / "docs").rglob("*.md"))
    files += sorted((REPO / ".github").rglob("*.yml"))
    files += sorted((REPO / "crates").glob("*/Cargo.toml"))
    return files


def test_no_published_text_names_the_old_repository_owner():
    hits = [str(p.relative_to(REPO)) for p in _text_files() if "gstvschlz" in p.read_text("utf-8")]
    assert not hits


def test_project_urls_point_at_the_organization():
    urls = _pyproject()["project"]["urls"]
    assert {"Homepage", "Documentation", "Repository", "Issues", "Changelog"} <= set(urls)
    for name, url in urls.items():
        assert url.startswith((SITE, GITHUB)), (name, url)
    assert urls["Documentation"] == SITE


def test_classifiers_match_a_first_public_release():
    classifiers = _pyproject()["project"]["classifiers"]
    status = [c for c in classifiers if c.startswith("Development Status")]
    assert status == ["Development Status :: 3 - Alpha"]


def test_mkdocs_site_and_repo_urls():
    text = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert f"site_url: {SITE}\n" in text
    assert f"repo_url: {GITHUB}\n" in text
    assert "repo_name: cratongen/segyrs\n" in text


def test_readme_links_work_on_pypi():
    """PyPI renders README.md without the repository, so every link must be absolute."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    targets += re.findall(r'(?:src|srcset)="([^"]+)"', text)
    assert targets
    relative = [t for t in targets if not t.startswith(("https://", "#"))]
    assert not relative


def test_docs_deploy_waits_for_a_public_repository():
    text = (REPO / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    deploy = text.split("\n  deploy:\n", 1)[1]
    assert "!github.event.repository.private" in deploy
    assert "mise run docs-build" in text.split("\n  deploy:\n", 1)[0]


# Built from the repository for development, not published in 0.1.0.
EXTENSIONS = {"segyrs-view": "segyrs-view-py"}
# Crates of the window and GPU stack that only the segyrs-view extension links.
VIEWER = {"eframe", "egui", "wgpu", "winit", "softbuffer"}
# Crates of the object-store and Zarr support that the segyrs wheel links.
CLOUD_AND_ZARR = {"segyrs-interop", "object_store", "tokio", "serde_json"}


def _workspace_version() -> str:
    cargo = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
    return cargo["workspace"]["package"]["version"]


def _reachable(root: str) -> set[str]:
    """Crates linked into ``root`` for this platform (normal edges, no dev or build)."""
    out = subprocess.run(
        ["cargo", "tree", "-p", root, "-e", "normal", "--prefix", "none", "--offline"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.split(" ")[0] for line in out.stdout.splitlines() if line.strip()}


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (mise install)")
def test_the_segyrs_wheel_links_cloud_and_zarr_but_no_viewer_stack():
    """Owner decision of 2026-11-21: cloud reads and Zarr export are built into segyrs."""
    core = _reachable("segyrs-py")
    assert core >= {"segyrs-draw"} | CLOUD_AND_ZARR
    assert not core & (VIEWER | {"segyrs-view"})
    view = _reachable("segyrs-view-py")
    assert view >= VIEWER
    assert not view & CLOUD_AND_ZARR
    assert not (REPO / "crates" / "segyrs-interop-py").exists()


def test_segyrs_is_the_only_published_project_and_the_viewer_builds_from_the_repository():
    """Owner decision of 2026-11-21: 0.1.0 publishes segyrs alone, with no extra that
    names an unpublished package; `mise run build` installs the viewer for development."""
    version = _workspace_version()
    project = _pyproject()["project"]
    assert project["dependencies"] == ["numpy>=2"]
    assert "optional-dependencies" not in project
    tasks = tomllib.loads((REPO / "mise.toml").read_text(encoding="utf-8"))["tasks"]
    assert any("crates/segyrs-view-py/Cargo.toml" in cmd for cmd in tasks["build"]["run"])
    for name in ("wheel", "sdist", "test-wheel", "test-sdist", "check-dist"):
        run = tasks[name]["run"]
        assert "crates/" not in "".join(run if isinstance(run, list) else [run]), name
    for dist, crate in EXTENSIONS.items():
        ext = tomllib.loads((REPO / "crates" / crate / "pyproject.toml").read_text("utf-8"))
        meta = ext["project"]
        assert meta["name"] == dist
        assert meta["dependencies"] == [f"segyrs=={version}"]
        assert meta["requires-python"] == project["requires-python"]
        assert meta["license"] == project["license"]
        assert "Development Status :: 2 - Pre-Alpha" in meta["classifiers"]
        assert "xperimental" in meta["description"]
        assert ext["tool"]["maturin"]["module-name"] == "_" + dist.replace("-", "_")
        assert meta["license-files"] == project["license-files"]
        for name in ("LICENSE-MIT", "LICENSE-APACHE"):
            assert (REPO / "crates" / crate / name).read_bytes() == (REPO / name).read_bytes()
        cargo = (REPO / "crates" / crate / "Cargo.toml").read_text("utf-8")
        floor = project["requires-python"].removeprefix(">=").replace(".", "")
        assert f'"abi3-py{floor}"' in cargo


def test_no_crate_is_published_to_crates_io_yet():
    """Owner decision of 2026-11-19: the Python packages first, crates.io later."""
    for manifest in sorted((REPO / "crates").glob("*/Cargo.toml")):
        package = tomllib.loads(manifest.read_text("utf-8"))["package"]
        assert package.get("publish") is False, manifest.parent.name
        assert package["license"] == {"workspace": True}, manifest.parent.name
        assert package["repository"] == {"workspace": True}, manifest.parent.name
        assert package["homepage"] == {"workspace": True}, manifest.parent.name
    cargo = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
    assert cargo["workspace"]["package"]["license"] == "MIT OR Apache-2.0"
    assert cargo["workspace"]["package"]["repository"] == GITHUB
    assert cargo["workspace"]["package"]["homepage"] == SITE
    assert _pyproject()["project"]["license"] == "MIT OR Apache-2.0"


def _minor(text: str | None) -> tuple[int, int]:
    major, minor = (text or "0.0").split(".")[:2]
    return int(major), int(minor)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (mise install)")
def test_each_crate_rust_version_covers_its_dependencies():
    """A crate's rust-version is at least the highest one its linked dependencies declare."""
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--offline", "--locked"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(out.stdout)
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    for member in meta["workspace_members"]:
        seen, todo = set(), [member]
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo += [
                d["pkg"]
                for d in nodes[node]["deps"]
                if any(k["kind"] in (None, "build") for k in d["dep_kinds"])
            ]
        seen.discard(member)
        need = max(seen, key=lambda i: _minor(packages[i].get("rust_version")))
        own = packages[member]
        assert _minor(own.get("rust_version")) >= _minor(packages[need].get("rust_version")), (
            f"{own['name']} declares rust-version {own.get('rust_version')}, but "
            f"{packages[need]['name']} needs {packages[need].get('rust_version')}"
        )


def test_the_msrv_task_checks_the_workspace_rust_version():
    workspace = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
    msrv = workspace["workspace"]["package"]["rust-version"]
    task = tomllib.loads((REPO / "mise.toml").read_text(encoding="utf-8"))["tasks"]["check-msrv"]
    libs, ext = task["run"]
    assert re.match(rf"mise exec rust@{re.escape(msrv)}\.\d+ -- cargo check", libs)
    for core in ("segyrs", "segyrs-draw"):
        assert f"-p {core} " in libs + " "
        manifest = (REPO / "crates" / core / "Cargo.toml").read_text(encoding="utf-8")
        assert "rust-version.workspace = true" in manifest
    py = tomllib.loads((REPO / "crates" / "segyrs-py" / "Cargo.toml").read_text("utf-8"))
    floor = py["package"]["rust-version"]
    assert _minor(floor) > _minor(msrv)
    assert re.match(rf"mise exec rust@{re.escape(floor)}\.\d+ -- cargo check", ext)
    assert ext.endswith(" -p segyrs-py")
    doc = (REPO / "docs" / "install-and-release.md").read_text(encoding="utf-8")
    assert f"Rust {floor} or newer" in doc


def test_the_release_checklist_names_the_gate_and_real_checks():
    text = (REPO / "docs" / "release-checklist.md").read_text(encoding="utf-8")
    assert "## minetypes adoption: gate lifted on 2026-11-21" in text
    assert "| `minetypes.SeismicVolume` |" in text
    assert "| PyPI project name | `segyrs` |" in text
    assert "segyrs[" not in text and "segyrs-interop-py" not in text
    tests = "".join(p.read_text("utf-8") for p in (REPO / "python" / "tests").glob("test_*.py"))
    named = set(re.findall(r"`(test_[a-z0-9_]+)`", text))
    assert named
    missing = sorted(n for n in named if f"def {n}(" not in tests)
    assert not missing
    tasks = tomllib.loads((REPO / "mise.toml").read_text(encoding="utf-8"))["tasks"]
    for task in re.findall(r"mise run ([a-z0-9-]+)", text):
        assert task in tasks, task


THIRD_PARTY = "THIRD-PARTY-LICENSES.txt"
WHEELS = {
    "segyrs": ("segyrs-py", REPO),
    **{d: (c, REPO / "crates" / c) for d, c in EXTENSIONS.items()},
}


TREE_FLAGS = ("--prefix", "none", "--offline", "--locked")


def _release_targets() -> list[str]:
    text = (REPO / ".github" / "workflows" / "release.yml").read_text("utf-8")
    return re.findall(r"\{ platform: [\w-]+, runner: [\w.-]+, target: ([\w-]+)", text)


def test_cargo_about_covers_every_release_target():
    about = tomllib.loads((REPO / "about.toml").read_text("utf-8"))
    targets = _release_targets()
    assert len(targets) == 6
    assert sorted(about["targets"]) == sorted(targets)
    assert about["ignore-build-dependencies"] is False


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (mise install)")
@pytest.mark.parametrize("dist", sorted(WHEELS))
def test_each_wheel_carries_the_licence_of_every_crate_it_links(dist):
    """Statically linked crates under MIT, BSD, ISC, Zlib, OFL and others need their
    notice beside the binary; rerun `mise run third-party-licenses` after a lockfile change."""
    crate, root = WHEELS[dist]
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    assert THIRD_PARTY in project["license-files"]
    text = (root / THIRD_PARTY).read_text("utf-8")
    targets = [arg for t in _release_targets() for arg in ("--target", t)]
    out = subprocess.run(
        ["cargo", "tree", "-p", crate, "-e", "normal,build", *targets, *TREE_FLAGS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    linked = {tuple(line.split(" ")[:2]) for line in out.stdout.splitlines() if line.strip()}
    assert len(linked) > 40
    missing = sorted(f"{n} {v}" for n, v in linked if f"\n  {n} {v.lstrip('v')}\n" not in text)
    assert not missing, f"{dist}: {THIRD_PARTY} lacks {missing[:10]}"
    for spdx in ("MIT", "Apache-2.0"):
        assert f"({spdx})" in text
    if dist == "segyrs-view":
        assert "(OFL-1.1)" in text and "(Ubuntu-font-1.0)" in text


@pytest.mark.parametrize("dist", sorted(WHEELS))
def test_an_installed_wheel_holds_its_third_party_licences(dist):
    import importlib.metadata as md
    import os

    try:
        files = md.distribution(dist).files or []
    except md.PackageNotFoundError:
        if dist == "segyrs" or os.environ.get("SEGYRS_REQUIRE_EXTRAS") == "1":
            raise
        pytest.skip(f"{dist} is not installed")
    if os.environ.get("SEGYRS_CHECK_BUILD_PATHS") != "1":
        pytest.skip("checks a wheel installed by `mise run test-wheel`")
    shipped = [f for f in files if f.name == THIRD_PARTY and "licenses" in f.parts]
    assert shipped, f"{dist} wheel has no {THIRD_PARTY}"
    root = WHEELS[dist][1]
    assert shipped[0].read_text() == (root / THIRD_PARTY).read_text("utf-8")


DATA = REPO / "test-data"


def _textual(path: Path) -> str:
    head = path.read_bytes()[:3200]
    return head.decode("latin-1") + head.decode("cp037")


def test_test_data_carries_its_own_licence_notice():
    """test-data/ is segyio material (LGPL-3.0) with F3 field data (CC BY-SA 3.0)."""
    notice = (DATA / "LICENSE.md").read_text("utf-8")
    needed = ("LICENSES/LGPL-3.0.txt", "LICENSES/GPL-3.0.txt", "by-sa/3.0", "dGB Earth Sciences")
    for text in needed:
        assert text in notice
    assert (
        (DATA / "LICENSES" / "LGPL-3.0.txt")
        .read_text("utf-8")
        .startswith("GNU LESSER GENERAL PUBLIC LICENSE\nVersion 3")
    )
    assert (
        (DATA / "LICENSES" / "GPL-3.0.txt")
        .read_text("utf-8")
        .startswith("GNU GENERAL PUBLIC LICENSE\nVersion 3")
    )
    fixtures = sorted(p for p in DATA.rglob("*") if p.suffix in {".sgy", ".su"})
    assert len(fixtures) > 80
    named = {"f3.sgy", "f3-lsb.sgy"} | {
        f"multiformats/{p.name}" for p in fixtures if p.parent.name == "multiformats"
    }
    assert len(named) == 30
    assert "the 28 files in `multiformats/`" in notice
    for rel in ("f3.sgy", "f3-lsb.sgy"):
        assert f"`{rel}`" in notice
    for path in fixtures:
        rel = path.relative_to(DATA).as_posix()
        assert "F3" not in _textual(path) or rel in named, f"{rel} holds F3 data"
    for path in fixtures:
        assert path.stat().st_size < 1_000_000, path
    for page in ("README.md", "docs/index.md"):
        text = (REPO / page).read_text("utf-8")
        assert f"{GITHUB}/blob/main/test-data/LICENSE.md" in text, page


def test_no_fixture_of_unknown_origin_and_segyrs_fixtures_are_reproducible(tmp_path):
    assert not (DATA / "delay-scalar.sgy").exists(), "source and licence unknown (LICENSE.md)"
    subprocess.run(
        [sys.executable, str(DATA / "segyrs" / "make_fixtures.py"), str(tmp_path)], check=True
    )
    made = sorted(p.name for p in tmp_path.iterdir())
    committed = sorted(p.name for p in (DATA / "segyrs").glob("*.sgy"))
    assert made == committed == ["delay-scalar.sgy"]
    for name in made:
        assert (tmp_path / name).read_bytes() == (DATA / "segyrs" / name).read_bytes(), name


@pytest.mark.parametrize("pattern", ["*.whl", "*.tar.gz"])
def test_built_packages_hold_no_test_data_and_ship_third_party_licences(pattern):
    import tarfile
    import zipfile

    built = sorted((REPO / "dist").glob(pattern))
    if not built:
        pytest.skip("no package in dist/ (mise run wheel, mise run sdist)")
    for path in built:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        else:
            with tarfile.open(path) as t:
                names = t.getnames()
        assert not [n for n in names if "test-data" in n or n.endswith((".sgy", ".su"))], path
        assert any(n.endswith(THIRD_PARTY) for n in names), f"{path.name} lacks {THIRD_PARTY}"


def test_a_code_contribution_starts_with_an_issue():
    guide = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "A code contribution starts with an issue" in guide
    assert "Closes #N" in guide
    assert "A code contribution starts with an issue" in (
        REPO / "docs" / "contributing.md"
    ).read_text(encoding="utf-8")
    templates = REPO / ".github" / "ISSUE_TEMPLATE"
    forms = sorted(p.name for p in templates.glob("*.yml") if p.name != "config.yml")
    assert forms == ["bug.yml", "proposal.yml"]
    for name in forms:
        text = (templates / name).read_text(encoding="utf-8")
        assert re.search(r"^name: \S", text, re.M) and re.search(r"^description: \S", text, re.M)
        assert "\nbody:\n" in text, name
    assert "starts with an issue" in (templates / "proposal.yml").read_text(encoding="utf-8")
    assert "blank_issues_enabled: false" in (templates / "config.yml").read_text(encoding="utf-8")


def test_status_pages_agree_that_m1_to_m5_are_implemented():
    roadmap = (REPO / "docs" / "roadmap.md").read_text(encoding="utf-8")
    assert "M1 to M5 are implemented" in roadmap
    assert "In progress" not in roadmap
    for page in (
        REPO / "README.md",
        REPO / "docs" / "index.md",
        REPO / "docs" / "install-and-release.md",
    ):
        text = page.read_text(encoding="utf-8").lower()
        for stale in ("pre-alpha", "not on pypi", "release is pending", "3.9 to"):
            assert stale not in text, (page.name, stale)


def test_published_benchmarks_name_their_date_harness_and_versions():
    page = (REPO / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    assert re.search(r"^\| Date \| 2026-\d\d-\d\d", page, re.M)
    assert "segyio 1.9.15" in page and "2.x" not in page
    harness = REPO / "benchmarks"
    readme = (harness / "README.md").read_text(encoding="utf-8")
    scripts = sorted(p.name for p in harness.glob("*.py"))
    assert scripts == sorted(set(re.findall(r"`(\w+\.py)", readme)))
    for name in scripts:
        assert f"`{name}" in page or name.startswith("gen_"), name
    readme_top = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Measured on 2026-" in readme_top and "2.x" not in readme_top
    for path in (REPO / "docs" / "index.md", REPO / "mkdocs.yml"):
        assert "absurdly" not in path.read_text(encoding="utf-8").lower(), path.name
