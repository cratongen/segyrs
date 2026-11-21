"""M5: object-store reads, Zarr v3 export, and the packaging ledger items.

No cloud credentials: remote reads go through object_store's local file store
(``file://``) and an in-process HTTP server that serves byte ranges."""

from __future__ import annotations

import http.server
import importlib
import json
import multiprocessing
import re
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA, grid, write_segy

import segyrs

REPO = Path(__file__).resolve().parents[2]


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """HEAD and ranged GET over the test-data directory. Counts GET requests."""

    root = DATA
    gets = 0

    def _body(self) -> bytes | None:
        path = self.root / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return None
        return path.read_bytes()

    def _head(self, status: int, length: int, extra: dict[str, str]) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", "Tue, 17 Nov 2026 00:00:00 GMT")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()

    def do_HEAD(self) -> None:
        body = self._body()
        if body is not None:
            self._head(200, len(body), {})

    def do_GET(self) -> None:
        type(self).gets += 1
        body = self._body()
        if body is None:
            return
        rng = self.headers.get("Range")
        if rng is None:
            self._head(200, len(body), {})
            self.wfile.write(body)
            return
        start_s, end_s = rng.removeprefix("bytes=").split("-")
        start = int(start_s)
        end = min(int(end_s) if end_s else len(body) - 1, len(body) - 1)
        if start >= len(body):
            self.send_error(416)
            return
        part = body[start : end + 1]
        self._head(206, len(part), {"Content-Range": f"bytes {start}-{end}/{len(body)}"})
        self.wfile.write(part)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def http_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    _RangeHandler.gets = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _same(a: segyrs.Volume, b: segyrs.Volume) -> None:
    assert a.sizes == b.sizes
    assert a.attrs == b.attrs
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    for name, col in a.headers.items():
        np.testing.assert_array_equal(col, b.headers[name])


def test_s37_open_file_url_matches_local_open():
    local = segyrs.open(DATA / "small.sgy")
    remote = segyrs.open((DATA / "small.sgy").resolve().as_uri())
    assert remote.traces_read == 0
    _same(local, remote)
    np.testing.assert_array_equal(local.isel(crossline=2).numpy(), remote.isel(crossline=2).numpy())


def test_s37_open_http_url_with_storage_options(http_url):
    local = segyrs.open(DATA / "small.sgy")
    remote = segyrs.open(
        f"{http_url}/small.sgy",
        storage_options={"allow_http": True, "block_size": 1024, "readahead": 2},
    )
    _same(local, remote)
    # A strided crossline selection is prefetched, then decoded.
    np.testing.assert_array_equal(local.sel(crossline=21).numpy(), remote.sel(crossline=21).numpy())
    assert _RangeHandler.gets > 0


def test_http_prestack_file(http_url):
    local = segyrs.open(DATA / "small-ps.sgy")
    remote = segyrs.open(f"{http_url}/small-ps.sgy", storage_options={"allow_http": "true"})
    _same(local, remote)


def test_unknown_storage_option_is_an_error(http_url):
    with pytest.raises(segyrs.SegyrsError, match="aws_regoin"):
        segyrs.open("s3://bucket/line.sgy", storage_options={"aws_regoin": "eu-west-1"})
    with pytest.raises(segyrs.SegyrsError, match="nonsense"):
        segyrs.open(f"{http_url}/small.sgy", storage_options={"nonsense": "1"})


def test_storage_options_on_a_local_path_raise():
    with pytest.raises(ValueError, match="URLs only"):
        segyrs.open(DATA / "small.sgy", storage_options={"aws_region": "x"})
    with pytest.raises(TypeError):
        segyrs.open((DATA / "small.sgy").as_uri(), storage_options=[("a", "b")])


def test_missing_object_raises(http_url):
    with pytest.raises(segyrs.SegyrsError):
        segyrs.open(f"{http_url}/absent.sgy", storage_options={"allow_http": True})


def _read_zarr(root: Path, name: str) -> np.ndarray:
    """Assemble a Zarr v3 array written with the bytes codec and no compressor."""
    meta = json.loads((root / name / "zarr.json").read_text())
    assert meta["zarr_format"] == 3
    assert meta["node_type"] == "array"
    assert [c["name"] for c in meta["codecs"]] == ["bytes"]
    shape = meta["shape"]
    chunk = meta["chunk_grid"]["configuration"]["chunk_shape"]
    dtype = np.dtype(meta["data_type"]).newbyteorder("<")
    out = np.full(shape, meta["fill_value"], dtype=dtype)
    grid = [-(-s // c) for s, c in zip(shape, chunk, strict=True)]
    for idx in np.ndindex(*grid):
        key = root / name / "c" / "/".join(str(i) for i in idx)
        if not key.exists():
            continue
        block = np.frombuffer(key.read_bytes(), dtype=dtype).reshape(chunk)
        sl = tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunk, shape, strict=True)
        )
        out[sl] = block[tuple(slice(0, s.stop - s.start) for s in sl)]
    return out


def test_x14_to_zarr_directory_round_trips(tmp_path):
    vol = segyrs.open(DATA / "small.sgy")
    summary = vol.to_zarr(tmp_path / "small.zarr")
    assert summary["dims"] == list(vol.dims)
    assert summary["shape"] == list(vol.shape)
    assert summary["data_type"] == "float32"
    assert summary["chunks_missing"] == 0
    root = tmp_path / "small.zarr"
    assert json.loads((root / "zarr.json").read_text())["node_type"] == "group"
    np.testing.assert_array_equal(_read_zarr(root, "data"), vol.numpy())
    np.testing.assert_array_equal(_read_zarr(root, "inline"), vol.coords["inline"])
    np.testing.assert_array_equal(_read_zarr(root, "crossline"), vol.coords["crossline"])
    assert _read_zarr(root, "trace_mask").all()


def test_to_zarr_explicit_chunks_and_depth(tmp_path):
    vol = segyrs.open(DATA / "small.sgy", depth=True)
    summary = vol.to_zarr(str(tmp_path / "c.zarr"), chunks=(2, 2, 7))
    assert summary["chunks"] == [2, 2, 7]
    assert summary["dims"][-1] == "depth"
    np.testing.assert_array_equal(_read_zarr(tmp_path / "c.zarr", "data"), vol.numpy())


def test_to_zarr_object_store_url(tmp_path):
    vol = segyrs.open(DATA / "small.sgy")
    dest = tmp_path / "remote.zarr"
    vol.to_zarr(dest.as_uri(), target_chunk_bytes=1000)
    np.testing.assert_array_equal(_read_zarr(dest, "data"), vol.numpy())
    with pytest.raises(segyrs.SegyrsError, match="block_size"):
        vol.to_zarr((tmp_path / "x.zarr").as_uri(), storage_options={"block_size": 1024})


def test_to_zarr_rejects_selections_and_local_options(tmp_path):
    vol = segyrs.open(DATA / "small.sgy")
    with pytest.raises(ValueError, match="whole file"):
        vol.isel(inline=0).to_zarr(tmp_path / "sel.zarr")
    with pytest.raises(ValueError, match="URLs only"):
        vol.to_zarr(tmp_path / "o.zarr", storage_options={"aws_region": "x"})
    with pytest.raises(ValueError, match="positive"):
        vol.to_zarr(tmp_path / "t.zarr", target_chunk_bytes=0)


def test_remote_file_exports_to_zarr(tmp_path, http_url):
    remote = segyrs.open(f"{http_url}/small.sgy", storage_options={"allow_http": True})
    remote.to_zarr(tmp_path / "r.zarr")
    local = segyrs.open(DATA / "small.sgy")
    np.testing.assert_array_equal(_read_zarr(tmp_path / "r.zarr", "data"), local.numpy())


def test_x34_missing_chunks_read_as_fill_with_mask_false(tmp_path):
    samples, il, xl = grid(3, 3, ns=6)
    keep = np.arange(9) != 4  # drop the centre trace
    path = write_segy(tmp_path / "hole.sgy", samples[keep], inline=il[keep], crossline=xl[keep])
    vol = segyrs.open(path)
    summary = vol.to_zarr(tmp_path / "hole.zarr", chunks=(1, 1, 6))
    assert summary["dims"] == ["inline", "crossline", "time"]
    assert summary["chunks_missing"] == 1
    assert not (tmp_path / "hole.zarr" / "data" / "c" / "1" / "1" / "0").exists()
    data = _read_zarr(tmp_path / "hole.zarr", "data")
    mask = _read_zarr(tmp_path / "hole.zarr", "trace_mask")
    assert not mask[1, 1] and mask.sum() == 8
    np.testing.assert_array_equal(data[1, 1], 0)
    np.testing.assert_array_equal(data.reshape(9, 6)[keep], samples[keep])


def _first_trace(location: str) -> list[float]:
    return segyrs.open(location).numpy().reshape(-1, 50)[0].tolist()


def test_s11_workers_reopen_by_path_and_url():
    path = str(DATA / "small.sgy")
    url = (DATA / "small.sgy").resolve().as_uri()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(2, mp_context=ctx) as pool:
        got = list(pool.map(_first_trace, [path, url]))
    want = _first_trace(path)
    assert got == [want, want]


def test_s43_installed_package_has_no_sidecar_library():
    pkg = Path(segyrs.__file__).parent
    native = [p.name for p in pkg.iterdir() if p.suffix in (".so", ".pyd", ".dylib", ".dll")]
    assert len(native) == 1 and native[0].startswith("_segyrs"), native
    for libs in ("segyrs.libs", ".dylibs", "segyrs.dylibs"):
        assert not (pkg.parent / libs).exists()
        assert not (pkg / libs).exists()
    for module in ("_segyrs_view",):
        try:
            ext = importlib.import_module(module)
        except ImportError:
            continue
        home = Path(ext.__file__).parent
        native = [p.name for p in home.iterdir() if p.suffix in (".so", ".pyd", ".dylib", ".dll")]
        assert len(native) == 1 and native[0].startswith(module), native
        dist = module.lstrip("_")
        for libs in (f"{dist}.libs", f"{module}.libs", ".dylibs"):
            assert not (home.parent / libs).exists()
            assert not (home / libs).exists()


def _needs_repo() -> None:
    if not (REPO / ".github" / "workflows" / "release.yml").exists():
        pytest.skip("needs the source checkout")


def _version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def _release_jobs() -> dict[str, str]:
    text = (REPO / ".github" / "workflows" / "release.yml").read_text()
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([a-z][a-z0-9-]*):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_s44_python_floor_and_newest_match_the_mise_pins():
    _needs_repo()
    pyproject = (REPO / "pyproject.toml").read_text()
    floor = _version(re.search(r'requires-python = ">=(3\.\d+)"', pyproject).group(1))
    pin_line = re.search(r"^python = (.+)$", (REPO / "mise.toml").read_text(), re.M).group(1)
    pins = sorted({_version(v) for v in re.findall(r'"(3\.\d+[.\d]*)"', pin_line)})
    assert pins[0] == floor
    classifiers = sorted(
        _version(v) for v in re.findall(r'"Programming Language :: Python :: (3\.\d+)"', pyproject)
    )
    assert classifiers == [(3, m) for m in range(floor[1], pins[-1][1] + 1)]
    cargo = (REPO / "crates" / "segyrs-py" / "Cargo.toml").read_text()
    assert f'"abi3-py{floor[0]}{floor[1]}"' in cargo
    jobs = _release_jobs()
    matrix = re.search(r"python: \[(.+)\]", jobs["test-linux"]).group(1)
    assert sorted(_version(v) for v in re.findall(r'"(3\.\d+)"', matrix)) == classifiers
    spot = re.search(r"python: \[(.+)\]", jobs["test-other"]).group(1)
    spot = {_version(v) for v in re.findall(r'"(3\.\d+)"', spot)}
    assert {floor, pins[-1]} <= spot
    abi = f"cp{floor[0]}{floor[1]}-abi3"
    for publish in ("publish-pypi", "publish-testpypi"):
        tags = set(re.findall(r"-(cp\d+-abi3)-", jobs[publish]))
        assert tags == {abi}, (publish, tags)
    mise = (REPO / "mise.toml").read_text()
    assert f"default='{floor[0]}.{floor[1]}'" in mise
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    wheel_runs = {_version(v) for v in re.findall(r"mise run test-wheel (3\.\d+)", ci)}
    assert wheel_runs == {floor, pins[-1]}


def test_s45_release_publishes_only_what_it_tests():
    _needs_repo()
    jobs = _release_jobs()
    built = set(re.findall(r"platform: ([a-z0-9_-]+),", jobs["wheels"]))
    tested = set()
    for name in ("test-linux", "test-other"):
        listed = re.search(r"platform: \[(.+)\]", jobs[name]).group(1)
        tested |= {p.strip() for p in listed.split(",")}
    assert built == tested
    assert len(built) == 6
    # Owner decision of 2026-11-21: 0.1.0 ships the segyrs project only.
    assert "crates/" not in jobs["sdist"] + jobs["wheels"]
    for other in ("segyrs-view", "segyrs_view", "segyrs-interop", "segyrs_interop", "segyrs[view]"):
        assert other not in "".join(jobs.values()), other
    assert '"${#whl[@]}" -eq 1' in jobs["wheels"]
    for publish in ("publish-pypi", "publish-testpypi"):
        assert "needs: [sdist, wheels, test-linux, test-other]" in jobs[publish]
        assert '"${#sdist[@]}" -eq 1' in jobs[publish]
        assert '"${#whl[@]}" -eq 6' in jobs[publish]
        assert "ls dist/segyrs-*-cp311-abi3-*" in jobs[publish]


def _job_if(job: str) -> str:
    cond = re.search(r"^    if: (>-\n(?:      .+\n)+|.+\n)", job, re.M).group(1)
    return " ".join(line.strip() for line in cond.removeprefix(">-").splitlines()).strip()


def test_release_publishes_only_from_this_repository_and_pypi_only_from_a_version_tag():
    _needs_repo()
    jobs = _release_jobs()
    pypi = _job_if(jobs["publish-pypi"])
    assert pypi == (
        "github.repository == 'cratongen/segyrs' && "
        "startsWith(github.ref, 'refs/tags/v') && "
        "(github.event_name == 'push' || inputs.publish == 'pypi')"
    ), pypi
    testpypi = _job_if(jobs["publish-testpypi"])
    assert testpypi.startswith("github.repository == 'cratongen/segyrs' && "), testpypi
    assert "id-token: write" not in "".join(
        body for name, body in jobs.items() if not name.startswith("publish-")
    )
    check = jobs["sdist"].split("- name: Tag matches the workspace version\n", 1)[1]
    assert check.startswith("        if: startsWith(github.ref, 'refs/tags/v')\n"), check[:80]


def test_release_wheels_build_from_the_committed_lockfile():
    _needs_repo()
    wheels = _release_jobs()["wheels"]
    builds = re.findall(r"^          args: (--release.*)$", wheels, re.M)
    assert len(builds) == 1
    assert all("--locked" in args for args in builds), builds
    mise = (REPO / "mise.toml").read_text()
    task = re.search(r"^\[tasks\.wheel\]\n(.*?)^\[", mise, re.M | re.S).group(1)
    local = re.findall(r'"(maturin build [^"]+)"', task)
    assert len(local) == 1
    assert all("--locked" in cmd for cmd in local), local


def test_the_segyrs_wheel_alone_reads_urls_writes_zarr_and_says_how_to_build_the_viewer(
    tmp_path,
):
    """Owner decision of 2026-11-21: cloud reads and Zarr export are in the segyrs
    wheel; the viewer window is not released on PyPI and is built from the repository."""
    code = (
        "import sys\n"
        "sys.modules['_segyrs_view'] = None\n"
        "import segyrs\n"
        f"vol = segyrs.open({(DATA / 'small.sgy').resolve().as_uri()!r})\n"
        "assert vol.sel(inline=3).plot().traces_read == 5\n"
        f"assert vol.to_zarr({str(tmp_path / 'z.zarr')!r})['chunks_missing'] == 0\n"
        "try:\n"
        "    segyrs.view(vol.isel(inline=1))\n"
        "    print('NO ERROR')\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert len(lines) == 1, out.stdout
    assert lines[0].startswith("segyrs.view"), lines[0]
    for text in ("not part of the segyrs 0.1.0 release on PyPI", "mise run build"):
        assert text in lines[0], lines[0]
    assert "pip install" not in lines[0], lines[0]
    assert (tmp_path / "z.zarr" / "zarr.json").is_file()
    assert "_segyrs_interop" not in out.stdout
