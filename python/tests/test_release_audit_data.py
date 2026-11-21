"""Release audit on large local files. Skipped unless ``SEGYRS_DATA`` names a
directory of SEG-Y files (the owner's segyrs-data folder). Nothing here reads a
whole file or builds a dense cube: every check samples traces or reads one line.

Run it under a memory limit, for example
``SEGYRS_DATA=../segyrs-data mise run test-data``.
Written files go to ``SEGYRS_AUDIT_TMP`` (default ``target/release-audit``),
never to a RAM-backed temporary directory.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import segyrs
from segyrs import _segyrs

sys.path.insert(0, str(Path(__file__).parent))
import _oracle

ROOT = os.environ.get("SEGYRS_DATA")
pytestmark = pytest.mark.skipif(not ROOT, reason="SEGYRS_DATA is not set")
FILES = sorted(Path(ROOT).rglob("*.sgy")) if ROOT else []
IDS = [p.name if p.parent == Path(ROOT) else f"{p.parent.name}/{p.name}" for p in FILES]
TMP = Path(os.environ.get("SEGYRS_AUDIT_TMP", Path(__file__).resolve().parents[2] / "target"))


def _order(path: Path) -> str:
    return "<" if "_le_" in path.name else ">"


def _seeded(n: int, k: int, seed: int = 20261120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = rng.choice(n, size=min(k, n), replace=False)
    return np.unique(np.concatenate([picks, [0, n - 1]])).astype(np.int64)


def _header_word(path: Path, lay, t: int, byte: int, width: int = 4) -> int:
    with open(path, "rb") as fh:
        fh.seek(lay.data_start + int(t) * lay.stride + byte - 1)
        raw = fh.read(width)
    return int.from_bytes(raw, "little" if lay.order == "<" else "big", signed=True)


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_open_is_lazy_and_sampled_traces_match_the_oracle(path):
    vol = segyrs.open(path)
    assert vol.traces_read == 0
    f = vol._src.file
    picks = _seeded(f.n_traces, 400)
    lay, want = _oracle.read_traces(path, _order(path), picks)
    assert (f.n_traces, f.n_samples, f.data_start) == (lay.n_traces, lay.ns, lay.data_start)
    got = np.asarray(f.read(picks, 0, f.n_samples))
    assert got.dtype == want.dtype
    assert got.tobytes() == want.tobytes()
    for t in picks[:50]:
        raw = f.header_bytes(int(t))
        for byte in (37, 181, 189, 193):
            got_word = int.from_bytes(
                raw[byte - 1 : byte + 3], "little" if lay.order == "<" else "big", signed=True
            )
            assert got_word == _header_word(path, lay, t, byte)
    window = np.asarray(f.read(picks, 7, 19))
    assert window.tobytes() == np.ascontiguousarray(want[:, 7:19]).tobytes()
    assert vol.traces_read == 2 * picks.size


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_sampled_traces_match_segyio(path):
    segyio = pytest.importorskip("segyio")
    endian = "little" if _order(path) == "<" else "big"
    f = _segyrs.File(str(path))
    picks = _seeded(f.n_traces, 200, seed=7)
    with segyio.open(path, ignore_geometry=True, endian=endian) as s:
        assert s.tracecount == f.n_traces
        theirs = np.stack([s.trace.raw[int(t)] for t in picks])
        lines = {b: np.array([s.header[int(t)][b] for t in picks]) for b in (189, 193, 37)}
    ours = np.asarray(f.read(picks, 0, f.n_samples))
    assert ours.dtype == theirs.dtype
    assert ours.tobytes() == theirs.tobytes()
    raw = [f.header_bytes(int(t)) for t in picks]
    for byte, want in lines.items():
        got = [int.from_bytes(r[byte - 1 : byte + 3], endian, signed=True) for r in raw]
        assert got == list(want)


GRID = [p for p in FILES if p.parent == Path(ROOT) or p.parent.name == "1000x1000x100"]
GRID_IDS = [f"{p.parent.name}/{p.name}" for p in GRID]


@pytest.mark.parametrize("path", GRID, ids=GRID_IDS)
def test_grid_lines_and_points_match_the_headers(path):
    vol = segyrs.open(path)
    assert vol.geometry.is_grid
    assert vol.geometry.source == "assumed"
    f = vol._src.file
    lay = _oracle.read_traces(path, _order(path), np.array([0]))[0]
    il = vol.coords["inline"]
    xl = vol.coords["crossline"]
    a, b = il[len(il) // 3], xl[(2 * len(xl)) // 3]
    for sel in ({"inline": a}, {"crossline": b}):
        part = vol.sel(**sel)
        got = np.asarray(part.numpy())
        idx = np.asarray(part._traces).ravel() if hasattr(part, "_traces") else None
        assert idx is not None and idx.size == got.shape[0]
        _, want = _oracle.read_traces(path, _order(path), idx)
        assert got.tobytes() == want.tobytes()
        key = 189 if "inline" in sel else 193
        words = {_header_word(path, lay, t, key) for t in idx[:: max(1, idx.size // 50)]}
        assert words == {next(iter(sel.values()))}
    pts = vol.sel_points(inline=[il[1], il[-2], il[5]], crossline=[xl[-1], xl[3], xl[0]])
    got = np.asarray(pts.numpy())
    for row, (i, x) in zip(got, [(il[1], xl[-1]), (il[-2], xl[3]), (il[5], xl[0])], strict=True):
        (t,) = f.lookup(int(i), int(x))
        assert _header_word(path, lay, t, 189) == i
        assert _header_word(path, lay, t, 193) == x
        assert row.tobytes() == _oracle.read_traces(path, _order(path), [t])[1][0].tobytes()


def _truth():
    p = Path(ROOT) / "irregular" / "truth.npz"
    if not p.exists():
        pytest.skip("no irregular/truth.npz")
    return np.load(p)


def test_unsorted_file_finds_every_sampled_label():
    truth = _truth()
    path = Path(ROOT) / "irregular" / "unsorted.sgy"
    vol = segyrs.open(path)
    assert vol.geometry.kind == "unsorted"
    f = vol._src.file
    picks = _seeded(f.n_traces, 300)
    for t in picks:
        i, x = 1000 + int(truth["unsorted_il"][t]), 2000 + int(truth["unsorted_xl"][t])
        assert list(f.lookup(i, x)) == [t]
        got = np.asarray(vol.sel(inline=i, crossline=x).numpy()).ravel()
        assert got.tobytes() == _oracle.read_traces(path, ">", [t])[1][0].tobytes()


def test_prestack_with_varying_fold_keeps_every_gather():
    truth = _truth()
    path = Path(ROOT) / "irregular" / "prestack_varfold.sgy"
    vol = segyrs.open(path)
    assert "offset" not in vol.dims
    f = vol._src.file
    il, xl, off = truth["prestack_il"], truth["prestack_xl"], truth["prestack_off"]
    assert f.n_traces == il.size
    rng = np.random.default_rng(3)
    for t in rng.choice(il.size, 50, replace=False):
        hits = f.lookup(1000 + int(il[t]), 2000 + int(xl[t]))
        same = np.flatnonzero((il == il[t]) & (xl == xl[t]))
        assert list(hits) == list(same)
        gather = np.asarray(vol.sel(inline=1000 + int(il[t]), crossline=2000 + int(xl[t])).numpy())
        assert gather.shape[0] == same.size
        assert gather.tobytes() == _oracle.read_traces(path, ">", same)[1].tobytes()
        assert np.array_equal(np.sort(off[same]), off[same])


def test_one_corrupt_header_is_caught_on_the_assumed_grid_and_opens_on_full_scan():
    truth = _truth()
    path = Path(ROOT) / "irregular" / "regular_one_bad.sgy"
    bad = int(truth["bad_trace"][0])
    vol = segyrs.open(path)
    assert vol.geometry.source == "assumed"
    with pytest.raises(segyrs.GeometryAssumptionError):
        vol.sel(inline=1500).numpy()
    full = segyrs.open(path, scan="full")
    assert not full.geometry.is_grid
    f = full._src.file
    assert list(f.lookup(1500, 7000)) == [bad]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(64 << 20):
            h.update(block)
    return h.hexdigest()


ROUND = [
    pytest.param(p, id=p.name) for p in FILES if p.parent.name in ("1000x1000x100", "irregular")
]


@pytest.mark.parametrize("path", ROUND)
def test_whole_file_round_trip_is_bit_for_bit(path):
    out = TMP / "release-audit" / f"copy-{path.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    vol = segyrs.open(path, scan="full")
    try:
        vol.to_segy(out, sync=False)
        assert out.stat().st_size == path.stat().st_size
        assert _sha(out) == _sha(path)
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.parametrize("path", GRID, ids=GRID_IDS)
def test_cropped_round_trip_reads_back_the_same_traces(path):
    out = TMP / "release-audit" / f"crop-{path.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    vol = segyrs.open(path)
    il = vol.coords["inline"]
    part = vol.sel(inline=slice(il[10], il[13]))
    try:
        part.to_segy(out, sync=False)
        back = segyrs.open(out)
        assert back.attrs["sample_format"] == vol.attrs["sample_format"]
        assert np.asarray(back.numpy()).tobytes() == np.asarray(part.numpy()).tobytes()
        src_f, dst_f = vol._src.file, back._src.file
        idx = np.sort(np.asarray(part._traces).ravel())  # written in file order
        for j in (0, idx.size // 2, idx.size - 1):
            assert dst_f.header_bytes(j) == src_f.header_bytes(int(idx[j]))
    finally:
        out.unlink(missing_ok=True)


_DIGEST = r"""
import hashlib, sys
import numpy as np
import segyrs
v = segyrs.open(sys.argv[1])
h = hashlib.sha256()
il, xl = v.coords["inline"], v.coords["crossline"]
h.update(np.ascontiguousarray(v.sel(crossline=xl[len(xl) // 2]).numpy()).tobytes())
part = v.sel(inline=slice(il[100], il[140]))
h.update(repr(sorted(part.stats().items())).encode())
h.update(np.ascontiguousarray(part.rms()).tobytes())
h.update(np.ascontiguousarray(v.isel(time=37).numpy()).tobytes())
print(h.hexdigest())
"""


@pytest.mark.parametrize("path", GRID, ids=GRID_IDS)
def test_thread_count_does_not_change_results_on_large_files(path):
    digests = []
    for threads in ("1", str(max(2, os.cpu_count() or 2))):
        env = dict(os.environ, RAYON_NUM_THREADS=threads)
        run = subprocess.run(
            [sys.executable, "-c", _DIGEST, str(path)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.append(run.stdout.strip())
    assert digests[0] == digests[1]
