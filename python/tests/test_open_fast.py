"""Assumed-grid open (scan="auto"), the full scan, and the index sidecar."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
from conftest import write_segy

import segyrs

NIL, NXL, NS = 100, 60, 4
N = NIL * NXL  # above the 4096-trace floor of the assumed-grid open
STRIDE = 240 + 4 * NS
DATA_START = 3600


def cube(path, inline, crossline, offset=None):
    n = len(inline)
    samples = np.arange(n * NS, dtype=np.float32).reshape(n, NS)
    cdp_x = np.arange(n, dtype=np.int32) * 10
    return write_segy(
        path,
        samples,
        inline=inline,
        crossline=crossline,
        offset=offset,
        words={(181, "i4"): cdp_x},
    )


def inline_major(path):
    t = np.arange(N)
    return cube(path, 1 + t // NXL, 10 + t % NXL)


def put_i32(path, trace, byte, value):
    with open(path, "r+b") as f:
        f.seek(DATA_START + trace * STRIDE + byte - 1)
        f.write(int(value).to_bytes(4, "big", signed=True))


def same_geometry(a, b):
    assert (a.kind, a.axes) == (b.kind, b.axes)
    for name in ("inlines", "crosslines", "offsets", "missing_pairs"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


LAYOUTS = {
    "inline_major": lambda t: (1 + t // NXL, 10 + t % NXL, None),
    "crossline_major_descending": lambda t: (500 - t % NIL, 2 * (t // NIL), None),
    "prestack": lambda t: (1 + t // 300, 1 + (t // 10) % 30, 100 * (t % 10)),
}


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_a_regular_file_opens_on_the_fast_path(tmp_path, layout):
    il, xl, off = LAYOUTS[layout](np.arange(N))
    path = cube(tmp_path / "c.sgy", il, xl, off)
    fast = segyrs.open(path)
    full = segyrs.open(path, scan="full")
    assert fast.geometry.source == "assumed"
    assert full.geometry.source == "scan"
    assert "source='assumed'" in repr(fast.geometry)
    same_geometry(fast.geometry, full.geometry)
    assert fast.dims == full.dims and fast.shape == full.shape
    assert fast.traces_read == 0
    assert np.array_equal(fast.numpy(), full.numpy())
    assert np.array_equal(fast.headers["cdp_x"], full.headers["cdp_x"])
    assert np.array_equal(fast.coords["cdp_x"], full.coords["cdp_x"])


def test_a_small_file_is_scanned_in_full(tmp_path):
    t = np.arange(40 * 30)
    path = cube(tmp_path / "small.sgy", 1 + t // 30, 1 + t % 30)
    assert segyrs.open(path).geometry.source == "scan"


def test_a_header_broken_mid_file_is_detected_on_read(tmp_path):
    good = inline_major(tmp_path / "good.sgy")
    bad = tmp_path / "bad.sgy"
    shutil.copy(good, bad)
    trace = 3337  # not among the sampled headers, so open still assumes the grid
    put_i32(bad, trace, 189, 9999)

    vol = segyrs.open(bad)
    assert vol.geometry.source == "assumed"
    broken_line = 1 + trace // NXL
    with pytest.raises(segyrs.GeometryAssumptionError, match=r'scan="full"') as err:
        vol.sel(inline=broken_line).numpy()
    assert f"trace {trace}" in str(err.value)
    assert isinstance(err.value, segyrs.GeometryError)
    # Traces the edit did not touch read correctly.
    ref = segyrs.open(good, scan="full")
    assert np.array_equal(vol.sel(inline=1).numpy(), ref.sel(inline=1).numpy())
    assert np.array_equal(vol.isel(inline=-1).numpy(), ref.isel(inline=-1).numpy())
    # Whole-file header access checks every trace.
    with pytest.raises(segyrs.GeometryAssumptionError):
        vol.headers["cdp_x"]
    with pytest.raises(segyrs.GeometryAssumptionError):
        vol.isel(time=0).numpy()
    # The full scan sees the real layout.
    full = segyrs.open(bad, scan="full")
    assert full.geometry.source == "scan"
    assert full.geometry.kind != "inline_major"


def test_a_layout_change_mid_file_falls_back_to_the_full_scan(tmp_path):
    t = np.arange(N)
    il, xl = 1 + t // NXL, 10 + t % NXL
    xl = np.where(il > NIL // 2, 10 + 2 * (t % NXL), xl)  # second half steps by 2
    path = cube(tmp_path / "c.sgy", il, xl)
    auto = segyrs.open(path)
    assert auto.geometry.source == "scan"
    same_geometry(auto.geometry, segyrs.open(path, scan="full").geometry)


def test_scan_must_be_auto_or_full(tmp_path):
    path = inline_major(tmp_path / "c.sgy")
    with pytest.raises(ValueError, match="scan"):
        segyrs.open(path, scan="fast")


def irregular(path):
    """A grid infer accepts but the arithmetic fast path does not."""
    t = np.arange(N)
    labels = 10 + np.arange(NXL) ** 2
    return cube(path, 1 + t // NXL, labels[t % NXL])


def sidecars(d):
    return sorted(p for p in os.listdir(d) if p.endswith(".segyrs-index"))


def test_the_index_sidecar_is_used_and_invalidated(tmp_path):
    path = irregular(tmp_path / "c.sgy")
    cache = tmp_path / "index"
    first = segyrs.open(path, index_cache=cache)
    assert first.geometry.source == "scan"
    assert first.geometry.kind == "inline_major"
    assert len(sidecars(cache)) == 1

    again = segyrs.open(path, index_cache=cache)
    assert again.geometry.source == "index_cache"
    same_geometry(again.geometry, first.geometry)
    assert np.array_equal(again.headers["cdp_x"], first.headers["cdp_x"])
    assert np.array_equal(again.numpy(), first.numpy())
    # A full scan never trusts the sidecar.
    assert segyrs.open(path, scan="full", index_cache=cache).geometry.source == "scan"

    # A newer modification time invalidates it.
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert segyrs.open(path, index_cache=cache).geometry.source == "scan"
    assert segyrs.open(path, index_cache=cache).geometry.source == "index_cache"

    # A header edit that keeps the size and the modification time is caught by
    # the fingerprint of the first and last trace headers.
    st = os.stat(path)
    put_i32(path, 0, 181, -7)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    edited = segyrs.open(path, index_cache=cache)
    assert edited.geometry.source == "scan"
    assert edited.headers["cdp_x"][0] == -7

    # A different size invalidates it.
    with open(path, "ab") as f:
        f.write(bytes(STRIDE))
    grown = segyrs.open(path, index_cache=cache)
    assert grown.geometry.source == "scan"
    assert grown.sizes == {"trace": N + 1, "time": NS}
    assert len(sidecars(cache)) == 1


def test_a_damaged_sidecar_is_ignored(tmp_path):
    path = irregular(tmp_path / "c.sgy")
    cache = tmp_path / "index"
    segyrs.open(path, index_cache=cache)
    (name,) = sidecars(cache)
    side = cache / name
    side.write_bytes(side.read_bytes()[:-5])
    assert segyrs.open(path, index_cache=cache).geometry.source == "scan"
    assert segyrs.open(path, index_cache=cache).geometry.source == "index_cache"


def test_index_cache_is_for_local_paths(tmp_path):
    path = inline_major(tmp_path / "c.sgy")
    with pytest.raises(ValueError, match="local paths"):
        segyrs.open(path.as_uri(), index_cache=tmp_path)
