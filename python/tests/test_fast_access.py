"""Single-trace selection cost, pointwise batch reads, and lazy header columns."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DATA, grid

import segyrs


def _ibm(words: np.ndarray) -> np.ndarray:
    """IBM hex float to float64, from the format definition (sign, base-16 exponent, fraction)."""
    w = words.astype(np.uint64)
    sign = np.where(w >> 31, -1.0, 1.0)
    exp = ((w >> 24) & 0x7F).astype(np.int64) - 64
    frac = (w & 0xFFFFFF).astype(np.float64) / float(1 << 24)
    return sign * frac * np.power(16.0, exp)


def _file_traces(path, endian: str = "big") -> np.ndarray:
    """Every trace in file order, decoded straight from the bytes (what segyio's
    ``f.trace[i]`` returns): IBM float or IEEE float32, one 240-byte header block."""
    vol = segyrs.open(path, endian=None if endian == "big" else endian)
    f = vol._src.file
    ns, n = f.n_samples, f.n_traces
    order = ">" if endian == "big" else "<"
    raw = np.fromfile(path, dtype=np.uint8, offset=f.data_start)
    body = raw.reshape(n, 240 + 4 * ns)[:, 240:].copy().view(order + "u4")
    if vol.attrs["sample_format"] == "ibm_float32":
        return _ibm(body).astype(np.float32)
    return body.view(order + "f4").astype(np.float32)


def _xline_major(segy, nil=4, nxl=3, ns=5):
    samples, il, xl = grid(nil, nxl, ns)
    perm = np.lexsort((il, xl))  # crossline-major file order
    return segy("xmajor.sgy", samples[perm], inline=il[perm], crossline=xl[perm]), samples


def test_single_trace_isel_on_a_crossline_sorted_file_keeps_the_inline_crossline_layout(segy):
    path, samples = _xline_major(segy)
    vol = segyrs.open(path)
    assert vol.geometry.kind == "crossline_major"
    assert vol.dims == ("inline", "crossline", "time")
    expect = samples.reshape(4, 3, 5)
    for i in range(4):
        for j in range(3):
            one = vol.isel(inline=i, crossline=j)
            assert one.dims == ("time",)
            np.testing.assert_array_equal(one.numpy(), expect[i, j])
    np.testing.assert_array_equal(vol.numpy(), expect)
    np.testing.assert_array_equal(vol.isel(crossline=1).numpy(), expect[:, 1])


def test_single_trace_isel_does_not_copy_the_trace_grid(segy):
    path, _ = _xline_major(segy)
    vol = segyrs.open(path)
    line = vol.isel(inline=2)
    assert np.shares_memory(line._traces, vol._traces)


@pytest.mark.parametrize("name,endian", [("small.sgy", "big"), ("small-lsb.sgy", "little")])
def test_isel_points_trace_matches_file_order_traces(name, endian):
    path = DATA / name
    ref = _file_traces(path, endian)
    vol = segyrs.open(path, endian=None if endian == "big" else endian)
    idx = np.random.default_rng(1).integers(0, len(ref), 40)
    pts = vol.isel_points(trace=idx)
    assert pts.dims == ("point", "time")
    out = pts.numpy()
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, ref[idx])
    np.testing.assert_array_equal(pts.coords["trace"], idx)
    np.testing.assert_array_equal(pts.isel(time=slice(3, 9)).numpy(), ref[idx, 3:9])


def test_isel_points_trace_on_a_crossline_sorted_file_is_file_order(segy):
    path, _ = _xline_major(segy)
    vol = segyrs.open(path)
    ref = _file_traces(path)
    idx = [11, 0, 5, 5]
    np.testing.assert_array_equal(vol.isel_points(trace=idx).numpy(), ref[idx])
    with pytest.raises(ValueError, match="trace="):
        vol.isel(inline=1).isel_points(trace=[0])


def test_sel_points_matches_one_sel_per_point_on_both_sort_orders(segy):
    for path in (DATA / "small.sgy", _xline_major(segy)[0]):
        vol = segyrs.open(path)
        il = vol.coords["inline"]
        xl = vol.coords["crossline"]
        rng = np.random.default_rng(2)
        pil = il[rng.integers(0, len(il), 30)]
        pxl = xl[rng.integers(0, len(xl), 30)]
        pts = vol.sel_points(inline=pil, crossline=pxl)
        assert pts.dims == ("point", "time")
        ref = np.stack(
            [vol.sel(inline=a, crossline=b).numpy() for a, b in zip(pil, pxl, strict=True)]
        )
        np.testing.assert_array_equal(pts.numpy(), ref)
        np.testing.assert_array_equal(pts.coords["inline"], pil)
        np.testing.assert_array_equal(pts.coords["crossline"], pxl)
        pos = vol.isel_points(
            inline=np.searchsorted(il, pil), crossline=np.searchsorted(xl, pxl)
        ).numpy()
        np.testing.assert_array_equal(pos, ref)


def test_points_keep_the_dims_left_out():
    vol = segyrs.open(DATA / "small-ps.sgy")
    il, xl = vol.coords["inline"], vol.coords["crossline"]
    pts = vol.sel_points(inline=[il[2], il[0]], crossline=[xl[1], xl[2]])
    assert pts.dims == ("point", "offset", "time")
    full = vol.numpy()
    np.testing.assert_array_equal(pts.numpy(), full[[2, 0], [1, 2]])
    mid = vol.isel_points(crossline=[0, 2], offset=[1, 0])
    assert mid.dims == ("point", "inline", "time")
    np.testing.assert_array_equal(mid.numpy(), np.moveaxis(full[:, [0, 2], [1, 0]], 1, 0))


def test_points_errors():
    vol = segyrs.open(DATA / "small.sgy")
    il, xl = vol.coords["inline"], vol.coords["crossline"]
    with pytest.raises(KeyError):
        vol.sel_points(inline=[il[0], 999], crossline=[xl[0], xl[1]])
    with pytest.raises(ValueError, match="length"):
        vol.isel_points(inline=[0, 1], crossline=[0])
    with pytest.raises(IndexError):
        vol.isel_points(inline=[5], crossline=[0])
    with pytest.raises(ValueError):
        vol.isel_points(time=[0])
    near = vol.sel_points(inline=[il[1] + 0.2], crossline=[xl[3]], method="nearest")
    np.testing.assert_array_equal(near.numpy(), vol.isel(inline=[1], crossline=3).numpy())
    gather = segyrs.open(DATA / "shot-gather.sgy")
    with pytest.raises(segyrs.GeometryError):
        gather.sel_points(inline=[1], crossline=[1])
    np.testing.assert_array_equal(
        gather.isel_points(trace=[3, 1]).numpy(), gather.isel(trace=[3, 1]).numpy()
    )


def test_vector_label_lookup_takes_the_first_of_duplicates_and_decreasing_labels():
    from segyrs._fast import _vector_labels

    labels = np.array([9, 7, 7, 3, 1], dtype=np.int32)
    np.testing.assert_array_equal(_vector_labels(labels, np.array([7, 1, 9, 3]), "x"), [1, 4, 0, 3])
    with pytest.raises(KeyError):
        _vector_labels(labels, np.array([2]), "x")
    with pytest.raises(KeyError):
        _vector_labels(labels[:0], np.array([2]), "x")


def test_headers_decode_one_column_on_first_access():
    vol = segyrs.open(DATA / "small.sgy")
    cache = vol._src.header_cache
    h = vol.headers
    assert isinstance(h, dict) and "source_x" in h and len(h) == len(cache.fields)
    assert "source_x" not in cache.cols
    col = h["source_x"]
    assert "source_x" in cache.cols and "group_x" not in cache.cols
    assert not col.flags.writeable
    full = vol._src.file.header_columns([(n, b, d) for n, (b, d) in cache.fields.items()])
    for name, ref in full.items():
        np.testing.assert_array_equal(h[name], ref)
        assert h[name].dtype == ref.dtype
    assert set(dict(vol.headers)) == set(full)
    assert all(isinstance(v, np.ndarray) for v in dict(vol.headers).values())
    assert all(isinstance(v, np.ndarray) for v in {**vol.headers}.values())
    assert all(isinstance(v, np.ndarray) for v in vol.headers.values())
    assert vol.headers.get("nope", 3) == 3
    sub = vol.isel(inline=1, crossline=[0, 2]).headers
    np.testing.assert_array_equal(sub["source_x"], full["source_x"][[5, 7]])
    table = vol.header_table()
    np.testing.assert_array_equal(table["cdp_x"], full["cdp_x"])


def test_a_header_map_override_is_not_served_from_the_open_scan(segy):
    samples, il, xl = grid(2, 2)
    path = segy(
        "h.sgy",
        samples,
        inline=il,
        crossline=xl,
        words={(181, "i4"): np.arange(4) + 10, (185, "i4"): np.arange(4) + 50},
    )
    np.testing.assert_array_equal(segyrs.open(path).headers["cdp_x"], np.arange(4) + 10)
    moved = segyrs.open(path, header_map={"cdp_x": (185, "int32")})
    np.testing.assert_array_equal(moved.headers["cdp_x"], np.arange(4) + 50)
    wide = segyrs.open(path, header_map={"cdp_x": (181, "int64")})
    assert wide.headers["cdp_x"].dtype == np.int64
