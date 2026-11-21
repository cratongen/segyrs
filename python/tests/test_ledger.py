"""Python tests named after the M1 ledger ids in .ai/research/upstream-issues.md."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA, grid, write_segy

import segyrs


def ibm_words(*words: int) -> np.ndarray:
    return np.array([words], dtype=np.uint32)


def patch(path: Path, offset: int, raw: bytes) -> None:
    buf = bytearray(path.read_bytes())
    buf[offset : offset + len(raw)] = raw
    path.write_bytes(bytes(buf))


# ---- sample formats ---------------------------------------------------------


def test_s01_ibm_is_float32_overflow_is_inf_never_nan(segy):
    path = segy("ibm.sgy", ibm_words(0x41100000, 0x61100000, 0xC1100000, 0, 0xE1100000), fmt=1)
    vol = segyrs.open(path)
    out = vol.numpy()
    assert out.dtype == np.float32
    assert vol.attrs["sample_format"] == "ibm_float32"
    np.testing.assert_array_equal(out[0], [1.0, np.inf, -1.0, 0.0, -np.inf])
    assert not np.isnan(out).any()


def test_s02_format4_needs_gain_and_both_conventions(segy):
    # byte 2 is the gain exponent 3, bytes 3-4 the signed magnitude 5.
    path = segy("f4.sgy", np.array([[0x00030005, 0x00018002]], dtype=np.uint32), fmt=4)
    with pytest.raises(segyrs.Format4NeedsGain):
        segyrs.open(path)
    rev0 = segyrs.open(path, gain="rev0").numpy()
    rev2 = segyrs.open(path, gain="rev2").numpy()
    assert rev0.dtype == np.float64
    np.testing.assert_array_equal(rev0[0], [40.0, -4.0])
    np.testing.assert_array_equal(rev2[0], [0.625, -1.0])
    with pytest.raises(ValueError):
        segyrs.open(path, gain="rev9")


def test_s03_sample_format_overrides_a_wrong_code(segy):
    path = segy("wrong.sgy", np.array([[1.5, -2.25]], dtype=np.float32), fmt=5)
    patch(path, 3224, (1).to_bytes(2, "big"))  # the header now claims IBM
    vol = segyrs.open(path, sample_format="ieee_float32")
    np.testing.assert_array_equal(vol.numpy()[0], [1.5, -2.25])
    assert vol.attrs["sample_format"] == "ieee_float32"
    assert vol.attrs["binary_format_code"] == 1
    with pytest.raises(ValueError):
        segyrs.open(path, sample_format="ibm64")


@pytest.mark.parametrize(
    ("fmt", "dtype", "values"),
    [
        (2, np.int32, [-(2**31), 2**31 - 1, 7]),
        (3, np.int16, [-32768, 32767, 7]),
        (5, np.float32, [1.5, -0.0, 3e38]),
        (6, np.float64, [1e300, -2.5, 1 / 3]),
        (7, np.int32, [-(2**23), 2**23 - 1, 40000]),
        (8, np.int8, [-128, 127, 7]),
        (9, np.int64, [-(2**63), 2**63 - 1, 7]),
        (10, np.uint32, [0, 2**32 - 1, 7]),
        (11, np.uint16, [0, 65535, 7]),
        (12, np.uint64, [0, 2**64 - 1, 7]),
        (15, np.uint32, [0, 2**24 - 1, 40000]),
        (16, np.uint8, [0, 255, 7]),
    ],
)
@pytest.mark.parametrize("little", [False, True])
def test_s04_every_format_decodes_to_its_native_dtype(segy, fmt, dtype, values, little):
    samples = np.array([values], dtype=dtype)
    path = segy(f"f{fmt}.sgy", samples, fmt=fmt, little=little)
    out = segyrs.open(path).numpy()
    assert out.dtype == dtype
    assert out.dtype.isnative
    np.testing.assert_array_equal(out, samples)


def test_s05_interval_and_sample_count_are_unsigned_16_bit(segy):
    ns = 40000
    path = segy("u16.sgy", np.zeros((1, ns), dtype=np.int8), fmt=8, dt=50000)
    vol = segyrs.open(path)
    assert vol.attrs["sample_interval_us"] == 50000
    assert vol.sizes["time"] == ns
    assert vol.coords["time"][1] == 50000
    assert vol.coords["time"][-1] == (ns - 1) * 50000


# ---- headers ----------------------------------------------------------------


def test_s07_header_columns_match_the_bytes_of_each_trace():
    vol = segyrs.open(DATA / "small.sgy")
    headers = vol.headers
    for t in (0, 7, 24):
        raw = vol.header_bytes(t)
        assert headers["cdp_x"][t] == int.from_bytes(raw[180:184], "big", signed=True)
        assert headers["inline"][t] == int.from_bytes(raw[188:192], "big", signed=True)
        assert headers["crossline"][t] == int.from_bytes(raw[192:196], "big", signed=True)
        assert headers["ns"][t] == int.from_bytes(raw[114:116], "big")


def test_s09_headers_are_a_dict_of_numpy_arrays_one_entry_per_trace():
    vol = segyrs.open(DATA / "small.sgy")
    headers = vol.headers
    assert isinstance(headers, dict)
    assert all(isinstance(v, np.ndarray) and len(v) == 25 for v in headers.values())
    assert headers["inline"].dtype == np.int32
    table = vol.header_table()
    assert table.shape == (25,)
    np.testing.assert_array_equal(table["crossline"], headers["crossline"])
    sub = vol.sel(inline=2).headers
    assert len(sub["inline"]) == 5
    assert set(sub["inline"]) == {2}


def test_s23_decoded_headers_are_native_and_raw_bytes_stay_in_file_order():
    big = segyrs.open(DATA / "small.sgy")
    little = segyrs.open(DATA / "small-lsb.sgy")
    for name in ("inline", "crossline", "cdp_x", "ns"):
        np.testing.assert_array_equal(big.headers[name], little.headers[name])
        assert little.headers[name].dtype.isnative
    assert little.header_bytes(0)[188:192] == (1).to_bytes(4, "little")
    assert big.header_bytes(0)[188:192] == (1).to_bytes(4, "big")


def test_s26_any_header_word_is_a_byte_and_a_dtype(segy):
    samples, il, xl = grid(2, 2)
    path = segy(
        "map.sgy",
        samples,
        inline=il,
        crossline=xl,
        words={(233, "i4"): [5, 6, 7, 8], (237, "i2"): [-1, -2, -3, -4]},
    )
    vol = segyrs.open(
        path, header_map={"word": (233, "int32"), "short": (237, "int16"), "plain": 233}
    )
    np.testing.assert_array_equal(vol.headers["word"], [5, 6, 7, 8])
    np.testing.assert_array_equal(vol.headers["short"], [-1, -2, -3, -4])
    np.testing.assert_array_equal(vol.headers["plain"], [5, 6, 7, 8])
    wide = segyrs.open(path, header_map={"cdp_x": (181, "int64")})
    assert wide.headers["cdp_x"].dtype == np.int64
    with pytest.raises(ValueError):
        segyrs.open(path, header_map={"late": (239, "int32")})
    with pytest.raises(ValueError):
        segyrs.open(path, header_map={"odd": (1, "float32")})


def test_x28_standard_fields_are_the_default_map_and_line_bytes_move_both(segy):
    samples, il, xl = grid(2, 3)
    path = segy("bytes.sgy", samples, words={(9, "i4"): il, (13, "i4"): xl}, inline=np.zeros(6))
    vol = segyrs.open(path)
    assert {"cdp", "scalco", "inline", "crossline", "offset", "trace_id"} <= set(vol.headers)
    assert vol.geometry.kind == "unstructured"
    moved = segyrs.open(path, inline_byte=9, crossline_byte=13)
    assert moved.dims == ("inline", "crossline", "time")
    assert moved.geometry.inline_byte == 9
    np.testing.assert_array_equal(moved.headers["inline"], il)
    np.testing.assert_array_equal(moved.sel(inline=2, crossline=11).numpy(), samples[4])


def test_s27_delay_scalar_moves_the_axis_and_a_varying_delay_is_a_coordinate(segy):
    fixture = segyrs.open(DATA / "segyrs/delay-scalar.sgy")
    assert fixture.coords["time"][0] == 1_000_000  # 10000 ms / 10, in microseconds
    samples = np.zeros((2, 3), dtype=np.float32)
    words = {(109, "i2"): [50, 70], (215, "i2"): [-10, -10]}
    vol = segyrs.open(segy("delay.sgy", samples, words=words, inline=[1, 1], crossline=[1, 2]))
    np.testing.assert_array_equal(vol.coords["time"], [5000, 9000, 13000])
    np.testing.assert_array_equal(vol.coords["delay"], [[5000, 7000]])


def test_s28_nonzero_time_stamp_becomes_datetime64(segy):
    samples = np.zeros((1, 2), dtype=np.float32)
    words = {
        (157, "i2"): 2024,
        (159, "i2"): 32,
        (161, "i2"): 3,
        (163, "i2"): 4,
        (165, "i2"): 5,
        (240 + 141, "i4"): 123_000_000,
    }
    path = segy("stamp.sgy", samples, words=words, revision=(2, 0), extra_blocks=1)
    stamp = segyrs.open(path).coords["timestamp"]
    assert stamp.dtype == np.dtype("datetime64[ns]")
    assert stamp[0] == np.datetime64("2024-02-01T03:04:05.123")
    plain = segyrs.open(segy("zero.sgy", samples))
    assert "timestamp" not in plain.coords


def test_x05_coordinate_scalar_is_float64_with_the_raw_word_beside_it(segy):
    samples = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
    words = {(181, "i4"): [12345] * 3, (71, "i2"): [10, -100, 0]}
    vol = segyrs.open(segy("scalco.sgy", samples, fmt=3, words=words))
    np.testing.assert_array_equal(vol.coords["cdp_x"], [123450.0, 123.45, 12345.0])
    assert vol.coords["cdp_x"].dtype == np.float64
    np.testing.assert_array_equal(vol.coords["cdp_x_raw"], [12345] * 3)
    assert vol.coords["cdp_x_raw"].dtype == np.int32
    np.testing.assert_array_equal(vol.numpy(), samples)


# ---- geometry and selection -------------------------------------------------


def test_s12_open_reads_no_samples_and_a_selection_reads_only_its_traces():
    vol = segyrs.open(DATA / "small.sgy")
    _ = vol.headers, vol.coords, vol.geometry, repr(vol)
    assert vol.traces_read == 0
    line = vol.sel(inline=2)
    assert vol.traces_read == 0
    assert line.numpy().shape == (5, 50)
    assert vol.traces_read == 5


def test_s14_unknown_sort_opens_as_trace_by_time(segy):
    samples, il, xl = grid(2, 3)
    order = [3, 0, 5, 1, 4, 2]
    vol = segyrs.open(segy("shuffled.sgy", samples[order], inline=il[order], crossline=xl[order]))
    assert vol.geometry.kind == "unsorted"
    assert vol.dims == ("trace", "time")
    assert vol.geometry.axes == ()
    np.testing.assert_array_equal(vol.coords["inline"], il[order])


def test_s15_label_sel_on_an_unsorted_file_hits_the_trace_with_that_pair(segy):
    samples, il, xl = grid(3, 4)
    rng = np.random.default_rng(15)
    order = rng.permutation(len(il))
    vol = segyrs.open(segy("unsorted.sgy", samples[order], inline=il[order], crossline=xl[order]))
    assert vol.geometry.kind == "unsorted"
    for i, x, expected in zip(il, xl, samples, strict=True):
        before = vol.traces_read
        got = vol.sel(inline=int(i), crossline=int(x))
        np.testing.assert_array_equal(got.numpy(), expected[None, :])
        assert vol.traces_read - before == 1
        assert got.attrs["sample_format"] == "ieee_float32"
    line = vol.sel(inline=2)
    assert line.sizes["trace"] == 4
    np.testing.assert_array_equal(np.sort(line.coords["crossline"]), [10, 11, 12, 13])
    with pytest.raises(KeyError):
        vol.sel(inline=9, crossline=9)


def test_s16_sparse_survey_lists_missing_pairs_and_a_duplicate_is_an_error(segy):
    samples, il, xl = grid(3, 3)
    keep = ~((il == 3) & (xl == 12))
    vol = segyrs.open(segy("sparse.sgy", samples[keep], inline=il[keep], crossline=xl[keep]))
    assert vol.geometry.kind == "sparse"
    np.testing.assert_array_equal(vol.geometry.missing_pairs, [[3, 12]])
    assert vol.coords["trace_mask"].all()
    np.testing.assert_array_equal(vol.sel(inline=3, crossline=11).numpy()[0], samples[7])
    dup = segy("dup.sgy", samples[:2], inline=[1, 1], crossline=[10, 10])
    with pytest.raises(segyrs.DuplicateKey, match="inline=1 crossline=10"):
        segyrs.open(dup)


def test_s17_sort_is_a_string_and_every_permutation_reads_the_same_labels(segy):
    assert segyrs.open(DATA / "small.sgy").geometry.kind == "inline_major"
    assert segyrs.open(DATA / "shot-gather.sgy").geometry.kind == "unstructured"
    samples, il, xl = grid(3, 2)
    order = np.lexsort((il, xl))  # crossline slowest
    xmajor = segyrs.open(segy("x.sgy", samples[order], inline=il[order], crossline=xl[order]))
    assert xmajor.geometry.kind == "crossline_major"
    assert xmajor.dims == ("inline", "crossline", "time")
    np.testing.assert_array_equal(xmajor.numpy().reshape(6, -1), samples)

    il3, xl3 = np.repeat([1, 2], 2), np.tile([10, 11], 2)
    off3 = np.repeat([100, 200], 4)
    omajor = segyrs.open(
        segy(
            "o.sgy",
            np.zeros((8, 2), np.float32),
            inline=np.tile(il3, 2),
            crossline=np.tile(xl3, 2),
            offset=off3,
        )
    )
    assert omajor.geometry.kind == "offset_major"
    assert omajor.geometry.axes == ("offset", "inline", "crossline")
    np.testing.assert_array_equal(omajor.coords["trace"][1, 0], [2, 6])  # (il 2, xl 10)

    # Increasing and decreasing line, crossline, and offset orders: each label
    # selection lands on the trace whose headers carry those labels.
    for path in sorted(DATA.glob("small-ps*.sgy")):
        vol = segyrs.open(path)
        assert vol.dims == ("inline", "crossline", "offset", "time"), path.name
        picked = vol.sel(inline=[1, 2, 3, 4], crossline=[1, 2, 3], offset=[1, 2])
        h = picked.headers
        want = np.meshgrid([1, 2, 3, 4], [1, 2, 3], [1, 2], indexing="ij")
        for name, labels in zip(("inline", "crossline", "offset"), want, strict=True):
            np.testing.assert_array_equal(h[name], labels.ravel(), err_msg=path.name)
        trace = picked.coords["trace"].ravel()
        np.testing.assert_array_equal(
            picked.numpy().reshape(24, -1)[5],
            vol.numpy().reshape(24, -1)[np.flatnonzero(vol.coords["trace"].ravel() == trace[5])[0]],
        )
    dec = segyrs.open(DATA / "small-ps-dec-il-xl-off.sgy").coords
    np.testing.assert_array_equal(dec["inline"], [4, 3, 2, 1])
    np.testing.assert_array_equal(dec["offset"], [2, 1])


def test_s18_offset_is_a_dim_and_a_label_slice_keeps_the_first_offset(segy):
    il = np.repeat([1, 2], 5)
    off = np.tile([1, 2, 3, 4, 5], 2)
    samples = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    vol = segyrs.open(segy("gather.sgy", samples, inline=il, crossline=np.ones(10), offset=off))
    assert vol.dims == ("inline", "crossline", "offset", "time")
    every_other = vol.sel(offset=slice(None, None, 2))
    np.testing.assert_array_equal(every_other.coords["offset"], [1, 3, 5])
    np.testing.assert_array_equal(every_other.numpy()[1, 0, :, 0], [5, 7, 9])
    np.testing.assert_array_equal(vol.sel(offset=slice(2, 4)).coords["offset"], [2, 3, 4])


def test_s19_extended_float64_offset_overrides_the_int32(segy):
    samples = np.array([[1.0], [2.0]], dtype=np.float32)
    words = {(37, "i4"): [12, 25], (240 + 129, "f8"): [12.5, 25.0]}
    path = segy("ext.sgy", samples, words=words, revision=(2, 0), extra_blocks=1)
    vol = segyrs.open(path)
    assert vol.dims == ("trace", "time")
    np.testing.assert_array_equal(vol.coords["offset"], [12.5, 25.0])
    np.testing.assert_array_equal(vol.coords["offset_raw"], [12, 25])
    hit = vol.sel(offset=12.4, method="nearest")
    np.testing.assert_array_equal(hit.numpy(), [[1.0]])
    np.testing.assert_array_equal(vol.headers["offset"], [12, 25])

    il = np.repeat([1, 2], 2)
    words3 = {(37, "i4"): [12, 25] * 2, (240 + 129, "f8"): [12.5, 25.0] * 2}
    samples3 = np.arange(4, dtype=np.float32)[:, None]
    cube = segyrs.open(
        segy(
            "ext3.sgy",
            samples3,
            inline=il,
            crossline=np.ones(4),
            offset=[12, 25] * 2,
            words=words3,
            revision=(2, 0),
            extra_blocks=1,
        )
    )
    np.testing.assert_array_equal(cube.coords["offset"], [12.5, 25.0])
    np.testing.assert_array_equal(cube.sel(offset=25.0).numpy()[:, 0, 0], [1.0, 3.0])


def test_s40_isel_is_positions_and_sel_is_labels_including_line_zero(segy):
    samples, il, xl = grid(4, 2, il0=0)
    vol = segyrs.open(segy("zero.sgy", samples, inline=il, crossline=xl))
    first3 = vol.isel(inline=slice(0, 3))
    np.testing.assert_array_equal(first3.coords["inline"], [0, 1, 2])
    np.testing.assert_array_equal(vol.sel(inline=0).numpy(), samples[:2])
    np.testing.assert_array_equal(vol.sel(inline=slice(0, 2)).coords["inline"], [0, 1, 2])
    np.testing.assert_array_equal(vol.isel(inline=[3, 0]).coords["inline"], [3, 0])
    np.testing.assert_array_equal(vol.isel(inline=-1).numpy(), samples[6:])
    with pytest.raises(IndexError):
        vol.isel(inline=4)


def test_s41_one_sample_is_a_0d_value_with_item():
    vol = segyrs.open(DATA / "f3.sgy")
    one = vol.isel(inline=0, crossline=0, time=0)
    assert one.dims == ()
    assert one.numpy().shape == ()
    assert isinstance(one.item(), int)
    assert one.item() == vol.numpy()[0, 0, 0]
    flat = segyrs.open(DATA / "shot-gather.sgy").isel(trace=0, time=0)
    assert isinstance(flat.item(), float)


def test_x16_absurd_line_span_is_geometry_error_before_allocation(segy):
    path = segy("span.sgy", np.zeros((2, 1), np.float32), inline=[1, 10**9], crossline=[1, 1])
    with pytest.raises(segyrs.GeometryError, match=r"min=1 max=1000000000 n_traces=2"):
        segyrs.open(path)


def test_x19_selecting_a_grid_that_was_not_inferred_is_geometry_error():
    vol = segyrs.open(DATA / "shot-gather.sgy")
    assert vol.geometry.kind == "unstructured"
    with pytest.raises(segyrs.GeometryError):
        vol.sel(inline=1)
    with pytest.raises(segyrs.GeometryError):
        vol.isel(crossline=0)
    grid_vol = segyrs.open(DATA / "small.sgy")
    with pytest.raises(segyrs.GeometryError):
        grid_vol.sel(offset=0)


def test_x20_sparse_files_use_the_trace_layout_and_the_mask(segy):
    samples, il, xl = grid(3, 3)
    keep = ~((il == 1) & (xl == 10))
    vol = segyrs.open(segy("sparse.sgy", samples[keep], inline=il[keep], crossline=xl[keep]))
    assert vol.dims == ("trace", "time")
    assert vol.numpy().shape == (8, 4)
    assert not np.isnan(vol.numpy()).any()
    assert vol.coords["trace_mask"].dtype == bool


def test_vertical_selection_by_label_and_position():
    vol = segyrs.open(DATA / "small.sgy")
    full = vol.numpy()
    window = vol.sel(time=slice(8000, 20000))
    np.testing.assert_array_equal(window.coords["time"], [8000, 12000, 16000, 20000])
    np.testing.assert_array_equal(window.numpy(), full[:, :, 2:6])
    stepped = vol.isel(time=slice(1, None, 7)).isel(time=slice(1, 3))
    np.testing.assert_array_equal(stepped.numpy(), full[:, :, 8:22:7])
    picked = vol.isel(time=[5, 1, 9]).isel(inline=[4, 0])
    np.testing.assert_array_equal(picked.numpy(), full[[4, 0]][:, :, [5, 1, 9]])
    np.testing.assert_array_equal(vol.isel(time=slice(None, None, -1)).numpy(), full[..., ::-1])
    near = vol.sel(time=4100, method="nearest")
    np.testing.assert_array_equal(near.numpy(), full[:, :, 1])
    with pytest.raises(KeyError):
        vol.sel(time=4100)
    assert segyrs.open(DATA / "small.sgy", depth=True).dims == ("inline", "crossline", "depth")


# ---- byte order and text ----------------------------------------------------


def test_s22_endian_signature_including_pairwise(segy):
    samples = np.array([[1.5, -2.0, 3.25]], dtype=np.float32)
    little = segy("le.sgy", samples, little=True, revision=(2, 0), signature=b"\x04\x03\x02\x01")
    assert segyrs.open(little).attrs["source_endian"] == "little"
    zero = segy("be.sgy", samples)
    assert segyrs.open(zero).attrs["source_endian"] == "big"

    big = segy("pair.sgy", samples, revision=(2, 0), signature=b"\x01\x02\x03\x04")
    raw = bytearray(big.read_bytes())
    body = np.frombuffer(bytes(raw[3200:]), dtype=np.uint8).reshape(-1, 2)[:, ::-1]
    raw[3200:] = body.tobytes()
    big.write_bytes(bytes(raw))
    vol = segyrs.open(big)
    assert vol.attrs["source_endian"] == "pairwise"
    np.testing.assert_array_equal(vol.numpy(), samples)
    with pytest.raises(segyrs.EndianMismatch):
        segyrs.open(little, endian="big")


def test_x04_both_endians_decode_to_the_same_native_values():
    for name in ("small", "f3"):
        big = segyrs.open(DATA / f"{name}.sgy")
        little = segyrs.open(DATA / f"{name}-lsb.sgy")
        assert big.attrs["source_endian"] == "big"
        assert little.attrs["source_endian"] == "little"
        a, b = big.numpy(), little.numpy()
        assert a.dtype.isnative and b.dtype.isnative
        np.testing.assert_array_equal(a, b)


def test_s20_each_textual_block_is_detected_as_ascii_or_ebcdic():
    assert segyrs.open(DATA / "segyrs/delay-scalar.sgy").attrs["textual_encoding"] == ["ascii"]
    vol = segyrs.open(DATA / "small.sgy")
    assert vol.attrs["textual_encoding"] == ["ebcdic"]
    assert vol.attrs["textual_header"][0].startswith("C")


def test_s21_cards_are_80_characters_40_per_block_with_no_rewrap():
    vol = segyrs.open(DATA / "multi-text.sgy")
    cards = vol.attrs["textual_header"]
    assert len(vol.attrs["textual_encoding"]) == 5
    assert len(cards) == 5 * 40
    assert all(len(card) == 80 for card in cards)


def test_x21_a_high_textual_byte_does_not_fail_open_and_raw_bytes_are_kept(segy):
    card = bytes([0xC3, 0x40, 0xF1, 0xFF, 0x30])
    path = segy("petrel.sgy", np.zeros((1, 2), np.float32), first_card=card)
    vol = segyrs.open(path)
    assert len(vol.attrs["textual_header"]) == 40
    assert vol.textual_header_bytes[0] == path.read_bytes()[:3200]
    assert segyrs.open(DATA / "text-embed-null.sgy").sizes["inline"] == 5


# ---- dtypes, masks, attrs ---------------------------------------------------


def test_x01_integers_stay_integers_and_dead_traces_are_a_mask(segy):
    vol = segyrs.open(DATA / "f3.sgy")
    assert vol.numpy().dtype == np.int16
    samples = np.array([[1, 2], [3, 4]], dtype=np.int16)
    path = segy("dead.sgy", samples, fmt=3, words={(29, "i2"): [1, 2]})
    live = segyrs.open(path)
    assert live.coords["trace_mask"].tolist() == [True, True]
    dead = segyrs.open(path, dead_trace="trace_id")
    assert dead.coords["trace_mask"].tolist() == [True, False]
    assert dead.numpy().dtype == np.int16
    np.testing.assert_array_equal(dead.numpy(), samples)
    explicit = segyrs.open(path, dead_trace=np.array([True, False]))
    assert explicit.coords["trace_mask"].tolist() == [False, True]
    with pytest.raises(ValueError):
        segyrs.open(path, dead_trace=np.array([True]))


def test_x08_volume_dtype_follows_the_sample_format():
    vol = segyrs.open(DATA / "small.sgy")
    assert vol.dtype == np.float32
    assert vol.numpy().dtype == np.float32
    assert vol.attrs["sample_format"] == "ibm_float32"


def test_x30_samples_are_lazy_and_the_chunk_shape_is_stated():
    vol = segyrs.open(DATA / "small.sgy")
    assert vol.attrs["chunk_shape"] == (1, 50)
    assert "lazy" in (segyrs.Volume.__doc__ or "")
    assert "lazy" in repr(vol)
    assert vol.traces_read == 0
    vol.isel(inline=0, crossline=0).numpy()
    assert vol.traces_read == 1


def test_x31_file_level_attrs_survive_selection():
    vol = segyrs.open(DATA / "small.sgy")
    keys = ("sample_format", "source_endian", "revision", "textual_header", "measurement_system")
    for sub in (vol.sel(time=0), vol.isel(inline=1, crossline=slice(1, 3)), vol.sel(inline=3)):
        for key in keys:
            assert sub.attrs[key] == vol.attrs[key]


def test_x32_an_all_zero_trace_is_live_data(segy):
    samples = np.array([[1, 2], [0, 0], [3, 4]], dtype=np.int32)
    vol = segyrs.open(segy("zeros.sgy", samples, fmt=2))
    np.testing.assert_array_equal(vol.numpy()[1], [0, 0])
    assert vol.coords["trace_mask"].all()


# ---- errors -----------------------------------------------------------------


def test_s25_sample_count_endian_and_trace_length_are_three_errors(segy):
    path = segy("ok.sgy", np.ones((2, 4), np.int8), fmt=8)
    short = path.with_name("short.sgy")
    short.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(segyrs.SampleCountMismatch, match="remainder="):
        segyrs.open(short)
    for cls in (
        segyrs.SampleCountMismatch,
        segyrs.EndianMismatch,
        segyrs.NonUniformTraceLength,
        segyrs.TraceLengthError,
    ):
        assert issubclass(cls, segyrs.SegyrsError)
    assert (
        len({segyrs.SampleCountMismatch, segyrs.EndianMismatch, segyrs.NonUniformTraceLength}) == 3
    )


def test_x23_per_trace_sample_count_disagreeing_is_non_uniform_trace_length(segy):
    path = segy(
        "mixed.sgy", np.ones((2, 4), np.int8), fmt=8, fixed_length=0, words={(115, "u2"): [4, 5]}
    )
    with pytest.raises(segyrs.NonUniformTraceLength, match="trace 1 binary_ns=4 trace_ns=5"):
        segyrs.open(path)


def test_x18_one_short_trace_is_trace_length_error_naming_the_index(segy):
    path = segy("short1.sgy", np.ones((3, 4), np.int8), fmt=8, fixed_length=0)
    raw = bytearray(path.read_bytes())
    t1 = 3600 + 244
    raw[t1 + 114 : t1 + 116] = (3).to_bytes(2, "big")
    del raw[t1 + 240 + 3]
    path.write_bytes(bytes(raw))
    with pytest.raises(segyrs.TraceLengthError, match="trace 1 expected=244 got=243"):
        segyrs.open(path)


def test_s39_typed_errors_carry_their_numbers(segy, tmp_path):
    path = segy("code.sgy", np.ones((1, 2), np.float32))
    patch(path, 3224, (13).to_bytes(2, "big"))
    with pytest.raises(segyrs.UnknownFormat, match="13"):
        segyrs.open(path)
    tiny = tmp_path / "tiny.sgy"
    tiny.write_bytes(b"\0" * 100)
    with pytest.raises(segyrs.TruncatedError, match="file has 100 bytes, need at least 3600"):
        segyrs.open(tiny)
    with pytest.raises(segyrs.SegyrsError):
        segyrs.open(tmp_path / "missing.sgy")
    assert issubclass(segyrs.GeometryError, segyrs.SegyrsError)
    assert issubclass(segyrs.UnsortedCoordinate, segyrs.SegyrsError)


# ---- API surface ------------------------------------------------------------


def test_s38_open_has_no_strict_or_ignore_geometry():
    with pytest.raises(TypeError):
        segyrs.open(DATA / "small.sgy", strict=True)
    with pytest.raises(TypeError):
        segyrs.open(DATA / "small.sgy", ignore_geometry=True)


def test_s42_non_ascii_and_pathlike_paths_open():
    plain = segyrs.open(DATA / "small.sgy").numpy()
    np.testing.assert_array_equal(segyrs.open(DATA / "小文件.sgy").numpy(), plain)
    np.testing.assert_array_equal(segyrs.open(str(DATA / "小文件.sgy")).numpy(), plain)


def test_s46_pyo3_binding_is_thread_safe_across_concurrent_reads():
    from segyrs import _segyrs

    assert _segyrs.__file__.endswith(".so") or _segyrs.__file__.endswith(".pyd")
    vol = segyrs.open(DATA / "small.sgy")
    expected = vol.numpy()
    out: list[np.ndarray] = []

    def work(i: int) -> None:
        out.append(vol.isel(inline=i % 5).numpy())

    threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 20
    for arr in out:
        assert any(np.array_equal(arr, expected[i]) for i in range(5))
    assert isinstance(vol.header_bytes(3), bytes) and len(vol.header_bytes(3)) == 240


def test_x33_open_and_read_start_no_process(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("segyrs started a process")

    monkeypatch.setattr(os, "fork", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(multiprocessing.Process, "start", refuse)
    vol = segyrs.open(DATA / "small.sgy")
    vol.numpy()
    _ = vol.headers, vol.coords
    assert multiprocessing.active_children() == []


def test_volume_is_made_by_open_only_and_converts_with_numpy():
    with pytest.raises(TypeError):
        segyrs.Volume()
    vol = segyrs.open(DATA / "tiny.sgy")
    np.testing.assert_array_equal(np.asarray(vol), vol.numpy())
    assert len(vol) == 3
    binary = vol.binary_header
    assert binary["format_code"] == 1
    assert binary["n_samples"] == 4
    assert len(vol.binary_header_bytes) == 400


def test_written_fixture_round_trips_through_the_writer_helper(tmp_path):
    samples, il, xl = grid(2, 3, ns=5)
    path = write_segy(tmp_path / "g.sgy", samples, inline=il, crossline=xl)
    vol = segyrs.open(path)
    assert vol.sizes == {"inline": 2, "crossline": 3, "time": 5}
    np.testing.assert_array_equal(vol.numpy().reshape(6, 5), samples)
