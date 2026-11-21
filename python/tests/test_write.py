"""M2 from Python: to_segy round trips, create, in-place edits, validate."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from conftest import DATA, grid

import segyrs


def _ieee(segy):
    samples, il, xl = grid(3, 4, ns=6)
    return segy("ieee.sgy", samples, fmt=5, inline=il, crossline=xl)


@pytest.mark.parametrize("source", ["ieee", "ibm", "int16"])
def test_x22_to_segy_round_trip_is_byte_identical(segy, tmp_path, source):
    path = {
        "ieee": lambda: _ieee(segy),
        "ibm": lambda: DATA / "small.sgy",
        "int16": lambda: DATA / "f3.sgy",
    }[source]()
    vol = segyrs.open(path)
    out = tmp_path / "copy.sgy"
    vol.to_segy(out)
    back = segyrs.open(out)
    assert back.attrs["sample_format"] == vol.attrs["sample_format"]
    assert back.sizes == vol.sizes
    np.testing.assert_array_equal(back.numpy(), vol.numpy())
    assert out.read_bytes() == path.read_bytes()


def test_s30_s36_selection_crop_keeps_headers_and_time(tmp_path):
    vol = segyrs.open(DATA / "small.sgy")
    sub = vol.isel(inline=slice(1, 3), crossline=[0, 2], time=slice(10, 20))
    out = tmp_path / "crop.sgy"
    sub.to_segy(out)
    back = segyrs.open(out)
    assert back.attrs["sample_format"] == "ibm_float32"
    np.testing.assert_array_equal(back.numpy(), sub.numpy())
    np.testing.assert_array_equal(back.coords["time"], sub.coords["time"])
    np.testing.assert_array_equal(back.coords["inline"], sub.coords["inline"])
    np.testing.assert_array_equal(back.coords["crossline"], sub.coords["crossline"])
    assert segyrs.validate(out)["n_traces"] == 4


def test_s35_to_segy_converts_only_when_asked(tmp_path):
    vol = segyrs.open(DATA / "small.sgy").isel(time=slice(5, 15))
    out = tmp_path / "ieee.sgy"
    vol.to_segy(out, sample_format="ieee_float32", endian="little")
    back = segyrs.open(out)
    assert back.attrs["sample_format"] == "ieee_float32"
    assert back.attrs["source_endian"] == "little"
    np.testing.assert_array_equal(back.numpy(), vol.numpy())
    np.testing.assert_array_equal(back.coords["time"], vol.coords["time"])


def test_s31_create_from_a_3d_array(tmp_path):
    data = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5) / 4
    out = tmp_path / "new.sgy"
    segyrs.create(
        out,
        data,
        sample_format="ibm_float32",
        sample_interval_us=2000,
        inline=[100, 101],
        crossline=[7, 8, 9],
        headers={"cdp_x": np.full(6, 1234.5), "scalco": -10},
        text="C 1 made by segyrs tests",
    )
    vol = segyrs.open(out)
    assert vol.dims == ("inline", "crossline", "time")
    assert vol.attrs["sample_format"] == "ibm_float32"
    assert vol.attrs["sample_interval_us"] == 2000
    np.testing.assert_array_equal(vol.coords["inline"], [100, 101])
    np.testing.assert_array_equal(vol.numpy(), data)
    np.testing.assert_array_equal(vol.coords["cdp_x"], np.full((2, 3), 1234.5))
    assert vol.headers["cdp_x"][0] == 12345
    assert vol.attrs["textual_header"][0].startswith("C 1 made by segyrs tests")


def test_x24_create_refuses_values_that_do_not_fit(tmp_path):
    data = np.zeros((2, 4), np.float32)
    with pytest.raises(segyrs.HeaderValueError):
        segyrs.create(
            tmp_path / "a.sgy",
            data,
            sample_format="ieee_float32",
            sample_interval_us=4000,
            headers={"trace_id": [1, 70000]},
        )
    with pytest.raises(segyrs.SampleValueError):
        segyrs.create(
            tmp_path / "b.sgy", data + 0.5, sample_format="int16", sample_interval_us=4000
        )
    assert not (tmp_path / "a.sgy").exists()
    assert not (tmp_path / "b.sgy").exists()


def test_create_refuses_a_fractional_interval(tmp_path):
    data = np.zeros((2, 4), np.float32)
    with pytest.raises(ValueError, match="not a whole number of microseconds"):
        segyrs.create(tmp_path / "a.sgy", data, sample_format="int16", sample_interval_us=333.4)
    assert not (tmp_path / "a.sgy").exists()
    segyrs.create(tmp_path / "b.sgy", data, sample_format="int16", sample_interval_us=2000.0)
    assert segyrs.validate(tmp_path / "b.sgy")["sample_interval_us"] == 2000


def test_create_says_why_a_scaled_coordinate_is_refused(tmp_path):
    data = np.zeros((2, 4), np.float32)
    with pytest.raises(segyrs.HeaderValueError, match="not a whole number; use a finer scalar"):
        segyrs.create(
            tmp_path / "c.sgy",
            data,
            sample_format="int16",
            sample_interval_us=2000,
            headers={"cdp_x": [0.5, 0.123], "scalco": -100},
        )


def test_s08_s32_edit_headers_and_traces_in_place(tmp_path):
    path = tmp_path / "small.sgy"
    shutil.copy(DATA / "small.sgy", path)
    before = segyrs.open(path).numpy().reshape(25, 50)
    with segyrs.edit(path) as ed:
        assert (ed.n_traces, ed.n_samples, ed.sample_format) == (25, 50, "ibm_float32")
        ed.set_header("cdp", np.arange(25) + 1000)
        ed.set_header("cdp_x", [10.5, 20.5], traces=[3, 4], scalars=-10)
        ed.write_traces(np.full(50, 2.0, np.float32), first=7)
    vol = segyrs.open(path)
    np.testing.assert_array_equal(vol.headers["cdp"], np.arange(25) + 1000)
    np.testing.assert_array_equal(vol.headers["cdp_x"][3:5], [105, 205])
    after = vol.numpy().reshape(25, 50)
    np.testing.assert_array_equal(after[7], np.full(50, 2.0, np.float32))
    np.testing.assert_array_equal(np.delete(after, 7, 0), np.delete(before, 7, 0))


def test_s35_edit_refuses_bad_input_and_leaves_the_file(tmp_path):
    path = tmp_path / "f3.sgy"
    shutil.copy(DATA / "f3.sgy", path)
    original = path.read_bytes()
    with segyrs.edit(path) as ed:
        with pytest.raises(segyrs.SampleValueError):
            ed.write_traces(np.full((1, ed.n_samples), 0.5), first=0)
        with pytest.raises(segyrs.TraceLengthError):
            ed.write_traces(np.zeros((1, 3), np.int16))
        with pytest.raises(segyrs.HeaderValueError):
            ed.set_header("trace_id", 40000)
    assert path.read_bytes() == original
    ed.close()
    with pytest.raises(ValueError, match="closed"):
        ed.sync()


def test_s34_validate_reports_a_corrupted_file(segy, tmp_path):
    good = _ieee(segy)
    layout = segyrs.validate(good)
    assert layout["n_traces"] == 12
    assert layout["sample_format"] == "ieee_float32"
    bad = tmp_path / "cut.sgy"
    bad.write_bytes(good.read_bytes()[:-10])
    with pytest.raises(segyrs.ValidationError) as err:
        segyrs.validate(bad)
    assert isinstance(err.value, segyrs.SegyrsError)
    assert err.value.problems
    assert any("sample count mismatch" in p for p in err.value.problems)
