"""M3 from Python: one test per whole-volume operation, on small generated files."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DATA, grid

import segyrs


@pytest.fixture
def cube(segy):
    """3 x 4 inline-major float32 grid, 6 samples, offsets 10 * trace."""
    samples, il, xl = grid(3, 4, ns=6)
    return segy("cube.sgy", samples, inline=il, crossline=xl, offset=10 * np.arange(12))


def test_resort_crossline_major(segy, tmp_path):
    samples, il, xl = grid(3, 4, ns=5)
    rng = np.random.default_rng(1)
    order = rng.permutation(12)
    path = segy("shuffled.sgy", samples[order], inline=il[order], crossline=xl[order])
    vol = segyrs.open(path)
    assert vol.dims == ("trace", "time")
    out = tmp_path / "sorted.sgy"
    report = vol.resort(out, by=("crossline", "inline"), budget=4096)
    assert report["strategy"] in ("memory", "gather", "external")
    back = segyrs.open(out)
    assert back.geometry.kind == "crossline_major"
    np.testing.assert_array_equal(back.numpy(), samples.reshape(3, 4, 5))
    with pytest.raises(ValueError, match="whole files"):
        vol.isel(trace=[0, 1]).resort(tmp_path / "x.sgy")


def test_x29_filter_on_header_fields(cube):
    vol = segyrs.open(cube)
    hit = vol.filter(offset=(20, 60), crossline=[11, 12])
    assert hit.dims == ("trace", "time")
    want = [t for t in range(12) if 20 <= 10 * t <= 60 and 10 + t % 4 in (11, 12)]
    np.testing.assert_array_equal(hit.coords["trace"], want)
    np.testing.assert_array_equal(hit.numpy(), vol.numpy().reshape(12, 6)[want])
    sub = vol.isel(inline=0).filter(offset=slice(None, 15))
    np.testing.assert_array_equal(sub.coords["trace"], [0, 1])


def test_s30_crop_rezero_time(tmp_path):
    vol = segyrs.open(DATA / "small.sgy").isel(time=slice(10, 20))
    out = tmp_path / "rezero.sgy"
    vol.to_segy(out, rezero_time=True, budget=1 << 16)
    back = segyrs.open(out)
    assert back.coords["time"][0] == 0
    np.testing.assert_array_equal(back.numpy(), vol.numpy())


def test_s10_time_slice_by_label(cube):
    vol = segyrs.open(cube)
    t = vol.coords["time"][3]
    sl = vol.sel(time=t)
    assert sl.dims == ("inline", "crossline")
    np.testing.assert_array_equal(sl.numpy(), vol.numpy()[..., 3])


def test_s10_slice_horizon(cube):
    vol = segyrs.open(cube)
    times = vol.coords["time"]
    horizon = np.full((3, 4), times[2], dtype=np.float64)
    horizon[0, 0] = np.nan
    horizon[2, 3] = times[5]
    data, valid = vol.slice_horizon(horizon, above=1, below=1)
    assert data.shape == valid.shape == (3, 4, 3)
    full = vol.numpy()
    np.testing.assert_array_equal(data[1, 1], full[1, 1, 1:4])
    assert not valid[0, 0].any()
    np.testing.assert_array_equal(valid[2, 3], [True, True, False])


def test_x15_x17_to_regular_fills_holes(segy, tmp_path):
    samples, il, xl = grid(3, 3, ns=4)
    keep = np.array([0, 1, 2, 3, 5, 6, 7, 8])  # (2, 11) is missing
    path = segy("holey.sgy", samples[keep], inline=il[keep], crossline=xl[keep])
    vol = segyrs.open(path)
    assert vol.dims == ("trace", "time")
    data, mask, coords = vol.to_regular(fill=-1.0)
    assert data.shape == (3, 3, 4)
    np.testing.assert_array_equal(coords["inline"], [1, 2, 3])
    np.testing.assert_array_equal(coords["crossline"], [10, 11, 12])
    assert not mask[1, 1] and mask.sum() == 8
    np.testing.assert_array_equal(data[1, 1], np.full(4, -1.0, np.float32))
    np.testing.assert_array_equal(data[2, 2], samples[8])

    out = tmp_path / "regular.sgy"
    assert vol.to_regular(out, fill=0.0, budget=4096) == {"cells": 9, "filled": 1}
    back = segyrs.open(out)
    assert back.geometry.kind == "inline_major"
    assert back.numpy().shape == (3, 3, 4)


def test_x35_concat(cube, segy, tmp_path):
    vol = segyrs.open(cube)
    samples, il, xl = grid(2, 4, ns=6, il0=4)
    more = segy("more.sgy", samples + 0.5, inline=il, crossline=xl)
    out = tmp_path / "both.sgy"
    assert segyrs.concat([vol, more], out, budget=4096) == 20
    back = segyrs.open(out)
    assert back.sizes == {"inline": 5, "crossline": 4, "time": 6}
    np.testing.assert_array_equal(
        back.numpy().reshape(20, 6),
        np.concatenate([vol.numpy().reshape(12, 6), samples + 0.5]),
    )
    other = segy("other.sgy", np.zeros((2, 7), np.float32))
    with pytest.raises(segyrs.FormatChangeError):
        segyrs.concat([cube, other], tmp_path / "bad.sgy")
    assert not (tmp_path / "bad.sgy").exists()


def test_stats_and_rms(cube):
    vol = segyrs.open(cube)
    full = vol.numpy().astype(np.float64)
    st = vol.stats()
    assert st["count"] == full.size and st["nan"] == 0
    assert (st["min"], st["max"]) == (full.min(), full.max())
    assert st["mean"] == pytest.approx(full.mean())
    assert st["std"] == pytest.approx(full.std())
    rms = vol.isel(inline=1).rms()
    assert rms.shape == (4,)
    np.testing.assert_allclose(rms, np.sqrt((full[1] ** 2).mean(axis=-1)))
    with pytest.raises(ValueError, match="whole traces"):
        vol.isel(time=0).stats()


def test_clip_keeps_format(tmp_path):
    vol = segyrs.open(DATA / "f3.sgy")
    out = tmp_path / "clip.sgy"
    full = vol.numpy()
    changed = vol.clip(-500, 600, out, budget=4096)
    assert changed == int(((full < -500) | (full > 600)).sum()) > 0
    back = segyrs.open(out)
    assert back.attrs["sample_format"] == "int16"
    np.testing.assert_array_equal(back.numpy(), np.clip(full, -500, 600))


def test_budget_too_small_is_refused_before_allocating(cube):
    vol = segyrs.open(cube)
    with pytest.raises(segyrs.SegyrsError, match="budget"):
        vol.to_regular(budget=16)
    with pytest.raises(ValueError):
        vol.to_regular(budget=1.5)
