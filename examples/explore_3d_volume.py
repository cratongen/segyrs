"""Exploring a 3D post-stack volume: the script behind docs/examples/explore-3d-volume.md.

Run it with `mise run example`. It writes its SEG-Y files to a scratch directory that it
deletes at the end, and its figures to the directory given by --figures (default: a
scratch directory). Every code block of the page is a verbatim part of this file, and every
result block of the page is a verbatim part of what it prints (python/tests/test_examples.py).
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import segyrs

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--figures", type=Path, help="directory for the PNG figures")
args = parser.parse_args()
workdir = Path(tempfile.mkdtemp(prefix="segyrs-example-"))
atexit.register(shutil.rmtree, workdir, ignore_errors=True)
figures = args.figures or workdir
figures.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20261120)
inlines = np.arange(1000, 1064)
crosslines = np.arange(2000, 2160, 2)
n_samples, dt_us, delay_ms = 251, 4000, 100
t = delay_ms / 1000 + np.arange(n_samples) * dt_us / 1e6

il, xl = np.meshgrid(inlines, crosslines, indexing="ij")
u, v = (il - 1000) / 63, (xl - 2000) / 158
horizons = [
    (0.36 + 0.08 * u, 1.0),
    (0.58 - 0.12 * np.exp(-((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.04), -0.8),
    (0.80 + 0.03 * v + 0.04 * (il >= 1040), 0.6),
]
reflectivity = np.zeros((*il.shape, n_samples))
for twt, rc in horizons:
    k = np.rint((twt - t[0]) * 1e6 / dt_us).astype(int)
    np.put_along_axis(reflectivity, k[..., None], rc, axis=-1)

tw = np.arange(-20, 21) * dt_us / 1e6
ricker = (1 - 2 * (np.pi * 25 * tw) ** 2) * np.exp(-((np.pi * 25 * tw) ** 2))
cube = np.apply_along_axis(np.convolve, -1, reflectivity, ricker, mode="same")
cube = (cube + 0.05 * rng.standard_normal(cube.shape)).astype(np.float32)

azimuth = np.deg2rad(30)
along_il, along_xl = (il - 1000) * 25.0, (xl - 2000) / 2 * 12.5
cdp_x = np.round(500_000 + along_il * np.cos(azimuth) - along_xl * np.sin(azimuth), 2)
cdp_y = np.round(6_000_000 + along_il * np.sin(azimuth) + along_xl * np.cos(azimuth), 2)

path = workdir / "survey3d.sgy"
segyrs.create(
    path,
    cube,
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    inline=inlines,
    crossline=crosslines,
    headers={"cdp_x": cdp_x, "cdp_y": cdp_y, "scalco": -100, "delay": delay_ms},
    text=[
        "C 1 SYNTHETIC 3D POST-STACK VOLUME, SEED 20261120. NOT FIELD DATA.",
        "C 2 INLINE BYTE 189, CROSSLINE BYTE 193, BOTH INT32.",
        "C 3 CDP X BYTE 181, CDP Y BYTE 185, METRES, SCALCO -100 (CENTIMETRES).",
        "C 4 TWO-WAY TIME IN MS, DELAY 100 MS, SAMPLE INTERVAL 4 MS.",
    ],
)
print(f"{path.stat().st_size} bytes")

vol = segyrs.open(path)
print(vol)
print("traces read:", vol.traces_read)
assert vol.traces_read == 0

print(vol.geometry)
print("source:", vol.geometry.source)
print("crossline labels:", vol.geometry.crosslines[:4], "...", vol.geometry.crosslines[-1])
for key in ("sample_format", "revision", "sample_interval_us", "measurement_system"):
    print(f"{key}: {vol.attrs[key]!r}")
for card in vol.attrs["textual_header"][:4]:
    print(card.rstrip())

section = vol.sel(inline=1030)
print(section.dims, section.shape)
print("traces read:", vol.traces_read)
data = section.numpy()
print("traces read:", section.traces_read)
print(data.dtype, data.shape)

ibm_tolerance = 2.0**-20 * np.abs(cube).max()
assert np.abs(data - cube[30]).max() <= ibm_tolerance
assert np.array_equal(data, vol.isel(inline=30).numpy())

xline = vol.sel(crossline=2080)
tslice = vol.sel(time=400_000)
print(xline.dims, xline.shape)
print(tslice.dims, tslice.shape)
assert np.abs(xline.numpy() - cube[:, 40]).max() <= ibm_tolerance
assert np.abs(tslice.numpy() - cube[:, :, 75]).max() <= ibm_tolerance

window = vol.sel(inline=slice(1020, 1040), crossline=2080, time=slice(300_000, 700_000))
print(window.dims, window.shape)
print(vol.coords["time"][:3], "...", vol.coords["time"][-1])
assert np.abs(window.numpy() - cube[20:41, 40, 50:151]).max() <= ibm_tolerance

pic = vol.sel(inline=1030).plot(width=640, height=400)
pic.save(figures / "inline-1030.png")
print(pic.text)

vol.sel(crossline=2080).plot(width=640, height=400).save(figures / "crossline-2080.png")
pic = vol.sel(time=580_000).plot(width=640, height=400)
pic.save(figures / "time-580ms.png")
print(pic.text)

print(sorted(vol.headers)[:6], "...", len(vol.headers), "fields")
print("scalco:", np.unique(vol.headers["scalco"]))
print("cdp_x word, trace 0:", vol.headers["cdp_x"][0])
print("cdp_x metres, trace 0:", vol.coords["cdp_x"][0, 0])

raw = vol.header_bytes(0)
print(len(raw), "bytes")
print("bytes 71-72 (scalco):", int.from_bytes(raw[70:72], "big", signed=True))
print("bytes 181-184 (cdp_x):", int.from_bytes(raw[180:184], "big", signed=True))
print("bytes 189-192 (inline):", int.from_bytes(raw[188:192], "big", signed=True))
print("bytes 193-196 (crossline):", int.from_bytes(raw[192:196], "big", signed=True))
assert int.from_bytes(raw[188:192], "big", signed=True) == vol.headers["inline"][0] == 1000

x, y = vol.coords["cdp_x"], vol.coords["cdp_y"]
inline_spacing = np.hypot(x[1, 0] - x[0, 0], y[1, 0] - y[0, 0])
crossline_spacing = np.hypot(x[0, 1] - x[0, 0], y[0, 1] - y[0, 0])
print(f"inline spacing {inline_spacing:.2f} m, crossline spacing {crossline_spacing:.2f} m")
assert np.allclose(x, cdp_x) and np.allclose(y, cdp_y)
assert np.allclose(vol.coords["cdp_x_raw"] / 100, vol.coords["cdp_x"])

order = rng.permutation(il.size)
traces = cube.reshape(-1, n_samples)[order]
unsorted_path = workdir / "unsorted.sgy"
segyrs.create(
    unsorted_path,
    traces,
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    inline=il.ravel()[order],
    crossline=xl.ravel()[order],
    headers={"delay": delay_ms},
)
messy = segyrs.open(unsorted_path)
print(messy)
print(messy.geometry)

trace = messy.sel(inline=1030, crossline=2080)
print(trace.dims, trace.shape, "file trace", trace.coords["trace"])
assert np.abs(trace.numpy()[0] - cube[30, 40]).max() <= ibm_tolerance
line = messy.sel(inline=1030)
print(line.dims, line.shape)
print(line.coords["crossline"][:5], line.coords["trace"][:5])

summary = messy.resort(workdir / "sorted.sgy", by=["inline", "crossline"])
print(summary)
tidy = segyrs.open(workdir / "sorted.sgy")
print(tidy.geometry)
assert np.abs(tidy.sel(inline=1030).numpy() - cube[30]).max() <= ibm_tolerance

keep = np.ones(il.size, bool)
keep[[100, 2500, 5000]] = False
sparse_path = workdir / "sparse.sgy"
segyrs.create(
    sparse_path,
    cube.reshape(-1, n_samples)[keep],
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    inline=il.ravel()[keep],
    crossline=xl.ravel()[keep],
)
holes = segyrs.open(sparse_path)
print(holes.geometry)
print(holes.geometry.missing_pairs)
dense, mask, coords = holes.to_regular(fill=0)
print(dense.shape, mask.shape, int((~mask).sum()), "empty slots")
assert not mask[1, 20] and np.all(dense[1, 20] == 0)
assert np.abs(dense[mask] - cube[mask]).max() <= ibm_tolerance


def show_error(call):
    try:
        call()
    except Exception as err:
        print(f"{type(err).__name__}: {err}")
    else:
        raise AssertionError("expected an error")


show_error(lambda: vol.sel(crossline=2081))
show_error(lambda: vol.sel(time=401_000))
show_error(lambda: vol.sel(iline=1030))
print(vol.sel(crossline=2081, method="nearest").coords["crossline"])

show_error(lambda: messy.isel(inline=0))

moved = workdir / "moved-keys.sgy"
segyrs.create(
    moved,
    traces,
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    headers={"field_record": il.ravel()[order], "cdp": xl.ravel()[order]},
)
print(segyrs.open(moved).geometry)
print(segyrs.open(moved, inline_byte=9, crossline_byte=21).geometry)

twice = workdir / "duplicate.sgy"
segyrs.create(
    twice,
    cube.reshape(-1, n_samples)[:3],
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    inline=[1000, 1000, 1000],
    crossline=[2000, 2002, 2000],
)
show_error(lambda: segyrs.open(twice))

padded = workdir / "padded.sgy"
shutil.copyfile(path, padded)
with open(padded, "ab") as f:
    f.write(bytes(100))
show_error(lambda: segyrs.open(padded))
os.truncate(padded, path.stat().st_size)
assert segyrs.open(padded).shape == vol.shape
