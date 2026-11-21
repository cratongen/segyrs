"""Pre-stack gathers and 2D lines: the script behind docs/examples/prestack-and-2d.md.

Run it with `mise run example`. It reads small files from test-data/, writes its own SEG-Y
files to a scratch directory that it deletes at the end, and writes its figures to the
directory given by --figures (default: a scratch directory). Every code block of the page
is a verbatim part of this file, and every result block of the page is a verbatim part of
what it prints (python/tests/test_examples.py).
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
test_data = Path(__file__).resolve().parents[1] / "test-data"

rng = np.random.default_rng(20261120)
n_samples, dt_us = 401, 4000
events = [(0.40, 2000.0, 1.0), (0.80, 2200.0, -0.7), (1.20, 2400.0, 0.5)]

tw = np.arange(-20, 21) * dt_us / 1e6
ricker = (1 - 2 * (np.pi * 25 * tw) ** 2) * np.exp(-((np.pi * 25 * tw) ** 2))


def synthetic(t0_shift, offset, delay_s=0.0):
    t0s, xs, delays = np.broadcast_arrays(t0_shift, offset, delay_s)
    reflectivity = np.zeros((*t0s.shape, n_samples))
    for t0, v_rms, rc in events:
        twt = np.sqrt((t0 + t0s) ** 2 + (xs / v_rms) ** 2)
        k = np.rint((twt - delays) * 1e6 / dt_us).astype(int)
        hit = np.nonzero((k >= 0) & (k < n_samples))
        reflectivity[(*hit, k[hit])] = rc
    traces = np.apply_along_axis(np.convolve, -1, reflectivity, ricker, mode="same")
    return (traces + 0.02 * rng.standard_normal(traces.shape)).astype(np.float32)


def pick(trace, twt):
    centre = round(twt * 1e6 / dt_us)
    lo, hi = max(centre - 5, 0), min(centre + 6, len(trace))
    return (lo + int(np.argmax(np.abs(trace[lo:hi])))) * dt_us


inlines = np.arange(100, 106)
crosslines = np.arange(200, 208)
offsets = np.arange(100, 2600, 200)
il, xl = np.meshgrid(inlines, crosslines, indexing="ij")
dip = 0.004 * (il - 100) + 0.002 * (xl - 200)
cube = synthetic(dip[..., None], offsets)
print(cube.shape, cube.dtype)

path = workdir / "prestack3d.sgy"
segyrs.create(
    path,
    cube,
    sample_format="ieee_float32",
    sample_interval_us=dt_us,
    inline=inlines,
    crossline=crosslines,
    offset=offsets,
    text=[
        "C 1 SYNTHETIC 3D PRE-STACK CDP GATHERS, SEED 20261120. NOT FIELD DATA.",
        "C 2 INLINE BYTE 189, CROSSLINE BYTE 193, OFFSET BYTE 37 (METRES), ALL INT32.",
        "C 3 TWO-WAY TIME, SAMPLE INTERVAL 4 MS, 401 SAMPLES, NO DELAY.",
    ],
)
print(f"{path.stat().st_size} bytes")

gathers = segyrs.open(path)
print(gathers)
print(gathers.geometry)

gather = gathers.sel(inline=102, crossline=204)
print(gather.dims, gather.shape)
print("offsets:", gather.coords["offset"])
data = gather.numpy()
assert np.array_equal(data, cube[2, 4])

for t0, v_rms, _ in events:
    expected = np.sqrt((t0 + dip[2, 4]) ** 2 + (offsets / v_rms) ** 2)
    picked = np.array([pick(tr, twt) for tr, twt in zip(data, expected, strict=True)])
    error_ms = np.abs(picked - expected * 1e6).max() / 1000
    print(f"t0 {t0 * 1000:.0f} ms, v_rms {v_rms:.0f} m/s: largest pick error {error_ms:.1f} ms")
    assert error_ms <= dt_us / 1000

pic = gather.plot(width=480, height=400)
pic.save(figures / "gather-il102-xl204.png")
print(pic.text)

near = gathers.sel(inline=102, offset=100)
print(near.dims, near.shape)
assert np.array_equal(near.numpy(), cube[2, :, 0])

cdps = np.arange(1001, 1041)
nominal = np.arange(150, 2550, 200)
fold = np.minimum(len(nominal), 2 + 2 * np.minimum(cdps - cdps[0], cdps[-1] - cdps))
print("fold per CDP:", fold[:8], "...", fold[-3:])

cdp = np.repeat(cdps, fold)
offset = np.concatenate([nominal[:n] for n in fold])
print(len(cdp), "traces")

cdp_x = 300_000 + (cdp - 1001) * 12.5
source_x = cdp_x - offset / 2
group_x = cdp_x + offset / 2
structure = 0.06 * np.sin((cdps - 1001) / 39 * np.pi)
traces = synthetic(structure[cdp - 1001], offset)

line_path = workdir / "line2d-gathers.sgy"
segyrs.create(
    line_path,
    traces,
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    headers={
        "cdp": cdp,
        "offset": offset,
        "cdp_x": cdp_x,
        "cdp_y": 7_000_000.0,
        "source_x": source_x,
        "group_x": group_x,
        "source_y": 7_000_000.0,
        "group_y": 7_000_000.0,
        "scalco": -100,
    },
    text=[
        "C 1 SYNTHETIC 2D PRE-STACK LINE, CDP SORTED, SEED 20261120. NOT FIELD DATA.",
        "C 2 CDP BYTE 21, OFFSET BYTE 37 (METRES), BOTH INT32.",
        "C 3 COORDINATES IN METRES, SCALCO -100. CDP INTERVAL 12.5 M.",
    ],
)

print(segyrs.open(line_path).geometry)

line = segyrs.open(line_path, inline_byte=21)
print(line)
print(line.geometry)
print("missing pairs:", line.geometry.missing_pairs.shape)

labels, counts = np.unique(line.coords["inline"], return_counts=True)
print("CDPs:", labels[0], "..", labels[-1], " fold:", counts[:8], "...", counts[-3:])
assert np.array_equal(labels, cdps) and np.array_equal(counts, fold)

edge = line.sel(inline=1003)
print(edge.dims, edge.shape)
print("offsets:", edge.coords["offset"], " file traces:", edge.coords["trace"])
ibm_tolerance = 2.0**-20 * np.abs(traces).max()
assert np.abs(edge.numpy() - traces[cdp == 1003]).max() <= ibm_tolerance

pic = line.sel(inline=1003).plot(width=480, height=400)
pic.save(figures / "gather-cdp1003.png")
print(pic.text)

sx, gx = line.coords["source_x"], line.coords["group_x"]
print("raw word, trace 0:", line.coords["source_x_raw"][0], " metres:", sx[0])
from_coordinates = np.abs(gx - sx)
print("largest offset disagreement:", np.abs(from_coordinates - line.coords["offset"]).max(), "m")
assert np.array_equal(from_coordinates, line.coords["offset"])
midpoint = (sx + gx) / 2
assert np.allclose(midpoint, line.coords["cdp_x"])
print("CDP interval:", np.unique(np.diff(np.unique(midpoint))), "m")

dense, mask, grid = line.to_regular(fill=0)
print(dense.shape, mask.shape, {k: len(v) for k, v in grid.items()})
print("live cells:", int(mask.sum()), " empty cells:", int((~mask).sum()))
assert np.array_equal(mask.sum(axis=(1, 2)), fold)
assert not mask[0, 0, -1] and np.all(dense[0, 0, -1] == 0)

summary = line.to_regular(workdir / "line2d-regular.sgy")
print(summary)
regular = segyrs.open(workdir / "line2d-regular.sgy", inline_byte=21, dead_trace="trace_id")
print(regular)
far = regular.sel(crossline=0, offset=2350)
print("live traces at 2350 m:", int(far.coords["trace_mask"].sum()), "of", far.shape[0])
pic = far.plot(width=640, height=400)
pic.save(figures / "line-offset-2350.png")
print(pic.text)

delay_ms = np.where(cdps <= 1020, 0, 100)
stack = synthetic(structure, 0.0, delay_s=delay_ms / 1000)
stack_path = workdir / "line2d-stack.sgy"
segyrs.create(
    stack_path,
    stack,
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    headers={
        "cdp": cdps,
        "cdp_x": 300_000 + (cdps - 1001) * 12.5,
        "scalco": -100,
        "delay": delay_ms,
    },
)

stacked = segyrs.open(stack_path, inline_byte=21)
print(stacked)
print("time axis:", stacked.coords["time"][:3], "...", stacked.coords["time"][-1])
delay = stacked.coords["delay"].ravel()
print("per-trace delay (us):", delay[18:23])

first = 0.40 + structure
samples = stacked.sel(crossline=0).numpy()
for k in (19, 20):
    peak = int(np.argmax(np.abs(samples[k])))
    label = stacked.coords["time"][peak]
    true = delay[k] + peak * dt_us
    print(
        f"CDP {cdps[k]}: strongest sample {peak}, time label {label / 1000:.0f} ms,"
        f" delay + {peak} x 4 ms = {true / 1000:.0f} ms"
    )
    assert abs(true - first[k] * 1e6) <= dt_us

pic = stacked.sel(crossline=0).plot(width=640, height=400)
pic.save(figures / "stack-as-stored.png")
print(pic.text)

shift = (delay - delay.min()) // dt_us
assert np.all((delay - delay.min()) % dt_us == 0)
aligned = np.zeros((len(cdps), n_samples + shift.max()), np.float32)
for k, s in enumerate(shift):
    aligned[k, s : s + n_samples] = samples[k]
aligned_path = workdir / "line2d-aligned.sgy"
segyrs.create(
    aligned_path,
    aligned,
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    headers={"cdp": cdps, "delay": int(delay.min() // 1000)},
)
flat = segyrs.open(aligned_path, inline_byte=21)
print(flat)
assert "delay" not in flat.coords
level = flat.sel(crossline=0).numpy()
assert all(abs(pick(level[k], first[k]) - first[k] * 1e6) <= dt_us for k in range(len(cdps)))
flat.sel(crossline=0).plot(width=640, height=400).save(figures / "stack-aligned.png")

scaled = segyrs.open(test_data / "segyrs" / "delay-scalar.sgy")
print("bytes 109-110:", scaled.headers["delay"], " bytes 215-216:", scaled.headers["time_scalar"])
print("time axis starts at", scaled.coords["time"][0], "us")
assert scaled.coords["time"][0] == 1_000_000

ibm = segyrs.open(test_data / "multiformats" / "Format1msb.sgy")
reference = ibm.numpy()
print(ibm)
print("amplitude range:", reference.min(), reference.max())
for name in [
    "Format1lsb",
    "Format5msb",
    "Format6msb",
    "Format2msb",
    "Format3msb",
    "Format7msb",
    "Format9msb",
]:
    other = segyrs.open(test_data / "multiformats" / f"{name}.sgy")
    values = other.numpy()
    same = np.array_equal(values.astype(np.float64), reference.astype(np.float64))
    attrs = other.attrs
    print(
        f"{name:11} code {attrs['binary_format_code']:>2} {attrs['sample_format']:13}"
        f" {attrs['source_endian']:6} -> {values.dtype!s:7} same values: {same}"
    )
    assert same

for name in ["Format8msb", "Format16msb"]:
    other = segyrs.open(test_data / "multiformats" / f"{name}.sgy")
    values = other.numpy()
    print(
        f"{name:11} {other.attrs['sample_format']:6} -> {values.dtype},"
        f" range {values.min()} .. {values.max()}"
    )

with open(test_data / "multiformats" / "Format1msb.sgy", "rb") as f:
    head = f.read(3600)
    n = int.from_bytes(head[3220:3222], "big")
    k = int(np.flatnonzero(reference.reshape(-1, n)[0])[0])
    f.seek(3600 + 240 + 4 * k)
    word = int.from_bytes(f.read(4), "big")
sign = -1.0 if word >> 31 else 1.0
exponent = (word >> 24) & 0x7F
fraction = (word & 0xFFFFFF) / 2**24
value = sign * fraction * 16.0 ** (exponent - 64)
print(
    f"trace 0 sample {k}: word 0x{word:08X}, sign {sign:+.0f}, exponent {exponent} - 64,"
    f" fraction 0x{word & 0xFFFFFF:06X} / 2^24 = {value}"
)
assert value == reference.reshape(-1, n)[0, k]

as_ieee = workdir / "f3-ieee.sgy"
ibm.to_segy(as_ieee, sample_format="ieee_float32")
converted = segyrs.open(as_ieee)
print(converted.attrs["sample_format"], converted.attrs["binary_format_code"])
assert np.array_equal(converted.numpy(), reference)


def show_error(call):
    try:
        call()
    except Exception as err:
        print(f"{type(err).__name__}: {err}")
    else:
        raise AssertionError("expected an error")


print(segyrs.validate(path))
show_error(lambda: segyrs.validate(test_data / "f3.sgy"))
print(segyrs.open(test_data / "f3.sgy").shape)

show_error(lambda: segyrs.open(test_data / "small.su"))

cut = workdir / "cut.sgy"
shutil.copyfile(path, cut)
os.truncate(cut, 3000)
show_error(lambda: segyrs.open(cut))

show_error(lambda: segyrs.open(test_data / "broken.sgy"))

show_error(lambda: segyrs.open(test_data / "decrement.sgy"))

odd = workdir / "odd-code.sgy"
shutil.copyfile(path, odd)
with open(odd, "r+b") as f:
    f.seek(3224)
    f.write((13).to_bytes(2, "big"))
show_error(lambda: segyrs.open(odd))
print(segyrs.open(odd, sample_format="ieee_float32").attrs["binary_format_code"])

signed = workdir / "wrong-signature.sgy"
shutil.copyfile(path, signed)
with open(signed, "r+b") as f:
    f.seek(3296)
    f.write(bytes([4, 3, 2, 1]))
show_error(lambda: segyrs.open(signed))
show_error(lambda: segyrs.open(signed, endian="big"))
with open(signed, "r+b") as f:
    f.seek(3296)
    f.write(bytes(4))
assert segyrs.open(signed).shape == gathers.shape

show_error(lambda: segyrs.open(test_data / "multiformats" / "Format4msb.sgy"))

show_error(lambda: segyrs.open(path, offset_byte=41))

show_error(lambda: gathers.sel(offset=200))
print(gathers.sel(offset=200, method="nearest").coords["offset"])

big = workdir / "assumed.sgy"
segyrs.create(
    big,
    np.zeros((65, 64, 4), np.int16),
    sample_format="int16",
    sample_interval_us=dt_us,
    inline=np.arange(1, 66),
    crossline=np.arange(1, 65),
)
with segyrs.edit(big) as editor:
    editor.set_header("crossline", [999], traces=[2000])
assumed = segyrs.open(big)
print(assumed.geometry)
show_error(lambda: assumed.sel(inline=32).numpy())
print(segyrs.open(big, scan="full").geometry)

show_error(lambda: gathers.to_segy(workdir / "int16.sgy", sample_format="int16"))
