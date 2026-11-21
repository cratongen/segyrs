"""Writing and editing SEG-Y: the script behind docs/examples/write-and-edit.md.

Run it with `mise run example`. It makes every array it writes from a fixed seed, writes its
SEG-Y files to a scratch directory that it deletes at the end, and writes its figures to the
directory given by --figures (default: a scratch directory). Every code block of the page
is a verbatim part of this file, and every result block of the page is a verbatim part of
what it prints (python/tests/test_examples.py).
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
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
inlines = np.arange(500, 532)
crosslines = np.arange(1000, 1048)
n_samples, dt_us, delay_ms = 301, 2000, 200
t = delay_ms / 1000 + np.arange(n_samples) * dt_us / 1e6

il, xl = np.meshgrid(inlines, crosslines, indexing="ij")
horizons = [
    (0.36 + 0.0020 * (il - 500) + 0.03 * np.sin((xl - 1000) / 47 * np.pi), 1.0),
    (0.62 - 0.0010 * (il - 500), -0.6),
]
reflectivity = np.zeros((*il.shape, n_samples))
for twt, rc in horizons:
    k = np.rint((twt - t[0]) * 1e6 / dt_us).astype(int)
    np.put_along_axis(reflectivity, k[..., None], rc, axis=-1)
tw = np.arange(-25, 26) * dt_us / 1e6
ricker = (1 - 2 * (np.pi * 30 * tw) ** 2) * np.exp(-((np.pi * 30 * tw) ** 2))
cube = np.apply_along_axis(np.convolve, -1, reflectivity, ricker, mode="same")
cube = (cube + 0.03 * rng.standard_normal(cube.shape)).astype(np.float32)
print(cube.shape, cube.dtype)

azimuth = np.deg2rad(20)
along_il = 12.5 * (xl - 1000)
along_xl = 25.0 * (il - 500)
cdp_x = np.round(650_000 + along_il * np.sin(azimuth) + along_xl * np.cos(azimuth), 2)
cdp_y = np.round(7_800_000 + along_il * np.cos(azimuth) - along_xl * np.sin(azimuth), 2)
print(f"inline 500, crossline 1000: x {cdp_x[0, 0]:.2f} m, y {cdp_y[0, 0]:.2f} m")
print(f"inline 531, crossline 1047: x {cdp_x[-1, -1]:.2f} m, y {cdp_y[-1, -1]:.2f} m")

path = workdir / "cube-ibm.sgy"
written = segyrs.create(
    path,
    cube,
    sample_format="ibm_float32",
    sample_interval_us=dt_us,
    inline=inlines,
    crossline=crosslines,
    headers={"cdp_x": cdp_x, "cdp_y": cdp_y, "scalco": -100, "delay": delay_ms},
    text=[
        "C 1 SYNTHETIC 3D POST-STACK VOLUME, SEED 20261120. NOT FIELD DATA.",
        "C 2 INLINE BYTES 189-192, CROSSLINE BYTES 193-196, BOTH INT32.",
        "C 3 CDP X BYTES 181-184, CDP Y BYTES 185-188, METRES, SCALCO -100.",
        "C 4 TWO-WAY TIME, 2 MS, 301 SAMPLES, FIRST SAMPLE AT 200 MS (BYTES 109-110).",
    ],
)
n_traces = len(inlines) * len(crosslines)
stride = 240 + 4 * n_samples
print(f"{written} bytes = 3600 + {n_traces} x (240 + {n_samples} x 4)")
assert written == path.stat().st_size == 3600 + n_traces * stride

raw = path.read_bytes()


def word(buf, byte, size):
    return int.from_bytes(buf[byte - 1 : byte - 1 + size], "big", signed=True)


print("bytes 3217-3218, sample interval (us):", word(raw, 3217, 2))
print("bytes 3221-3222, samples per trace:   ", word(raw, 3221, 2))
print("bytes 3225-3226, sample format code:  ", word(raw, 3225, 2))
print("bytes 3501-3502, revision:            ", raw[3500], raw[3501])
assert (word(raw, 3217, 2), word(raw, 3221, 2), word(raw, 3225, 2)) == (dt_us, n_samples, 1)

k = (510 - 500) * len(crosslines) + (1020 - 1000)
header = raw[3600 + k * stride : 3600 + k * stride + 240]
inline, crossline = word(header, 189, 4), word(header, 193, 4)
scalco, x_word, y_word = word(header, 71, 2), word(header, 181, 4), word(header, 185, 4)
print(f"trace {k}: inline {inline}, crossline {crossline}, scalco {scalco}")
print(f"  cdp_x word {x_word} -> {x_word / -scalco} m, cdp_y word {y_word} -> {y_word / -scalco} m")
delay, ns, interval = word(header, 109, 2), word(header, 115, 2), word(header, 117, 2)
print(f"  delay {delay} ms, samples {ns}, interval {interval} us")
assert (inline, crossline) == (510, 1020)
assert (x_word / -scalco, y_word / -scalco) == (cdp_x[10, 20], cdp_y[10, 20])

print(segyrs.validate(path))
vol = segyrs.open(path)
print(vol)
print(vol.attrs["textual_encoding"], vol.attrs["textual_header"][3].rstrip())

back = vol.numpy()
relative = np.abs(back - cube) / np.abs(cube)
print(f"largest relative change from float32 to IBM: {relative.max():.3g} (2^-21 = {2.0**-21:.3g})")
assert np.all(np.abs(back - cube) <= 2.0**-21 * np.abs(cube))
assert np.array_equal(vol.coords["cdp_x"], cdp_x) and np.array_equal(vol.coords["cdp_y"], cdp_y)
assert vol.coords["time"][0] == delay_ms * 1000

pic = vol.sel(inline=510).plot(width=640, height=400)
pic.save(figures / "write-inline-510.png")
print(pic.text)

ieee_path = workdir / "cube-ieee.sgy"
vol.to_segy(ieee_path, sample_format="ieee_float32")
ieee = segyrs.open(ieee_path)
print(ieee.attrs["sample_format"], ieee.attrs["binary_format_code"], ieee_path.stat().st_size)
assert np.array_equal(ieee.numpy(), back)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ibm_again = workdir / "cube-ibm-again.sgy"
ieee.to_segy(ibm_again, sample_format="ibm_float32")
print("IBM -> IEEE -> IBM identical:", sha256(ibm_again) == sha256(path))
assert sha256(ibm_again) == sha256(path)

scale = 32767 / float(np.abs(back).max())
counts = np.round(back.astype(np.float64) * scale).astype(np.int16)
int_path = workdir / "cube-int16.sgy"
segyrs.create(
    int_path,
    counts,
    sample_format="int16",
    sample_interval_us=dt_us,
    inline=inlines,
    crossline=crosslines,
    headers={"cdp_x": cdp_x, "cdp_y": cdp_y, "scalco": -100, "delay": delay_ms},
)
restored = segyrs.open(int_path).numpy() / scale
error = np.abs(restored - back)
rms_amplitude = np.sqrt((back.astype(np.float64) ** 2).mean())
print(f"int16: {scale:.1f} counts per unit amplitude, {int_path.stat().st_size} bytes")
print(f"  largest error {error.max():.3g} = {error.max() * scale:.3f} counts")
print(f"  largest error / rms amplitude = {error.max() / rms_amplitude:.3g}")
assert error.max() <= 0.5 / scale * (1 + 1e-9)

lsb_path = workdir / "cube-lsb.sgy"
vol.to_segy(lsb_path, endian="little")
lsb = segyrs.open(lsb_path)
print(lsb.attrs["source_endian"], lsb.attrs["revision"], lsb.binary_header["byte_order_signature"])
assert np.array_equal(lsb.numpy(), back)
big_again = workdir / "cube-big-again.sgy"
lsb.to_segy(big_again, endian="big")
assert sha256(big_again) == sha256(path)


def loudest(volume):
    rms = volume.rms()
    i, j = np.unravel_index(rms.argmax(), rms.shape)
    print(f"trace RMS: median {np.median(rms):.3f}, largest {rms.max():.3f}", end=" ")
    print(f"at inline {inlines[i]}, crossline {crosslines[j]}")
    return inlines[i], crosslines[j]


loudest(vol)
bad = (512 - 500) * len(crosslines) + (1030 - 1000)
spikes = np.zeros(n_samples, np.float32)
spikes[::7] = rng.choice([-25.0, 25.0], size=len(spikes[::7]))
with segyrs.edit(path) as editor:
    editor.write_traces(back.reshape(n_traces, n_samples)[bad] + spikes, first=bad)
assert loudest(segyrs.open(path)) == (512, 1030)

before = path.read_bytes()

opened = segyrs.open(path)
cards = bytearray(opened.textual_header_bytes[0])
card = "C 6 TRACE AT INLINE 512 CROSSLINE 1030 KILLED (SPIKES), TRACE ID 2."
cards[400:480] = card.ljust(80).encode("cp037")
binary = bytearray(opened.binary_header_bytes)
binary[54:56] = (1).to_bytes(2, "big")
trace_id = np.ones(n_traces, np.int64)
trace_id[bad] = 2

with segyrs.edit(path) as editor:
    editor.write_textual_block(0, bytes(cards))
    editor.write_binary_header(bytes(binary))
    editor.set_headers({"cdp": np.arange(1, n_traces + 1), "trace_id": trace_id})
    editor.write_traces(np.zeros(n_samples, np.float32), first=bad)

after = path.read_bytes()
assert len(after) == len(before)
changed = np.flatnonzero(np.frombuffer(before, np.uint8) != np.frombuffer(after, np.uint8))
in_text = changed[changed < 3200]
in_binary = changed[(changed >= 3200) & (changed < 3600)]
offset_in_trace = (changed[changed >= 3600] - 3600) % stride
trace_of = (changed[changed >= 3600] - 3600) // stride
header_bytes = np.unique(offset_in_trace[offset_in_trace < 240] + 1)
sample_traces = np.unique(trace_of[offset_in_trace >= 240])
print("changed bytes:", len(changed))
print("  textual cards:", np.unique(in_text // 80 + 1))
print("  binary-header bytes:", in_binary + 1)
print("  trace-header bytes:", header_bytes)
print("  traces with changed samples:", sample_traces)
assert set(np.unique(in_text // 80 + 1)) == {6}
assert set(in_binary + 1) <= {3255, 3256}
assert set(header_bytes) <= {21, 22, 23, 24, 29, 30}
assert list(sample_traces) == [bad]

edited = segyrs.open(path, dead_trace="trace_id")
print(segyrs.validate(path)["n_traces"], "traces,", edited.attrs["measurement_system"])
print(edited.attrs["textual_header"][5].rstrip())
dead = np.flatnonzero(~edited.coords["trace_mask"].ravel())
print("dead traces:", dead, " cdp:", edited.headers["cdp"][dead])
assert list(dead) == [bad]
untouched = np.delete(edited.numpy().reshape(n_traces, n_samples), bad, axis=0)
assert np.array_equal(untouched, np.delete(back.reshape(n_traces, n_samples), bad, axis=0))
assert np.all(edited.numpy()[12, 30] == 0)

pic = edited.sel(inline=512).plot(width=640, height=400)
pic.save(figures / "write-inline-512-killed.png")
print(pic.text)

copy_path = workdir / "copy.sgy"
edited.to_segy(copy_path)
print("unchanged copy identical:", sha256(copy_path) == sha256(path))
assert sha256(copy_path) == sha256(path)

crop_path = workdir / "crop.sgy"
edited.sel(inline=slice(510, 519), time=slice(300_000, 500_000)).to_segy(crop_path)
crop = segyrs.open(crop_path)
print(crop)
expected = back[10:20, :, 50:151].copy()
expected[2, 30] = 0
assert np.array_equal(crop.numpy(), expected)
assert np.array_equal(crop.coords["cdp_x"], cdp_x[10:20])
assert np.array_equal(crop.headers["cdp"], edited.headers["cdp"][10 * 48 : 20 * 48])
assert crop.coords["time"][0] == 300_000

n_shots, n_channels, shot_n, shot_dt_us = 5, 48, 601, 1000
v, depth = 2000.0, 500.0
shot_x = 1000.0 + 200.0 * np.arange(n_shots)
offset = np.tile(np.r_[-25 * np.arange(24, 0, -1), 25 * np.arange(1, 25)], (n_shots, 1))
receiver_x = shot_x[:, None] + offset
direct = np.abs(offset) / v
reflected = np.sqrt((2 * depth / v) ** 2 + (offset / v) ** 2)
print("signed offsets of shot 0:", offset[0, :3], "...", offset[0, -3:])

gathers = np.zeros((n_shots, n_channels, shot_n))
for arrival, distance, amplitude in [
    (direct, np.abs(offset), 1.0),
    (reflected, v * reflected, 0.6),
]:
    k = np.rint(arrival * 1e6 / shot_dt_us).astype(int)
    np.put_along_axis(gathers, k[..., None], (amplitude * 100 / distance)[..., None], axis=-1)
stw = np.arange(-40, 41) * shot_dt_us / 1e6
wavelet = (1 - 2 * (np.pi * 30 * stw) ** 2) * np.exp(-((np.pi * 30 * stw) ** 2))
gathers = np.apply_along_axis(np.convolve, -1, gathers, wavelet, mode="same").astype(np.float32)

record = np.repeat(np.arange(1, n_shots + 1), n_channels)
channel = np.tile(np.arange(1, n_channels + 1), n_shots)
sequence = np.arange(1, n_shots * n_channels + 1)
shot_path = workdir / "shots.sgy"
segyrs.create(
    shot_path,
    gathers.reshape(-1, shot_n),
    sample_format="ieee_float32",
    sample_interval_us=shot_dt_us,
    headers={
        "trace_seq_line": sequence,
        "trace_seq_file": sequence,
        "field_record": record,
        "trace_in_record": channel,
        "source_point": record,
        "trace_id": 11,
        "offset": offset.ravel(),
        "group_elev": 0.0,
        "source_elev": 0.0,
        "scalel": -100,
        "scalco": -100,
        "source_x": np.repeat(shot_x, n_channels),
        "source_y": 0.0,
        "group_x": receiver_x.ravel(),
        "group_y": 0.0,
        "coord_units": 1,
        "delay": 0,
        "cdp_x": ((np.repeat(shot_x, n_channels) + receiver_x.ravel()) / 2),
        "cdp_y": 0.0,
        "inline": record,
        "crossline": channel,
    },
    text=[
        "C 1 SYNTHETIC 2D SHOT GATHERS, ANALYTIC ARRIVALS, NO NOISE. NOT FIELD DATA.",
        "C 2 FIELD RECORD BYTES 9-12, CHANNEL BYTES 13-16, SIGNED OFFSET BYTES 37-40.",
        "C 3 ONE LAYER, V 2000 M/S, FLAT REFLECTOR AT 500 M. TRACE ID 11 (PRESSURE).",
        "C 4 COORDINATES IN METRES, SCALCO AND SCALEL -100. 1 MS, 601 SAMPLES.",
    ],
)
binary = bytearray(segyrs.open(shot_path).binary_header_bytes)
binary[54:56] = (1).to_bytes(2, "big")
with segyrs.edit(shot_path) as editor:
    editor.write_binary_header(bytes(binary))

shots = segyrs.open(shot_path, inline_byte=9, crossline_byte=13)
print(shots)
print(shots.geometry)
assert np.array_equal(shots.numpy(), gathers)
sx, gx = shots.coords["source_x"], shots.coords["group_x"]
assert np.array_equal(gx - sx, shots.coords["offset"])

gather = shots.sel(inline=3).numpy()
for name, arrival in [("direct", direct[2]), ("reflection", reflected[2])]:
    picked = []
    for trace, when in zip(gather, arrival, strict=True):
        centre = round(when * 1e6 / shot_dt_us)
        window = trace[centre - 10 : centre + 11]
        picked.append((centre - 10 + int(np.argmax(np.abs(window)))) * shot_dt_us / 1000)
    worst = np.abs(np.array(picked) - arrival * 1000).max()
    print(f"shot 3 {name}: largest pick error {worst:.2f} ms")
    assert worst <= shot_dt_us / 1000

pic = shots.sel(inline=3).plot(width=640, height=400)
pic.save(figures / "shot-3.png")
print(pic.text)


def show_error(call):
    try:
        call()
    except Exception as err:
        print(f"{type(err).__name__}: {err}")
    else:
        raise AssertionError("expected an error")


def attempt_create(**changes):
    options = dict(
        sample_format="ibm_float32", sample_interval_us=dt_us, inline=inlines, crossline=crosslines
    )
    options.update(changes)
    data = options.pop("data", cube)
    segyrs.create(workdir / "refused.sgy", data, **options)


show_error(lambda: attempt_create(sample_interval_us=1000 / 3))
show_error(lambda: attempt_create(headers={"cdp_x": cdp_x + 0.001, "scalco": -100}))
show_error(lambda: attempt_create(headers={"cdp_x": cdp_x, "scalco": -10000}))
with_nan = cube.copy()
with_nan[3, 4, 100] = np.nan
show_error(lambda: attempt_create(data=with_nan))
show_error(lambda: vol.to_segy(workdir / "refused.sgy", sample_format="int16"))
assert not (workdir / "refused.sgy").exists()

snapshot = path.read_bytes()
with segyrs.edit(path) as editor:
    show_error(lambda: editor.write_traces(np.zeros(n_samples - 1, np.float32), first=0))
    two_traces = np.zeros((2, n_samples), np.float32)
    show_error(lambda: editor.write_traces(two_traces, first=n_traces - 1))
    show_error(lambda: editor.set_header("delay", 40_000))
    changed_format = bytearray(snapshot[3200:3600])
    changed_format[24:26] = (5).to_bytes(2, "big")
    show_error(lambda: editor.write_binary_header(bytes(changed_format)))
assert path.read_bytes() == snapshot

padded = workdir / "padded.sgy"
shutil.copyfile(path, padded)
with open(padded, "ab") as f:
    f.write(bytes(100))
show_error(lambda: segyrs.validate(padded))
