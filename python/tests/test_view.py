"""M4 from Python: the notebook picture (Volume.plot, _repr_mimebundle_) and segyrs.view."""

from __future__ import annotations

import os
import struct
import subprocess
import sys

import numpy as np
import pytest
from conftest import DATA, grid

import segyrs

MAGIC = b"\x89PNG\r\n\x1a\n"
# Plot area 800 x 480 plus the margins (64 + 16, 28 + 40).
NOTEBOOK_SIZE = (880, 548)


def png_size(png: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR, after checking the signature."""
    assert png[:8] == MAGIC
    assert png[12:16] == b"IHDR"
    return struct.unpack(">II", png[16:24])


def test_s06_quick_start_inline_reads_only_that_inline():
    """The documented quick start: one lazy inline through the plot helper (S-06, #19)."""
    vol = segyrs.open(DATA / "small.sgy")
    pic = vol.sel(inline=3).plot()
    assert png_size(pic.png) == NOTEBOOK_SIZE
    assert pic.text.startswith("segyrs section: inline 3 | x crossline (5 positions)")
    # Only that inline's 5 traces were read, and nothing went through numpy.
    assert pic.traces_read == vol.sizes["crossline"]
    assert vol.traces_read == 0


def test_repr_of_a_3d_volume_is_its_first_inline():
    vol = segyrs.open(DATA / "small.sgy")
    bundle = vol._repr_mimebundle_()
    assert png_size(bundle["image/png"]) == NOTEBOOK_SIZE
    assert len(bundle["image/png"]) > 1000
    assert "segyrs section: inline 1 |" in bundle["text/plain"]
    assert "p98 of drawn view" in bundle["text/plain"]


def test_repr_of_a_prestack_volume_is_its_first_gather_plane():
    vol = segyrs.open(DATA / "small-ps.sgy")
    assert vol.ndim == 4
    bundle = vol._repr_mimebundle_()
    assert png_size(bundle["image/png"]) == NOTEBOOK_SIZE
    assert "segyrs section: inline 1, offset 1 |" in bundle["text/plain"]


@pytest.mark.parametrize(
    ("select", "want"),
    [
        ({"inline": 2}, "inline 3 | x crossline (5 positions), y time (ms) (50 positions)"),
        ({"crossline": 1}, "crossline 21 | x inline (5 positions)"),
        ({"time": 3}, "time (ms) 12 | x inline (5 positions), y crossline (5 positions)"),
        (
            {"inline": 2, "crossline": slice(1, 4), "time": slice(5, 20)},
            "inline 3 | x crossline (3 positions), y time (ms) (15 positions)",
        ),
    ],
)
def test_repr_of_a_2d_selection_draws_that_plane(select, want):
    sub = segyrs.open(DATA / "small.sgy").isel(select)
    assert sub.ndim == 2
    bundle = sub._repr_mimebundle_()
    assert png_size(bundle["image/png"]) == NOTEBOOK_SIZE
    assert want in bundle["text/plain"]


def test_plot_a_grid_whose_offset_word_varies_per_trace(tmp_path):
    path = tmp_path / "shots.sgy"
    offsets = np.tile(np.arange(-100, 101, 50), 3)
    segyrs.create(
        path,
        np.ones((15, 20), np.float32),
        sample_format="ieee_float32",
        sample_interval_us=1000,
        headers={"field_record": np.repeat([1, 2, 3], 5), "trace_in_record": np.tile(range(5), 3)},
        offset=offsets,
    )
    vol = segyrs.open(path, inline_byte=9, crossline_byte=13)
    assert vol.dims == ("inline", "crossline", "time")
    pic = vol.sel(inline=2).plot(width=64, height=48)
    assert pic.text.startswith("segyrs section: inline 2 | x crossline (5 positions)")


def test_plot_follows_sel_by_label_and_the_clip():
    vol = segyrs.open(DATA / "small.sgy")
    pic = vol.sel(crossline=22, time=slice(8000, 40000)).plot(vmin=-2, vmax=1.5)
    assert "crossline 22 | x inline (5 positions), y time (ms) (9 positions)" in pic.text
    assert "clip ±2 (set)" in pic.text


def test_repr_without_a_plane_says_why():
    vol = segyrs.open(DATA / "small.sgy")
    one_trace = vol.isel(inline=1, crossline=1)._repr_mimebundle_()
    assert "image/png" not in one_trace
    assert "no picture: a picture needs two free dims" in one_trace["text/plain"]
    scattered = vol.isel(inline=2, crossline=[0, 1, 3])._repr_mimebundle_()
    assert (
        "no picture: the crossline selection is not an ascending run of positions"
        " with a constant step" in scattered["text/plain"]
    )
    backwards = vol.isel(inline=2, crossline=slice(None, None, -1))._repr_mimebundle_()
    assert "image/png" not in backwards


def dense_line(segy, n=400, ns=60):
    """One inline of ``n`` crosslines: too dense for wiggles on a 200 px plot."""
    samples, il, xl = grid(1, n, ns)
    rng = np.random.default_rng(83)
    samples = rng.standard_normal(samples.shape).astype(np.float32)
    path = segy("dense.sgy", samples, inline=il, crossline=xl)
    return segyrs.open(path)


def test_83_wiggle_fallback_is_explicit(segy):
    vol = dense_line(segy)
    pic = vol.plot(wiggle="line", width=200)
    assert pic.note == (
        "wiggle needs >= 3 px per trace (400 traces on 200 px); showing density; try wiggle_every=6"
    )
    assert pic.text.endswith("| " + pic.note)
    # The image shows density, the same as no wiggle at all apart from the caption.
    assert pic.traces_read == 400


def test_83_wiggle_every_draws_wiggles(segy):
    vol = dense_line(segy)
    off = vol.plot(width=200, density=False)
    pic = vol.plot(wiggle_every=8, width=200, density=False)
    assert pic.note == "wiggle_every=8: one trace in 8"
    assert pic.traces_read == 50
    assert "x crossline (50 positions)" in pic.text
    assert png_size(pic.png) == png_size(off.png)
    assert pic.png != off.png
    area = vol.plot(wiggle="area", wiggle_every=8, width=200, density=False)
    assert area.png != pic.png
    with pytest.raises(segyrs.SegyrsError, match="slice"):
        segyrs.open(DATA / "small.sgy").isel(time=3).plot(wiggle_every=2)


def test_plot_rejects_bad_options():
    sub = segyrs.open(DATA / "small.sgy").isel(inline=0)
    with pytest.raises(ValueError, match="colormap"):
        sub.plot(colormap="jet")
    with pytest.raises(ValueError, match="wiggle_every"):
        sub.plot(wiggle_every=0)


def headless_view(tmp_path, call: str) -> str:
    """Run ``segyrs.view(<call>)`` in a child process that cannot reach a display.

    Every display variable is removed and ``XDG_RUNTIME_DIR`` points at an empty
    directory, so no X11 or Wayland socket is found and no window can open on the
    desktop running the tests. Prints ``RAISED <type> <message>`` or ``OPENED``.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "WINIT_UNIX_BACKEND")
    }
    runtime = tmp_path / "no-display"
    runtime.mkdir(mode=0o700)
    env["XDG_RUNTIME_DIR"] = str(runtime)
    code = (
        "import segyrs\n"
        f"vol = segyrs.open({str(DATA / 'small.sgy')!r})\n"
        "try:\n"
        f"    segyrs.view({call})\n"
        "    print('OPENED')\n"
        "except Exception as e:\n"
        "    print('RAISED', type(e).__name__, e)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "OPENED" not in out.stdout, "a window opened"
    return out.stdout


@pytest.mark.usefixtures("view_extra")
def test_view_without_a_display_raises_cleanly(tmp_path):
    """No display server: SegyrsError naming the notebook image, no window, no crash."""
    out = headless_view(tmp_path, "vol.isel(inline=1)")
    assert "RAISED SegyrsError" in out
    assert "cannot open a window" in out
    assert "notebook image" in out


@pytest.mark.usefixtures("view_extra")
def test_view_checks_the_plane_before_opening(tmp_path):
    out = headless_view(tmp_path, "vol.isel(inline=1, crossline=1)")
    assert "RAISED SegyrsError" in out
    assert "sel" in out


@pytest.mark.usefixtures("view_extra")
def test_view_starts_on_a_stepped_window(tmp_path):
    """The selection's window and step pass the checks and reach the window
    stage; the state itself is checked in segyrs-view's smoke tests."""
    out = headless_view(
        tmp_path, "vol.isel(inline=2, crossline=slice(1, 5, 2), time=slice(5, 40)), wiggle_every=2"
    )
    assert "cannot open a window" in out


def test_view_refuses_a_scattered_selection(tmp_path):
    out = headless_view(tmp_path, "vol.isel(inline=2, crossline=[0, 1, 3])")
    assert "RAISED _NotDrawable" in out
    assert "constant step" in out


def test_stepped_selection_draws_decimated():
    vol = segyrs.open(DATA / "small.sgy")
    sub = vol.isel(inline=2, crossline=slice(0, 5, 2), time=slice(3, 40, 5))
    pic = sub.plot()
    assert "x crossline (3 positions), y time (ms) (8 positions)" in pic.text
    assert pic.traces_read == 3
    full = vol.isel(inline=2, crossline=slice(0, 5), time=slice(3, 40)).plot()
    assert pic.png != full.png
    bundle = sub._repr_mimebundle_()
    assert png_size(bundle["image/png"]) == NOTEBOOK_SIZE
    # A list with a constant step is the same selection as the slice.
    same = vol.isel(inline=2, crossline=[0, 2, 4], time=slice(3, 40, 5)).plot()
    assert same.png == pic.png
    # A stepped time slice: inline and crossline both decimated.
    sl = vol.isel(time=7, inline=slice(1, 5, 2), crossline=slice(0, 5, 3)).plot()
    assert "x inline (2 positions), y crossline (2 positions)" in sl.text
    assert sl.traces_read == 4


def test_drawer_shares_the_volume_file():
    """The drawer wraps the Volume's own File (one header scan), not a path."""
    vol = segyrs.open(DATA / "small.sgy")
    vol.sel(inline=3).plot()
    src = vol._src
    assert isinstance(src.drawer, segyrs._segyrs.Drawer)
    with pytest.raises(TypeError):
        segyrs._segyrs.Drawer(str(DATA / "small.sgy"))


def test_remote_volume_draws():
    """A Volume opened through the object store (file://) draws the same picture."""
    local = segyrs.open(DATA / "small.sgy").sel(inline=3).plot()
    remote = segyrs.open((DATA / "small.sgy").resolve().as_uri()).sel(inline=3).plot()
    assert remote.png == local.png
    assert remote.traces_read == local.traces_read


@pytest.mark.usefixtures("view_extra")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="display variables are Linux")
def test_headless_view_says_why_without_build_paths_and_repeats_the_cause():
    code = (
        "import segyrs\n"
        "for _ in range(2):\n"
        "    try:\n"
        f"        segyrs.view({str(DATA / 'small.sgy')!r})\n"
        "    except segyrs.SegyrsError as e:\n"
        "        print(e)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    first, second = out.stdout.splitlines()
    for line in (first, second):
        assert ".rs:" not in line and ".cargo" not in line, line
        assert "DISPLAY" in line, line
    assert "restart Python" in second
