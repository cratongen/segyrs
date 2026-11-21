"""Pictures of a Volume: the notebook image (``Volume.plot``, ``_repr_mimebundle_``)
and the native window (``segyrs.view``). Both draw in Rust; no matplotlib."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _segyrs
from ._segyrs import SegyrsError

if TYPE_CHECKING:
    from ._volume import Volume


def _viewer() -> ModuleType:
    """The ``segyrs-view`` extension, or an ImportError that says how to build it."""
    try:
        return importlib.import_module("_segyrs_view")
    except ImportError as e:
        raise ImportError(
            "segyrs.view needs the experimental viewer window (the segyrs-view extension), "
            "which is not installed. It is not part of the segyrs 0.1.0 release on PyPI and "
            "will be published later. Build it from a clone of "
            "https://github.com/cratongen/segyrs with `mise run build`, which installs segyrs "
            "and the viewer into the clone's .venv. Volume.plot() draws without it.",
            name="_segyrs_view",
        ) from e


# Plot-area cap of the notebook image (segyrs_draw::NOTEBOOK_PIXELS).
NOTEBOOK_PIXELS = (800, 480)


class Picture:
    """One drawn plane: PNG bytes and a one-line description. Experimental.

    :meth:`Volume.plot` makes one. It displays inline in Jupyter.

    Attributes
    ----------
    png : bytes
        The PNG file.
    text : str
        What was drawn, the clip and where it came from.
    note : str or None
        Set when a wiggle was asked for: why it fell back to density, or
        how it was decimated.
    traces_read : int
        Traces whose samples were read to draw it.

    Examples
    --------
    >>> pic = segyrs.open("survey.sgy").sel(inline=3).plot()
    >>> pic.text.split(" | ")[0]
    'segyrs section: inline 3'
    """

    __slots__ = ("note", "png", "text", "traces_read")

    def __init__(self, png: bytes, text: str, note: str | None, traces_read: int):
        """Not public: :meth:`Volume.plot` makes a Picture.

        Parameters
        ----------
        png : bytes
            The PNG file.
        text : str
            The one-line description.
        note : str or None
            The wiggle note.
        traces_read : int
            Traces read to draw it.
        """
        self.png = png
        self.text = text
        self.note = note
        self.traces_read = traces_read

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the PNG to ``path``.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file. An existing file is replaced.

        Returns
        -------
        None

        Raises
        ------
        OSError
            When the file cannot be written.

        Examples
        --------
        >>> segyrs.open("survey.sgy").sel(inline=3).plot().save("inline-3.png")
        """
        with open(path, "wb") as f:
            f.write(self.png)

    def _repr_png_(self) -> bytes:
        return self.png

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict[str, Any]:
        return {"image/png": self.png, "text/plain": self.text}

    def __repr__(self) -> str:
        return self.text


class _NotDrawable(ValueError):
    pass


def _drawer(vol: Volume) -> _segyrs.Drawer:
    """The Volume's open file, wrapped for drawing once. It shares the File's
    header scan and trace index, so nothing is scanned twice, and a remote
    (object-store) Volume draws through the same reader."""
    src = vol._src
    drawer = getattr(src, "drawer", None)
    if drawer is None:
        drawer = _segyrs.Drawer(src.file, depth=src.vertical == "depth")
        src.drawer = drawer
    return drawer


def _run(p: np.ndarray, d: str, what: str) -> tuple[tuple[str, int, int, int], str | None]:
    """``(d, start, stop, step)`` for ascending positions with a constant step,
    or the reason they are not (a scattered list, a descending slice)."""
    start = int(p[0])
    step = int(p[1] - p[0]) if len(p) > 1 else 1
    stop = start + step * (len(p) - 1) + 1
    if step < 1 or not np.array_equal(p, np.arange(start, stop, step)):
        return (d, start, start + 1, 1), (
            f"the {d} selection is not an ascending run of {what} with a constant step"
        )
    return (d, start, stop, step), None


def _plane(
    vol: Volume,
) -> tuple[list[tuple[str, int]], list[tuple[str, int, int, int]], str | None]:
    """The selection as positions in the opened file.

    Returns the fixed ``(dim, position)`` pairs, the ``(dim, start, stop, step)``
    window of every free dim, and why it cannot be drawn (or ``None``). A free
    dim must be ascending positions with a constant step (a slice such as
    ``slice(0, 1000, 10)``); the picture reads every ``step``-th position.
    """
    src = vol._src
    g = src.geometry
    axes = tuple(g.axes) if g.is_grid else ("trace",)
    counts = (
        [
            len({"inline": g.inlines, "crossline": g.crosslines, "offset": g.offsets}[a])
            for a in axes
        ]
        if g.is_grid
        else [src.n]
    )
    dims = tuple(d for d in ("inline", "crossline", "offset", "trace") if d in axes)
    traces = vol._traces
    if traces.size == 0:
        return [], [], "the selection is empty"
    pos = np.unravel_index(traces, counts)
    fixed: list[tuple[str, int]] = []
    ranges: list[tuple[str, int, int, int]] = []
    problem = None
    for d in dims:
        p = pos[axes.index(d)]
        if d not in vol._tdims:
            fixed.append((d, int(p.flat[0])))
            continue
        k = vol._tdims.index(d)
        line = p[(0,) * k + (slice(None),) + (0,) * (p.ndim - k - 1)]
        window, why = _run(line, d, "positions")
        problem = problem or why
        ranges.append(window)
    v = vol._vsel
    vertical = src.vertical
    if isinstance(v, int):
        fixed.append((vertical, v))
    else:
        run = np.asarray(v, dtype=np.int64)
        if not len(run):
            return fixed, ranges, f"the {vertical} selection is empty"
        window, why = _run(run, vertical, "samples")
        problem = problem or why
        ranges.append(window)
    if len(ranges) < 2:
        problem = f"a picture needs two free dims; this selection has dims {vol.dims}"
    return fixed, ranges, problem


def _clip(vmin: float | None, vmax: float | None) -> float | None:
    vals = [abs(float(v)) for v in (vmin, vmax) if v is not None]
    return max(vals) if vals else None


def _live(vol: Volume) -> np.ndarray | None:
    src = vol._src
    return None if src.dead_trace is None else np.ascontiguousarray(src.trace_mask())


def plot(
    vol: Volume,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: str = "grey",
    polarity: str = "normal",
    gain: float = 1.0,
    wiggle: str | None = None,
    wiggle_every: int = 1,
    density: bool = True,
    width: int = NOTEBOOK_PIXELS[0],
    height: int = NOTEBOOK_PIXELS[1],
) -> Picture:
    """Draw this Volume's plane to a PNG. Reads only that plane's traces.

    A 2D selection draws that section or slice. A stepped selection
    (``isel(crossline=slice(0, 1000, 10))``) reads and draws every step-th
    position only; a scattered list of positions has no picture. A 3D (or 4D) Volume draws its
    first plane: the first position of the slowest grid axis (and of offset),
    leaving crossline and time free (roadmap decision 5). With no ``vmin`` or
    ``vmax`` the clip is the symmetric p98 of the drawn view; otherwise it is
    ``max(|vmin|, |vmax|)``.

    ``colormap`` is ``"grey"``, ``"seismic"`` or ``"amber"``; ``polarity`` is
    ``"normal"`` or ``"reversed"``; ``wiggle`` is ``"line"`` or ``"area"``.
    Wiggles need at least 3 px per trace; below that the picture shows density
    and ``Picture.note`` says so. ``wiggle_every=N`` draws one trace in N (and
    reads only those), so a dense section still shows wiggles; it implies
    ``wiggle="line"``. ``width`` and ``height`` size the plot area in pixels.
    """
    fixed, ranges, problem = _plane(vol)
    if problem:
        raise _NotDrawable(problem)
    if wiggle is None:
        wiggle = "line" if wiggle_every > 1 else "off"
    png, text, note, read = _drawer(vol).picture(
        fixed,
        ranges,
        clip=_clip(vmin, vmax),
        live=_live(vol),
        width=width,
        height=height,
        colormap=colormap,
        polarity=polarity,
        gain=gain,
        wiggle=wiggle,
        wiggle_every=wiggle_every,
        density=density,
    )
    return Picture(png, text, note, read)


def mimebundle(vol: Volume) -> dict[str, Any]:
    """``image/png`` and ``text/plain`` for Jupyter. A stepped selection draws
    decimated; one that has no picture (one dim, a scattered list of
    positions) shows its text and the reason."""
    try:
        pic = plot(vol)
    except (_NotDrawable, SegyrsError) as e:
        return {"text/plain": f"{vol!r}\n  no picture: {e}"}
    return {"image/png": pic.png, "text/plain": f"{vol!r}\n  {pic.text}"}


def view(
    target: Volume | str | os.PathLike[str],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: str = "grey",
    polarity: str = "normal",
    gain: float = 1.0,
    wiggle: str | None = None,
    wiggle_every: int = 1,
    renderer: str = "auto",
    title: str | None = None,
) -> str:
    """Open the native viewer window and block until it closes. Experimental.

    The window is the ``segyrs-view`` extension, which is not part of the
    0.1.0 release on PyPI: build it from the repository with ``mise run build``.

    The window starts on the Volume's selection: its selected-away dims
    stay fixed (on a 3D Volume, the first plane of the slowest axis), it
    opens zoomed to the selected window of the two free dims, and a stepped
    selection is read one position in step. Zoom out, pan and step from
    there; changing the plane drops the steps. The GIL is released while it
    is open, and Ctrl-C closes it and raises ``KeyboardInterrupt``. It must
    run on the main thread.

    Parameters
    ----------
    target : Volume or str or os.PathLike
        A Volume, or the path of a file to open.
    vmin, vmax : float, optional
        Clip, as in :meth:`Volume.plot`.
    colormap : {"grey", "seismic", "amber"}, default "grey"
        Colour map of the density display.
    polarity : {"normal", "reversed"}, default "normal"
        ``"reversed"`` flips the sign of the samples.
    gain : float, default 1.0
        Multiplier applied before the clip.
    wiggle : {None, "off", "line", "area"}, optional
        Wiggle traces, as in :meth:`Volume.plot`.
    wiggle_every : int, default 1
        Draw one trace in N of a section. ``N > 1`` implies
        ``wiggle="line"``.
    renderer : {"auto", "gpu", "cpu"}, default "auto"
        ``"auto"`` uses wgpu when an adapter exists, else a CPU window.
    title : str, optional
        Window title. ``None`` shows ``segyrs``.

    Returns
    -------
    str
        The presenter that ran, ``"gpu"`` or ``"cpu"``.

    Raises
    ------
    ImportError
        The ``segyrs-view`` extension is not installed.
    SegyrsError
        On a machine with no display, saying why. Use
        :meth:`Volume.plot` or the notebook image there.
    ValueError
        A selection that cannot be drawn, or an unknown option value.

    Examples
    --------
    >>> segyrs.view("survey.sgy", colormap="seismic")  # doctest: +SKIP
    'gpu'
    """
    from ._volume import Volume, open

    vol = target if isinstance(target, Volume) else open(target)
    fixed, ranges, problem = _plane(vol)
    if problem and "two free dims" not in problem:
        raise _NotDrawable(problem)
    if wiggle is None:
        wiggle = "line" if wiggle_every > 1 else "off"
    ext = _viewer()
    src = vol._src
    live = _live(vol)
    args = dict(src.open_args[1])
    args["gain_convention"] = args.pop("gain")
    return ext.view(
        src.file.reader(),
        os.path.basename(src.file.path),
        fixed,
        ranges,
        **args,
        depth=src.vertical == "depth",
        clip=_clip(vmin, vmax),
        live=None if live is None else live.tolist(),
        colormap=colormap,
        polarity=polarity,
        gain=gain,
        wiggle=wiggle,
        wiggle_every=wiggle_every,
        renderer=renderer,
        title=title,
    )
