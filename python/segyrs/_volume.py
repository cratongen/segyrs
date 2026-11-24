"""The labelled lazy ``Volume`` and ``segyrs.open``."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _fast, _interop, _segyrs
from ._protocol import produce
from ._segyrs import GeometryError

if TYPE_CHECKING:
    from ._view import Picture

_GRID_KINDS = ("inline_major", "crossline_major", "offset_major")
_LINE_DIMS = ("inline", "crossline", "offset")
_HEADER_DTYPES = ("int8", "int16", "int32", "int64", "uint16", "uint32", "float64")

# Header words that carry a coordinate. The scaled float64 value takes the name,
# and the integer header word stays beside it as ``<name>_raw`` (X-05).
_SCALCO = ("cdp_x", "cdp_y", "source_x", "source_y", "group_x", "group_y")
_SCALEL = ("group_elev", "source_elev", "source_depth", "group_datum", "source_datum")

# Rev 2 binary header: (name, offset in the 400-byte block, numpy kind).
_BINARY_FIELDS = (
    ("job_id", 0, "i4"),
    ("line_number", 4, "i4"),
    ("reel_number", 8, "i4"),
    ("traces_per_ensemble", 12, "i2"),
    ("aux_traces_per_ensemble", 14, "i2"),
    ("sample_interval", 16, "u2"),
    ("sample_interval_orig", 18, "u2"),
    ("n_samples", 20, "u2"),
    ("n_samples_orig", 22, "u2"),
    ("format_code", 24, "u2"),
    ("ensemble_fold", 26, "i2"),
    ("sorting_code", 28, "i2"),
    ("vertical_sum", 30, "i2"),
    ("sweep_freq_start", 32, "i2"),
    ("sweep_freq_end", 34, "i2"),
    ("sweep_length", 36, "i2"),
    ("sweep_type", 38, "i2"),
    ("sweep_channel", 40, "i2"),
    ("sweep_taper_start", 42, "i2"),
    ("sweep_taper_end", 44, "i2"),
    ("taper_type", 46, "i2"),
    ("correlated", 48, "i2"),
    ("gain_recovered", 50, "i2"),
    ("amplitude_recovery", 52, "i2"),
    ("measurement_system", 54, "i2"),
    ("impulse_polarity", 56, "i2"),
    ("vibratory_polarity", 58, "i2"),
    ("ext_traces_per_ensemble", 60, "i4"),
    ("ext_aux_traces_per_ensemble", 64, "i4"),
    ("ext_n_samples", 68, "i4"),
    ("ext_sample_interval", 72, "f8"),
    ("ext_sample_interval_orig", 80, "f8"),
    ("ext_n_samples_orig", 88, "i4"),
    ("ext_ensemble_fold", 92, "i4"),
    ("fixed_length", 302, "i2"),
    ("n_extended_textual", 304, "i2"),
    ("max_extra_trace_headers", 306, "i4"),
    ("time_basis", 310, "i2"),
    ("n_traces", 312, "u8"),
    ("first_trace_offset", 320, "u8"),
    ("n_trailer_stanzas", 328, "i4"),
)


def _pair_swap(raw: bytes) -> bytes:
    a = np.frombuffer(raw, dtype=np.uint8)
    return a.reshape(-1, 2)[:, ::-1].tobytes()


def _word(raw: bytes, off: int, kind: str, endian: str) -> Any:
    """One header word in the file's byte order, as a Python number."""
    width = int(kind[1:])
    chunk = raw[off : off + width]
    if endian == "pairwise":
        # Pairwise swap restores big-endian order. A 2-byte word is then little-endian.
        chunk = _pair_swap(chunk)
        order = ">"
    else:
        order = ">" if endian == "big" else "<"
    return np.frombuffer(chunk, dtype=order + kind)[0].item()


def _scaled(raw: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """SEG-Y scalar rule: positive multiplies, negative divides, zero means 1 (S-27, X-05)."""
    out = raw.astype(np.float64)
    s = scalar.astype(np.float64)
    pos, neg = s > 0, s < 0
    out[pos] *= s[pos]
    out[neg] /= -s[neg]
    return out


def _exact(values: np.ndarray) -> np.ndarray:
    """Integer microseconds stay int64. A fractional value keeps the float64."""
    rounded = np.rint(values)
    if values.size and np.all(rounded == values) and np.all(np.abs(values) < 2**53):
        return rounded.astype(np.int64)
    return values


@dataclass(frozen=True, repr=False, eq=False)
class Geometry:
    """What open learned about the trace layout. Sorting is a report, not a failure.

    A file whose headers fit no grid still opens: its ``kind`` is
    ``"unstructured"`` and the Volume has dims ``(trace, time)``. Read it
    from :attr:`Volume.geometry`; segyrs makes it, you do not.

    Attributes
    ----------
    kind : str
        ``"inline_major"``, ``"crossline_major"``, ``"offset_major"`` or
        ``"unstructured"``.
    inline_byte, crossline_byte, offset_byte : int
        1-based header bytes the keys were read from.
    inlines, crosslines, offsets : numpy.ndarray
        Sorted distinct key values. Empty on an unstructured file.
    axes : tuple of str
        Grid axes slowest first, the order of the traces in the file.
    missing_pairs : numpy.ndarray
        ``(inline, crossline)`` pairs of the grid with no trace.
    source : str
        ``"scan"``: every trace header was read at open. ``"assumed"``: a
        regular grid predicted from sampled headers; reads check the traces
        they touch. ``"index_cache"``: loaded from a valid index sidecar.

    Examples
    --------
    >>> vol = segyrs.open("survey.sgy")
    >>> vol.geometry.kind, vol.geometry.axes
    ('inline_major', ('inline', 'crossline'))
    >>> vol.geometry.inlines
    array([1, 2, 3, 4, 5], dtype=int32)
    """

    kind: str
    inline_byte: int
    crossline_byte: int
    offset_byte: int
    inlines: np.ndarray
    crosslines: np.ndarray
    offsets: np.ndarray
    axes: tuple[str, ...]
    missing_pairs: np.ndarray
    #: ``"scan"``: every trace header was read at open. ``"assumed"``: a regular
    #: grid predicted from sampled headers; reads check the traces they touch.
    #: ``"index_cache"``: loaded from a valid index sidecar.
    source: str = "scan"

    @property
    def is_grid(self) -> bool:
        """True when the traces fill a regular inline and crossline grid.

        Examples
        --------
        >>> segyrs.open("survey.sgy").geometry.is_grid
        True
        """
        return self.kind in _GRID_KINDS

    def __repr__(self) -> str:
        parts = [
            f"kind={self.kind!r}",
            f"axes={self.axes!r}",
            f"inlines={len(self.inlines)}",
            f"crosslines={len(self.crosslines)}",
            f"offsets={len(self.offsets)}",
        ]
        if len(self.missing_pairs):
            parts.append(f"missing_pairs={len(self.missing_pairs)}")
        if self.source != "scan":
            parts.append(f"source={self.source!r}")
        parts.append(
            f"bytes=(inline={self.inline_byte}, crossline={self.crossline_byte},"
            f" offset={self.offset_byte})"
        )
        return f"Geometry({', '.join(parts)})"


Geometry.__init__.__doc__ = "Not public: segyrs builds a Geometry at open."


class _Source:
    """File-level state shared by a Volume and every selection made from it."""

    def __init__(self, file: _segyrs.File, fields, dead_trace, vertical: str):
        self.file = file
        self.n = file.n_traces
        self.ns = file.n_samples
        self.fields = fields
        self.vertical = vertical
        g = file.geometry()
        self.geometry = Geometry(
            kind=g["kind"],
            inline_byte=g["inline_byte"],
            crossline_byte=g["crossline_byte"],
            offset_byte=g["offset_byte"],
            inlines=g["inlines"],
            crosslines=g["crosslines"],
            offsets=g["offsets"],
            axes=tuple(g["axes"]),
            missing_pairs=g["missing_pairs"],
            source=g["source"],
        )
        # An assumed grid defers the header scan to first use (and checks it then).
        self._scanned = None if g["source"] == "assumed" else file.scanned_columns()
        self.dead_trace = dead_trace
        self.header_cache = _fast.HeaderCache(
            file,
            fields,
            self._scanned or {},
            {
                "inline": g["inline_byte"],
                "crossline": g["crossline_byte"],
                "offset": g["offset_byte"],
            },
        )
        self._trace_coords: dict[str, np.ndarray] | None = None
        self._axis: np.ndarray | None = None
        self._ext_offset: np.ndarray | bool | None = False

        blocks = file.textual_header()
        self.textual_bytes = [raw for _, _, raw in blocks]
        measurement = {1: "meters", 2: "feet"}.get(file.measurement_system)
        self.attrs = {
            "sample_format": file.sample_format,
            "binary_format_code": file.binary_format_code,
            "source_endian": file.source_endian,
            "revision": file.revision,
            "textual_header": [card for _, cards, _ in blocks for card in cards],
            "textual_encoding": [enc for enc, _, _ in blocks],
            "measurement_system": measurement,
            "sample_interval_us": file.sample_interval_us,
            "n_samples": self.ns,
            "n_traces": self.n,
            "chunk_shape": (1, self.ns),
            "lazy": True,
        }
        if file.gain is not None:
            self.attrs["gain"] = file.gain

    @property
    def scanned(self) -> dict[str, np.ndarray]:
        if self._scanned is None:
            self._scanned = self.file.scanned_columns()
        return self._scanned

    def headers(self) -> dict[str, np.ndarray]:
        """Every header column, decoding each on first access (read-only arrays)."""
        cache = self.header_cache
        return _fast.LazyHeaders(cache.names(), cache.get)

    def axis(self) -> np.ndarray:
        """``delay + i * interval`` in microseconds, from trace 0 and the binary header."""
        if self._axis is None:
            delay = 0.0
            if self.n:
                raw = self.file.header_bytes(0)
                endian = self.file.source_endian
                lag = _word(raw, 108, "i2", endian)
                scalar = _word(raw, 214, "i2", endian)
                delay = float(_scaled(np.array([lag]), np.array([scalar]))[0]) * 1000.0
            dt = self.file.sample_interval_us
            self._axis = _exact(delay + np.arange(self.ns, dtype=np.float64) * dt)
        return self._axis

    def trace_coords(self) -> dict[str, np.ndarray]:
        """Per-trace coordinates, file order, computed once from one header scan."""
        if self._trace_coords is not None:
            return self._trace_coords
        stamp_words = ("year", "day", "hour", "minute", "second")
        h = self.header_cache.get(
            [*_SCALCO, *_SCALEL, "scalco", "scalel", "delay", "time_scalar", *stamp_words]
        )
        s = self.scanned
        c: dict[str, np.ndarray] = {
            "inline": s["inline"],
            "crossline": s["crossline"],
            "offset": s["offset"],
        }
        ext = self.extended_offset()
        if ext is not None:
            c["offset"] = ext
            c["offset_raw"] = s["offset"]
        for name in _SCALCO:
            c[name] = _scaled(h[name], h["scalco"])
            c[name + "_raw"] = h[name]
        for name in _SCALEL:
            c[name] = _scaled(h[name], h["scalel"])
            c[name + "_raw"] = h[name]
        c["trace_mask"] = self.trace_mask()
        delay = _scaled(h["delay"], h["time_scalar"]) * 1000.0
        if delay.size and np.any(delay != delay[0]):
            c["delay"] = _exact(delay)
        stamp = self._timestamps(h)
        if stamp is not None:
            c["timestamp"] = stamp
        for v in c.values():
            v.setflags(write=False)
        self._trace_coords = c
        return c

    def trace_mask(self) -> np.ndarray:
        dead = self.dead_trace
        if dead is None:
            return np.ones(self.n, dtype=bool)
        if isinstance(dead, str):
            if dead != "trace_id":
                raise ValueError('dead_trace must be None, "trace_id", or a bool array')
            # Trace identification code 2 is a dead trace in every revision.
            return self.scanned["trace_id"] != 2
        mask = np.asarray(dead)
        if mask.dtype != bool or mask.shape != (self.n,):
            raise ValueError(f"dead_trace array must be bool with shape ({self.n},)")
        return ~mask

    def extended_offset(self) -> np.ndarray | None:
        """Rev 2 trace header extension 1, bytes 129-136: float64 offset (S-19).

        Read once, on first use. ``None`` when the file has no extension 1 block
        or every value is zero, so the int32 at bytes 37-40 stays the offset.
        """
        if self._ext_offset is False:
            col = None
            if self.file.revision not in ("0", "1") and self.file.extra_trace_header_blocks:
                col = self.file.header_columns([("offset_ext", 369, "float64")])["offset_ext"]
                if not np.any(col != 0) or not np.all(np.isfinite(col)):
                    col = None
            self._ext_offset = col
        return self._ext_offset

    def _timestamps(self, h: Mapping[str, np.ndarray]) -> np.ndarray | None:
        """A datetime64 coordinate when the time-of-recording words are set (S-28)."""
        parts = [h[k].astype(np.int64) for k in ("year", "day", "hour", "minute", "second")]
        if not any(np.any(p) for p in parts):
            return None
        year, day, hour, minute, second = parts
        nanos = np.zeros(self.n, dtype=np.int64)
        if self.file.revision not in ("0", "1") and self.file.extra_trace_header_blocks >= 1:
            # Extension 1, bytes 141-144: nanoseconds to add to the second.
            ext = self.file.header_columns([("ns_add", 240 + 141, "int32")])["ns_add"]
            nanos = ext.astype(np.int64)
        stamp = (year - 1970).astype("datetime64[Y]").astype("datetime64[ns]")
        stamp = stamp + (np.maximum(day, 1) - 1).astype("timedelta64[D]")
        stamp = stamp + hour.astype("timedelta64[h]") + minute.astype("timedelta64[m]")
        stamp = stamp + second.astype("timedelta64[s]") + nanos.astype("timedelta64[ns]")
        unset = (year == 0) & (day == 0) & (hour == 0) & (minute == 0) & (second == 0)
        stamp[unset] = np.datetime64("NaT", "ns")
        return stamp


def _check_dim_type(dim: str, geometry: Geometry) -> None:
    if dim in ("inline", "crossline") and geometry.kind == "unstructured":
        raise GeometryError(
            f"no {dim} grid was inferred (geometry kind {geometry.kind!r});"
            f" select by position on 'trace' instead"
        )


def _label_positions(labels: np.ndarray, key: Any, method: str | None, dim: str) -> Any:
    """Labels to positions. Returns an int, a slice, or an int array."""
    if isinstance(key, slice):
        lo, hi, step = key.start, key.stop, key.step
        if step is not None and not isinstance(step, (int, np.integer)):
            raise TypeError(f"{dim}: a label slice step is an int count of positions")
        mask = np.ones(len(labels), dtype=bool)
        decreasing = len(labels) > 1 and bool(np.all(np.diff(labels) < 0))
        if decreasing:
            lo, hi = hi, lo
        if lo is not None:
            mask &= labels >= lo
        if hi is not None:
            mask &= labels <= hi
        pos = np.flatnonzero(mask)
        if step is not None:
            pos = pos[::step]
        if len(pos) and (step is None or step > 0):
            stride = step or 1
            if len(pos) == 1 or np.all(np.diff(pos) == stride):
                return slice(int(pos[0]), int(pos[-1]) + 1, stride)
        return pos
    if np.ndim(key) == 0:
        if method == "nearest":
            if not len(labels):
                raise KeyError(f"{dim} has no labels")
            return int(np.argmin(np.abs(labels - key)))
        if method is not None:
            raise ValueError(f'method must be None or "nearest", not {method!r}')
        hit = np.flatnonzero(labels == key)
        if not len(hit):
            raise _fast.label_error(dim, key, labels)
        return int(hit[0])
    return _fast.label_positions_many(labels, key, method, dim, _label_positions)


def _match(values: np.ndarray, key: Any, method: str | None, dim: str) -> np.ndarray:
    """Boolean mask of a coordinate along ``trace`` against a label key."""
    if isinstance(key, slice):
        if key.step is not None:
            raise ValueError(f"{dim}: a step is not defined on a coordinate along 'trace'")
        mask = np.ones(len(values), dtype=bool)
        if key.start is not None:
            mask &= values >= key.start
        if key.stop is not None:
            mask &= values <= key.stop
        return mask
    if np.ndim(key) == 0:
        if method == "nearest":
            if not len(values):
                return np.zeros(0, dtype=bool)
            return values == values[np.argmin(np.abs(values - key))]
        if method is not None:
            raise ValueError(f'method must be None or "nearest", not {method!r}')
        return values == key
    return np.isin(values, np.asarray(key))


class Volume:
    """A labelled, lazy view of a SEG-Y file.

    Make one with :func:`segyrs.open`. Samples are lazy: open reads the
    textual, binary and trace headers it needs for geometry and no sample
    bytes. ``sel`` and ``isel`` return a new Volume without reading samples.
    :meth:`numpy` reads only the traces in the selection, and only the
    sample window it covers, from the memory map, decoding on all cores with
    the GIL released. Chunks are whole traces in on-disk order:
    ``attrs["chunk_shape"]`` is ``(1, n_samples)``.

    Dims are ``(inline, crossline[, offset], time)`` on a complete grid with
    constant fold, otherwise ``(trace, time)`` with ``inline``,
    ``crossline`` and ``offset`` as coordinates along ``trace``.
    ``segyrs.open(path, depth=True)`` names the vertical dim ``depth``.

    Examples
    --------
    >>> vol = segyrs.open("survey.sgy")
    >>> vol.dims
    ('inline', 'crossline', 'time')
    >>> vol.sel(inline=3).numpy().shape
    (5, 50)
    """

    __slots__ = ("_attrs", "_labels", "_root", "_scalars", "_src", "_tdims", "_traces", "_vsel")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Not public: a Volume comes from :func:`segyrs.open`.

        Raises
        ------
        TypeError
            Always.
        """
        raise TypeError("use segyrs.open(path) to make a Volume")

    @classmethod
    def _make(cls, src, tdims, traces, labels, vsel, scalars, attrs, root) -> Volume:
        self = object.__new__(cls)
        self._src = src
        self._tdims = tdims
        self._traces = traces
        self._labels = labels
        self._vsel = vsel
        self._scalars = scalars
        self._attrs = attrs
        self._root = root
        return self

    # ---- shape --------------------------------------------------------------

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names, slowest first, the vertical dim last.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.dims
        ('inline', 'crossline', 'time')
        >>> vol.sel(inline=3).dims
        ('crossline', 'time')
        """
        if isinstance(self._vsel, int):
            return self._tdims
        return (*self._tdims, self._src.vertical)

    @property
    def shape(self) -> tuple[int, ...]:
        """Size of each dim, in the order of :attr:`dims`.

        Examples
        --------
        >>> segyrs.open("survey.sgy").shape
        (5, 5, 50)
        """
        if isinstance(self._vsel, int):
            return self._traces.shape
        return (*self._traces.shape, len(self._vsel))

    @property
    def sizes(self) -> dict[str, int]:
        """Size of each dim by name.

        Examples
        --------
        >>> segyrs.open("survey.sgy").sizes
        {'inline': 5, 'crossline': 5, 'time': 50}
        """
        return dict(zip(self.dims, self.shape, strict=True))

    @property
    def ndim(self) -> int:
        """Number of dims.

        Examples
        --------
        >>> segyrs.open("survey.sgy").ndim
        3
        """
        return len(self.dims)

    @property
    def size(self) -> int:
        """Number of samples in the selection.

        Examples
        --------
        >>> segyrs.open("survey.sgy").size
        1250
        """
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype :meth:`numpy` returns, set by the sample format.

        IBM float is ``float32``; integer formats stay integers.

        Examples
        --------
        >>> segyrs.open("survey.sgy").dtype
        dtype('float32')
        """
        return np.dtype(self._src.file.dtype)

    @property
    def nbytes(self) -> int:
        """Bytes :meth:`numpy` would return for this selection.

        Nothing is read to answer it, so check it before reading a large selection.

        Examples
        --------
        >>> segyrs.open("survey.sgy").nbytes
        5000
        """
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of a 0-d Volume")
        return self.shape[0]

    # ---- metadata -----------------------------------------------------------

    @property
    def attrs(self) -> dict[str, Any]:
        """File-level facts. They survive ``sel`` and ``isel``.

        Keys include ``sample_format``, ``binary_format_code``,
        ``sample_interval_us``, ``n_samples``, ``n_traces``, ``revision``,
        ``source_endian``, ``textual_header`` (decoded cards) and
        ``chunk_shape``.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.attrs["sample_format"], vol.attrs["sample_interval_us"]
        ('ibm_float32', 4000)
        """
        return self._attrs

    @property
    def geometry(self) -> Geometry:
        """The :class:`Geometry` open found: sorting, line keys, missing traces.

        Examples
        --------
        >>> segyrs.open("survey.sgy").geometry.kind
        'inline_major'
        """
        return self._src.geometry

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """Dim labels, the vertical axis, and per-trace coordinates shaped like the traces.

        The vertical axis is in microseconds (or depth units with
        ``depth=True``). Coordinates scaled by ``scalco`` or ``scalel`` are
        float64, with the raw header word beside them as ``<name>_raw``.
        ``trace_mask`` is true for live traces and ``trace`` holds the file
        trace numbers.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.coords["crossline"]
        array([20, 21, 22, 23, 24], dtype=int32)
        >>> vol.coords["time"][:3]
        array([   0, 4000, 8000])
        >>> vol.sel(inline=2).coords["trace"]
        array([5, 6, 7, 8, 9])
        """
        out: dict[str, np.ndarray] = {}
        for dim in self._tdims:
            out[dim] = self._labels[dim]
        axis = self._src.axis()
        out[self._src.vertical] = axis[self._vsel_index()]
        for name, value in self._scalars.items():
            out[name] = np.asarray(value)
        skip = set(self._tdims) | set(self._scalars)
        for name, full in self._src.trace_coords().items():
            if name in skip:
                continue
            out[name] = full[self._traces]
        if "trace" not in skip:
            out["trace"] = self._traces.copy()
        return out

    @property
    def headers(self) -> dict[str, np.ndarray]:
        """Trace headers as a dict of numpy arrays, one entry per trace.

        Lazy per column: every field name is a key at once, and
        ``headers[name]`` decodes only that column, in Rust on all cores, the
        first time any Volume of this file asks for it. ``items()`` and
        ``values()`` decode every missing column in one parallel scan. On the
        opened Volume the arrays are in file order, read-only and shared. On a
        selection they hold the selected traces in the selection's order.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.headers["crossline"][:6]
        array([20, 21, 22, 23, 24, 20], dtype=int32)
        >>> vol.sel(inline=2).headers["inline"]
        array([2, 2, 2, 2, 2], dtype=int32)
        """
        cache = self._src.header_cache
        if self._root:
            return _fast.LazyHeaders(cache.names(), cache.get)
        flat = self._traces.reshape(-1)
        return _fast.LazyHeaders(
            cache.names(), lambda names: {n: c[flat] for n, c in cache.get(names).items()}
        )

    def header_table(self) -> np.ndarray:
        """The header columns packed into one numpy structured array.

        The table is built for this call; use :attr:`headers` to read one
        column without decoding the others.

        Returns
        -------
        numpy.ndarray
            One record per trace, one field per header column.

        Raises
        ------
        GeometryAssumptionError
            On a file opened with ``scan="auto"``, when a header contradicts the
            assumed grid.

        Examples
        --------
        >>> table = segyrs.open("survey.sgy").header_table()
        >>> len(table), table["inline"][-1]
        (25, np.int32(5))
        """
        cols = self.headers.copy()
        table = np.empty(
            len(next(iter(cols.values()))) if cols else 0,
            dtype=[(k, v.dtype) for k, v in cols.items()],
        )
        for k, v in cols.items():
            table[k] = v
        return table

    def header_bytes(self, trace: int) -> bytes:
        """On-disk bytes of the standard trace header of one file trace.

        Parameters
        ----------
        trace : int
            File trace number, from 0, whatever the selection.

        Returns
        -------
        bytes
            The 240 header bytes, unchanged.

        Raises
        ------
        SegyrsError
            When ``trace`` is not a trace of the file.

        Examples
        --------
        >>> raw = segyrs.open("survey.sgy").header_bytes(0)
        >>> len(raw), int.from_bytes(raw[188:192], "big")
        (240, 1)
        """
        return self._src.file.header_bytes(trace)

    @property
    def binary_header_bytes(self) -> bytes:
        """The 400 bytes at file offset 3200, unchanged.

        Examples
        --------
        >>> len(segyrs.open("survey.sgy").binary_header_bytes)
        400
        """
        return self._src.file.binary_header_bytes()

    @property
    def binary_header(self) -> dict[str, int | float]:
        """The rev 2 binary header fields, decoded from the file's byte order.

        Examples
        --------
        >>> bh = segyrs.open("survey.sgy").binary_header
        >>> bh["sample_interval"], bh["n_samples"], bh["format_code"]
        (4000, 50, 1)
        """
        raw = self.binary_header_bytes
        endian = self._src.file.source_endian
        out = {name: _word(raw, off, kind, endian) for name, off, kind in _BINARY_FIELDS}
        out["byte_order_signature"] = int.from_bytes(raw[96:100], "big")
        out["revision_major"] = raw[300]
        out["revision_minor"] = raw[301]
        return out

    @property
    def textual_header_bytes(self) -> list[bytes]:
        """The raw 3200-byte textual blocks, primary first, beside the decoded cards.

        ``attrs["textual_header"]`` holds the decoded cards.

        Examples
        --------
        >>> blocks = segyrs.open("survey.sgy").textual_header_bytes
        >>> len(blocks), len(blocks[0])
        (1, 3200)
        """
        return list(self._src.textual_bytes)

    # ---- selection ----------------------------------------------------------

    def isel(self, indexers: Mapping[str, Any] | None = None, **kwargs: Any) -> Volume:
        """Select by position.

        An int drops the dim. A slice, an int list or a boolean mask keeps it.
        Negative positions count from the end. No sample is read.

        Parameters
        ----------
        indexers : mapping of str to int, slice or array, optional
            Positions by dim name. Use it for dim names that are not Python
            identifiers.
        **kwargs
            The same, as keywords: ``isel(inline=0, time=slice(0, 10))``.

        Returns
        -------
        Volume
            The selection.

        Raises
        ------
        IndexError
            A position outside the dim.
        TypeError
            A position that is not an int, a slice or an int array.
        ValueError
            A name that is not a dim of this Volume (the message suggests the
            nearest one), or a vertical dim selected twice to one sample.
        GeometryError
            ``inline``, ``crossline``, ``offset`` or ``trace`` on a Volume that
            does not have that dim.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.isel(inline=0, crossline=slice(1, 3)).shape
        (2, 50)
        >>> vol.isel(inline=-1).coords["inline"]
        array(5, dtype=int32)
        """
        keys = dict(indexers or {}, **kwargs)
        traces = self._traces
        tdims = list(self._tdims)
        labels = dict(self._labels)
        scalars = dict(self._scalars)
        vsel = self._vsel
        root = self._root
        for dim, key in keys.items():
            if dim in tdims:
                ax = tdims.index(dim)
                n = traces.shape[ax]
                key = _positions(key, n, dim)
                index = (slice(None),) * ax + (key,)
                # Basic indexing keeps an int or a slice a view. np.take would copy
                # the whole (transposed, non-contiguous) trace grid on every call.
                traces = traces[index]
                if isinstance(key, int):
                    scalars[dim] = labels.pop(dim)[key]
                    tdims.pop(ax)
                else:
                    labels[dim] = labels[dim][key]
                root = False
            elif dim == self._src.vertical:
                if isinstance(vsel, int):
                    raise ValueError(f"{dim} was already selected to one sample")
                vsel = _compose(vsel, _positions(key, len(vsel), dim))
            elif dim in _LINE_DIMS or dim == "trace":
                _check_dim_type(dim, self.geometry)
                raise GeometryError(
                    f"{dim!r} is not a dimension of this Volume (dims {self.dims});"
                    f" geometry kind is {self.geometry.kind!r}"
                )
            else:
                raise ValueError(
                    f"{dim!r} is not a dimension (dims {self.dims}){_fast.did_you_mean(dim, self)}"
                )
        return Volume._make(
            self._src, tuple(tdims), traces, labels, vsel, scalars, dict(self._attrs), root
        )

    def sel(
        self,
        indexers: Mapping[str, Any] | None = None,
        method: str | None = None,
        **kwargs: Any,
    ) -> Volume:
        """Select by label.

        On a grid, labels are the line, offset and vertical coordinate values.
        Label slices include both ends. On the ``(trace, time)`` layout,
        ``inline``, ``crossline`` and ``offset`` filter the traces through the
        trace index, so an unsorted file answers ``sel(inline=...,
        crossline=...)`` without a rewrite. No sample is read.

        Parameters
        ----------
        indexers : mapping of str to label, slice or array, optional
            Labels by dim name.
        method : {None, "nearest"}, optional
            ``None`` needs exact labels. ``"nearest"`` picks the closest label.
        **kwargs
            The same, as keywords: ``sel(inline=3, time=slice(0, 100000))``.

        Returns
        -------
        Volume
            The selection.

        Raises
        ------
        KeyError
            A label that is not on the axis. The message gives the labels that
            exist.
        ValueError
            A name that is not a dim (the message suggests the nearest one), or
            an unknown ``method``.
        GeometryError
            ``inline``, ``crossline`` or ``offset`` on a Volume that does not
            have that dim.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.sel(inline=[1, 3], crossline=slice(21, 23)).shape
        (2, 3, 50)
        >>> vol.sel(inline=2.8, method="nearest").coords["inline"]
        array(3, dtype=int32)
        >>> vol.sel(time=8000).dims
        ('inline', 'crossline')
        """
        keys = dict(indexers or {}, **kwargs)
        out = self
        by_trace: dict[str, Any] = {}
        positional: dict[str, Any] = {}
        for dim, key in keys.items():
            if dim in self._tdims:
                positional[dim] = _label_positions(self._labels[dim], key, method, dim)
            elif dim == self._src.vertical:
                if isinstance(self._vsel, int):
                    raise ValueError(f"{dim} was already selected to one sample")
                axis = self._src.axis()[self._vsel_index()]
                positional[dim] = _label_positions(axis, key, method, dim)
            elif "trace" in self._tdims and dim in _LINE_DIMS:
                _check_dim_type(dim, self.geometry)
                by_trace[dim] = key
            elif dim in _LINE_DIMS:
                raise GeometryError(
                    f"{dim!r} is not a dimension of this Volume (dims {self.dims});"
                    f" geometry kind is {self.geometry.kind!r}"
                )
            else:
                raise ValueError(
                    f"{dim!r} is not a dimension (dims {self.dims}){_fast.did_you_mean(dim, self)}"
                )
        if positional:
            out = out.isel(positional)
        if by_trace:
            out = out._filter_traces(by_trace, method)
        return out

    def isel_points(self, indexers: Mapping[str, Any] | None = None, **kwargs: Any) -> Volume:
        """Pointwise selection by position: one trace per point, read in one call.

        Each keyword is a trace dim with an equal-length list of positions;
        point ``k`` is ``(inline[k], crossline[k], ...)``. Dims left out stay
        whole. :meth:`numpy` then reads every trace in one parallel decode with
        the GIL released.

        On the Volume :func:`segyrs.open` returned, ``trace=`` alone takes file
        trace numbers in file order, whatever the layout or sort order.

        Parameters
        ----------
        indexers : mapping of str to int array, optional
            Positions by trace dim name.
        **kwargs
            The same, as keywords.

        Returns
        -------
        Volume
            Dims ``(point, *other trace dims, time)``.

        Raises
        ------
        ValueError
            No dim given, lists of unequal length, or a name that is not a
            trace dim.
        IndexError
            A position outside its dim.
        GeometryError
            A grid dim on the ``(trace, time)`` layout; use :meth:`sel` there.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> pts = vol.isel_points(inline=[0, 4], crossline=[0, 4])
        >>> pts.dims, pts.shape
        (('point', 'time'), (2, 50))
        >>> vol.isel_points(trace=[24, 0]).coords["trace"]
        array([24,  0])
        """
        return _fast.isel_points(self, dict(indexers or {}, **kwargs))

    def sel_points(
        self,
        indexers: Mapping[str, Any] | None = None,
        method: str | None = None,
        **kwargs: Any,
    ) -> Volume:
        """Pointwise selection by label, as :meth:`isel_points` takes positions.

        Parameters
        ----------
        indexers : mapping of str to label array, optional
            Labels by trace dim name.
        method : {None, "nearest"}, optional
            ``"nearest"`` picks the closest label per point.
        **kwargs
            The same, as keywords.

        Returns
        -------
        Volume
            Dims ``(point, *other trace dims, time)``.

        Raises
        ------
        KeyError
            A label that is not on the grid.
        TypeError
            A slice instead of a list of labels.
        ValueError
            No dim given, lists of unequal length, or a name that is not a
            trace dim.
        GeometryError
            A grid dim on the ``(trace, time)`` layout.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> pts = vol.sel_points(inline=[1, 5], crossline=[20, 24])
        >>> pts.coords["trace"]
        array([ 0, 24])
        """
        return _fast.sel_points(self, dict(indexers or {}, **kwargs), method)

    def _filter_traces(self, keys: dict[str, Any], method: str | None) -> Volume:
        src = self._src
        current = self._labels["trace"]
        il, xl = keys.get("inline"), keys.get("crossline")
        pos: np.ndarray | None = None
        if (
            il is not None
            and xl is not None
            and np.ndim(il) == 0
            and np.ndim(xl) == 0
            and not isinstance(il, slice)
            and not isinstance(xl, slice)
            and method is None
        ):
            # The parallel (inline, crossline) index: no scan over every trace (S-15).
            hits = src.file.lookup(int(il), int(xl))
            pos = hits if self._root else np.flatnonzero(np.isin(current, hits))
            keys = {k: v for k, v in keys.items() if k not in ("inline", "crossline")}
        if pos is None:
            pos = np.arange(len(current))
        if keys:
            coords = dict(src.scanned)
            if src.extended_offset() is not None:
                coords["offset"] = src.extended_offset()
            mask = np.ones(len(pos), dtype=bool)
            for dim, key in keys.items():
                mask &= _match(coords[dim][current[pos]], key, method, dim)
            pos = pos[mask]
        if not len(pos):
            asked = {
                k: _fast.plain(v) for k, v in (keys or {"inline": il, "crossline": xl}).items()
            }
            raise KeyError(f"no trace matches {asked}")
        return self.isel({"trace": pos})

    def _vsel_index(self) -> Any:
        v = self._vsel
        if isinstance(v, range):
            return slice(v.start, v.stop, v.step) if len(v) else slice(0, 0)
        return v

    # ---- data ---------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Read and decode the current selection, and nothing else.

        The dtype follows the sample format: integers stay integers and IBM
        float is float32. No mask and no scale are applied.

        Returns
        -------
        numpy.ndarray
            Shaped like :attr:`shape`. A new array the caller owns.

        Raises
        ------
        GeometryAssumptionError
            On a file opened with ``scan="auto"``, when a trace read contradicts
            the assumed grid.
        SegyrsIOError
            When the file system fails a read.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> section = vol.sel(inline=3).numpy()
        >>> section.shape, section.dtype
        ((5, 50), dtype('float32'))
        """
        flat = np.ascontiguousarray(self._traces.reshape(-1), dtype=np.int64)
        v = self._vsel
        if isinstance(v, int):
            lo, hi, local = v, v + 1, 0
        elif isinstance(v, range):
            if not len(v):
                lo, hi, local = 0, 0, slice(None)
            elif v.step > 0:
                lo, hi, local = v.start, v[-1] + 1, slice(None, None, v.step)
            else:
                lo, hi = v[-1], v[0] + 1
                local = np.asarray(v) - lo
        else:
            lo, hi = int(v.min()), int(v.max()) + 1
            local = v - lo
        _interop.prefetch(self._src.file, flat)
        block = self._src.file.read(flat, lo, hi)
        return block[:, local].reshape(self.shape)

    @property
    def traces_read(self) -> int:
        """Traces whose sample bytes this file has read so far. Open leaves it at 0.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.traces_read
        0
        >>> _ = vol.sel(inline=3).numpy()
        >>> vol.traces_read
        5
        """
        return self._src.file.samples_read

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def item(self) -> Any:
        """The one selected sample as a Python number.

        Returns
        -------
        int or float
            The sample value.

        Raises
        ------
        ValueError
            When the selection holds more than one sample.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> round(vol.sel(inline=3, crossline=22, time=8000).item(), 4)
        3.22
        """
        return self.numpy().item()

    def __minetypes__(self, *, budget: int | float | None = None) -> dict:
        """Protocol version 1 for this selection.

        Decodes the selection into ``data``. That array is a copy, and its
        size is :attr:`nbytes`. ``budget`` is checked before the copy.

        An ``(inline, crossline, time)`` grid is ``seismic_volume``. A grid
        with ``offset``, or a ``(trace, time)`` volume, is
        ``prestack_volume``. ``sample_unit`` is ``"us"``. This call does
        not rescale the sample axis.

        Parameters
        ----------
        budget : int or float, optional
            RAM for the copy. ``None`` is half the available RAM, an int is
            bytes, a float in (0, 1] is a fraction of the available RAM.

        Returns
        -------
        dict
            Keys defined by protocol version 1 for the kind above.

        Raises
        ------
        BudgetError
            ``nbytes`` does not fit ``budget``. No sample bytes are read.
        SegyrsValueError
            The selection has no sample axis, a dropped line axis, a
            ``point`` axis, or a dtype the protocol does not list.

        Examples
        --------
        >>> payload = segyrs.open("survey.sgy").__minetypes__()
        >>> payload["kind"], payload["data"].shape, payload["sample_unit"]
        ('seismic_volume', (5, 5, 50), 'us')
        >>> window = segyrs.open("survey.sgy").sel(inline=slice(1, 2)).__minetypes__()
        >>> window["data"].shape, window["inlines"].tolist()
        ((2, 5, 50), [1, 2])
        >>> segyrs.open("survey.sgy").__minetypes__(budget=10)
        Traceback (most recent call last):
        ...
        segyrs.BudgetError: the selection needs 5000 bytes, the RAM budget is 10 bytes
        """
        return produce(self, budget)

    # ---- writing (M2) and whole-volume operations (M3): bodies in _write and _ops

    def to_segy(
        self,
        path: str | os.PathLike[str],
        *,
        sample_format: str | None = None,
        endian: str | None = None,
        rezero_time: bool = False,
        budget: int | float | None = None,
        sync: bool = True,
    ) -> None:
        """Write this selection as a new SEG-Y file: the crop of ``sel`` or ``isel``.

        The traces are the selected ones, slowest file axis first (a
        crossline-major file stays crossline-major) and each axis in selection
        order, with their own headers byte for byte. An unselected Volume copies
        the file byte for byte. The sample window is the selected contiguous
        ``time`` range, and each kept sample keeps its time: the delay word
        moves with the window. The file appears complete or not at all.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file.
        sample_format : str, optional
            Convert the samples, for example ``"ieee_float32"``. ``None`` keeps
            the source format, so IBM stays IBM bit for bit.
        endian : {None, "big", "little", "pairwise"}, optional
            Change the byte order. ``None`` keeps the source order.
        rezero_time : bool, default False
            Write a zero delay, so the first kept sample is at time 0.
        budget : int or float, optional
            RAM for the streaming copy: ``None`` is half the available RAM, an
            int is bytes, a float in (0, 1] a fraction of the available RAM.
        sync : bool, default True
            Flush the data and the directory before returning.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            An empty selection, a stepped or reversed time window, or an
            unknown ``sample_format`` or ``endian``.
        SampleValueError
            A sample the target format cannot hold exactly. Nothing is written.
        SegyrsIOError
            When the file system refuses the write.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.sel(inline=slice(2, 4), time=slice(0, 100000)).to_segy("crop.sgy")
        >>> segyrs.open("crop.sgy").shape
        (3, 5, 26)
        """
        from ._write import to_segy

        to_segy(
            self,
            path,
            sample_format=sample_format,
            endian=endian,
            rezero_time=rezero_time,
            budget=budget,
            sync=sync,
        )

    def filter(self, **conditions: Any) -> Volume:
        """Keep the traces whose header fields pass every condition.

        Field names are the standard header names; ``inline``, ``crossline``
        and ``offset`` follow the bytes the file was opened with. The headers
        are scanned in Rust on all cores. No sample is read.

        Parameters
        ----------
        **conditions
            ``field=(lo, hi)`` or ``field=slice(lo, hi)`` is an inclusive range
            (``None`` leaves an end open). A list or array is a set of values,
            a scalar one value.

        Returns
        -------
        Volume
            A ``(trace, time)`` Volume in file axis order, each axis in
            selection order.

        Raises
        ------
        ValueError
            No condition, or a range with a step.
        SegyrsValueError
            A field name that does not exist.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> kept = vol.filter(inline=(2, 3), crossline=[20, 24])
        >>> kept.dims, kept.shape
        (('trace', 'time'), (4, 50))
        """
        from ._ops import filter

        return filter(self, **conditions)

    def resort(
        self,
        path: str | os.PathLike[str],
        by: str | Sequence[str] = ("inline", "crossline"),
        *,
        descending: bool | Sequence[bool] = False,
        strategy: str = "auto",
        sorting_code: int | None = None,
        tmp_dir: str | os.PathLike[str] | None = None,
        budget: int | float | None = None,
        sync: bool = True,
    ) -> dict[str, Any]:
        """Write the whole file sorted by header fields. Ties keep file order.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file.
        by : str or sequence of str, default ("inline", "crossline")
            Header fields, most significant first.
        descending : bool or sequence of bool, default False
            One flag for every key, or one for all.
        strategy : {"auto", "memory", "gather", "external"}, default "auto"
            How to sort: in RAM, by gathers, or with spill runs on disk.
            ``"auto"`` picks from the file size and ``budget``.
        sorting_code : int, optional
            Value for the binary-header trace sorting code (bytes 3229-3230).
        tmp_dir : str or os.PathLike, optional
            Directory for the spill runs of the external sort. ``None`` uses
            the destination's directory.
        budget : int or float, optional
            RAM for the sort: ``None`` is half the available RAM, an int is
            bytes, a float in (0, 1] a fraction of the available RAM.
        sync : bool, default True
            Flush the data and the directory before returning.

        Returns
        -------
        dict
            What ran: ``strategy``, ``already_sorted``, ``runs``,
            ``spilled_bytes``.

        Raises
        ------
        ValueError
            On a selection (resort works on whole files), an unknown
            ``strategy``, or ``descending`` of the wrong length.
        SegyrsValueError
            A field name that does not exist.
        SegyrsIOError
            When the file system refuses the write.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> vol.resort("by-crossline.sgy", by=("crossline", "inline"))["strategy"]
        'memory'
        >>> segyrs.open("by-crossline.sgy").geometry.kind
        'crossline_major'
        """
        from ._ops import resort

        return resort(
            self,
            path,
            by,
            descending=descending,
            strategy=strategy,
            sorting_code=sorting_code,
            tmp_dir=tmp_dir,
            budget=budget,
            sync=sync,
        )

    def slice_horizon(
        self, horizon: Any, *, above: int = 0, below: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Samples along a horizon: ``above + 1 + below`` per trace.

        Parameters
        ----------
        horizon : array_like
            One ``time`` (or ``depth``) value per trace, shaped like the trace
            dims, in the units of ``coords["time"]``. The nearest sample is
            used. NaN marks no pick. A scalar is one flat horizon.
        above, below : int, default 0
            Samples to take above and below the pick.

        Returns
        -------
        values : numpy.ndarray
            Shaped ``(*trace dims, above + 1 + below)``. Samples outside a
            trace are 0.
        valid : numpy.ndarray of bool
            False where a sample is outside its trace or the pick is NaN.

        Raises
        ------
        ValueError
            A horizon that does not broadcast to the trace dims.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> values, valid = vol.slice_horizon(40000.0, above=1, below=1)
        >>> values.shape, bool(valid.all())
        ((5, 5, 3), True)
        """
        from ._ops import slice_horizon

        return slice_horizon(self, horizon, above=above, below=below)

    def to_regular(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        fill: float = 0.0,
        budget: int | float | None = None,
        sync: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]] | dict[str, int]:
        """A dense ``(inline, crossline[, offset], time)`` grid of these traces.

        Use it on a sparse or unsorted file to get a full cube.

        Parameters
        ----------
        path : str or os.PathLike, optional
            ``None`` returns the grid in RAM. A path streams the grid to a new
            SEG-Y file, empty cells as dead traces.
        fill : float, default 0.0
            Sample value of the cells with no trace.
        budget : int or float, optional
            RAM allowed: ``None`` is half the available RAM, an int is bytes, a
            float in (0, 1] a fraction of the available RAM. Checked before
            anything is allocated.
        sync : bool, default True
            With ``path``: flush the data and the directory before returning.

        Returns
        -------
        tuple or dict
            Without ``path``: ``(data, mask, coords)``, where ``mask`` is false
            where no trace exists and ``data`` holds ``fill`` there. With
            ``path``: ``{"cells": ..., "filled": ...}``.

        Raises
        ------
        ValueError
            On a selection of samples rather than whole traces.
        BudgetError
            When the grid does not fit ``budget``. Nothing is allocated.
        GeometryError
            When the trace keys span an absurd grid.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> data, mask, coords = vol.filter(crossline=(21, 23)).to_regular()
        >>> data.shape, bool(mask.all())
        ((5, 3, 50), True)
        """
        from ._ops import to_regular

        return to_regular(self, path, fill=fill, budget=budget, sync=sync)

    def stats(self) -> dict[str, float]:
        """Summary statistics over the selected traces, in one parallel pass.

        Returns
        -------
        dict of str to float
            ``count``, ``nan``, ``infinite``, ``min``, ``max``, ``mean``,
            ``std``, ``variance`` and ``rms``. NaN and infinite samples are
            counted and left out of the others.

        Raises
        ------
        ValueError
            On a selection of samples rather than whole traces.

        Examples
        --------
        >>> s = segyrs.open("survey.sgy").stats()
        >>> s["count"], round(s["max"], 2)
        (1250, 5.24)
        """
        from ._ops import stats

        return stats(self)

    def rms(self) -> np.ndarray:
        """RMS of each selected trace over its finite samples.

        Returns
        -------
        numpy.ndarray
            float64, shaped like the trace dims.

        Raises
        ------
        ValueError
            On a selection of samples rather than whole traces.

        Examples
        --------
        >>> segyrs.open("survey.sgy").rms().shape
        (5, 5)
        """
        from ._ops import rms

        return rms(self)

    def clip(
        self,
        lo: float,
        hi: float,
        path: str | os.PathLike[str],
        *,
        budget: int | float | None = None,
        sync: bool = True,
    ) -> int:
        """Write the selected traces with samples limited to ``[lo, hi]``.

        The output keeps the source format. A sample inside keeps its bytes;
        one outside becomes the nearest value of the source format inside the
        bounds.

        Parameters
        ----------
        lo, hi : float
            Inclusive bounds, ``lo <= hi``.
        path : str or os.PathLike
            Destination file.
        budget : int or float, optional
            RAM for the streaming write: ``None`` is half the available RAM, an
            int is bytes, a float in (0, 1] a fraction of the available RAM.
        sync : bool, default True
            Flush the data and the directory before returning.

        Returns
        -------
        int
            Number of samples changed.

        Raises
        ------
        ValueError
            On a selection of samples rather than whole traces.
        SegyrsValueError
            When ``lo > hi``.
        SegyrsIOError
            When the file system refuses the write.

        Examples
        --------
        >>> vol = segyrs.open("survey.sgy")
        >>> changed = vol.clip(2.0, 4.0, "clipped.sgy")
        >>> float(segyrs.open("clipped.sgy").stats()["max"])
        4.0
        """
        from ._ops import clip

        return clip(self, lo, hi, path, budget=budget, sync=sync)

    def to_zarr(
        self,
        path: str | os.PathLike[str],
        *,
        chunks: Sequence[int] | None = None,
        target_chunk_bytes: int | None = None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write the whole file as a Zarr v3 group. Experimental.

        Arrays: ``data``, ``trace_mask``, the dim coordinates, ``cdp_x`` and
        ``cdp_y``. Chunks with no trace are not written. Runs on all cores with
        the GIL released. Only a Volume from :func:`segyrs.open` can be
        exported, not a selection.

        Parameters
        ----------
        path : str or os.PathLike
            A directory, or an object-store URL such as ``s3://bucket/x.zarr``.
        chunks : sequence of int, optional
            Explicit chunk shape, one entry per dim.
        target_chunk_bytes : int, optional
            Default chunks are whole traces in on-disk order of about this many
            bytes (8 MiB when ``None``).
        storage_options : mapping, optional
            URLs only: object-store keys, as for :func:`segyrs.open`.

        Returns
        -------
        dict
            ``dims``, ``shape``, ``chunks``, ``data_type``,
            ``chunks_written`` and ``chunks_missing``.

        Raises
        ------
        ValueError
            On a selection, or with ``storage_options`` on a local path.
        SegyrsValueError
            A ``chunks`` shape of the wrong length, or an unknown store key.
        SegyrsIOError
            When the store refuses a write.

        Examples
        --------
        >>> summary = segyrs.open("survey.sgy").to_zarr("survey.zarr")
        >>> summary["dims"], summary["chunks_written"]
        (['inline', 'crossline', 'time'], 1)
        """
        return _interop.to_zarr(
            self,
            path,
            chunks=chunks,
            target_chunk_bytes=target_chunk_bytes,
            storage_options=storage_options,
        )

    # ---- pictures (M4, in _view.py) -------------------------------------------

    def plot(
        self,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        colormap: str = "grey",
        polarity: str = "normal",
        gain: float = 1.0,
        wiggle: str | None = None,
        wiggle_every: int = 1,
        density: bool = True,
        width: int = 800,
        height: int = 480,
    ) -> Picture:
        """Draw this Volume's plane to a :class:`Picture` (PNG plus text). Experimental.

        Reads only that plane's traces. A 2D selection draws that section or
        slice. A stepped selection (``isel(crossline=slice(0, 1000, 10))``)
        reads and draws every step-th position only; a scattered list of
        positions has no picture. A 3D (or 4D) Volume draws its first plane:
        the first position of the slowest grid axis (and of offset), leaving
        crossline and time free. In Jupyter a Volume displays this picture.

        Parameters
        ----------
        vmin, vmax : float, optional
            With neither, the clip is the symmetric 98th percentile of the
            drawn view. Otherwise it is ``max(|vmin|, |vmax|)``.
        colormap : {"grey", "seismic", "amber"}, default "grey"
            Colour map of the density display.
        polarity : {"normal", "reversed"}, default "normal"
            ``"reversed"`` flips the sign of the samples.
        gain : float, default 1.0
            Multiplier applied before the clip.
        wiggle : {None, "off", "line", "area"}, optional
            Wiggle traces over (or instead of) the density display. They need
            at least 3 px per trace; below that the picture shows density and
            :attr:`Picture.note` says so.
        wiggle_every : int, default 1
            Draw one trace in N, and read only those. ``N > 1`` implies
            ``wiggle="line"``.
        density : bool, default True
            Draw the colour density under the wiggles.
        width, height : int, default 800, 480
            Plot area in pixels.

        Returns
        -------
        Picture
            The PNG and a description of what was drawn.

        Raises
        ------
        ValueError
            A selection with fewer than two free dims, a scattered or
            descending selection, or an unknown option value.

        Examples
        --------
        >>> pic = segyrs.open("survey.sgy").sel(inline=3).plot()
        >>> pic.traces_read, pic.png[1:4]
        (5, b'PNG')
        >>> pic.save("inline-3.png")
        """
        from ._view import plot

        return plot(
            self,
            vmin=vmin,
            vmax=vmax,
            colormap=colormap,
            polarity=polarity,
            gain=gain,
            wiggle=wiggle,
            wiggle_every=wiggle_every,
            density=density,
            width=width,
            height=height,
        )

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict[str, Any]:
        from ._view import mimebundle

        return mimebundle(self)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{d}: {n}" for d, n in self.sizes.items())
        lines = [
            f"<segyrs.Volume ({sizes})> {self.dtype} {self._attrs['sample_format']},"
            f" lazy, chunks of whole traces"
        ]
        for dim in self._tdims:
            lab = self._labels[dim]
            span = f"{lab[0]} .. {lab[-1]}" if len(lab) else "empty"
            lines.append(f"  {dim:<10} {span}")
        if not isinstance(self._vsel, int):
            axis = self._src.axis()[self._vsel_index()]
            span = f"{axis[0]} .. {axis[-1]}" if len(axis) else "empty"
            lines.append(f"  {self._src.vertical:<10} {span} us")
        for name, value in self._scalars.items():
            lines.append(f"  {name:<10} {value} (selected)")
        lines.append(f"  geometry   {self.geometry.kind}")
        return "\n".join(lines)


def _file_order(vol: Volume) -> np.ndarray:
    """The selected trace numbers flattened slowest file axis first.

    A crossline-major file then writes crossline-major, each axis in selection
    order, and an unselected volume is the file's own trace order.
    """
    axes = vol._src.geometry.axes
    traces = vol._traces
    if traces.ndim > 1 and all(d in axes for d in vol._tdims):
        order = sorted(range(traces.ndim), key=lambda i: axes.index(vol._tdims[i]))
        traces = traces.transpose(order)
    return np.ascontiguousarray(traces.reshape(-1), dtype=np.int64)


def _positions(key: Any, n: int, dim: str) -> Any:
    """Normalise a positional key: int (bounds-checked), slice, or int array."""
    if isinstance(key, slice):
        return key
    if np.ndim(key) == 0:
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)):
            raise TypeError(f"{dim}: position must be an int, slice, or int array")
        i = int(key)
        if not -n <= i < n:
            raise IndexError(f"{dim} position {i} is outside 0..{n}")
        return i % n
    arr = np.asarray(key)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise IndexError(f"{dim}: boolean mask needs length {n}")
        return np.flatnonzero(arr)
    if not np.issubdtype(arr.dtype, np.integer) and arr.size:
        raise TypeError(f"{dim}: positions must be integers")
    arr = arr.astype(np.int64).ravel()
    if arr.size and (arr.min() < -n or arr.max() >= n):
        raise IndexError(f"{dim} positions are outside 0..{n}")
    return arr % n if n else arr


def _compose(vsel: Any, key: Any) -> Any:
    """Apply a positional key to the current vertical selection."""
    if isinstance(vsel, range):
        if isinstance(key, (int, slice)):
            return vsel[key]
        return np.asarray(vsel, dtype=np.int64)[key]
    if isinstance(key, int):
        return int(vsel[key])
    return vsel[key]


def _header_fields(file: _segyrs.File, header_map, inline_byte, crossline_byte, offset_byte):
    fields = {name: (byte, dtype) for name, byte, dtype in file.standard_fields()}
    fields["inline"] = (inline_byte, "int32")
    fields["crossline"] = (crossline_byte, "int32")
    fields["offset"] = (offset_byte, "int32")
    limit = 240 * (1 + file.extra_trace_header_blocks)
    for name, spec in (header_map or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("header_map keys are field names")
        byte, dtype = (spec, "int32") if isinstance(spec, (int, np.integer)) else spec
        dtype = np.dtype(dtype).name
        if dtype not in _HEADER_DTYPES:
            raise ValueError(f"header_map[{name!r}]: dtype {dtype} is not one of {_HEADER_DTYPES}")
        width = np.dtype(dtype).itemsize
        if not 1 <= int(byte) <= limit - width + 1:
            raise ValueError(
                f"header_map[{name!r}]: byte {byte} width {width} does not fit"
                f" the {limit}-byte trace header"
            )
        fields[name] = (int(byte), dtype)
    return [(name, byte, dtype) for name, (byte, dtype) in fields.items()]


def open(
    path: str | os.PathLike[str],
    *,
    storage_options: Mapping[str, Any] | None = None,
    sample_format: str | None = None,
    endian: str | None = None,
    gain: str | None = None,
    inline_byte: int = 189,
    crossline_byte: int = 193,
    offset_byte: int = 37,
    header_map: Mapping[str, int | tuple[int, str]] | None = None,
    dead_trace: str | np.ndarray | None = None,
    depth: bool = False,
    scan: str = "auto",
    index_cache: str | os.PathLike[str] | None = None,
) -> Volume:
    """Open a SEG-Y file as a lazy :class:`Volume`.

    Open maps the file and reads the textual header, the binary header, and
    the trace-header words that geometry needs, in parallel with the GIL
    released. It reads no sample bytes and starts no process. On a large
    regular file it reads only a sample of the trace headers: see ``scan``.

    Parameters
    ----------
    path : str or os.PathLike
        A filesystem path, including non-ASCII paths, or an object-store
        URL: ``s3://``, ``gs://``, ``az://``, ``abfs(s)://``,
        ``http(s)://`` or ``file://``. A path is memory-mapped. A URL is
        read through Rust ``object_store`` with a parallel byte-range cache;
        opening scans every trace header, so it reads most of the object
        once. URL reads are experimental.
    storage_options : mapping, optional
        URLs only. Store keys (``aws_region``, ``google_service_account``,
        ``azure_storage_account_name``, ``allow_http``, ...) and reader
        cache keys (``block_size``, ``readahead``, ``coalesce_gap``,
        ``concurrency``, ``cache_bytes``, ``request_timeout``,
        ``retries``). Credentials also come from the environment.
    sample_format : str, optional
        Override the binary-header format code, for example
        ``"ibm_float32"`` or ``"int16"``. ``attrs["binary_format_code"]``
        keeps the file's code.
    endian : {None, "big", "little", "pairwise"}, optional
        ``None`` reads the rev 2 signature, then falls back to detection.
        The others force the byte order.
    gain : {None, "rev0", "rev2"}, optional
        Format 4 only, and then required.
    inline_byte, crossline_byte, offset_byte : int, default 189, 193, 37
        1-based byte positions of the int32 inline, crossline and offset
        words.
    header_map : mapping, optional
        Extra or replacement header fields: ``{"name": (byte, dtype)}`` or
        ``{"name": byte}`` for an int32.
    dead_trace : {None, "trace_id"} or numpy.ndarray of bool, optional
        ``None``: every stored trace is live, zeros included.
        ``"trace_id"``: trace identification code 2 is dead. Or a bool
        array, true where the trace is dead. The result is
        ``coords["trace_mask"]``.
    depth : bool, default False
        Name the vertical dim ``depth`` instead of ``time``.
    scan : {"auto", "full"}, default "auto"
        ``"auto"``: a local file of more than 4096 traces whose first, last
        and 1024 sampled headers fit a grid with constant inline, crossline
        and offset steps opens from those headers, without reading the
        rest. ``geometry.source`` is then ``"assumed"``. Every read checks
        the keys of the traces it touches, and whole-file header access
        checks every trace; a trace that contradicts the grid raises
        :class:`GeometryAssumptionError`. Any other file, and any URL, is
        scanned in full. ``"full"``: read every trace header now.
    index_cache : str or os.PathLike, optional
        Local paths only. A directory for header-index sidecars. A full
        scan stores the scanned columns there, and the next open of the
        unchanged file (same size, modification time, and leading and
        trailing headers) loads them instead (``geometry.source ==
        "index_cache"``). ``None`` writes nothing.

    Returns
    -------
    Volume
        The lazy, labelled view of the file.

    Raises
    ------
    PathNotFoundError, PathPermissionError, PathIsDirectoryError
        The path is wrong. These are also ``FileNotFoundError``,
        ``PermissionError`` and ``IsADirectoryError``.
    NotSegyError
        The bytes are not SEG-Y. The message says what they look like.
    UnknownFormat
        The format code is unknown; pass ``sample_format``.
    Format4NeedsGain
        Format 4 without ``gain``.
    SampleCountMismatch, NonUniformTraceLength, TruncatedError
        The file size does not match its headers.
    EndianMismatch
        The rev 2 signature and the decoded sizes disagree.
    DuplicateKey
        Two traces share one grid slot.
    GeometryError
        The trace keys span an absurd grid. Nothing is allocated.
    ValueError
        An option value segyrs does not know, ``storage_options`` on a
        local path, or a ``header_map`` field that does not fit the header.
    TypeError
        ``depth`` is not a bool.

    Examples
    --------
    >>> vol = segyrs.open("survey.sgy")
    >>> vol.sizes
    {'inline': 5, 'crossline': 5, 'time': 50}
    >>> vol = segyrs.open("survey.sgy", header_map={"il_copy": (189, "int32")})
    >>> vol.headers["il_copy"][:3]
    array([1, 1, 1], dtype=int32)
    """
    if not isinstance(depth, bool):
        raise TypeError("depth must be a bool")
    if index_cache is not None and _interop.is_url(path):
        raise ValueError("index_cache applies to local paths only")
    file = _segyrs.File(
        _interop.location(path),
        storage_options=_interop.normalize_options(storage_options),
        sample_format=sample_format,
        endian=endian,
        gain=gain,
        inline_byte=inline_byte,
        crossline_byte=crossline_byte,
        offset_byte=offset_byte,
        scan=scan,
        index_cache=None if index_cache is None else os.fspath(index_cache),
    )
    fields = _header_fields(file, header_map, inline_byte, crossline_byte, offset_byte)
    src = _Source(file, fields, dead_trace, "depth" if depth else "time")
    src.open_args = (
        os.fspath(path),
        {
            "sample_format": sample_format,
            "endian": endian,
            "gain": gain,
            "inline_byte": inline_byte,
            "crossline_byte": crossline_byte,
            "offset_byte": offset_byte,
            "scan": scan,
        },
    )
    src.trace_mask()  # validate dead_trace now, not at the first coords access
    g = src.geometry
    n = src.n
    if g.is_grid:
        line_labels = {"inline": g.inlines, "crossline": g.crosslines, "offset": g.offsets}
        tdims = ("inline", "crossline", "offset") if "offset" in g.axes else ("inline", "crossline")
        counts = [len(line_labels[a]) for a in g.axes]
        traces = np.arange(n, dtype=np.int64).reshape(counts)
        traces = traces.transpose([g.axes.index(d) for d in tdims])
        labels = {d: line_labels[d] for d in tdims}
        if "offset" in tdims:
            ext = src.extended_offset()
            if ext is not None:
                first = traces[(0,) * (len(tdims) - 1)]
                labels["offset"] = ext[first]
    else:
        tdims = ("trace",)
        traces = np.arange(n, dtype=np.int64)
        labels = {"trace": traces}
    for v in labels.values():
        v.setflags(write=False)
    return Volume._make(src, tdims, traces, labels, range(src.ns), {}, dict(src.attrs), True)
