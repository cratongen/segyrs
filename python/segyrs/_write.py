"""Writing SEG-Y (M2): create, Volume.to_segy, edit and validate."""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _segyrs
from ._protocol import prepare

if TYPE_CHECKING:
    from ._volume import Volume

# Coordinate words and the scalar word that scales them (X-24, X-05).
_SCALED_BY = {
    **{n: "scalco" for n in ("cdp_x", "cdp_y", "source_x", "source_y", "group_x", "group_y")},
    **{
        n: "scalel"
        for n in ("group_elev", "source_elev", "source_depth", "group_datum", "source_datum")
    },
}


def _fields() -> dict[str, tuple[int, str]]:
    fields = {name: (byte, dtype) for name, byte, dtype in _segyrs.File.standard_fields()}
    fields.setdefault("inline", (189, "int32"))
    fields.setdefault("crossline", (193, "int32"))
    fields.setdefault("offset", (37, "int32"))
    return fields


def _field(key: Any, fields: Mapping[str, tuple[int, str]]) -> tuple[int, str, str]:
    """``"name"`` or ``(byte, dtype)`` to ``(byte, dtype, name)``."""
    if isinstance(key, str):
        if key not in fields:
            raise KeyError(f"{key!r} is not a standard header field; pass (byte, dtype)")
        byte, dtype = fields[key]
        return byte, dtype, key
    byte, dtype = key
    return int(byte), np.dtype(dtype).name, f"byte_{int(byte)}"


def _values(values: Any, n: int, name: str) -> np.ndarray:
    """One value per trace as int64 or float64: a scalar fills every trace."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = np.full(n, arr.item())
    arr = arr.reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name}: {arr.size} values for {n} traces")
    if arr.dtype.kind in "biu":
        return np.ascontiguousarray(arr, dtype=np.int64)
    if arr.dtype.kind == "f":
        return np.ascontiguousarray(arr, dtype=np.float64)
    raise TypeError(f"{name}: header values must be numbers, not {arr.dtype}")


def _overlay(key: Any, values: Any, n: int, fields, scalars=None) -> tuple:
    byte, dtype, name = _field(key, fields)
    vals = _values(values, n, name)
    if scalars is not None:
        sc = np.ascontiguousarray(np.broadcast_to(np.asarray(scalars), (n,)), dtype=np.int16)
        return (byte, dtype, name, vals.astype(np.float64), sc)
    return (byte, dtype, name, vals, None)


def _overlays(headers: Mapping[Any, Any], n: int) -> list[tuple]:
    """Overlays for ``create``: float coordinates are scaled by ``scalco``/``scalel`` when given."""
    fields = _fields()
    out = []
    for key, values in headers.items():
        scalar = _SCALED_BY.get(key) if isinstance(key, str) else None
        scalars = None
        if scalar in headers and np.asarray(values).dtype.kind == "f":
            scalars = _values(headers[scalar], n, scalar)
        out.append(_overlay(key, values, n, fields, scalars))
    return out


def _cards(text: str | Sequence[str] | None) -> list[str] | None:
    if text is None:
        return None
    lines = text.splitlines() if isinstance(text, str) else list(text)
    if len(lines) > 40:
        raise ValueError(f"the textual header holds 40 cards, not {len(lines)}")
    return lines


def _stamp_length_unit(path: str, unit: str | None) -> None:
    """Write ``m`` or ``ft`` into the binary-header measurement system."""
    code = {"m": 1, "ft": 2}.get(unit or "")
    if code is None:
        return
    file = _segyrs.File(path)
    endian = file.source_endian
    raw = bytearray(file.binary_header_bytes())
    del file
    gc.collect()
    order = "little" if endian in ("little", "pairwise") else "big"
    raw[54:56] = int(code).to_bytes(2, order, signed=True)
    with edit(path) as editor:
        editor.write_binary_header(bytes(raw))


def create(
    path: str | os.PathLike[str],
    data: Any,
    *,
    sample_format: str,
    sample_interval_us: int | float | None = None,
    inline: Any = None,
    crossline: Any = None,
    offset: Any = None,
    headers: Mapping[Any, Any] | None = None,
    text: str | Sequence[str] | None = None,
    endian: str | None = None,
    sync: bool = True,
) -> int:
    """Write a new SEG-Y file from an array or a protocol volume.

    Parameters
    ----------
    path : str or os.PathLike
        Destination. It appears complete or not at all.
    data : numpy.ndarray or protocol object
        ``(trace, time)``, ``(inline, crossline, time)`` or
        ``(inline, crossline, offset, time)`` samples of any numeric dtype.
        A dict, or any object with ``__minetypes__``, of kind
        ``seismic_volume`` or ``prestack_volume`` is accepted too.
    sample_format : str
        Required, for example ``"ibm_float32"``, ``"ieee_float32"`` or
        ``"int16"``. segyrs does not pick one.
    sample_interval_us : int, optional
        Sample interval in integer microseconds, written as given.
        Required for an array. For a protocol object the sample axis
        supplies it, and a value you pass has to equal that step.
    inline, crossline, offset : array_like, optional
        For 3D and 4D data, the axis labels (default ``1..n``). For 2D
        data, one value per trace. Written at bytes 189, 193 and 37.
        On a protocol object these come from the object.
    headers : mapping, optional
        More trace-header fields: ``{"cdp_x": values, (byte, dtype):
        values}``, one value per trace or one for all. Float coordinates
        are stored with the ``scalco`` (or ``scalel``) you pass in the same
        mapping.
    text : str or sequence of str, optional
        Textual header cards (a string with one card per line, or a list
        of at most 40), written as EBCDIC. ``None`` writes a default
        header.
    endian : {None, "big", "little", "pairwise"}, optional
        Byte order. ``None`` is big-endian, as the standard says.
    sync : bool, default True
        Flush the data and the directory before returning.

    Returns
    -------
    int
        Bytes written.

    Raises
    ------
    ValueError
        ``data`` that is not 2D, 3D or 4D, a ``sample_interval_us`` that is
        not a whole number, labels of the wrong length, more than 40 cards,
        or an unknown ``sample_format`` or ``endian``.
    SampleValueError
        A value the format cannot hold exactly (a fraction for an integer
        format, NaN for IBM).
    HeaderValueError
        A header value that does not fit its field.
    SpecError
        An inconsistent request, such as ``sample_interval_us=0`` with
        traces, which :func:`validate` would reject. Nothing is written.
    SegyrsValueError
        A protocol object segyrs cannot store, or ``inline``,
        ``crossline``, ``offset`` or ``headers`` passed with one.
    TypeError
        An array written with no ``sample_interval_us``.

    Examples
    --------
    >>> data = np.zeros((2, 3, 4), np.float32)
    >>> segyrs.create("new.sgy", data, sample_format="ieee_float32",
    ...               sample_interval_us=2000, inline=[10, 11])
    5136
    >>> segyrs.open("new.sgy").coords["inline"]
    array([10, 11], dtype=int32)
    >>> payload = segyrs.open("survey.sgy").__minetypes__()
    >>> segyrs.create("from_protocol.sgy", payload, sample_format="ieee_float32",
    ...               sample_interval_us=4000) > 0
    True
    >>> segyrs.open("from_protocol.sgy").shape
    (5, 5, 50)
    """
    prepared = prepare(data, sample_interval_us)
    length_unit = None
    if prepared is not None:
        if inline is not None or crossline is not None or offset is not None or headers is not None:
            raise _segyrs.SegyrsValueError(
                "inline, crossline, offset and headers come from the protocol object"
            )
        data = prepared.data
        sample_interval_us = prepared.interval_us
        inline = prepared.inline
        crossline = prepared.crossline
        offset = prepared.offset
        headers = prepared.headers
        length_unit = prepared.length_unit
    elif sample_interval_us is None:
        raise TypeError("create() missing required keyword-only argument: 'sample_interval_us'")
    data = np.asarray(data)
    if data.ndim not in (2, 3, 4):
        raise ValueError(f"data must be 2D, 3D or 4D, not {data.ndim}D")
    if int(sample_interval_us) != sample_interval_us:
        raise ValueError(
            f"sample_interval_us={sample_interval_us} is not a whole number of microseconds;"
            " the binary header stores the interval as an integer"
        )
    lead = data.shape[:-1]
    n = int(np.prod(lead, dtype=np.int64))
    labels: dict[str, Any] = {}
    if data.ndim == 2:
        for name, v in (("inline", inline), ("crossline", crossline), ("offset", offset)):
            if v is not None:
                labels[name] = v
    else:
        names = ("inline", "crossline", "offset")[: data.ndim - 1]
        given = {"inline": inline, "crossline": crossline, "offset": offset}
        axes = []
        for i, name in enumerate(names):
            size = lead[i]
            ax = np.arange(1, size + 1) if given[name] is None else np.asarray(given[name])
            if ax.shape != (size,):
                raise ValueError(f"{name} labels: {ax.size} for an axis of {size}")
            axes.append(ax)
        grids = np.meshgrid(*axes, indexing="ij")
        labels = {name: grids[i].reshape(-1) for i, name in enumerate(names)}
        if data.ndim == 3 and offset is not None:
            labels["offset"] = offset
    merged: dict[Any, Any] = dict(labels)
    merged.update(headers or {})
    samples = np.ascontiguousarray(data.reshape(n, data.shape[-1]))
    written = _segyrs.create(
        os.fspath(path),
        samples,
        int(sample_interval_us),
        sample_format,
        endian=endian,
        overlays=_overlays(merged, n),
        text=_cards(text),
        sync=sync,
    )
    _stamp_length_unit(os.fspath(path), length_unit)
    return written


def _window(vol: Volume) -> tuple[int, int]:
    v = vol._vsel
    if isinstance(v, int):
        return v, v + 1
    if isinstance(v, range) and (len(v) <= 1 or v.step == 1):
        return (v.start, v.start + len(v)) if len(v) else (0, 0)
    raise ValueError(
        "to_segy writes a contiguous, increasing sample window; select one with a slice"
    )


def to_segy(
    vol: Volume,
    path: str | os.PathLike[str],
    *,
    sample_format: str | None = None,
    endian: str | None = None,
    rezero_time: bool = False,
    budget: int | float | None = None,
    sync: bool = True,
) -> int:
    """Body of :meth:`Volume.to_segy`. Returns the number of traces or bytes written."""
    from ._volume import _file_order

    file = vol._src.file
    traces = _file_order(vol)
    start, stop = _window(vol)
    if stop <= start or not len(traces):
        raise ValueError("nothing to write: the selection is empty")
    same_format = sample_format in (None, file.sample_format, str(file.binary_format_code))
    same_endian = endian in (None, file.source_endian)
    if same_format and same_endian:
        # M3 crop streams raw trace bytes through the M2 FileSink: IBM stays IBM
        # bit for bit, the delay moves with the window (S-36), memory follows `budget`.
        return _segyrs.crop(
            file,
            os.fspath(path),
            traces,
            start,
            stop,
            rezero_time=rezero_time,
            budget=budget,
            sync=sync,
        )
    overlays = []
    if rezero_time or start:
        h = vol._src.headers()
        flat = traces
        if rezero_time:
            overlays.append((109, "int16", "delay", np.zeros(len(flat), np.int64), None))
        else:
            from ._volume import _scaled

            dt_ms = file.sample_interval_us / 1000.0
            delay = _scaled(h["delay"][flat], h["time_scalar"][flat]) + start * dt_ms
            scal = np.ascontiguousarray(h["time_scalar"][flat], dtype=np.int16)
            overlays.append((109, "int16", "delay", np.ascontiguousarray(delay), scal))
    root = vol._root and start == 0 and stop == file.n_samples
    return _segyrs.write_from(
        file,
        os.fspath(path),
        traces=None if root else traces,
        first_sample=start,
        n_samples=stop - start,
        sample_format=sample_format,
        endian=endian,
        overlays=overlays,
        sync=sync,
    )


def validate(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Check a file's structure and return its layout.

    Passing does not prove that every other program accepts the file.

    Parameters
    ----------
    path : str or os.PathLike
        The file to check.

    Returns
    -------
    dict
        ``revision``, ``sample_format``, ``binary_format_code``,
        ``endian``, ``n_traces``, ``n_samples``, ``sample_interval_us``,
        ``extended_textual_blocks``, ``extra_trace_header_blocks`` and
        ``data_start``.

    Raises
    ------
    ValidationError
        Listing every problem found in ``err.problems``: unknown format
        code, textual blocks that are not whole, a length that is not
        headers plus whole traces, a zero sample count or interval, illegal
        revision or fixed-length values, or a trace header whose sample
        count disagrees with the binary header.
    PathNotFoundError, PathPermissionError, PathIsDirectoryError
        The path is wrong.

    Examples
    --------
    >>> layout = segyrs.validate("survey.sgy")
    >>> layout["n_traces"], layout["sample_format"]
    (25, 'ibm_float32')
    """
    return _segyrs.validate(os.fspath(path))


class Editor:
    """In-place edits that never change a file's layout.

    Make one with :func:`segyrs.edit`. Every call checks all of its input
    first, then writes, so an error leaves the file as it was. A change of
    sample format or sample count is a new file: use
    :meth:`Volume.to_segy`.

    Examples
    --------
    >>> with segyrs.edit("new.sgy") as ed:
    ...     ed.set_header("cdp", np.arange(ed.n_traces))
    >>> segyrs.open("new.sgy").headers["cdp"]
    array([0, 1, 2, 3, 4, 5], dtype=int32)
    """

    def __init__(self, path: str | os.PathLike[str]):
        """Open ``path`` for editing. Prefer :func:`segyrs.edit`.

        Parameters
        ----------
        path : str or os.PathLike
            An existing SEG-Y file this user may write.

        Raises
        ------
        PathNotFoundError, PathPermissionError, PathIsDirectoryError
            The path is wrong or read-only.
        NotSegyError, UnknownFormat
            The file is not SEG-Y that segyrs can edit.
        """
        self._ed = _segyrs.Editor(os.fspath(path))
        self._fields = _fields()

    def __enter__(self) -> Editor:
        """Return the editor for a ``with`` block.

        Returns
        -------
        Editor
            This editor.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Sync the edits when the block ends without an exception, then close.

        Parameters
        ----------
        exc_type, exc, tb
            The exception leaving the block, or ``None``.
        """
        try:
            if exc_type is None and not self._ed.closed:
                self._ed.sync()
        finally:
            self._ed.close()

    @property
    def n_traces(self) -> int:
        """Number of traces in the file.

        Examples
        --------
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.n_traces
        6
        """
        return self._ed.n_traces

    @property
    def n_samples(self) -> int:
        """Samples per trace.

        Examples
        --------
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.n_samples
        4
        """
        return self._ed.n_samples

    @property
    def sample_format(self) -> str:
        """The file's sample format, which every write keeps.

        Examples
        --------
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.sample_format
        'ieee_float32'
        """
        return self._ed.sample_format

    def set_header(
        self,
        field: Any,
        values: Any,
        *,
        traces: Any = None,
        scalars: Any = None,
    ) -> None:
        """Write one trace-header field on every trace, or on ``traces``.

        Parameters
        ----------
        field : str or (int, str)
            A standard name (``"cdp_x"``) or ``(byte, dtype)``.
        values : array_like
            One value per trace written, or one for all.
        traces : array_like of int, optional
            File trace numbers to write. ``None`` writes every trace.
        scalars : array_like of int, optional
            Encode float coordinates in real units with this SEG-Y scalar (a
            negative scalar multiplies on write). The scalar word itself
            (``scalco`` or ``scalel``) is not written: write it in the same
            :meth:`set_headers` call. It scales every coordinate of the trace.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            An unknown field name.
        ValueError
            The wrong number of values, or an editor that is closed.
        HeaderValueError
            A value that does not fit the field. Nothing is written.

        Examples
        --------
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.set_header("cdp", [101, 102], traces=[0, 1])
        >>> segyrs.open("new.sgy").headers["cdp"][:3]
        array([101, 102,   0], dtype=int32)
        """
        self.set_headers({field: values}, traces=traces, scalars={field: scalars})

    def set_headers(
        self,
        values: Mapping[Any, Any],
        *,
        traces: Any = None,
        scalars: Mapping[Any, Any] | None = None,
    ) -> None:
        """Write several fields in one pass over the file.

        Parameters
        ----------
        values : mapping
            ``{field: values}``, each field as in :meth:`set_header`.
        traces : array_like of int, optional
            File trace numbers to write. ``None`` writes every trace.
        scalars : mapping, optional
            ``{field: scalar}`` for float coordinates encoded with a SEG-Y
            scalar. Write the scalar word (``scalco`` or ``scalel``) in
            ``values`` too, or the file reads the encoded integers.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            An unknown field name.
        ValueError
            The wrong number of values, or an editor that is closed.
        HeaderValueError
            A value that does not fit its field. Nothing is written.

        Examples
        --------
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.set_headers(
        ...         {"cdp_x": [100.5, 200.25], "scalco": -100},
        ...         traces=[0, 1],
        ...         scalars={"cdp_x": -100},
        ...     )
        >>> segyrs.open("new.sgy").coords["cdp_x"].ravel()[:2]
        array([100.5 , 200.25])
        """
        scalars = scalars or {}
        idx = None
        n = self.n_traces
        if traces is not None:
            idx = np.ascontiguousarray(np.asarray(traces).reshape(-1), dtype=np.int64)
            n = len(idx)
        overlays = [_overlay(k, v, n, self._fields, scalars.get(k)) for k, v in values.items()]
        self._ed.write_header_columns(overlays, idx)

    def write_traces(self, samples: Any, first: int = 0) -> None:
        """Overwrite traces ``first, first + 1, ...`` with ``samples``.

        Samples are encoded in the file's own format.

        Parameters
        ----------
        samples : array_like
            One trace, or a 2D ``(trace, sample)`` array.
        first : int, default 0
            File trace number of the first trace written.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            ``samples`` that is not 1D or 2D, or an editor that is closed.
        TraceLengthError
            A row whose length is not the file's sample count.
        SampleValueError
            A value the file's format cannot hold. Nothing is written.
        SpecError
            Traces past the end of the file.

        Examples
        --------
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.write_traces(np.ones((2, 4), np.float32), first=1)
        >>> segyrs.open("new.sgy").isel(inline=0, crossline=1).numpy()
        array([1., 1., 1., 1.], dtype=float32)
        """
        arr = np.asarray(samples)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError("samples must be one trace or a 2D (trace, sample) array")
        self._ed.write_traces(int(first), np.ascontiguousarray(arr))

    def write_binary_header(self, data: bytes) -> None:
        """Replace the 400-byte binary header.

        Parameters
        ----------
        data : bytes
            The new header, exactly 400 bytes.

        Returns
        -------
        None

        Raises
        ------
        SpecError
            ``data`` is not 400 bytes.
        FormatChangeError
            The new header changes the layout (format, sample count, interval
            or extension blocks). Nothing is written.

        Examples
        --------
        >>> header = bytearray(segyrs.open("new.sgy").binary_header_bytes)
        >>> header[0:4] = (42).to_bytes(4, "big")
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.write_binary_header(bytes(header))
        >>> segyrs.open("new.sgy").binary_header["job_id"]
        42
        """
        self._ed.write_binary_header(bytes(data))

    def write_textual_block(self, index: int, data: bytes) -> None:
        """Replace one 3200-byte textual block.

        Parameters
        ----------
        index : int
            0 for the primary block, then the extended blocks.
        data : bytes
            Exactly 3200 bytes, in the encoding the file uses.

        Returns
        -------
        None

        Raises
        ------
        SpecError
            A block that does not exist, or ``data`` that is not 3200 bytes.

        Examples
        --------
        >>> card = "C 1 EDITED".ljust(80).encode("cp037")
        >>> with segyrs.edit("new.sgy") as ed:
        ...     ed.write_textual_block(0, card * 40)
        >>> segyrs.open("new.sgy").attrs["textual_header"][0].strip()
        'C 1 EDITED'
        """
        self._ed.write_textual_block(int(index), bytes(data))

    def sync(self) -> None:
        """One ``fdatasync`` for the edits so far. Leaving the ``with`` block does it too.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            The editor is closed.
        SegyrsIOError
            When the file system fails the sync.

        Examples
        --------
        >>> ed = segyrs.edit("new.sgy")
        >>> ed.set_header("cdp", 1)
        >>> ed.sync()
        >>> ed.close()
        """
        self._ed.sync()

    def close(self) -> None:
        """Close the file without a sync. Closing twice does nothing.

        Examples
        --------
        >>> ed = segyrs.edit("new.sgy")
        >>> ed.close()
        """
        self._ed.close()


def edit(path: str | os.PathLike[str]) -> Editor:
    """Open a file for in-place edits, as a context manager.

    Parameters
    ----------
    path : str or os.PathLike
        An existing SEG-Y file this user may write.

    Returns
    -------
    Editor
        Use it in a ``with`` block: leaving the block syncs and closes.

    Raises
    ------
    PathNotFoundError, PathPermissionError, PathIsDirectoryError
        The path is wrong or read-only.
    NotSegyError, UnknownFormat
        The file is not SEG-Y that segyrs can edit.

    Examples
    --------
    >>> with segyrs.edit("new.sgy") as ed:
    ...     ed.set_header("cdp", np.arange(ed.n_traces))
    ...     ed.write_traces(np.zeros((1, ed.n_samples), np.float32), first=5)
    """
    return Editor(path)
