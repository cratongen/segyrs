"""Whole-volume operations (M3) on a :class:`Volume`.

Every call runs in Rust on all cores with the GIL released. Operations that
produce a file stream it chunk by chunk through the M2 writer, so memory stays
inside ``budget``: ``None`` is half the available RAM, an int is bytes, and a
float in (0, 1] is that fraction of the available RAM.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _segyrs
from ._volume import _file_order

if TYPE_CHECKING:
    from ._volume import Volume


def _flat(vol: Volume) -> np.ndarray:
    return np.ascontiguousarray(vol._traces.reshape(-1), dtype=np.int64)


def _whole_traces(vol: Volume, what: str) -> None:
    v = vol._vsel
    if not (isinstance(v, range) and v == range(vol._src.ns)):
        raise ValueError(f"{what} works on whole traces; select traces, not a {vol._src.vertical}")


def _whole_file(vol: Volume, what: str) -> None:
    if not vol._root:
        raise ValueError(f"{what} works on whole files; write the selection with to_segy first")


def _predicate(field: str, key: Any) -> tuple:
    if isinstance(key, slice):
        if key.step is not None:
            raise ValueError(f"{field}: a filter range has no step")
        key = (key.start, key.stop)
    if isinstance(key, tuple):
        lo, hi = key
        return (field, None if lo is None else float(lo), None if hi is None else float(hi), None)
    values = np.asarray(key, dtype=np.float64).reshape(-1)
    return (field, None, None, [float(x) for x in values])


def filter(vol: Volume, **conditions: Any) -> Volume:
    """Body of :meth:`Volume.filter`."""
    from ._volume import Volume

    if not conditions:
        raise ValueError("filter needs at least one field=condition")
    preds = [_predicate(f, k) for f, k in conditions.items()]
    hits = _segyrs.filter(vol._src.file, preds)
    flat = _file_order(vol)
    kept = flat[np.isin(flat, hits)]
    kept.setflags(write=False)
    return Volume._make(
        vol._src, ("trace",), kept, {"trace": kept}, vol._vsel, {}, dict(vol._attrs), False
    )


def resort(
    vol: Volume,
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
    """Body of :meth:`Volume.resort`."""
    _whole_file(vol, "resort")
    keys = [by] if isinstance(by, str) else list(by)
    desc = [descending] * len(keys) if isinstance(descending, bool) else list(descending)
    if len(desc) != len(keys):
        raise ValueError("descending needs one bool per key")
    return _segyrs.resort(
        vol._src.file,
        os.fspath(path),
        [(k, desc[i]) for i, k in enumerate(keys)],
        strategy=strategy,
        sorting_code=sorting_code,
        tmp_dir=None if tmp_dir is None else os.fspath(tmp_dir),
        budget=budget,
        sync=sync,
    )


def slice_horizon(
    vol: Volume, horizon: Any, *, above: int = 0, below: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Body of :meth:`Volume.slice_horizon`."""
    tshape = vol._traces.shape
    h = np.asarray(horizon, dtype=np.float64)
    if h.shape != tshape:
        h = np.broadcast_to(h, tshape)
    axis = vol._src.axis()
    dt = float(vol._src.file.sample_interval_us)
    pos = (h.reshape(-1) - float(axis[0])) / dt if len(axis) else h.reshape(-1)
    index = np.full(pos.shape, np.iinfo(np.int64).min, dtype=np.int64)
    ok = np.isfinite(pos)
    index[ok] = np.rint(pos[ok]).astype(np.int64)
    data, valid = _segyrs.slice_horizon(vol._src.file, _flat(vol), index, int(above), int(below))
    width = data.shape[1]
    return data.reshape(*tshape, width), valid.reshape(*tshape, width)


def to_regular(
    vol: Volume,
    path: str | os.PathLike[str] | None = None,
    *,
    fill: float = 0.0,
    budget: int | float | None = None,
    sync: bool = True,
) -> Any:
    """Body of :meth:`Volume.to_regular`."""
    _whole_traces(vol, "to_regular")
    traces = _flat(vol)
    if path is not None:
        cells, filled = _segyrs.to_regular_file(
            vol._src.file, os.fspath(path), traces, float(fill), budget=budget, sync=sync
        )
        return {"cells": cells, "filled": filled}
    samples, shape, mask, il, xl, off = _segyrs.to_regular(
        vol._src.file, traces, float(fill), budget
    )
    shape = tuple(shape)
    coords: dict[str, np.ndarray] = {"inline": il, "crossline": xl}
    if len(shape) == 4:
        coords["offset"] = off
    coords[vol._src.vertical] = vol._src.axis()
    return samples.reshape(shape), mask.reshape(shape[:-1]), coords


def concat(
    volumes: Sequence[Any],
    path: str | os.PathLike[str],
    *,
    budget: int | float | None = None,
    sync: bool = True,
) -> int:
    """Write whole files one after the other into ``path``.

    Nothing is cast. The output keeps the first file's textual and binary
    headers, and trace bytes are copied as they are on disk.

    Parameters
    ----------
    volumes : sequence of Volume or path
        Opened Volumes (not selections) or paths, in output order.
    path : str or os.PathLike
        Destination file.
    budget : int or float, optional
        RAM for the streaming copy: ``None`` is half the available RAM, an
        int is bytes, a float in (0, 1] a fraction of the available RAM.
    sync : bool, default True
        Flush the data and the directory before returning.

    Returns
    -------
    int
        Number of traces written.

    Raises
    ------
    FormatChangeError
        Sample format, interval, sample count, byte order or trace-header
        length differ between the files. Nothing is written.
    ValueError
        No file, or a selection instead of a whole file.
    SegyrsIOError
        When the file system refuses the write.

    Examples
    --------
    >>> segyrs.concat(["survey.sgy", "survey.sgy"], "twice.sgy")
    50
    >>> segyrs.validate("twice.sgy")["n_traces"]
    50
    """
    from ._volume import Volume

    files = []
    for v in volumes:
        if isinstance(v, Volume):
            _whole_file(v, "concat")
            files.append(v._src.file)
        else:
            files.append(_segyrs.File(os.fspath(v)))
    if not files:
        raise ValueError("concat needs at least one file")
    return _segyrs.concat(files, os.fspath(path), budget=budget, sync=sync)


def stats(vol: Volume) -> dict[str, float]:
    """Body of :meth:`Volume.stats`."""
    _whole_traces(vol, "stats")
    return _segyrs.stats(vol._src.file, _flat(vol))


def rms(vol: Volume) -> np.ndarray:
    """Body of :meth:`Volume.rms`."""
    _whole_traces(vol, "rms")
    return _segyrs.rms(vol._src.file, _flat(vol)).reshape(vol._traces.shape)


def clip(
    vol: Volume,
    lo: float,
    hi: float,
    path: str | os.PathLike[str],
    *,
    budget: int | float | None = None,
    sync: bool = True,
) -> int:
    """Body of :meth:`Volume.clip`."""
    _whole_traces(vol, "clip")
    return _segyrs.clip(
        vol._src.file,
        os.fspath(path),
        float(lo),
        float(hi),
        _file_order(vol),
        budget=budget,
        sync=sync,
    )
