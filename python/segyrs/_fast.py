"""Lazy header columns and pointwise trace selection (``Volume.isel_points``/``sel_points``)."""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._volume import Volume

_LAZY = object()
_ALIASES = {"iline": "inline", "il": "inline", "xline": "crossline", "xl": "crossline"}


class LazyHeaders(dict):
    """A dict of header columns that decodes a column on first access.

    Every name is a key from the start, so ``in``, ``len`` and iteration cost
    nothing. ``h[name]`` decodes that one column (in Rust, on all cores) and
    keeps it. ``values()``, ``items()``, ``copy()``, ``==`` and ``repr`` decode
    every missing column in one parallel scan first.
    """

    __slots__ = ("_fetch",)

    def __init__(self, names: Iterable[str], fetch: Callable[[list[str]], dict[str, np.ndarray]]):
        super().__init__(dict.fromkeys(names, _LAZY))
        self._fetch = fetch

    def _load(self, names: Iterable[str]) -> None:
        get = super().__getitem__
        missing = [k for k in names if get(k) is _LAZY]
        if missing:
            super().update(self._fetch(missing))

    def _load_all(self) -> None:
        self._load(list(super().keys()))

    def __getitem__(self, key: str) -> np.ndarray:
        value = super().__getitem__(key)
        if value is _LAZY:
            self._load([key])
            value = super().__getitem__(key)
        return value

    def __iter__(self):
        # Overriding __iter__ also keeps dict(h) and {**h} off CPython's fast
        # path, which would copy the placeholders: they go through __getitem__.
        return super().__iter__()

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default  # noqa: SIM401 (this is get)

    def values(self):
        self._load_all()
        return super().values()

    def items(self):
        self._load_all()
        return super().items()

    def copy(self) -> dict[str, np.ndarray]:
        self._load_all()
        return dict(super().items())

    def pop(self, key, *default):
        if key in self:
            self._load([key])
        return super().pop(key, *default)

    def popitem(self):
        self._load_all()
        return super().popitem()

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        return super().setdefault(key, default)

    def __eq__(self, other: object) -> bool:
        self._load_all()
        if isinstance(other, LazyHeaders):
            other._load_all()
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]

    def __or__(self, other):
        return self.copy() | other

    def __ror__(self, other):
        return dict(other) | self.copy()

    def __reduce__(self):
        return (dict, (self.copy(),))

    def __repr__(self) -> str:
        self._load_all()
        return super().__repr__()


class HeaderCache:
    """Decoded header columns of one file, filled on demand (one parallel scan per call)."""

    def __init__(self, file, fields, scanned: dict[str, np.ndarray], line_bytes: dict[str, int]):
        self.file = file
        self.fields = {name: (byte, dtype) for name, byte, dtype in fields}
        self.cols: dict[str, np.ndarray] = {}
        # Open already decoded these words. Reuse one only when the requested field
        # is the same word with the same dtype as the scanned one.
        scanned_spec = {name: (byte, "int32") for name, byte in line_bytes.items()}
        for name, byte, dtype in file.standard_fields():
            if name not in scanned_spec:
                scanned_spec[name] = (byte, dtype)
        for name, col in scanned.items():
            want = self.fields.get(name)
            if want is not None and want == scanned_spec.get(name) and col.dtype == want[1]:
                col = col.view()
                col.setflags(write=False)
                self.cols[name] = col

    def names(self) -> list[str]:
        return list(self.fields)

    def get(self, names: Iterable[str]) -> dict[str, np.ndarray]:
        names = list(names)
        missing = [n for n in names if n not in self.cols]
        if missing:
            cols = self.file.header_columns([(n, *self.fields[n]) for n in missing])
            for name, col in cols.items():
                col.setflags(write=False)
                self.cols[name] = col
        return {n: self.cols[n] for n in names}


def plain(value: Any) -> Any:
    """A numpy scalar as the Python number it holds, for messages."""
    return value.item() if isinstance(value, np.generic) else value


def did_you_mean(dim: str, vol: Volume) -> str:
    """``"; did you mean 'inline'?"`` for a near miss of a dim name, else ``""``."""
    names = list(dict.fromkeys((*vol.dims, "inline", "crossline", "offset", "trace")))
    alias = _ALIASES.get(dim.lower()) if isinstance(dim, str) else None
    close = [alias] if alias else difflib.get_close_matches(str(dim), names, n=1)
    return f"; did you mean {close[0]!r}?" if close else ""


def label_error(dim: str, key: Any, labels: np.ndarray) -> KeyError:
    """``KeyError`` naming the missing label and the labels that exist."""
    msg = f"{dim}={plain(key)!r} is not a label"
    if not len(labels):
        return KeyError(f"{msg}; {dim} has no labels")
    lo, hi = plain(labels.min()), plain(labels.max())
    steps = np.unique(np.diff(np.sort(labels))) if len(labels) > 1 else []
    every = f" step {plain(steps[0])}" if len(steps) == 1 else ""
    return KeyError(f"{msg}; the {len(labels)} {dim} labels run {lo}..{hi}{every}")


def _vector_labels(labels: np.ndarray, keys: np.ndarray, dim: str) -> np.ndarray:
    """Exact label lookup for many keys at once: the first position of each label."""
    order = np.argsort(labels, kind="stable")
    ranked = labels[order]
    at = np.searchsorted(ranked, keys, side="left")
    at = np.minimum(at, max(len(ranked) - 1, 0))
    hit = ranked[at] == keys if len(ranked) else np.zeros(keys.shape, dtype=bool)
    if not np.all(hit):
        raise label_error(dim, keys[~hit].ravel()[0], labels)
    return order[at].astype(np.int64)


def label_positions_many(labels: np.ndarray, keys: Any, method: str | None, dim: str, one):
    """Positions of an array of labels. ``one`` is the scalar lookup for ``method``."""
    keys = np.asarray(keys).ravel()
    if method is None and keys.dtype.kind in "iuf" and labels.dtype.kind in "iuf":
        return _vector_labels(labels, keys, dim)
    return np.array([one(labels, k, method, dim) for k in keys], np.int64)


def _points(vol: Volume, pos: dict[str, np.ndarray]) -> Volume:
    from ._volume import Volume

    lens = {len(p) for p in pos.values()}
    if len(lens) > 1:
        raise ValueError(f"point lists differ in length: { {d: len(p) for d, p in pos.items()} }")
    tdims = vol._tdims
    axes = [tdims.index(d) for d in pos]
    rest = [d for d in tdims if d not in pos]
    traces = np.moveaxis(vol._traces, axes, list(range(len(axes))))
    traces = traces[tuple(pos[d] for d in pos)]
    n = traces.shape[0]
    labels = {"point": np.arange(n, dtype=np.int64)}
    labels.update({d: vol._labels[d] for d in rest})
    for v in labels.values():
        v.setflags(write=False)
    return Volume._make(
        vol._src,
        ("point", *rest),
        traces,
        labels,
        vol._vsel,
        dict(vol._scalars),
        dict(vol._attrs),
        False,
    )


def _check_keys(vol: Volume, keys: dict[str, Any]) -> None:
    if not keys:
        raise ValueError("give at least one dim with a list of points")
    for dim in keys:
        if dim not in vol._tdims:
            from ._volume import _LINE_DIMS, GeometryError

            if dim in _LINE_DIMS or dim == "trace":
                raise GeometryError(
                    f"{dim!r} is not a trace dim of this Volume (dims {vol.dims}); "
                    f"on the (trace, time) layout use sel({dim}=...)"
                )
            raise ValueError(
                f"{dim!r} is not a trace dim (dims {vol.dims}){did_you_mean(dim, vol)}"
            )


def isel_points(vol: Volume, keys: dict[str, Any]) -> Volume:
    """Implementation of :meth:`Volume.isel_points`."""
    from ._volume import _positions

    if "trace" in keys and "trace" not in vol._tdims:
        if not vol._root or len(keys) > 1:
            raise ValueError(
                "trace= takes file trace numbers on the Volume segyrs.open returned, alone"
            )
        traces = _positions(np.atleast_1d(keys["trace"]), vol._src.n, "trace")
        from ._volume import Volume

        labels = {"point": np.arange(len(traces), dtype=np.int64)}
        labels["point"].setflags(write=False)
        return Volume._make(
            vol._src,
            ("point",),
            traces,
            labels,
            vol._vsel,
            dict(vol._scalars),
            dict(vol._attrs),
            False,
        )
    _check_keys(vol, keys)
    pos = {}
    for dim, key in keys.items():
        n = vol._traces.shape[vol._tdims.index(dim)]
        pos[dim] = np.atleast_1d(_positions(np.atleast_1d(key), n, dim))
    return _points(vol, pos)


def sel_points(vol: Volume, keys: dict[str, Any], method: str | None) -> Volume:
    """Implementation of :meth:`Volume.sel_points`."""
    from ._volume import _label_positions

    _check_keys(vol, keys)
    pos = {}
    for dim, key in keys.items():
        if isinstance(key, slice):
            raise TypeError(f"{dim}: sel_points takes a list of labels, not a slice")
        pos[dim] = np.atleast_1d(
            _label_positions(vol._labels[dim], np.atleast_1d(key), method, dim)
        )
    return _points(vol, pos)
