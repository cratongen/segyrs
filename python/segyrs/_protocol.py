"""Protocol version 1 for ``seismic_volume`` and ``prestack_volume``.

``Volume.__minetypes__`` builds the dict. ``segyrs.create`` reads one. Neither
path imports another library. The sample axis stays in the producer's unit on
the way out (``"us"``). On the way in, a time axis in ``"us"``, ``"ms"`` or
``"s"`` becomes the integer microsecond interval SEG-Y stores.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from . import _segyrs

_DATA = {np.dtype(k) for k in ("float32", "float64", "int8", "int16", "int32")}
_TIME_TO_US = {"us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}
_SCALARS = (1, -1, -10, -100, -1000, -10000, 10, 100, 1000, 10000)
_I32 = 2**31 - 1


class Prepared(NamedTuple):
    data: np.ndarray
    interval_us: int
    inline: Any
    crossline: Any
    offset: Any
    headers: dict[str, Any]
    length_unit: str | None


def read_protocol(obj: Any) -> dict | None:
    """The protocol dict, or None when ``obj`` is raw array data."""
    if isinstance(obj, np.ndarray):
        return None
    method = getattr(obj, "__minetypes__", None)
    if callable(method):
        got = method()
        if not isinstance(got, dict):
            raise _segyrs.SegyrsValueError("__minetypes__ must return a dict")
        return got
    if isinstance(obj, Mapping):
        return dict(obj)
    return None


def prepare(obj: Any, interval_us: int | float | None) -> Prepared | None:
    spec = read_protocol(obj)
    if spec is None:
        return None
    return _consume(spec, interval_us)


def produce(vol: Any, budget: int | float | None) -> dict:
    """Dict for ``vol``. Checks ``budget`` before decoding samples."""
    kind, layout = _kind(vol)
    if any(size < 1 for size in vol.shape):
        raise _segyrs.SegyrsValueError(f"data: each axis length must be >= 1, not {vol.shape}")
    if vol.dtype not in _DATA:
        raise _segyrs.SegyrsValueError(
            f"data: dtype {vol.dtype} is not float32, float64, int8, int16 or int32"
        )
    samples = _f64(vol._src.axis()[vol._vsel_index()])
    _finite(samples, "samples")
    _monotonic(samples, "samples", increasing=True)
    _segyrs.check_budget(vol.nbytes, budget)
    data = vol.numpy()
    if not data.flags.c_contiguous or not _native(data):
        data = np.ascontiguousarray(data)
    data.setflags(write=False)
    mask = _indexed(vol._src.trace_coords()["trace_mask"], vol._traces)
    if mask.dtype != np.bool_:
        mask = np.ascontiguousarray(mask, dtype=bool)
        mask.setflags(write=False)
    _no_live_inf(data, mask)
    unit = {"meters": "m", "feet": "ft"}.get(vol.attrs.get("measurement_system"))
    out: dict[str, Any] = {
        "protocol": 1,
        "kind": kind,
        "columns": {},
        "units": {},
        "categories": {},
        "crs": None,
        "length_unit": unit,
    }
    if layout is not None:
        out["layout"] = layout
    out["data"] = data
    if layout == "traces":
        out["inlines"] = _per_trace(vol, "inline")
        out["crosslines"] = _per_trace(vol, "crossline")
        out["offsets"] = _per_trace_f64(vol, "offset")
    else:
        out["inlines"] = _axis_i64(vol, "inline")
        out["crosslines"] = _axis_i64(vol, "crossline")
        _monotonic(out["inlines"], "inlines", increasing=False)
        _monotonic(out["crosslines"], "crosslines", increasing=False)
        if layout == "grid":
            out["offsets"] = _axis_f64(vol, "offset")
            _finite(out["offsets"], "offsets")
            _monotonic(out["offsets"], "offsets", increasing=False)
    out["samples"] = samples
    out["trace_mask"] = mask
    out["affine"] = None
    cdp_x, cdp_y = _cdp(vol, layout)
    _live_finite(cdp_x, _cdp_mask(mask, layout), "cdp_x")
    _live_finite(cdp_y, _cdp_mask(mask, layout), "cdp_y")
    out["cdp_x"] = cdp_x
    out["cdp_y"] = cdp_y
    if layout == "traces":
        out["source_coords"] = _xyz(vol, "source_x", "source_y", "source_elev")
        out["receiver_coords"] = _xyz(vol, "group_x", "group_y", "group_elev")
        _live_finite(out["source_coords"], mask, "source_coords")
        _live_finite(out["receiver_coords"], mask, "receiver_coords")
    elif layout == "grid":
        out["source_coords"] = None
        out["receiver_coords"] = None
    out["vertical_domain"] = "time"
    out["sample_unit"] = "us"
    out["attrs"] = {}
    return out


def _consume(spec: dict, interval_us: int | float | None) -> Prepared:
    protocol = _key(spec, "protocol")
    if isinstance(protocol, bool) or protocol != 1:
        raise _segyrs.SegyrsValueError(f"protocol: segyrs accepts protocol 1, not {protocol!r}")
    kind = _key(spec, "kind")
    if kind not in ("seismic_volume", "prestack_volume"):
        raise _segyrs.SegyrsValueError(
            f"kind: segyrs.create accepts seismic_volume or prestack_volume, not {kind!r}"
        )
    for name in ("columns", "units", "categories"):
        value = _key(spec, name)
        if not isinstance(value, dict) or value:
            raise _segyrs.SegyrsValueError(f"{name}: a {kind} carries an empty dict")
    _crs(_key(spec, "crs"))
    length_unit = _length_unit(_key(spec, "length_unit"))
    _attrs(_key(spec, "attrs"))
    domain = _key(spec, "vertical_domain")
    unit = _key(spec, "sample_unit")
    _domain(domain, unit)
    layout = None
    if kind == "prestack_volume":
        layout = _key(spec, "layout")
        if layout not in ("grid", "traces"):
            raise _segyrs.SegyrsValueError(f"layout: {layout!r} is not 'grid' or 'traces'")
    data = _array(spec, "data", None, None)
    if data.dtype not in _DATA:
        raise _segyrs.SegyrsValueError(
            f"data: dtype {data.dtype} is not float32, float64, int8, int16 or int32"
        )
    if data.ndim != (4 if layout == "grid" else 3 if layout is None else 2):
        raise _segyrs.SegyrsValueError(
            f"data: shape {data.shape} is not the {kind} {layout or 'post-stack'} shape"
        )
    if any(size < 1 for size in data.shape):
        raise _segyrs.SegyrsValueError(f"data: each axis length must be >= 1, not {data.shape}")
    n_s = data.shape[-1]
    samples = _array(spec, "samples", np.float64, (n_s,))
    _finite(samples, "samples")
    _monotonic(samples, "samples", increasing=True)
    interval, t0 = _interval(samples, unit, interval_us)
    if layout == "grid":
        n_il, n_xl, n_off, _ = data.shape
        inlines = _array(spec, "inlines", np.int64, (n_il,))
        crosslines = _array(spec, "crosslines", np.int64, (n_xl,))
        offsets = _array(spec, "offsets", np.float64, (n_off,))
        mask = _array(spec, "trace_mask", np.bool_, (n_il, n_xl, n_off))
        _finite(offsets, "offsets")
        _monotonic(offsets, "offsets", increasing=False)
        offsets = _whole_numbers(offsets, "offsets")
        inline, crossline, offset = inlines, crosslines, offsets
    elif layout == "traces":
        n_tr = data.shape[0]
        inlines = _optional_1d(spec, "inlines", np.int64, n_tr)
        crosslines = _optional_1d(spec, "crosslines", np.int64, n_tr)
        offsets = _optional_1d(spec, "offsets", np.float64, n_tr)
        mask = _array(spec, "trace_mask", np.bool_, (n_tr,))
        if offsets is not None:
            _finite(offsets, "offsets")
            offsets = _whole_numbers(offsets, "offsets")
        inline, crossline, offset = inlines, crosslines, offsets
    else:
        n_il, n_xl, _ = data.shape
        inlines = _array(spec, "inlines", np.int64, (n_il,))
        crosslines = _array(spec, "crosslines", np.int64, (n_xl,))
        mask = _array(spec, "trace_mask", np.bool_, (n_il, n_xl))
        inline, crossline, offset = inlines, crosslines, None
    if inlines is not None:
        _fits_i32(inlines, "inlines")
    if crosslines is not None:
        _fits_i32(crosslines, "crosslines")
    if layout != "traces":
        _monotonic(inlines, "inlines", increasing=False)
        _monotonic(crosslines, "crosslines", increasing=False)
    _no_live_inf(data, mask)
    headers = _headers(spec, data.shape, layout, mask, t0)
    return Prepared(data, interval, inline, crossline, offset, headers, length_unit)


def _kind(vol: Any) -> tuple[str, str | None]:
    if isinstance(vol._vsel, int):
        raise _segyrs.SegyrsValueError(
            f"data: __minetypes__ needs a sample axis; this selection has dims {vol.dims}"
        )
    tdims = vol._tdims
    if tdims == ("inline", "crossline"):
        return "seismic_volume", None
    if tdims == ("inline", "crossline", "offset"):
        return "prestack_volume", "grid"
    if tdims == ("trace",):
        return "prestack_volume", "traces"
    raise _segyrs.SegyrsValueError(
        f"kind: dims {vol.dims} are not a seismic_volume or a prestack_volume"
    )


def _headers(spec: dict, shape: tuple[int, ...], layout: str | None, mask: np.ndarray, t0: int):
    headers: dict[str, Any] = {}
    if t0:
        stored, scalar = _fit_i16(t0 / 1000.0, t0)
        headers["delay"] = np.full(int(np.prod(shape[:-1])), stored, dtype=np.int64)
        headers["time_scalar"] = scalar
    trace_id = np.where(mask, 1, 2).astype(np.int64, copy=False)
    headers["trace_id"] = trace_id
    cdp_x, cdp_y = _consume_cdp(spec, shape, layout)
    source = receiver = None
    if layout == "traces":
        source = _optional_array(spec, "source_coords", np.float64, (shape[0], 3))
        receiver = _optional_array(spec, "receiver_coords", np.float64, (shape[0], 3))
        if source is not None:
            _finite(source, "source_coords")
        if receiver is not None:
            _finite(receiver, "receiver_coords")
        if _key(spec, "affine") is not None:
            raise _segyrs.SegyrsValueError("affine: a traces layout has affine None")
    elif layout == "grid":
        for name in ("source_coords", "receiver_coords"):
            if _key(spec, name) is not None:
                raise _segyrs.SegyrsValueError(f"{name}: a pre-stack grid carries {name} as None")
    named = [("cdp_x", cdp_x), ("cdp_y", cdp_y)]
    if source is not None:
        named += [("source_x", source[:, 0]), ("source_y", source[:, 1])]
        headers["source_elev"] = source[:, 2]
    if receiver is not None:
        named += [("group_x", receiver[:, 0]), ("group_y", receiver[:, 1])]
        headers["group_elev"] = receiver[:, 2]
    horiz = [(name, values) for name, values in named if values is not None]
    if horiz:
        headers["scalco"] = _scalar_for(horiz)
        for name, values in horiz:
            headers[name] = values
    elev = [
        (name, headers[name])
        for name in ("source_elev", "group_elev")
        if isinstance(headers.get(name), np.ndarray)
    ]
    if elev:
        headers["scalel"] = _scalar_for(elev)
    return headers


def _consume_cdp(spec, shape, layout):
    affine = _affine(_key(spec, "affine"))
    if layout == "grid":
        n_il, n_xl, n_off, _ = shape
        bin_shape = (n_il, n_xl)
    elif layout == "traces":
        n_il = n_xl = n_off = None
        bin_shape = (shape[0],)
    else:
        n_il, n_xl, _ = shape
        n_off = None
        bin_shape = (n_il, n_xl)
    x = _key(spec, "cdp_x")
    y = _key(spec, "cdp_y")
    if (x is None) ^ (y is None):
        raise _segyrs.SegyrsValueError("cdp_x and cdp_y: set both, or set neither")
    if affine is not None and x is not None:
        raise _segyrs.SegyrsValueError("affine: set this or the CDP pair, not both")
    if affine is None and x is None:
        return None, None
    if affine is not None:
        if layout == "traces":
            raise _segyrs.SegyrsValueError("affine: a traces layout has affine None")
        x, y = _affine_grid(affine, n_il, n_xl)
    else:
        x = _array_value(x, "cdp_x", np.float64, bin_shape)
        y = _array_value(y, "cdp_y", np.float64, bin_shape)
        x, y = _finite_live_pair(x, y, spec, shape, layout)
    if n_off is not None:
        x = np.broadcast_to(x[:, :, None], (n_il, n_xl, n_off))
        y = np.broadcast_to(y[:, :, None], (n_il, n_xl, n_off))
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _affine_grid(affine, n_il, n_xl):
    x0, ax_il, ax_xl, y0, ay_il, ay_xl = affine
    ii = np.arange(n_il, dtype=np.float64)[:, None]
    jj = np.arange(n_xl, dtype=np.float64)[None, :]
    return x0 + ii * ax_il + jj * ax_xl, y0 + ii * ay_il + jj * ay_xl


def _finite_live_pair(x, y, spec, shape, layout):
    mask = spec["trace_mask"]
    live = mask if layout != "grid" else mask.any(axis=-1)
    for name, values in (("cdp_x", x), ("cdp_y", y)):
        bad = ~np.isfinite(values)
        if np.any(bad & live):
            raise _segyrs.SegyrsValueError(f"{name}: a live trace is not finite")
        if np.any(bad):
            values = np.array(values, copy=True)
            values[bad] = 0.0
            if name == "cdp_x":
                x = values
            else:
                y = values
    return x, y


def _interval(samples: np.ndarray, unit: str, given: int | float | None) -> tuple[int, int]:
    factor = _TIME_TO_US[unit]
    if samples.size == 1:
        if given is None:
            raise _segyrs.SegyrsValueError(
                "sample_interval_us: one sample does not determine the interval"
            )
        step = _whole_us(given)
        us0 = samples[0] * factor
        nearest = round(float(us0))
        if abs(us0 - nearest) > 1e-3:
            raise _segyrs.SegyrsValueError(
                "samples: the first sample is not a whole number of microseconds"
            )
        return step, int(nearest)
    us = samples * factor
    nearest = np.rint(us)
    if float(np.max(np.abs(us - nearest))) > 1e-3:
        raise _segyrs.SegyrsValueError("samples: the step is not a whole number of microseconds")
    ticks = nearest.astype(np.int64)
    diffs = np.diff(ticks)
    if not np.all(diffs == diffs[0]) or int(diffs[0]) <= 0:
        raise _segyrs.SegyrsValueError("samples: the axis is not a uniform step in time")
    step = int(diffs[0])
    if given is not None and _whole_us(given) != step:
        raise _segyrs.SegyrsValueError(
            f"sample_interval_us: {given} does not match the sample axis step of {step} us"
        )
    return step, int(ticks[0])


def _whole_us(interval: int | float) -> int:
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
        raise _segyrs.SegyrsValueError(
            f"sample_interval_us: {interval!r} is not a whole number of microseconds"
        )
    if int(interval) != interval or int(interval) <= 0:
        raise _segyrs.SegyrsValueError(
            f"sample_interval_us: {interval} is not a positive whole number of microseconds"
        )
    return int(interval)


def _domain(domain: Any, unit: Any) -> None:
    if domain not in ("time", "depth"):
        raise _segyrs.SegyrsValueError(f"vertical_domain: {domain!r} is not 'time' or 'depth'")
    if domain == "time" and unit not in _TIME_TO_US:
        raise _segyrs.SegyrsValueError(f"sample_unit: {unit!r} is not 'us', 'ms' or 's'")
    if domain == "depth":
        if unit not in ("m", "ft"):
            raise _segyrs.SegyrsValueError(f"sample_unit: {unit!r} is not 'm' or 'ft'")
        raise _segyrs.SegyrsValueError(
            "sample_unit: segyrs stores the sample interval as integer microseconds"
            f" and does not relabel {unit!r} as time"
        )


def _cdp(vol, layout: str | None):
    coords = vol._src.trace_coords()
    x = _indexed(coords["cdp_x"], vol._traces)
    y = _indexed(coords["cdp_y"], vol._traces)
    if layout == "grid":
        x = _bin_coord(x, "cdp_x")
        y = _bin_coord(y, "cdp_y")
    return _f64(x), _f64(y)


def _bin_coord(values: np.ndarray, key: str) -> np.ndarray:
    first = values[:, :, 0]
    if values.shape[-1] == 1:
        return first
    rest = values[:, :, 1:]
    base = first[:, :, None]
    same = rest == base
    both_nan = np.isnan(rest) & np.isnan(base)
    if not np.all(same | both_nan):
        raise _segyrs.SegyrsValueError(f"{key}: a pre-stack grid bin has more than one {key}")
    return np.ascontiguousarray(first)


def _cdp_mask(mask: np.ndarray, layout: str | None) -> np.ndarray:
    if layout == "grid":
        return np.any(mask, axis=-1)
    return mask


def _per_trace(vol, name: str) -> np.ndarray | None:
    if name not in vol._src.trace_coords():
        return None
    return _i64(_indexed(vol._src.trace_coords()[name], vol._traces).reshape(-1))


def _per_trace_f64(vol, name: str) -> np.ndarray | None:
    if name not in vol._src.trace_coords():
        return None
    return _f64(_indexed(vol._src.trace_coords()[name], vol._traces).reshape(-1))


def _axis_i64(vol, name: str) -> np.ndarray:
    return _i64(vol._labels[name])


def _axis_f64(vol, name: str) -> np.ndarray:
    return _f64(vol._labels[name])


def _xyz(vol, xname: str, yname: str, zname: str) -> np.ndarray:
    coords = vol._src.trace_coords()
    columns = [
        _f64(_indexed(coords[name], vol._traces).reshape(-1)) for name in (xname, yname, zname)
    ]
    out = np.column_stack(columns)
    out.setflags(write=False)
    return out


def _indexed(full: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Values of ``full`` at ``index``. A contiguous increasing run is a view."""
    index = np.asarray(index)
    flat = np.ascontiguousarray(index).reshape(-1)
    if flat.size == 0:
        out = np.empty(index.shape, dtype=full.dtype)
    elif flat.size == 1 or np.all(flat[1:] - flat[:-1] == 1):
        start, stop = int(flat[0]), int(flat[-1]) + 1
        if 0 <= start <= stop <= len(full):
            out = full[start:stop].reshape(index.shape)
            if not out.flags.c_contiguous or not _native(out):
                out = np.ascontiguousarray(out)
        else:
            out = np.ascontiguousarray(full[index])
    else:
        out = np.ascontiguousarray(full[index])
    out.setflags(write=False)
    return out


def _i64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype == np.int64 and values.flags.c_contiguous and _native(values):
        if values.flags.writeable:
            values = values.view()
            values.setflags(write=False)
        return values
    out = np.ascontiguousarray(values, dtype=np.int64)
    out.setflags(write=False)
    return out


def _f64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype == np.float64 and values.flags.c_contiguous and _native(values):
        if values.flags.writeable:
            values = values.view()
            values.setflags(write=False)
        return values
    out = np.ascontiguousarray(values, dtype=np.float64)
    out.setflags(write=False)
    return out


def _native(arr: np.ndarray) -> bool:
    order = arr.dtype.byteorder
    if order in ("=", "|"):
        return True
    native = "<" if sys.byteorder == "little" else ">"
    return order == native


def _key(spec: dict, key: str) -> Any:
    if key not in spec:
        raise _segyrs.SegyrsValueError(f"{key}: missing")
    return spec[key]


def _array(spec: dict, key: str, dtype, shape) -> np.ndarray:
    return _array_value(_key(spec, key), key, dtype, shape)


def _array_value(value, key: str, dtype, shape) -> np.ndarray:
    if dtype is None:
        if not isinstance(value, np.ndarray):
            raise _segyrs.SegyrsValueError(f"{key}: expected a numpy array")
        if not value.flags.c_contiguous:
            raise _segyrs.SegyrsValueError(f"{key}: the array is not C-contiguous")
        if not _native(value):
            raise _segyrs.SegyrsValueError(f"{key}: the array is not native-endian")
        return value
    if not isinstance(value, np.ndarray):
        raise _segyrs.SegyrsValueError(f"{key}: expected a numpy array")
    if value.dtype != np.dtype(dtype):
        raise _segyrs.SegyrsValueError(f"{key}: dtype {value.dtype} is not {np.dtype(dtype).name}")
    if shape is not None and value.shape != shape:
        raise _segyrs.SegyrsValueError(f"{key}: shape {value.shape} is not {shape}")
    if not value.flags.c_contiguous:
        raise _segyrs.SegyrsValueError(f"{key}: the array is not C-contiguous")
    if not _native(value):
        raise _segyrs.SegyrsValueError(f"{key}: the array is not native-endian")
    if np.dtype(dtype) == np.bool_ and not np.all(value.view(np.uint8) <= 1):
        raise _segyrs.SegyrsValueError(f"{key}: bool bytes must be 0x00 or 0x01")
    return value


def _optional_1d(spec, key, dtype, n: int):
    value = _key(spec, key)
    if value is None:
        return None
    return _array_value(value, key, dtype, (n,))


def _optional_array(spec, key, dtype, shape):
    value = _key(spec, key)
    if value is None:
        return None
    return _array_value(value, key, dtype, shape)


def _affine(value):
    if value is None:
        return None
    if isinstance(value, str | bytes | Mapping):
        raise _segyrs.SegyrsValueError("affine: expected 6 finite floats, or None")
    try:
        seq = list(value)
    except TypeError as exc:
        raise _segyrs.SegyrsValueError("affine: expected 6 finite floats, or None") from exc
    if len(seq) != 6 or any(isinstance(v, bool | np.bool_) for v in seq):
        raise _segyrs.SegyrsValueError("affine: expected 6 finite floats, or None")
    try:
        arr = np.asarray(seq, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _segyrs.SegyrsValueError("affine: expected 6 finite floats, or None") from exc
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise _segyrs.SegyrsValueError("affine: expected 6 finite floats, or None")
    return tuple(float(v) for v in arr)


def _crs(value) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise _segyrs.SegyrsValueError("crs: an empty string is not a CRS; pass None")


def _length_unit(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _segyrs.SegyrsValueError("length_unit: an empty unit is not allowed; pass None")
    return value.strip()


def _attrs(value) -> None:
    if not isinstance(value, dict) or not _json_ok(value):
        raise _segyrs.SegyrsValueError("attrs: expected a JSON object with no NaN and no Inf")


def _json_ok(value) -> bool:
    if value is None or isinstance(value, str | bool):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, list | tuple):
        return all(_json_ok(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_ok(item) for key, item in value.items())
    return False


def _finite(values: np.ndarray, key: str) -> None:
    if not np.all(np.isfinite(values)):
        raise _segyrs.SegyrsValueError(f"{key}: values must be finite")


def _monotonic(values: np.ndarray, key: str, *, increasing: bool) -> None:
    if values.size < 2:
        return
    up = bool(np.all(values[1:] > values[:-1]))
    if increasing:
        ok, word = up, "strictly increasing"
    else:
        ok = up or bool(np.all(values[1:] < values[:-1]))
        word = "strictly monotonic"
    if not ok:
        raise _segyrs.SegyrsValueError(f"{key}: values are not {word}")


def _fits_i32(values: np.ndarray, key: str) -> None:
    if np.any((values < -_I32 - 1) | (values > _I32)):
        raise _segyrs.SegyrsValueError(f"{key}: a value does not fit an int32 header")


def _whole_numbers(values: np.ndarray, key: str) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise _segyrs.SegyrsValueError(f"{key}: values must be finite")
    nearest = np.rint(values)
    if float(np.max(np.abs(values - nearest))) > 1e-6:
        raise _segyrs.SegyrsValueError(
            f"{key}: {key} is not a whole number, and the SEG-Y offset word is an int32"
        )
    if np.any((nearest < -_I32 - 1) | (nearest > _I32)):
        raise _segyrs.SegyrsValueError(f"{key}: a value does not fit an int32 header")
    return np.ascontiguousarray(nearest, dtype=np.int64)


def _live_finite(values: np.ndarray, mask: np.ndarray, key: str) -> None:
    bad = ~np.isfinite(values)
    if values.ndim == mask.ndim + 1:
        bad = bad.any(axis=-1)
    if np.any(bad & mask):
        raise _segyrs.SegyrsValueError(f"{key}: a live trace is not finite")


def _no_live_inf(data: np.ndarray, mask: np.ndarray) -> None:
    if data.dtype.kind != "f":
        return
    rows = np.reshape(data, (-1, data.shape[-1]))
    live = np.reshape(mask, (-1,))
    for start in range(0, live.size, 4096):
        chosen = live[start : start + 4096]
        if not chosen.any():
            continue
        if np.isinf(rows[start : start + 4096][chosen]).any():
            raise _segyrs.SegyrsValueError("data: a live trace contains infinity")


def _fit_i16(delay_ms: float, t0_us: int) -> tuple[int, int]:
    for scalar in _SCALARS:
        raw = delay_ms * (-scalar) if scalar < 0 else delay_ms / scalar
        whole = round(raw)
        if not -32768 <= whole <= 32767:
            continue
        scaled = whole * scalar if scalar > 0 else whole / (-scalar)
        if abs(scaled * 1000.0 - t0_us) < 1e-4:
            return whole, scalar
    raise _segyrs.SegyrsValueError(f"samples: start {t0_us} us does not fit the delay header")


def _scalar_for(named: list[tuple[str, np.ndarray]]) -> int:
    for scalar in _SCALARS:
        if all(_exact_raw(values, scalar) for _, values in named):
            return scalar
    names = ", ".join(name for name, _ in named)
    raise _segyrs.SegyrsValueError(f"{named[0][0]}: {names} do not fit one SEG-Y scalar")


def _exact_raw(values: np.ndarray, scalar: int) -> bool:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    raw = flat * (-scalar) if scalar < 0 else flat / scalar
    whole = np.rint(raw)
    tol = np.maximum(1e-6, np.abs(raw) * (4.0 * np.finfo(np.float64).eps))
    return bool(
        np.all(np.isfinite(raw))
        and np.all(np.abs(raw - whole) <= tol)
        and np.all(np.abs(whole) <= _I32)
    )
