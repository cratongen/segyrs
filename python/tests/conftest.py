"""Shared fixtures: the read-only files in test-data/ and a small SEG-Y writer for cases
the fixtures do not cover. The writer follows the rev 0, 1, and 2 byte layouts."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "test-data"


def _extra(module: str) -> None:
    """Skip unless the optional extension is installed. With
    SEGYRS_REQUIRE_EXTRAS=1 (`mise run test`) a missing one fails instead."""
    try:
        importlib.import_module(module)
    except ImportError:
        if os.environ.get("SEGYRS_REQUIRE_EXTRAS") == "1":
            raise
        pytest.skip(f"{module} is not installed (built by mise run build, not released yet)")


@pytest.fixture
def view_extra() -> None:
    """The ``segyrs-view`` extension (built by ``mise run build``, not released yet)."""
    _extra("_segyrs_view")


# Format code to the on-disk numpy kind (format 1 and 4 are written as raw words).
FORMAT_KIND = {
    1: "u4",
    2: "i4",
    3: "i2",
    4: "u4",
    5: "f4",
    6: "f8",
    8: "i1",
    9: "i8",
    10: "u4",
    11: "u2",
    12: "u8",
    16: "u1",
}


def write_segy(
    path: Path,
    samples: np.ndarray,
    *,
    fmt: int = 5,
    inline=None,
    crossline=None,
    offset=None,
    dt: int = 4000,
    little: bool = False,
    revision: tuple[int, int] = (1, 0),
    signature: bytes = b"\0\0\0\0",
    extra_blocks: int = 0,
    words: dict[tuple[int, str], np.ndarray] | None = None,
    fixed_length: int = 1,
    first_card: bytes = b"C 1 ",
) -> Path:
    """Write ``samples`` (traces by samples) as a SEG-Y file.

    ``words`` maps ``(1-based byte, numpy kind)`` to one value per trace and may
    address extension blocks (byte > 240). Format 7 and 15 take int32 or uint32
    values and are packed to three bytes.
    """
    samples = np.asarray(samples)
    ntr, ns = samples.shape
    order = "<" if little else ">"
    blocks = 1 + extra_blocks
    text = bytearray(b" " * 3200)
    text[: len(first_card)] = first_card
    binary = bytearray(400)

    def put(buf, off, kind, value):
        raw = np.array(value, dtype=order + kind).tobytes()
        buf[off : off + len(raw)] = raw

    put(binary, 16, "u2", dt)
    put(binary, 20, "u2", min(ns, 65535))
    put(binary, 24, "u2", fmt)
    binary[96:100] = signature
    binary[300], binary[301] = revision
    put(binary, 302, "i2", fixed_length)
    put(binary, 306, "i4", extra_blocks)

    if fmt in (7, 15):
        wide = samples.astype(order + ("i4" if fmt == 7 else "u4")).view(np.uint8)
        wide = wide.reshape(ntr, ns, 4)
        body = wide[:, :, 1:] if not little else wide[:, :, :3]
        body = np.ascontiguousarray(body).reshape(ntr, ns * 3)
    else:
        body = np.ascontiguousarray(samples.astype(order + FORMAT_KIND[fmt])).view(np.uint8)
        body = body.reshape(ntr, -1)

    headers = np.zeros((ntr, 240 * blocks), dtype=np.uint8)
    cols = {(115, "u2"): np.full(ntr, min(ns, 65535))}
    if inline is not None:
        cols[(189, "i4")] = inline
    if crossline is not None:
        cols[(193, "i4")] = crossline
    if offset is not None:
        cols[(37, "i4")] = offset
    cols.update(words or {})
    for (byte, kind), values in cols.items():
        raw = np.asarray(values, dtype=order + kind).reshape(ntr, 1).view(np.uint8)
        headers[:, byte - 1 : byte - 1 + raw.shape[1]] = raw
    traces = np.concatenate([headers, body], axis=1)
    path.write_bytes(bytes(text) + bytes(binary) + traces.tobytes())
    return path


@pytest.fixture
def segy(tmp_path):
    """``segy(name, samples, **kw)`` writes a file under tmp_path and returns its path."""

    def make(name: str, samples, **kw) -> Path:
        return write_segy(tmp_path / name, samples, **kw)

    return make


def grid(nil: int, nxl: int, ns: int = 4, il0: int = 1, xl0: int = 10):
    """An inline-major grid: samples hold ``100 * trace + sample`` as float32."""
    il, xl = np.meshgrid(np.arange(il0, il0 + nil), np.arange(xl0, xl0 + nxl), indexing="ij")
    trace = np.arange(nil * nxl)
    samples = (100 * trace[:, None] + np.arange(ns)[None, :]).astype(np.float32)
    return samples, il.ravel(), xl.ravel()
