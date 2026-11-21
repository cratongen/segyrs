"""Object-store reads and Zarr v3 export (M5, docs/cloud-and-zarr.md)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _segyrs

if TYPE_CHECKING:
    from ._volume import Volume

# A scheme of two or more characters; a one-letter scheme is a Windows drive.
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+://")

PathOrUrl = str | os.PathLike[str]


def is_url(location: Any) -> bool:
    """True for ``s3://``, ``gs://``, ``az://``, ``http(s)://``, ``file://`` and other URLs."""
    return isinstance(location, str) and _URL.match(location) is not None


def location(path: PathOrUrl) -> str:
    """The string to hand to Rust: a URL unchanged, a path through ``os.fspath``."""
    return path if is_url(path) else os.fspath(path)


def normalize_options(options: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Normalise an fsspec-style ``storage_options`` dict to string keys and values.

    ``True``/``False`` become ``"true"``/``"false"``; other values go through ``str``.
    Unknown keys are not dropped here: Rust rejects them by name.
    """
    if options is None:
        return None
    if not isinstance(options, Mapping):
        raise TypeError("storage_options must be a dict of option names to values")
    out = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise TypeError(f"storage option names are strings, got {key!r}")
        if isinstance(value, bool):
            value = "true" if value else "false"
        out[key] = str(value)
    return out


def prefetch(file: _segyrs.File, traces: np.ndarray) -> None:
    """Fetch ``traces`` in coalesced parallel requests when the file is remote."""
    _segyrs.prefetch(file, traces)


def to_zarr(
    vol: Volume,
    path: PathOrUrl,
    *,
    chunks: Sequence[int] | None = None,
    target_chunk_bytes: int | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Implementation of :meth:`Volume.to_zarr`."""
    if not vol._root:
        raise ValueError(
            "to_zarr writes the whole file; call it on the Volume segyrs.open returned"
        )
    if chunks is not None:
        chunks = [int(c) for c in chunks]
    if target_chunk_bytes is not None:
        target_chunk_bytes = int(target_chunk_bytes)
        if target_chunk_bytes <= 0:
            raise ValueError("target_chunk_bytes must be positive")
    return _segyrs.zarr_export(
        vol._src.file,
        location(path),
        chunks=chunks,
        target_chunk_bytes=target_chunk_bytes,
        depth=vol._src.vertical == "depth",
        storage_options=normalize_options(storage_options),
    )
