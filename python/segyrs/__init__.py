"""Clean-room SEG-Y reading, writing and manipulation, with labelled exploration.

``segyrs.open`` returns a lazy :class:`Volume`. Numpy is the only runtime
dependency. Sample decoding and header scans run in Rust on every core with the
GIL released.
"""

from importlib import metadata as _metadata

from ._ops import concat
from ._segyrs import (
    BudgetError,
    DuplicateKey,
    EndianMismatch,
    Format4NeedsGain,
    FormatChangeError,
    GeometryAssumptionError,
    GeometryError,
    HeaderValueError,
    NonUniformTraceLength,
    NotSegyError,
    PathIsDirectoryError,
    PathNotFoundError,
    PathPermissionError,
    SampleCountMismatch,
    SampleValueError,
    SegyrsError,
    SegyrsIOError,
    SegyrsValueError,
    SpecError,
    TraceLengthError,
    TruncatedError,
    UnknownFormat,
    UnsortedCoordinate,
    ValidationError,
)
from ._view import Picture, view
from ._volume import Geometry, Volume, open
from ._write import Editor, create, edit, validate

try:
    __version__ = _metadata.version("segyrs")
except _metadata.PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"

__all__ = [
    "BudgetError",
    "DuplicateKey",
    "Editor",
    "EndianMismatch",
    "Format4NeedsGain",
    "FormatChangeError",
    "Geometry",
    "GeometryAssumptionError",
    "GeometryError",
    "HeaderValueError",
    "NonUniformTraceLength",
    "NotSegyError",
    "PathIsDirectoryError",
    "PathNotFoundError",
    "PathPermissionError",
    "Picture",
    "SampleCountMismatch",
    "SampleValueError",
    "SegyrsError",
    "SegyrsIOError",
    "SegyrsValueError",
    "SpecError",
    "TraceLengthError",
    "TruncatedError",
    "UnknownFormat",
    "UnsortedCoordinate",
    "ValidationError",
    "Volume",
    "concat",
    "create",
    "edit",
    "open",
    "validate",
    "view",
]
