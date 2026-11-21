"""Write the segyrs-made fixtures in this folder. Run from the repository root:
mise exec -- .venv/bin/python test-data/segyrs/make_fixtures.py
python/tests/test_packaging.py checks that the committed files match this script."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _ibm(value: int) -> int:
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    mag, exp = abs(value), 70
    while mag >= 1 << 24:
        if mag % 16:
            raise ValueError(f"{value} is not exact in IBM float")
        mag, exp = mag // 16, exp + 1
    while mag < 1 << 20:
        mag, exp = mag * 16, exp - 1
    return sign | exp << 24 | mag


def _textual(lines: list[str]) -> bytes:
    rows = [f"C{i + 1:2d} {lines[i] if i < len(lines) else ''}" for i in range(40)]
    return "".join(row.ljust(80)[:80] for row in rows).encode("ascii")


def delay_scalar() -> bytes:
    """One IBM trace of 251 samples at 4 ms whose delay is 10000 with time scalar -10
    (1000 ms), in a rev 1 file with an ASCII textual header."""
    ns, dt = 251, 4000
    text = _textual(
        [
            "segyrs test fixture delay-scalar.sgy, made by make_fixtures.py",
            "Recording delay 10000 (bytes 109-110) with time scalar -10 (bytes 215-216)",
            "Coordinate scalar -100, inline 2500 at byte 189, crossline 1883 at byte 193",
            "Samples are the ramp 0, 1, 2, ... in IBM float (format 1)",
        ]
    )
    binary = bytearray(400)
    for byte, fmt, value in [
        (3217, ">h", dt),
        (3221, ">h", ns),
        (3225, ">h", 1),
        (3255, ">h", 1),
        (3501, ">H", 0x0100),
        (3503, ">h", 1),
    ]:
        struct.pack_into(fmt, binary, byte - 3201, value)
    trace = bytearray(240)
    for byte, fmt, value in [
        (1, ">i", 1),
        (71, ">h", -100),
        (89, ">h", 1),
        (109, ">h", 10000),
        (115, ">h", ns),
        (117, ">h", dt),
        (181, ">i", 12_345_600),
        (185, ">i", 98_765_400),
        (189, ">i", 2500),
        (193, ">i", 1883),
        (215, ">h", -10),
    ]:
        struct.pack_into(fmt, trace, byte - 1, value)
    samples = struct.pack(f">{ns}I", *(_ibm(v) for v in range(ns)))
    return text + bytes(binary) + bytes(trace) + samples


FIXTURES = {"delay-scalar.sgy": delay_scalar}

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    for name, make in FIXTURES.items():
        (out / name).write_bytes(make())
