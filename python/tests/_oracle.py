"""An independent SEG-Y decoder for the release-audit tests.

It is written from the SEG-Y rev 0, 1 and 2 byte layouts and shares no code
with segyrs. It reads a whole small file into memory, so it is only for the
fixtures and for sampled traces of large files (``read_traces``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WIDTH = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4, 6: 8, 7: 3, 8: 1, 9: 8, 10: 4, 11: 2, 12: 8, 15: 3, 16: 1}
_END_TEXT = re.compile(rb"\(\(\s*seg\s*:\s*endtext\s*\)\)", re.IGNORECASE)
_EBCDIC_END = "((SEG: EndText))".encode("cp037")


@dataclass
class Layout:
    order: str
    fmt: int
    ns: int
    data_start: int
    header_len: int
    n_traces: int
    revision: int

    @property
    def stride(self) -> int:
        return self.header_len + self.ns * WIDTH[self.fmt]


def _u(raw: bytes, off: int, n: int, order: str, signed: bool = False) -> int:
    return int.from_bytes(raw[off : off + n], "big" if order == ">" else "little", signed=signed)


def _is_end(block: bytes) -> bool:
    return bool(_END_TEXT.search(block)) or _EBCDIC_END.lower() in block.lower()


def layout(head: bytes, size: int, order: str, read_block) -> Layout:
    """``read_block(offset)`` returns the 3200 bytes at ``offset``."""
    fmt = _u(head, 3224, 2, order)
    if fmt not in WIDTH:
        raise ValueError(f"format {fmt}")
    rev = head[3500]
    ns = _u(head, 3220, 2, order)
    if rev >= 2 and _u(head, 3268, 4, order):
        ns = _u(head, 3268, 4, order)
    ntext = _u(head, 3504, 2, order, signed=True)
    start = 3600
    if ntext > 0:
        start += 3200 * ntext
    elif ntext < 0:
        while True:
            if start + 3200 > size:
                raise ValueError("no EndText stanza")
            block = read_block(start)
            start += 3200
            if _is_end(block):
                break
    payload = size - start
    for extra in _extra_candidates(head, order, rev):
        header_len = 240 * (1 + extra)
        stride = header_len + ns * WIDTH[fmt]
        if payload >= 0 and payload % stride == 0:
            return Layout(order, fmt, ns, start, header_len, payload // stride, rev)
    raise ValueError("size is not headers plus whole traces")


def _extra_candidates(head: bytes, order: str, rev: int) -> list[int]:
    if rev < 2:
        return [0]
    out = []
    for n in (4, 2):
        v = _u(head, 3506, n, order)
        if 0 < v <= 65535 and v not in out:
            out.append(v)
    return [*out, 0]


def ibm_to_float32(words: np.ndarray) -> np.ndarray:
    """IBM hexadecimal float words (uint32, native) to float32, round to nearest."""
    w = words.astype(np.uint64)
    sign = np.where(w >> 31, -1.0, 1.0)
    exp = ((w >> 24) & 0x7F).astype(np.int64) - 64
    frac = (w & 0xFFFFFF).astype(np.float64) / float(1 << 24)
    with np.errstate(over="ignore"):
        value = sign * np.ldexp(frac, (4 * exp).astype(np.int32))
        return value.astype(np.float32)


def decode(raw: bytes, fmt: int, order: str, ns: int, gain: str | None = None) -> np.ndarray:
    """``raw`` holds whole traces of ``ns`` samples; returns (n, ns) in the natural dtype."""
    if ns == 0:
        return np.zeros((0, 0), np.float32)
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, ns * WIDTH[fmt])
    simple = {2: "i4", 3: "i2", 5: "f4", 6: "f8", 8: "i1", 9: "i8", 10: "u4", 11: "u2", 12: "u8"}
    simple[16] = "u1"
    if fmt in simple:
        dt = np.dtype(order + simple[fmt])
        return a.copy().view(dt).reshape(a.shape[0], ns).astype(dt.newbyteorder("="))
    if fmt == 1:
        words = a.copy().view(np.dtype(order + "u4")).reshape(a.shape[0], ns).astype(np.uint32)
        return ibm_to_float32(words)
    if fmt in (7, 15):
        b = a.reshape(a.shape[0], ns, 3).astype(np.uint32)
        if order == ">":
            v = (b[..., 0] << 16) | (b[..., 1] << 8) | b[..., 2]
        else:
            v = (b[..., 2] << 16) | (b[..., 1] << 8) | b[..., 0]
        if fmt == 15:
            return v
        return np.where(v & 0x800000, v.astype(np.int64) - (1 << 24), v).astype(np.int32)
    if fmt == 4:
        b = a.reshape(a.shape[0], ns, 4)
        g = b[..., 1].astype(np.int64) if order == ">" else b[..., 2].astype(np.int64)
        m = b[..., 2:4] if order == ">" else b[..., 0:2]
        mant = m.copy().view(np.dtype(order + "i2")).reshape(a.shape[0], ns).astype(np.float64)
        e = g if gain == "rev0" else -g
        return np.ldexp(mant, e.astype(np.int32)).astype(np.float32)
    raise ValueError(f"format {fmt}")


def read_file(path: Path, order: str, gain: str | None = None):
    """Layout, (n_traces, ns) samples and (n_traces, header_len) raw headers of a small file."""
    raw = Path(path).read_bytes()
    lay = layout(raw[:3600], len(raw), order, lambda off: raw[off : off + 3200])
    body = np.frombuffer(raw, np.uint8, count=lay.n_traces * lay.stride, offset=lay.data_start)
    body = body.reshape(lay.n_traces, lay.stride)
    headers = body[:, : lay.header_len]
    samples = decode(body[:, lay.header_len :].tobytes(), lay.fmt, order, lay.ns, gain)
    return lay, samples.reshape(lay.n_traces, lay.ns), headers


def read_traces(path: Path, order: str, traces: np.ndarray, gain: str | None = None):
    """Layout and samples of selected traces, read with positional reads."""
    size = Path(path).stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(3600)

        def block(off: int) -> bytes:
            fh.seek(off)
            return fh.read(3200)

        lay = layout(head, size, order, block)
        out = []
        for t in traces:
            fh.seek(lay.data_start + int(t) * lay.stride + lay.header_len)
            out.append(fh.read(lay.ns * WIDTH[lay.fmt]))
    return lay, decode(b"".join(out), lay.fmt, order, lay.ns, gain)
