"""Generate ~1e9-cell SEG-Y test volumes from one anisotropic Matern Gaussian random field.

Local test data for segyrs (not part of the repo). Usage:
    mise exec uv -- uv run --no-project --with numpy --with scipy python gen_matern.py [n]
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft

OUT = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000  # inline = crossline = N
NS = int(sys.argv[2]) if len(sys.argv) > 2 else N  # samples per trace
TAG = f"{N}" if NS == N else f"{N}x{N}x{NS}"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "" if NS == N else TAG)
NU = 1.5  # Matern smoothness
RANGE = (40.0, 40.0, 8.0)  # correlation length in cells: inline, crossline, sample
DT_US = 4000
IL0, XL0 = 1000, 2000
X0, Y0, BIN = 500_000.0, 6_700_000.0, 12.5
SCALCO = -100  # coordinates stored in centimetres
WORKERS = os.cpu_count()


def log(msg, t0=[time.perf_counter()]):
    print(f"[{time.perf_counter() - t0[0]:7.1f}s] {msg}", flush=True)


def matern_field(n, ns):
    """Unit-variance float32 field via FFT filtering of white noise with sqrt(Matern spectrum)."""
    shape = (n, n, ns)
    noise = np.empty(shape, np.float32)
    seeds = np.random.SeedSequence(20261117).spawn(WORKERS)

    def fill(i):
        lo, hi = i * n // WORKERS, (i + 1) * n // WORKERS
        noise[lo:hi] = np.random.default_rng(seeds[i]).standard_normal((hi - lo, n, ns), np.float32)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(fill, range(WORKERS)))
    log("white noise ready")

    spec = sfft.rfftn(noise, workers=WORKERS, overwrite_x=True)
    del noise
    log("forward FFT done")

    k0 = (sfft.fftfreq(n) * RANGE[0]).astype(np.float32) ** 2
    k1 = (sfft.fftfreq(n) * RANGE[1]).astype(np.float32) ** 2
    k2 = (sfft.rfftfreq(ns) * RANGE[2]).astype(np.float32) ** 2
    c = np.float32((2 * np.pi) ** 2 / (2 * NU))
    p = np.float32(-(NU + 1.5) / 2)  # sqrt of the spectral density exponent
    for lo in range(0, n, 25):
        hi = min(lo + 25, n)
        q = k0[lo:hi, None, None] + k1[None, :, None] + k2[None, None, :]
        spec[lo:hi] *= (1 + c * q) ** p
    log("Matern filter applied")

    field = sfft.irfftn(spec, s=shape, workers=WORKERS, overwrite_x=True).astype(np.float32, copy=False)
    del spec
    field -= np.float32(field.mean(dtype=np.float64))
    field /= np.float32(field.std(dtype=np.float64))
    log(f"inverse FFT done, std normalised (min {field.min():.2f}, max {field.max():.2f})")
    return field


def ieee_to_ibm(x):
    """Vectorised IEEE float32 -> IBM System/360 single (uint32 words)."""
    a = np.abs(x.astype(np.float64))
    word = np.zeros(x.shape, np.uint32)
    nz = a > 0
    e = np.floor(np.log2(a[nz]) / 4).astype(np.int64) + 1
    frac = a[nz] / np.exp2(4.0 * e)
    lo = frac < 1 / 16
    e[lo] -= 1
    frac[lo] *= 16
    mant = np.rint(frac * 2**24).astype(np.int64)
    over = mant >= 2**24
    mant[over] >>= 4
    e[over] += 1
    w = (np.uint64(1) << 31) * (x[nz] < 0) | ((e + 64).astype(np.uint64) << 24) | mant.astype(np.uint64)
    word[nz] = w.astype(np.uint32)
    return word


def textual(lines):
    cards = [f"C{i + 1:2d} {s}"[:80].ljust(80) for i, s in enumerate(lines)]
    cards += [f"C{i + 1:2d}".ljust(80) for i in range(len(cards), 40)]
    return "".join(cards).encode("cp037")


def write(path, field, fmt, endian, order, rev2):
    n, ns = field.shape[0], field.shape[2]
    e = ">" if endian == "big" else "<"
    sdtype = {5: e + "f4", 1: e + "u4", 3: e + "i2"}[fmt]
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(textual([
            "SEGYRS LOCAL TEST VOLUME. NOT REAL DATA.",
            f"ANISOTROPIC MATERN GAUSSIAN FIELD NU={NU} RANGES IL/XL/T={RANGE} CELLS",
            f"GRID {n} INLINES x {n} CROSSLINES x {ns} SAMPLES, DT {DT_US} US",
            f"FORMAT {fmt} {endian.upper()}-ENDIAN, {order.upper()}-SORTED, REV {'2' if rev2 else '1'}",
            f"INLINE BYTES 189-192 FROM {IL0}, CROSSLINE 193-196 FROM {XL0}",
            f"CDP X/Y BYTES 181-188, SCALCO {SCALCO}, BIN {BIN} M",
            "SEED 20261117",
        ]))
        b = np.zeros(400, np.uint8)

        def put(arr, off, val, t):
            v = np.asarray(val).astype(e + t)
            sz = v.dtype.itemsize
            arr[..., off:off + sz] = v.reshape(-1).view(np.uint8).reshape(v.shape + (sz,))

        put(b, 12, n, "i2")       # 3213 traces per ensemble
        put(b, 16, DT_US, "i2")   # 3217 sample interval
        put(b, 20, ns, "i2")      # 3221 samples per trace
        put(b, 24, fmt, "i2")     # 3225 format code
        put(b, 28, 1, "i2")       # 3229 ensemble fold
        put(b, 30, 2 if order == "inline" else 3, "i2")  # 3231 sorting code (rev1 table; best effort)
        put(b, 54, 1, "i2")       # 3255 measurement system: metres
        if rev2:
            put(b, 96, 16909060, "i4")  # 3297 byte-order signature 0x01020304
        b[300], b[301] = (2, 0) if rev2 else (1, 0)  # 3501 revision major/minor
        put(b, 302, 1, "i2")      # 3503 fixed-length traces
        f.write(b.tobytes())

        idx = np.arange(n)
        for s in range(n):
            if order == "inline":
                il = np.full(n, s); xl = idx; samples = field[s]
            else:
                il = idx; xl = np.full(n, s); samples = field[:, s, :]
            h = np.zeros((n, 240), np.uint8)
            seq = s * n + idx + 1
            put(h, 0, seq, "i4"); put(h, 4, seq, "i4")
            put(h, 8, IL0 + il, "i4"); put(h, 12, idx + 1, "i4")
            put(h, 20, (IL0 + il) * 10000 + XL0 + xl, "i4")
            put(h, 28, 1, "i2")                     # trace id: seismic
            put(h, 70, SCALCO, "i2")
            put(h, 114, ns, "i2"); put(h, 116, DT_US, "i2")
            put(h, 180, np.rint((X0 + BIN * xl) * 100), "i4")
            put(h, 184, np.rint((Y0 + BIN * il) * 100), "i4")
            put(h, 188, IL0 + il, "i4"); put(h, 192, XL0 + xl, "i4")
            if fmt == 5:
                d = samples.astype(sdtype)
            elif fmt == 1:
                d = ieee_to_ibm(samples).astype(sdtype)
            else:
                d = np.clip(np.rint(samples * 1000), -32767, 32767).astype(sdtype)
            f.write(np.concatenate([h, np.ascontiguousarray(d).view(np.uint8).reshape(n, -1)], axis=1).tobytes())
    os.replace(tmp, path)
    expect = 3600 + n * n * (240 + ns * np.dtype(sdtype).itemsize)
    assert os.path.getsize(path) == expect, (path, os.path.getsize(path), expect)
    log(f"wrote {os.path.basename(path)} ({expect / 1e9:.2f} GB)")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    npy = os.path.join(OUTDIR, f"matern_field_{TAG}.npy")
    if os.path.exists(npy):
        field = np.load(npy, mmap_mode="r")
        log("reusing cached field")
    else:
        field = matern_field(N, NS)
        np.save(npy, field)
        log(f"cached field -> {os.path.basename(npy)}")
    # IBM round-trip sanity on a slab before writing gigabytes.
    x = np.asarray(field[0, :8], np.float32)
    w = ieee_to_ibm(x).astype(np.uint64)
    back = (1 - 2 * (w >> 31).astype(np.float64)) * (w & 0xFFFFFF) / 2**24 * 16.0 ** (((w >> 24) & 0x7F).astype(np.float64) - 64)
    assert np.allclose(back, x, rtol=1e-6, atol=0), "IBM encoder broken"
    write(os.path.join(OUTDIR, "matern_ieee_be_inline.sgy"), field, 5, "big", "inline", False)
    write(os.path.join(OUTDIR, "matern_ibm_be_crossline.sgy"), field, 1, "big", "crossline", False)
    write(os.path.join(OUTDIR, "matern_int16_le_rev2.sgy"), field, 3, "little", "inline", True)


if __name__ == "__main__":
    main()
