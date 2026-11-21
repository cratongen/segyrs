"""Irregular SEG-Y test files from the cached 1000x1000x100 Matern field (local only, not in the repo).

Usage: python gen_irregular.py
Writes into irregular/:
  sparse_holes.sgy      inline-sorted, an irregular survey outline plus random holes (~35% of slots empty)
  unsorted.sgy          every slot of the 1000x1000 grid once, in random trace order
  prestack_varfold.sgy  CDP-sorted pre-stack, 400x400 CDPs, fold 1..24 varying smoothly, offsets 100 m apart
  regular_one_bad.sgy   regular inline-sorted grid with ONE trace near the middle whose crossline header is wrong
  truth.npz             what each file holds (il, xl, offset per trace; the bad trace index)
IEEE float32 big-endian, rev 1, inline at 189, crossline at 193, offset at 37.
"""
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "1000x1000x100", "matern_field_1000x1000x100.npy")
OUT = os.path.join(HERE, "irregular")
DT_US, IL0, XL0 = 4000, 1000, 2000
X0, Y0, BIN = 500_000.0, 6_700_000.0, 12.5
CHUNK = 20_000


def log(msg, t0=[time.perf_counter()]):
    print(f"[{time.perf_counter() - t0[0]:6.1f}s] {msg}", flush=True)


def textual(lines):
    cards = [f"C{i + 1:2d} {s}"[:80].ljust(80) for i, s in enumerate(lines)]
    cards += [f"C{i + 1:2d}".ljust(80) for i in range(len(cards), 40)]
    return "".join(cards).encode("cp037")


def put(arr, off, val, t):
    v = np.asarray(val).astype(">" + t)
    sz = v.dtype.itemsize
    arr[..., off:off + sz] = v.reshape(-1).view(np.uint8).reshape(v.shape + (sz,))


def write(path, title, il, xl, off, samples_for):
    """il, xl, off: int arrays, one entry per trace in file order. samples_for(idx) -> (len(idx), ns) float32."""
    ntr, ns = len(il), samples_for(np.arange(1)).shape[1]
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(textual([
            "SEGYRS LOCAL IRREGULAR TEST FILE. NOT REAL DATA.",
            title,
            f"{ntr} TRACES x {ns} SAMPLES, DT {DT_US} US, IEEE FLOAT32 BIG-ENDIAN",
            "INLINE 189-192, CROSSLINE 193-196, OFFSET 37-40, CDP X/Y 181-188 SCALCO -100",
        ]))
        b = np.zeros(400, np.uint8)
        put(b, 16, DT_US, "i2")
        put(b, 20, ns, "i2")
        put(b, 24, 5, "i2")
        put(b, 54, 1, "i2")
        b[300] = 1
        put(b, 302, 1, "i2")
        f.write(b.tobytes())
        for lo in range(0, ntr, CHUNK):
            idx = np.arange(lo, min(lo + CHUNK, ntr))
            h = np.zeros((len(idx), 240), np.uint8)
            put(h, 0, idx + 1, "i4")
            put(h, 4, idx + 1, "i4")
            put(h, 20, (IL0 + il[idx]) * 10000 + XL0 + xl[idx], "i4")
            put(h, 28, 1, "i2")
            put(h, 36, off[idx], "i4")
            put(h, 70, -100, "i2")
            put(h, 114, ns, "i2")
            put(h, 116, DT_US, "i2")
            put(h, 180, np.rint((X0 + BIN * xl[idx]) * 100), "i4")
            put(h, 184, np.rint((Y0 + BIN * il[idx]) * 100), "i4")
            put(h, 188, IL0 + il[idx], "i4")
            put(h, 192, XL0 + xl[idx], "i4")
            d = np.ascontiguousarray(samples_for(idx).astype(">f4")).view(np.uint8)
            f.write(np.concatenate([h, d], axis=1).tobytes())
    os.replace(tmp, path)
    expect = 3600 + ntr * (240 + 4 * ns)
    assert os.path.getsize(path) == expect
    log(f"wrote {os.path.basename(path)}: {ntr} traces, {expect / 1e9:.2f} GB")


def main():
    os.makedirs(OUT, exist_ok=True)
    field = np.load(SRC, mmap_mode="r")
    n = field.shape[0]
    rng = np.random.default_rng(20261117)
    truth = {}

    # 1. Sparse: an irregular outline (a wobbly ellipse) with random holes inside.
    ii, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    theta = np.arctan2(xx - n / 2, ii - n / 2)
    radius = 0.47 * n * (1 + 0.12 * np.sin(5 * theta) + 0.05 * np.cos(11 * theta))
    keep = np.hypot(ii - n / 2, xx - n / 2) < radius
    keep &= rng.random((n, n)) > 0.08
    il, xl = ii[keep], xx[keep]
    write(os.path.join(OUT, "sparse_holes.sgy"), "SPARSE: IRREGULAR OUTLINE PLUS RANDOM HOLES, INLINE-SORTED",
          il, xl, np.zeros_like(il), lambda idx: np.asarray(field[il[idx], xl[idx]]))
    truth.update(sparse_il=il, sparse_xl=xl)

    # 2. Unsorted: the full grid, every slot once, random order.
    perm = rng.permutation(n * n)
    il, xl = perm // n, perm % n
    write(os.path.join(OUT, "unsorted.sgy"), "UNSORTED: FULL 1000x1000 GRID, RANDOM TRACE ORDER",
          il, xl, np.zeros_like(il), lambda idx: np.asarray(field[il[idx], xl[idx]]))
    truth.update(unsorted_il=il, unsorted_xl=xl)

    # 3. Pre-stack, varying fold: 400x400 CDPs, fold from a smooth surface 1..24, offsets 100 m apart.
    m = 400
    ci, cx = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    fold = np.clip(np.rint(12.5 + 11.5 * np.sin(ci / 37.0) * np.cos(cx / 53.0)), 1, 24).astype(np.int64)
    rep = fold.ravel()
    il = np.repeat(ci.ravel(), rep)
    xl = np.repeat(cx.ravel(), rep)
    starts = np.repeat(np.cumsum(rep) - rep, rep)
    k = np.arange(len(il)) - starts
    off = 100 * (k + 1)
    write(os.path.join(OUT, "prestack_varfold.sgy"), "PRE-STACK, CDP-SORTED, FOLD 1..24 VARYING, OFFSET STEP 100 M",
          il, xl, off,
          lambda idx: np.asarray(field[il[idx] * 2, xl[idx] * 2]) * (1 - 0.01 * k[idx, None]))
    truth.update(prestack_il=il, prestack_xl=xl, prestack_off=off)

    # 4. Regular grid with one wrong crossline header near the middle (a typo-like corruption).
    ii, xx = np.divmod(np.arange(n * n), n)
    bad = 500 * n + 617  # inline 1500, slot of crossline 2617
    xl = xx.copy()
    xl[bad] = 5000  # crossline 7000: off the grid
    write(os.path.join(OUT, "regular_one_bad.sgy"), "REGULAR INLINE-SORTED GRID, ONE CORRUPT CROSSLINE HEADER (TRACE 500617)",
          ii, xl, np.zeros_like(ii), lambda idx: np.asarray(field[ii[idx], xx[idx]]))
    truth.update(bad_trace=np.array([bad]))

    np.savez(os.path.join(OUT, "truth.npz"), **truth)
    log("truth.npz written")


if __name__ == "__main__":
    main()
