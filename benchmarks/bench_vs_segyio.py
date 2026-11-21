"""segyrs vs segyio on the local Matern volumes: same operations, interleaved rounds, medians.

Usage: bench_vs_segyio.py PYTHON ROUNDS [OUT.json]
Every measurement runs in a fresh subprocess. "cold" evicts the file from the page
cache with posix_fadvise(DONTNEED) before the timed call; "warm" reads it once first.
"""
import json
import os
import statistics
import subprocess
import sys

DATA = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ieee_be_inline": ("matern_ieee_be_inline.sgy", {}),
    "ibm_be_xline": ("matern_ibm_be_crossline.sgy", {}),
    "int16_le_rev2": ("matern_int16_le_rev2.sgy", {"endian": "little"}),
}
OPS = ["open", "inline", "crossline", "time_slice", "header_column", "traces_1000", "cube"]

CHILD = r'''
import os, sys, time, json, numpy as np
lib, path, op, mode, endian = sys.argv[1:6]
IL, XL, K = 1500, 2500, 500
def evict():
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def warm():
    with open(path, "rb", buffering=0) as f:
        while f.readinto(buf): pass
buf = bytearray(64 << 20)
if lib == "segyio":
    import segyio
    def opener():
        f = segyio.open(path, "r", iline=189, xline=193, endian=endian)
        if os.environ.get("SEGYIO_MMAP"): f.mmap()  # segyio's documented read speed-up
        return f
    def run(f):
        if op == "inline": return f.iline[IL]
        if op == "crossline": return f.xline[XL]
        if op == "time_slice": return f.depth_slice[K]
        if op == "header_column": return f.attributes(181)[:]
        if op == "traces_1000":
            idx = np.random.default_rng(0).integers(0, f.tracecount, 1000)
            return np.stack([f.trace[int(i)] for i in idx])
        if op == "cube": return segyio.tools.cube(f)
else:
    import segyrs
    def opener(): return segyrs.open(path)
    def run(v):
        if op == "inline": return v.sel(inline=IL).numpy()
        if op == "crossline": return v.sel(crossline=XL).numpy()
        if op == "time_slice": return v.isel(time=K).numpy()
        if op == "header_column": return v.headers["cdp_x"]
        if op == "traces_1000":
            idx = np.random.default_rng(0).integers(0, 1000 * 1000, 1000)
            xl_major = v.geometry.kind == "crossline_major"  # trace i in file order, as segyio's f.trace[i]
            return np.stack([v.isel(inline=int(i) % 1000 if xl_major else int(i) // 1000,
                                    crossline=int(i) // 1000 if xl_major else int(i) % 1000).numpy() for i in idx])
        if op == "cube": return v.numpy()
evict() if mode == "cold" else warm()
if op == "open":
    t = time.perf_counter(); f = opener(); dt = time.perf_counter() - t; out = None
elif mode == "cold":
    # Cold = open + operation from an evicted cache. Evicting after open is not
    # possible for segyrs: its header scan maps pages that hold samples too.
    t = time.perf_counter(); f = opener(); out = run(f); dt = time.perf_counter() - t
else:
    f = opener()
    t = time.perf_counter(); out = run(f); dt = time.perf_counter() - t
if os.environ.get("BENCH_DUMP") and out is not None:
    np.save(os.environ["BENCH_DUMP"], np.asarray(out))
print(json.dumps({"s": dt}))
'''


def one(py, lib, path, op, mode, endian, dump=None):
    env = dict(os.environ, BENCH_DUMP=dump) if dump else os.environ
    r = subprocess.run([py, "-c", CHILD, lib, path, op, mode, endian],
                       capture_output=True, text=True, env=env, check=False)
    if r.returncode:
        return {"error": r.stderr.strip().splitlines()[-1]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    py, rounds = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA, "bench_vs_segyio.json")
    res = {}
    for key, (name, kw) in FILES.items():
        path = os.path.join(DATA, name)
        endian = kw.get("endian", "big")
        for op in OPS:
            for mode in ("cold", "warm"):
                n = max(3, rounds // 2) if op == "cube" else rounds
                t = {"segyio": [], "segyrs": []}
                for r in range(n):
                    order = ("segyio", "segyrs") if r % 2 == 0 else ("segyrs", "segyio")
                    for lib in order:
                        x = one(py, lib, path, op, mode, endian)
                        t[lib].append(x.get("s", x.get("error")))
                row = {lib: (statistics.median(v) if all(isinstance(s, float) for s in v) else v)
                       for lib, v in t.items()}
                res[f"{key}/{op}/{mode}"] = {"median_s": row, "rounds": n, "all": t}
                a, b = row["segyio"], row["segyrs"]
                ratio = f"{a / b:7.1f}x" if isinstance(a, float) and isinstance(b, float) and b else "   n/a"
                fmt = lambda s: f"{s:9.4f}" if isinstance(s, float) else f"{str(s)[:40]:>9}"
                print(f"{key:15s} {op:13s} {mode:4s} segyio {fmt(a)} s  segyrs {fmt(b)} s  {ratio}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
