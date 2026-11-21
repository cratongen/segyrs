"""Irregular files: segyrs (scan=auto, scan=full, index_cache) vs segyio (mmap, ignore_geometry).

Usage: bench_irregular.py PYTHON ROUNDS
Every measurement is a fresh subprocess; sides alternate each round; medians + ranges.
cold = posix_fadvise(DONTNEED) on the file, then open + op timed. warm = file read once, open, op timed.
Ops: open, inline (all traces of one inline label), header (cdp_x column), all (every trace as (ntr, ns)).
"""
import json
import os
import statistics
import subprocess
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "irregular")
FILES = {"sparse_holes": 1500, "unsorted": 1500, "prestack_varfold": 1200, "regular_one_bad": 1500}
SIDES = ["segyio", "segyrs_auto", "segyrs_full", "segyrs_cache"]
OPS = ["open", "inline", "header", "all"]

CHILD = r'''
import os, sys, time, json, numpy as np
side, path, op, mode, label, cache = sys.argv[1:7]
def evict():
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def warm():
    buf = bytearray(64 << 20)
    with open(path, "rb", buffering=0) as f:
        while f.readinto(buf): pass
label = int(label)
if side == "segyio":
    import segyio
    def opener():
        f = segyio.open(path, "r", ignore_geometry=True); f.mmap(); return f
    def run(f):
        if op == "inline":
            idx = np.flatnonzero(f.attributes(189)[:] == label)
            return np.stack([f.trace[int(i)] for i in idx])
        if op == "header": return f.attributes(181)[:]
        if op == "all": return f.trace.raw[:]
else:
    import segyrs
    kw = {"scan": "full"} if side == "segyrs_full" else {}
    if side == "segyrs_cache": kw = {"index_cache": cache}
    def opener(): return segyrs.open(path, **kw)
    def run(v):
        if op == "inline":
            s = v.sel(inline=label).numpy()
            return s.reshape(-1, s.shape[-1])
        if op == "header": return v.headers["cdp_x"]
        if op == "all":
            a = v.numpy(); return a.reshape(-1, a.shape[-1])
evict() if mode == "cold" else warm()
info = {}
try:
    t = time.perf_counter()
    f = opener()
    if op == "open":
        dt = time.perf_counter() - t; out = None
    else:
        if mode == "warm": t = time.perf_counter()
        out = run(f); dt = time.perf_counter() - t
    if side != "segyio" and op == "open":
        info = {"dims": list(f.dims), "shape": list(f.shape), "kind": f.geometry.kind,
                "source": getattr(f.geometry, "source", "scan")}
    if os.environ.get("DUMP") and out is not None: np.save(os.environ["DUMP"], np.asarray(out))
    print(json.dumps({"s": dt, **info}))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {str(e)[:160]}"}))
'''


def one(py, side, path, op, mode, label, cache, dump=None):
    env = dict(os.environ, DUMP=dump) if dump else os.environ
    r = subprocess.run([py, "-c", CHILD, side, path, op, mode, str(label), cache],
                       capture_output=True, text=True, env=env, check=False)
    lines = r.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else {"error": r.stderr.strip().splitlines()[-1][:200]}


def main():
    py, rounds = sys.argv[1], int(sys.argv[2])
    cache = os.path.join(DATA, "index_cache")
    res = {}
    for name, label in FILES.items():
        path = os.path.join(DATA, name + ".sgy")
        # prime the sidecar once (a full scan that writes it), outside timing
        one(py, "segyrs_cache", path, "open", "warm", label, cache)
        for op in OPS:
            for mode in ("cold", "warm"):
                n = max(3, rounds // 2) if op == "all" else rounds
                t = {s: [] for s in SIDES}
                extra = {}
                for r in range(n):
                    order = SIDES[r % len(SIDES):] + SIDES[:r % len(SIDES)]
                    for s in order:
                        x = one(py, s, path, op, mode, label, cache)
                        t[s].append(x.get("s", x.get("error")))
                        if "dims" in x: extra[s] = {k: x[k] for k in ("dims", "shape", "kind", "source")}
                med = {s: (statistics.median(v) if all(isinstance(q, float) for q in v) else v[0]) for s, v in t.items()}
                rng = {s: ([min(v), max(v)] if all(isinstance(q, float) for q in v) else None) for s, v in t.items()}
                res[f"{name}/{op}/{mode}"] = {"median_s": med, "range_s": rng, "rounds": n, "open_info": extra}
                cells = "  ".join(f"{s} {m:8.4f}" if isinstance(m, float) else f"{s} {str(m)[:60]}" for s, m in med.items())
                print(f"{name:17s} {op:6s} {mode:4s} {cells}", flush=True)
                for s, e in extra.items():
                    if mode == "warm" and op == "open":
                        print(f"    {s}: {e}", flush=True)
    with open(os.path.join(os.path.dirname(DATA), "bench_irregular.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
