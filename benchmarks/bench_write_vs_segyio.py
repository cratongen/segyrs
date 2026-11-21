"""segyrs vs segyio write paths, interleaved rounds, medians. Usage: bench_write_vs_segyio.py PYTHON ROUNDS SRC OUTDIR

copy        : whole file to a new SEG-Y, same format (segyio: create from spec + bulk trace/header copy;
              segyrs: Volume.to_segy). Both end with the data on disk (fsync for segyio, sync=True for segyrs).
edit_header : set one header word (offset, byte 37) on every trace of a copy, in place.
Each measurement runs in a fresh subprocess; outputs are deleted after each run.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys

CHILD = r'''
import os, sys, time, json, numpy as np
lib, op, src, out = sys.argv[1:5]
if op == "edit_header":
    import shutil; shutil.copyfile(src, out)
t = time.perf_counter()
if lib == "segyio":
    import segyio
    if op == "copy":
        with segyio.open(src, "r", iline=189, xline=193, ignore_geometry=False) as f:
            spec = segyio.tools.metadata(f)
            with segyio.create(out, spec) as g:
                g.text[0] = f.text[0]
                g.bin = f.bin
                g.header = f.header
                g.trace = f.trace
        fd = os.open(out, os.O_RDONLY); os.fsync(fd); os.close(fd)
    else:
        with segyio.open(out, "r+", ignore_geometry=True) as f:
            n = f.tracecount
            for i in range(n):
                f.header[i] = {segyio.TraceField.offset: 25}
        fd = os.open(out, os.O_RDONLY); os.fsync(fd); os.close(fd)
else:
    import segyrs
    if op == "copy":
        segyrs.open(src).to_segy(out, sync=True)
    else:
        with segyrs.edit(out) as ed:
            ed.set_header("offset", np.full(ed.n_traces, 25, np.int32))
dt = time.perf_counter() - t
print(json.dumps({"s": dt}))
'''


def one(py, lib, op, src, out):
    r = subprocess.run([py, "-c", CHILD, lib, op, src, out], capture_output=True, text=True, check=False)
    for p in (out, out + ".part"):
        if os.path.exists(p):
            os.remove(p)
    if r.returncode:
        return {"error": r.stderr.strip().splitlines()[-1]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    py, rounds, src, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    os.makedirs(outdir, exist_ok=True)
    res = {}
    for op in ("copy", "edit_header"):
        t = {"segyio": [], "segyrs": []}
        for r in range(rounds):
            for lib in (("segyio", "segyrs") if r % 2 == 0 else ("segyrs", "segyio")):
                x = one(py, lib, op, src, os.path.join(outdir, f"out_{lib}.sgy"))
                t[lib].append(x.get("s", x.get("error")))
        row = {k: statistics.median(v) if all(isinstance(s, float) for s in v) else v for k, v in t.items()}
        res[op] = {"median_s": row, "rounds": rounds, "all": t}
        a, b = row["segyio"], row["segyrs"]
        ratio = f"{a / b:.1f}x" if isinstance(a, float) and isinstance(b, float) else "n/a"
        print(f"{os.path.basename(src)} {op:12s} segyio {a} s  segyrs {b} s  {ratio}", flush=True)
    shutil.rmtree(outdir, ignore_errors=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
