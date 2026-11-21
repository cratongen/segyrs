"""What segyrs and segyio return on the irregular files, checked against truth.npz and the field."""
import os

import numpy as np
import segyio
import segyrs

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "irregular")
field = np.load(os.path.join(os.path.dirname(D), "1000x1000x100", "matern_field_1000x1000x100.npy"), mmap_mode="r")
T = np.load(os.path.join(D, "truth.npz"))


def show(tag, fn):
    try:
        print(f"  {tag}: {fn()}")
    except Exception as e:
        print(f"  {tag}: RAISES {type(e).__name__}: {str(e)[:200]}")


for name in ["sparse_holes", "unsorted", "prestack_varfold", "regular_one_bad"]:
    p = os.path.join(D, name + ".sgy")
    print(f"== {name}")
    for scan in ("auto", "full"):
        show(f"segyrs scan={scan} open", lambda: (lambda v: f"dims={v.dims} shape={v.shape} kind={v.geometry.kind} source={getattr(v.geometry, 'source', 'scan')}")(segyrs.open(p, scan=scan)))
    show("segyio open with geometry", lambda: (lambda f: f"sorting={f.sorting} ilines={len(f.ilines)} xlines={len(f.xlines)} offsets={len(f.offsets)}")(segyio.open(p, iline=189, xline=193)))

    if name in ("sparse_holes", "unsorted"):
        il_t, xl_t = T[f"{name.split('_')[0]}_il"], T[f"{name.split('_')[0]}_xl"]
        for scan in ("auto", "full"):
            def chk(scan=scan):
                v = segyrs.open(p, scan=scan).sel(inline=1500)
                got = v.numpy().reshape(-1, 100)
                c = v.coords
                xl = np.asarray(c["crossline"]).reshape(-1) - 2000
                want = np.asarray(field[500, xl])
                exp_n = int(np.sum(il_t == 500))
                return f"inline 1500: {got.shape[0]} traces (truth {exp_n}), data equal={np.array_equal(got, want)}"
            show(f"segyrs scan={scan} sel(inline=1500)", chk)
    if name == "prestack_varfold":
        il_t, xl_t, off_t = T["prestack_il"], T["prestack_xl"], T["prestack_off"]
        def chk():
            v = segyrs.open(p).sel(inline=1200, crossline=1200 - 1000 + 2000 - 200)
            return f"cdp (1200, 2000): {v.numpy().reshape(-1, 100).shape[0]} traces (truth {int(np.sum((il_t == 200) & (xl_t == 0)))})"
        show("segyrs gather", chk)
    if name == "regular_one_bad":
        bad = int(T["bad_trace"][0])
        for scan in ("auto", "full"):
            def chk(scan=scan):
                v = segyrs.open(p, scan=scan)
                out = v.sel(inline=1500).numpy().reshape(-1, 100)
                return f"inline 1500 read {out.shape[0]} traces"
            show(f"segyrs scan={scan} read inline 1500 (holds the bad trace)", chk)
            def chk2(scan=scan):
                v = segyrs.open(p, scan=scan)
                out = v.sel(inline=1200).numpy().reshape(-1, 100)
                return f"inline 1200 read {out.shape[0]} traces, equal={np.array_equal(out, np.asarray(field[200]))}"
            show(f"segyrs scan={scan} read inline 1200 (clean)", chk2)
        def sio():
            f = segyio.open(p, iline=189, xline=193)
            a = f.iline[1500]
            return f"iline[1500] shape {a.shape}, equal to field={np.array_equal(a, np.asarray(field[500]))}, xlines max={f.xlines.max()}"
        show("segyio iline[1500] with geometry", sio)
