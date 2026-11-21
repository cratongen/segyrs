<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://cratongen.github.io/segyrs/brand/wordmark-white.svg">
    <img alt="segyrs" src="https://cratongen.github.io/segyrs/brand/wordmark-black.svg" width="372">
  </picture>
</p>

**segyrs is my personal tool for handling SEG-Y files. I wrote it because I had issues handling SEG-Y files with the existing libraries. It is a Rust core with Python bindings.**

segyrs reads and writes SEG-Y rev 0, 1, 2.0 and 2.1 on every core. You open a file, select by label or position, and get numpy arrays back. It draws its own pictures in a notebook, and in a native GPU window when you build it from the repository. Numpy is the only Python runtime dependency: no xarray, no dask, no matplotlib, and no command-line program.

segyrs is alpha software, and 0.1.0 is its first public release: the API may still change between minor versions. Reading, writing, in-place edits, the streaming operations and the notebook picture work in Rust and Python, and every planned milestone is in ([Roadmap](https://cratongen.github.io/segyrs/roadmap/)). Object-store reads and Zarr export are in the same wheel and are experimental. The viewer window is not part of the 0.1.0 release: it builds from the repository and will be published later.

[Docs](https://cratongen.github.io/segyrs/) · [User guide](https://cratongen.github.io/segyrs/user-guide/) · [Benchmarks](https://cratongen.github.io/segyrs/benchmarks/) · [Design](https://cratongen.github.io/segyrs/design/) · [Roadmap](https://cratongen.github.io/segyrs/roadmap/)

## Install

```bash
pip install segyrs
```

That installs one abi3 wheel for CPython 3.11 and newer on Linux (x86_64, aarch64, musl), macOS (Intel, Apple silicon) and Windows x64: reading, selection, headers, writing, in-place edits, the streaming operations, the notebook picture, `segyrs.open` on `s3://`, `gs://`, `az://` and `http(s)://` URLs, and `Volume.to_zarr`. Numpy is its only dependency. segyrs is the only package of the 0.1.0 release.

[Install and release](https://cratongen.github.io/segyrs/install-and-release/) has the platform table. The native viewer window (`segyrs.view`) is not part of the 0.1.0 release on PyPI; it will be published later. To build segyrs and the viewer from a checkout, use [mise](https://mise.jdx.dev/):

```bash
mise install       # the pinned Rust, Python, maturin and uv
mise run build     # segyrs and the viewer, into .venv
```

## Quick start

No SEG-Y file at hand? This writes a small one, 50 inlines by 60 crosslines of 200 samples:

<!-- doc-block: setup -->
```python
import numpy as np
import segyrs

cube = np.random.default_rng(0).standard_normal((50, 60, 200)).astype(np.float32)
segyrs.create("survey.sgy", cube, sample_format="ibm_float32", sample_interval_us=4000,
              inline=np.arange(1000, 1050), crossline=np.arange(2000, 2060))
```

Then open it, or your own file with its own labels:

```python
vol = segyrs.open("survey.sgy")      # headers only; no sample byte is read yet
vol.dims                             # ("inline", "crossline", "time")
vol.geometry.kind                    # "inline_major", "unsorted", "sparse", ...

il = vol.sel(inline=1020)            # by label: one inline, still lazy
il.numpy()                           # reads that inline's traces, nothing else
vol.isel(crossline=0, time=slice(0, 100)).numpy()   # by position

vol.headers["cdp_x"]                 # one numpy column per header field

pts = vol.sel_points(inline=[1001, 1020, 1049], crossline=[2005, 2030, 2059])
pts.numpy()                          # scattered traces in one parallel read

il.plot().save("il1020.png")         # drawn in Rust, no matplotlib
# segyrs.view(vol)                   # native wgpu window, built from the repository; needs a display

vol.sel(inline=slice(1010, 1030)).to_segy("crop.sgy")   # IBM stays IBM
with segyrs.edit("crop.sgy") as ed:                     # in place, same layout
    ed.set_header("offset", np.full(ed.n_traces, 25, np.int32))
segyrs.validate("crop.sgy")                             # the layout, or ValidationError
```

In Jupyter a 2D selection shows itself as a picture. `segyrs.view` opens a window on the same plane. The [user guide](https://cratongen.github.io/segyrs/user-guide/) covers every option.

## What segyrs fixes

segyrs is a clean-room rewrite. It follows the SEG technical standards, and uses segyio only as a behavioural reference: its documentation and its issues, never its code. Each upstream problem is a GitHub issue here with a ledger id (`S-` for segyio, `X-` for xarray, segysak and mdio), and a test carries the same id. The [segyio ledger](https://github.com/cratongen/segyrs/issues?q=is%3Aissue+label%3Aupstream%3Asegyio) and the [xarray ledger](https://github.com/cratongen/segyrs/issues?q=is%3Aissue+label%3Aupstream%3Axarray) list them all.

| segyio | segyrs |
| --- | --- |
| An unsorted or sparse file does not open with geometry (S-14, S-16) | It opens. `geometry.kind` names the layout, and `sel(inline=..., crossline=...)` still finds the trace (S-15) |
| A file larger than RAM cannot be wrapped as a lazy array (S-12) | Open reads no samples. Each selection reads its own traces |
| Header columns mean a Python loop per trace (S-07, S-09) | `vol.headers["cdp_x"]` decodes one column on all cores |
| IBM float overflow can become NaN (S-01) | It becomes infinity. Integers stay integers, and every rev 2 format decodes (S-04) |
| The rev 2 byte-order word is not read, and raw header buffers stay big-endian (S-22, S-23) | The rev 2 signature picks the byte order. Decoded values are native, raw bytes stay in file order |
| One error message covers three different problems (S-25) | `SampleCountMismatch`, `EndianMismatch` and `NonUniformTraceLength` |
| Writing a numpy cube needs a spec object (S-31) | `segyrs.create` takes the array. `to_segy` keeps the source format (S-35) |
| A cropped window can lose its time (S-36) | The delay moves with the window |
| No picture in the quick start (S-06) | `vol.sel(inline=...)` is a picture in Jupyter |

[Design](https://cratongen.github.io/segyrs/design/) ties each rule to its ledger id and the upstream title.

## Benchmarks

Measured on 2026-11-17 on one machine (Intel i9-13980HX, 32 threads, 30 GB RAM, Micron 2400 NVMe), every number a median of interleaved rounds, each measurement in a fresh process. Cold means the file was evicted from the page cache first. The other side is segyio 1.9.15 with `f.mmap()`, its documented speed-up. The [benchmarks page](https://cratongen.github.io/segyrs/benchmarks/) has every row, the method, the load, the losses and the reasons for them, and [`benchmarks/`](https://github.com/cratongen/segyrs/tree/main/benchmarks) holds the scripts and the commands that produced them.

1000 x 1000 traces of 1000 samples (4.24 GB, IEEE float32, inline-sorted):

| Operation | segyio | segyrs |
| --- | --- | --- |
| Open, cold | 0.2965 s | 0.0335 s |
| One crossline, cold | 1.0386 s | 0.0420 s |
| Whole cube into numpy, warm | 0.8441 s | 0.2505 s |
| One inline, warm | 0.0014 s | 0.0020 s |
| 1000 random traces, one call each, warm | 0.0097 s | 0.0140 s |

Writing, on 1000000 traces: a whole-file copy synced to disk took 0.349 s against 24.647 s, and setting one header word on every trace in place took 0.486 s against 3.975 s.

segyio wins the last two rows of the table: a warm single read is mostly per-call overhead, and segyrs pays for a labelled `Volume` on each call. It also opens an irregular file faster when you pass `ignore_geometry=True`, because it then reads no trace headers. segyrs scans and checks them so the file opens with its real layout. `index_cache` cuts that cost on the next open.

## Documentation

- [Docs home](https://cratongen.github.io/segyrs/)
- [User guide](https://cratongen.github.io/segyrs/user-guide/): open, `Volume`, selection, headers, geometry, errors
- [Exploring a 3D volume](https://cratongen.github.io/segyrs/examples/explore-3d-volume/): a worked example on a seeded synthetic volume, from open to the errors you meet
- [Pre-stack gathers and 2D lines](https://cratongen.github.io/segyrs/examples/prestack-and-2d/): gathers with varying fold, 2D lines, per-trace delays, sample formats, and the typed errors of a file you do not trust
- [Writing and editing SEG-Y](https://cratongen.github.io/segyrs/examples/write-and-edit/): create from numpy arrays, convert formats, edit in place, prove the round trip, and write shot gathers
- [Writing SEG-Y](https://cratongen.github.io/segyrs/writing/) and [Manipulation at scale](https://cratongen.github.io/segyrs/manipulation/)
- [Viewer and notebook](https://cratongen.github.io/segyrs/viewer/) and [Cloud and Zarr](https://cratongen.github.io/segyrs/cloud-and-zarr/)
- [Sample formats](https://cratongen.github.io/segyrs/formats/), [Design](https://cratongen.github.io/segyrs/design/), [Benchmarks](https://cratongen.github.io/segyrs/benchmarks/)
- [Install and release](https://cratongen.github.io/segyrs/install-and-release/), [Roadmap](https://cratongen.github.io/segyrs/roadmap/), [Contributing](https://cratongen.github.io/segyrs/contributing/), [Brand](https://cratongen.github.io/segyrs/brand/)

The same pages build into a site with `mise run docs`.

## Licence

Copyright 2026 the segyrs authors. segyrs is licensed under the Apache License 2.0 (`Apache-2.0`): [LICENSE](https://github.com/cratongen/segyrs/blob/main/LICENSE). Each wheel also carries `THIRD-PARTY-LICENSES.txt`, the licences of the Rust crates linked into it. The test files in `test-data/` are not under the segyrs licence: they come from segyio (LGPL-3.0) and include F3 Netherlands field data (CC BY-SA 3.0), and they are not in the packages. See [test-data/LICENSE.md](https://github.com/cratongen/segyrs/blob/main/test-data/LICENSE.md).
