---
title: segyrs
---

<div class="sg-hero" markdown>

![segyrs](brand/wordmark-black.svg#only-light){ width="372" }
![segyrs](brand/wordmark-white.svg#only-dark){ width="372" }

<p class="sg-pitch">segyrs is my personal tool for handling SEG-Y files. I wrote it because I had issues handling SEG-Y files with the existing libraries. It is a Rust core with Python bindings.</p>

[User guide](user-guide.md){ .md-button .md-button--primary }
[Benchmarks](benchmarks.md){ .md-button }
[GitHub](https://github.com/cratongen/segyrs){ .md-button }

</div>

segyrs reads and writes SEG-Y rev 0, rev 1, rev 2.0 and rev 2.1. The Rust library does the work; Python calls it and gets numpy arrays back. Numpy is the only Python runtime dependency. There is no xarray, no dask, no matplotlib and no command-line program: the labelled volume, the streaming kernels and the pictures belong to segyrs.

segyrs is alpha software, and 0.1.0 is its first public release: the API may still change between minor versions. Reading, writing, in-place edits, the streaming operations and the notebook picture work in Rust and Python, and every planned milestone is in ([Roadmap](roadmap.md)). `pip install segyrs` installs everything in the release, including the experimental object-store reads and Zarr export. The viewer window is not part of the 0.1.0 release: it builds from the repository and will be published later ([Install and release](install-and-release.md)).

## Quick start

```bash
pip install segyrs
```

No SEG-Y file at hand? This writes a small one, 50 inlines by 60 crosslines of 200 samples. The examples in these pages use it:

<!-- doc-block: setup -->
```python
import numpy as np
import segyrs

cube = np.random.default_rng(0).standard_normal((50, 60, 200)).astype(np.float32)
segyrs.create("survey.sgy", cube, sample_format="ibm_float32", sample_interval_us=4000,
              inline=np.arange(1000, 1050), crossline=np.arange(2000, 2060))
```

```python
vol = segyrs.open("survey.sgy")           # reads headers, not samples
section = vol.sel(inline=1020).numpy()    # reads one inline's traces
vol.sel(inline=1020).plot().save("il1020.png")
```

## Pages

| Page | What it covers |
| --- | --- |
| [User guide](user-guide.md) | `segyrs.open`, `Volume`, `sel` and `isel`, scattered points, headers, geometry, errors |
| [Writing SEG-Y](writing.md) | `create`, `to_segy`, in-place `edit`, `validate`, IBM encoding, the round-trip proof |
| [Manipulation at scale](manipulation.md) | Windows, filter, crop, resort, dense grids, concat and the streaming kernels, inside a RAM budget |
| [Exploring a 3D volume](examples/explore-3d-volume.md) | A worked example: open lazily, read the geometry report, select and draw sections and a time slice, decode trace headers and scalars, unsorted and sparse files, and the errors you meet |
| [Pre-stack gathers and 2D lines](examples/prestack-and-2d.md) | A worked example: CDP gathers with constant and varying fold, offsets checked against moveout and coordinates, 2D lines, per-trace delays, IBM float and other sample formats, and how to check a file you do not trust |
| [Writing and editing SEG-Y](examples/write-and-edit.md) | A worked example: a file from numpy arrays with its geometry headers checked byte by byte, sample format and byte order conversions, in-place header and sample edits with a byte diff that proves what changed, round trips, shot gathers for the geophysics hand-off, and the errors that stop a bad write |
| [Viewer and notebook](viewer.md) | The notebook picture, `Volume.plot`, and the native wgpu window from `segyrs.view` (experimental, built from the repository, not in the 0.1.0 release) |
| [Cloud and Zarr](cloud-and-zarr.md) | `segyrs.open` on `s3://`, `gs://`, `az://` and `http(s)://` URLs, and `Volume.to_zarr` (experimental, in the segyrs wheel) |
| [Sample formats](formats.md) | Every binary-header format code, the revision that added it, and the numpy dtype you get |
| [Benchmarks](benchmarks.md) | Measured timings against segyio, wins and losses, and the correctness checks on irregular files |
| [Design](design.md) | The rules behind the API, each tied to the upstream issue it answers |
| [Install and release](install-and-release.md) | Wheels, platforms, CI, the release workflow and this site |
| [Release checklist](release-checklist.md) | What is ready for 0.1.0, the owner steps, and the minetypes adoption that follows |
| [Roadmap](roadmap.md) | M1 to M5: goals, features, non-goals, exit criteria |
| [Contributing](contributing.md) | Issues first, Conventional Commits, the clean-room rule, mise tasks |
| [Brand](brand.md) | The mark, the palette, the type and how to use them |

## Licence

Copyright 2026 the segyrs authors. segyrs is dual-licensed under MIT OR Apache-2.0; pick either. See [LICENSE-MIT](https://github.com/cratongen/segyrs/blob/main/LICENSE-MIT) and [LICENSE-APACHE](https://github.com/cratongen/segyrs/blob/main/LICENSE-APACHE). Each wheel also carries `THIRD-PARTY-LICENSES.txt`, the licences of the Rust crates linked into it. The test files in `test-data/` are not under the segyrs licence: they come from segyio (LGPL-3.0) and include F3 Netherlands field data (CC BY-SA 3.0), and they are not in the packages. See [test-data/LICENSE.md](https://github.com/cratongen/segyrs/blob/main/test-data/LICENSE.md).
