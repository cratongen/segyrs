# Benchmarks

Every number on this page was read from a benchmark log. Nothing is rounded up, estimated or carried over from another machine. Where segyio wins, the row stays in the table and the reason follows it.

## Method

| Item | Setting |
| --- | --- |
| Date | 2026-11-17: the irregular-file run from 19:56 to 19:59, the regular-file and write runs from 20:00 to 20:14 (UTC-3), segyrs 0.1.0 built from that day's checkout |
| Machine | Intel i9-13980HX, 32 threads, 30 GB RAM, Micron 2400 NVMe |
| Harness | [`benchmarks/`](https://github.com/cratongen/segyrs/tree/main/benchmarks) in the repository: the generators, the three timing scripts and `check_irregular.py`, as they ran. Its README has the commands |
| segyio | segyio 1.9.15 with numpy 2.5.5 (both libraries in one Python 3.13 environment). Regular files: `segyio.open(path, iline=189, xline=193)`, once with `f.mmap()` (its documented read speed-up, the tuned side) and once without. Irregular files: `ignore_geometry=True` plus `f.mmap()` |
| segyrs | `segyrs.open(path)` with defaults. On the regular files `scan="auto"` checks about a thousand sampled headers and assumes the grid; on the irregular files it falls back to the full scan. The irregular run adds `scan="full"` and `index_cache` |
| Process | Every measurement runs in a fresh Python subprocess, so no run inherits a warm library or a cache from the one before |
| Rounds | 5 rounds per row (3 for the whole cube and for reading every trace). The sides alternate which goes first each round. The figure is the median |
| Cold | `posix_fadvise(POSIX_FADV_DONTNEED)` on the file, then open plus the operation, timed together |
| Warm | The file is read once into the page cache, then opened; only the operation is timed (for "open", the open itself) |
| Load | Load average 6.70 when the regular-file run started, and 2.33 to 17.93 across the irregular-file run, which counts the benchmark's own I/O waits |

The test files are synthetic: a Matérn random field written as SEG-Y, with the inline number at byte 189 and the crossline number at byte 193. The ratio column is segyio's time divided by segyrs's time, so a ratio above 1 means segyrs is faster and below 1 means segyio is faster.

The operations, each as both libraries write it:

| Row | segyio | segyrs |
| --- | --- | --- |
| One inline | `f.iline[1500]` | `vol.sel(inline=1500).numpy()` |
| One crossline | `f.xline[2500]` | `vol.sel(crossline=2500).numpy()` |
| Time slice | `f.depth_slice[500]` | `vol.isel(time=500).numpy()` |
| One header column | `f.attributes(181)[:]` | `vol.headers["cdp_x"]` |
| 1000 random traces | `f.trace[i]` in a loop | `vol.isel(inline=..., crossline=...).numpy()` in a loop |
| Whole cube | `segyio.tools.cube(f)` | `vol.numpy()` |

## Regular files against segyio with `mmap`

Three 1000 x 1000 grids of 1000 samples per trace, the same field in three encodings, against segyio's tuned configuration. Script: `bench_vs_segyio.py` with `SEGYIO_MMAP=1`.

**IEEE float32, big-endian, inline-sorted (4.24 GB)**

| Operation | Cold segyio | Cold segyrs | Cold ratio | Warm segyio | Warm segyrs | Warm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open | 0.2965 s | 0.0335 s | 8.8x | 0.0059 s | 0.0039 s | 1.5x |
| One inline (1000 traces) | 0.2663 s | 0.0414 s | 6.4x | 0.0014 s | 0.0020 s | 0.7x |
| One crossline (1000 traces) | 1.0386 s | 0.0420 s | 24.7x | 0.0042 s | 0.0017 s | 2.4x |
| Time slice (one sample of every trace) | 1.6577 s | 1.4742 s | 1.1x | 0.0975 s | 0.0447 s | 2.2x |
| One header column (`cdp_x`) | 1.6299 s | 1.4338 s | 1.1x | 0.0889 s | 0.0421 s | 2.1x |
| 1000 random traces, one call each | 1.0593 s | 0.6743 s | 1.6x | 0.0097 s | 0.0140 s | 0.7x |
| Whole cube into numpy | 2.0625 s | 1.6225 s | 1.3x | 0.8441 s | 0.2505 s | 3.4x |

**IBM float32, big-endian, crossline-sorted (4.24 GB)**

| Operation | Cold segyio | Cold segyrs | Cold ratio | Warm segyio | Warm segyrs | Warm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open | 0.2657 s | 0.0267 s | 10.0x | 0.0059 s | 0.0034 s | 1.7x |
| One inline (1000 traces) | 1.1854 s | 0.0412 s | 28.8x | 0.0046 s | 0.0019 s | 2.4x |
| One crossline (1000 traces) | 0.2708 s | 0.0440 s | 6.2x | 0.0024 s | 0.0019 s | 1.3x |
| Time slice (one sample of every trace) | 1.5502 s | 1.4134 s | 1.1x | 0.0826 s | 0.0425 s | 1.9x |
| One header column (`cdp_x`) | 1.6083 s | 1.3679 s | 1.2x | 0.0744 s | 0.0419 s | 1.8x |
| 1000 random traces, one call each | 1.0392 s | 0.7831 s | 1.3x | 0.0108 s | 0.0152 s | 0.7x |
| Whole cube into numpy | 2.9078 s | 1.6278 s | 1.8x | 1.7055 s | 0.2826 s | 6.0x |

**int16, little-endian, rev 2 (2.24 GB)**

| Operation | Cold segyio | Cold segyrs | Cold ratio | Warm segyio | Warm segyrs | Warm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open | 0.3214 s | 0.0320 s | 10.1x | 0.0055 s | 0.0036 s | 1.5x |
| One inline (1000 traces) | 0.2575 s | 0.0373 s | 6.9x | 0.0010 s | 0.0007 s | 1.4x |
| One crossline (1000 traces) | 0.8328 s | 0.0341 s | 24.4x | 0.0036 s | 0.0008 s | 4.6x |
| Time slice (one sample of every trace) | 0.9584 s | 0.7981 s | 1.2x | 0.0617 s | 0.0393 s | 1.6x |
| One header column (`cdp_x`) | 0.9238 s | 0.7789 s | 1.2x | 0.0547 s | 0.0406 s | 1.3x |
| 1000 random traces, one call each | 0.8808 s | 0.6037 s | 1.5x | 0.0090 s | 0.0125 s | 0.7x |
| Whole cube into numpy | 1.1718 s | 0.8895 s | 1.3x | 0.4485 s | 0.1459 s | 3.1x |

## Regular files against default segyio

The same files and operations, with segyio opened without `f.mmap()`. Script: `bench_vs_segyio.py` without `SEGYIO_MMAP`. The difference shows in the warm rows that touch every trace: without `mmap`, segyio's warm time slice and header column take 0.3414 s to 0.5374 s across the three files, and with it they take 0.0547 s to 0.0975 s. segyrs maps the file itself; you never call `mmap`.

**IEEE float32, big-endian, inline-sorted (4.24 GB)**

| Operation | Cold segyio | Cold segyrs | Cold ratio | Warm segyio | Warm segyrs | Warm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open | 0.2321 s | 0.0304 s | 7.6x | 0.0058 s | 0.0036 s | 1.6x |
| One inline (1000 traces) | 0.2639 s | 0.0426 s | 6.2x | 0.0020 s | 0.0021 s | 0.9x |
| One crossline (1000 traces) | 0.9645 s | 0.0493 s | 19.6x | 0.0024 s | 0.0018 s | 1.3x |
| Time slice (one sample of every trace) | 1.7630 s | 1.5040 s | 1.2x | 0.5364 s | 0.0419 s | 12.8x |
| One header column (`cdp_x`) | 1.6807 s | 1.5310 s | 1.1x | 0.5348 s | 0.0425 s | 12.6x |
| 1000 random traces, one call each | 0.8600 s | 0.7888 s | 1.1x | 0.0088 s | 0.0143 s | 0.6x |
| Whole cube into numpy | 2.2674 s | 1.6504 s | 1.4x | 1.2179 s | 0.2505 s | 4.9x |

**IBM float32, big-endian, crossline-sorted (4.24 GB)**

| Operation | Cold segyio | Cold segyrs | Cold ratio | Warm segyio | Warm segyrs | Warm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open | 0.2211 s | 0.0342 s | 6.5x | 0.0058 s | 0.0041 s | 1.4x |
| One inline (1000 traces) | 0.9926 s | 0.0474 s | 20.9x | 0.0036 s | 0.0018 s | 2.0x |
| One crossline (1000 traces) | 0.2260 s | 0.0414 s | 5.5x | 0.0025 s | 0.0023 s | 1.1x |
| Time slice (one sample of every trace) | 1.6118 s | 1.4012 s | 1.2x | 0.5374 s | 0.0449 s | 12.0x |
| One header column (`cdp_x`) | 1.9944 s | 1.4067 s | 1.4x | 0.5180 s | 0.0440 s | 11.8x |
| 1000 random traces, one call each | 0.9083 s | 0.7691 s | 1.2x | 0.0090 s | 0.0172 s | 0.5x |
| Whole cube into numpy | 3.1894 s | 1.6645 s | 1.9x | 2.1366 s | 0.2768 s | 7.7x |

**int16, little-endian, rev 2 (2.24 GB)**

| Operation | Cold segyio | Cold segyrs | Cold ratio | Warm segyio | Warm segyrs | Warm ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open | 0.2424 s | 0.0269 s | 9.0x | 0.0056 s | 0.0037 s | 1.5x |
| One inline (1000 traces) | 0.2238 s | 0.0394 s | 5.7x | 0.0012 s | 0.0007 s | 1.8x |
| One crossline (1000 traces) | 0.6008 s | 0.0421 s | 14.3x | 0.0018 s | 0.0009 s | 2.1x |
| Time slice (one sample of every trace) | 1.0692 s | 0.7732 s | 1.4x | 0.3541 s | 0.0405 s | 8.7x |
| One header column (`cdp_x`) | 1.0236 s | 0.7889 s | 1.3x | 0.3414 s | 0.0373 s | 9.2x |
| 1000 random traces, one call each | 0.7887 s | 0.5310 s | 1.5x | 0.0070 s | 0.0124 s | 0.6x |
| Whole cube into numpy | 1.2119 s | 0.9009 s | 1.3x | 0.6693 s | 0.1420 s | 4.7x |

## Writing

Both libraries write from Python, each measurement in a fresh process, 5 rounds with the sides alternating, medians with the range of the rounds. The command is `bench_write_vs_segyio.py "$PY" 5 1000x1000x100/matern_ieee_be_inline.sgy out` from [`benchmarks/`](https://github.com/cratongen/segyrs/tree/main/benchmarks). The source is the IEEE inline-sorted 1000 x 1000 x 100 file `1000x1000x100/matern_ieee_be_inline.sgy`: 1000000 traces of 100 samples, 0.64 GB. Outputs are deleted after each run. The load average was 3.64 when the run ended. A lint job (`cargo clippy`, about 26 s) ran on the same machine during part of this run. An earlier run of the same script on a quiet machine, before the readahead fix, gave the same picture: copy 24.2 s against 0.33 s, header edit 3.96 s against 0.48 s.

| Operation | segyio | segyrs | Ratio |
| --- | ---: | ---: | ---: |
| Copy the whole file to a new SEG-Y, same format, synced to disk | 24.647 s (23.245 to 32.579 s) | 0.349 s (0.324 to 0.543 s) | 70.6x |
| Set one header word (`offset`) on every trace, in place, synced | 3.975 s (3.862 to 4.102 s) | 0.486 s (0.464 to 0.497 s) | 8.2x |

segyio copies with `segyio.create` from the source's spec, then assigns `text[0]`, `bin`, `header` and `trace`, then `fsync`. segyrs runs `segyrs.open(src).to_segy(out, sync=True)`, which streams the raw trace bytes through the writer, the same path as a crop. The header edit is a Python loop of `f.header[i] = {offset: 25}` in segyio, and one `ed.set_header("offset", values)` call in segyrs.

## Where segyio wins, and why

**Warm single reads in a loop.** "1000 random traces, one call each" is slower in segyrs on every file, warm. Each `isel(...).numpy()` builds a labelled `Volume`, crosses from Python into Rust, and returns a new array, once per trace. segyio's `f.trace[i]` does less per call. On the IEEE file a warm single inline is also slower: 0.0020 s against 0.0014 s for segyio with `mmap`, and 0.0021 s against 0.0020 s without. When the file is cold, the disk dominates and segyrs wins every row.

For scattered traces, segyrs has a call built for the job: `vol.sel_points(inline=[...], crossline=[...])` or `vol.isel_points(trace=[...])` reads every point in one parallel decode with the GIL released. This benchmark loops over `isel` on purpose, to match segyio's loop, so the `sel_points` time is not measured here.

**Opening an irregular file.** segyio opened with `ignore_geometry=True` reads no trace header at all: it takes the trace count from the file size and returns. That open costs 0.0031 s cold. segyrs reads and checks every trace header of a file it cannot assume to be a regular grid, then builds the `(inline, crossline)` index, so the file opens with its real layout and `sel(inline=...)` works on it. On these files one header per trace touches most pages, so a cold full scan costs about as much as reading the file.

The work moves rather than disappears. segyio with `ignore_geometry` knows nothing about the file, so finding one inline means reading the whole inline column first, and that cold read costs more than segyrs's whole open plus the inline on the cached path below.

**What `index_cache` changes.** `segyrs.open(path, index_cache=dir)` keeps the scanned header columns (44 bytes per trace) in a sidecar file. The next open loads the sidecar instead of the headers. On the irregular files it cut the cold open from 0.2394 s to 0.0943 s (sparse), 0.3127 s to 0.0937 s (unsorted) and 0.6355 s to 0.2200 s (pre-stack), and with it, segyrs reads one inline cold faster than segyio does on all three. The sidecar is checked against the file's size, modification time and a fingerprint of its headers, and a stale one is replaced, never trusted. See [How much open reads](user-guide.md#how-much-open-reads-scan-and-index_cache).

**Reading every trace of an irregular file, cold.** segyio's `f.trace.raw[:]` is one sequential pass. segyrs's `vol.numpy()` on a `(trace, time)` volume is slower cold on every irregular file, with or without the sidecar.

## Irregular files

Four files cut from a 1000 x 1000 x 100 field, IEEE float32, big-endian, rev 1. Script: `bench_irregular.py`. segyio has to open them with `ignore_geometry=True`: with geometry on it refuses three of them (see [Correctness](#correctness)).

| File | What it is | Traces | Size |
| --- | --- | ---: | ---: |
| `sparse_holes` | Inline-sorted, an irregular survey outline plus random holes, about 35% of slots empty | 637995 | 0.41 GB |
| `unsorted` | Every slot of the grid once, in random trace order | 1000000 | 0.64 GB |
| `prestack_varfold` | CDP-sorted pre-stack, 400 x 400 CDPs, fold 1 to 24 varying smoothly | 2025842 | 1.30 GB |
| `regular_one_bad` | A regular inline-sorted grid where one trace near the middle has a wrong crossline header | 1000000 | 0.64 GB |

Each segyrs column is one way to open: `scan="auto"` (the default), `scan="full"`, and `index_cache` with a sidecar written once before timing. The segyio side finds an inline by reading the inline header column and then each matching trace. A `cdp_x` time of 0.0000 s means segyrs had decoded that column during the open scan, so the read was a lookup below the log's 0.1 ms resolution. "Raises" is `GeometryAssumptionError`, explained under [Correctness](#correctness).

**sparse_holes**

| Operation | Mode | segyio (`ignore_geometry`) | segyrs `scan="auto"` | segyrs `scan="full"` | segyrs `index_cache` |
| --- | --- | ---: | ---: | ---: | ---: |
| Open | cold | 0.0031 s | 0.2394 s | 0.2335 s | 0.0943 s |
| Open | warm | 0.0004 s | 0.0950 s | 0.0997 s | 0.0807 s |
| Every trace of one inline | cold | 0.1356 s | 0.2303 s | 0.2328 s | 0.1013 s |
| Every trace of one inline | warm | 0.0120 s | 0.0027 s | 0.0030 s | 0.0046 s |
| `cdp_x` column | cold | 0.1274 s | 0.2289 s | 0.2349 s | 0.0907 s |
| `cdp_x` column | warm | 0.0070 s | 0.0000 s | 0.0000 s | 0.0000 s |
| Every trace | cold | 0.1637 s | 0.2418 s | 0.2432 s | 0.2481 s |
| Every trace | warm | 0.0632 s | 0.0184 s | 0.0174 s | 0.0193 s |

**unsorted**

| Operation | Mode | segyio (`ignore_geometry`) | segyrs `scan="auto"` | segyrs `scan="full"` | segyrs `index_cache` |
| --- | --- | ---: | ---: | ---: | ---: |
| Open | cold | 0.0032 s | 0.3127 s | 0.3041 s | 0.0937 s |
| Open | warm | 0.0003 s | 0.1057 s | 0.1118 s | 0.0832 s |
| Every trace of one inline | cold | 0.1961 s | 0.3200 s | 0.3270 s | 0.1096 s |
| Every trace of one inline | warm | 0.0242 s | 0.0042 s | 0.0041 s | 0.0056 s |
| `cdp_x` column | cold | 0.2014 s | 0.3058 s | 0.3096 s | 0.0974 s |
| `cdp_x` column | warm | 0.0109 s | 0.0000 s | 0.0000 s | 0.0000 s |
| Every trace | cold | 0.2524 s | 0.3640 s | 0.3359 s | 0.3049 s |
| Every trace | warm | 0.0955 s | 0.0279 s | 0.0294 s | 0.0290 s |

**prestack_varfold**

| Operation | Mode | segyio (`ignore_geometry`) | segyrs `scan="auto"` | segyrs `scan="full"` | segyrs `index_cache` |
| --- | --- | ---: | ---: | ---: | ---: |
| Open | cold | 0.0032 s | 0.6355 s | 0.6227 s | 0.2200 s |
| Open | warm | 0.0003 s | 0.2344 s | 0.2387 s | 0.1990 s |
| Every trace of one inline | cold | 0.3937 s | 0.6183 s | 0.6309 s | 0.2222 s |
| Every trace of one inline | warm | 0.0274 s | 0.0074 s | 0.0076 s | 0.0087 s |
| `cdp_x` column | cold | 0.3888 s | 0.6277 s | 0.6449 s | 0.2202 s |
| `cdp_x` column | warm | 0.0214 s | 0.0000 s | 0.0000 s | 0.0000 s |
| Every trace | cold | 0.5056 s | 0.6835 s | 0.6851 s | 0.6019 s |
| Every trace | warm | 0.1835 s | 0.0532 s | 0.0533 s | 0.0560 s |

**regular_one_bad**

| Operation | Mode | segyio (`ignore_geometry`) | segyrs `scan="auto"` | segyrs `scan="full"` | segyrs `index_cache` |
| --- | --- | ---: | ---: | ---: | ---: |
| Open | cold | 0.0032 s | 0.0288 s | 0.3506 s | 0.0168 s |
| Open | warm | 0.0003 s | 0.0033 s | 0.1693 s | 0.0035 s |
| Every trace of one inline | cold | 0.2007 s | raises | 0.4061 s | raises |
| Every trace of one inline | warm | 0.0293 s | raises | 0.0047 s | raises |
| `cdp_x` column | cold | 0.1860 s | raises | 0.4256 s | raises |
| `cdp_x` column | warm | 0.0265 s | raises | 0.0000 s | raises |
| Every trace | cold | 0.2590 s | raises | 0.4386 s | raises |
| Every trace | warm | 0.1064 s | raises | 0.0276 s | raises |

## Correctness

Speed only counts when the answer is right. The same four irregular files, checked against the truth the generator recorded (`check_irregular.py`):

| File | segyio with geometry | segyrs |
| --- | --- | --- |
| `sparse_holes` | Refuses to open: `ValueError: Inlines inconsistent, expect all inlines to be unique` | Opens as `(trace, time)`, kind `sparse`, 637995 traces. `sel(inline=1500)` returns 849 traces, the true count, with samples equal to the field |
| `unsorted` | Refuses to open: `RuntimeError: unable to find sorting.Check iline, (189) and xline (193) in case you are sure the file is a 3D sorted volume` | Opens as `(trace, time)`, kind `unsorted`, 1000000 traces. `sel(inline=1500)` returns 1000 traces, samples equal to the field |
| `prestack_varfold` | Refuses to open: `ValueError: Inlines inconsistent, expect all inlines to be unique` | Opens as `(trace, time)`, kind `sparse`, 2025842 traces. `sel(inline=1200, crossline=2000)` returns the 4 traces of that CDP, the true fold |
| `regular_one_bad` | Opens as a clean 1000 x 1000 grid with crosslines up to 2999. `f.iline[1500]` returns 1000 traces and no warning, although one of them says crossline 7000 in its header | `scan="auto"` opens on the assumed grid, reads inline 1200 correctly, and raises `GeometryAssumptionError` on inline 1500. `scan="full"` opens the file as it is: `(trace, time)`, kind `sparse` |

The last row is the one that matters most. segyio places traces by their position in the file, so a header that disagrees with the position goes unnoticed: here the samples happen to sit in the right slot, but if the header were right and the trace misplaced, the same call would return the wrong trace without a word. segyrs checks the key of every trace it reads against the grid it assumed. The error names the trace and both keys:

```text
GeometryAssumptionError: trace 500617 has (inline, crossline, offset) = (1500, 7000, 0),
but the regular grid assumed at open puts (1500, 2617, 0) there; reopen with a full
header scan (Scan::Full, scan="full" in Python)
```

The open checked the last trace and 1024 headers spread over the file, missed the bad one, and assumed the grid. Nothing was returned for inline 1500: the read checked each trace before handing back samples. A full scan reports the file as it is, and `sel(inline=1500)` on it returns all 1000 traces of that inline.

## Earlier measurements

[Design: What open reads](design.md#what-open-reads) has the measurements behind the open path: `scan="auto"` against a full scan, and the readahead guard against `MADV_RANDOM`. [Writing SEG-Y: Performance](writing.md#performance) has the write and in-place edit benchmarks in Rust.
