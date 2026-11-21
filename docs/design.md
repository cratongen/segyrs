# Design

These are the rules behind the [user guide](user-guide.md). Each one is tied to a ledger id. The title after the id is the upstream GitHub issue title. `S-` is equinor/segyio. `X-` is xarray, segysak, or mdio. segyrs does not copy those codebases. The ids name the behaviour the tests lock in.

Where a rule says the work uses every core, or sizes a buffer from available RAM, that describes the code, not a measurement. The timings in [What open reads](#what-open-reads) come from benchmarks on the development machine, with the method next to each table. The comparison with segyio is on [Benchmarks](benchmarks.md).

## Dtype

The numpy dtype is the sample format after a decode to native endian. Integers stay integers. IBM float is float32 plus `attrs["sample_format"] == "ibm_float32"`. IEEE float32 stays float32. There is no default cast to float64.

| Id | Upstream title | Rule in segyrs |
| --- | --- | --- |
| [S-01] | IBM360Converter converts to IEEE NaN for some values | IBM float decodes to float32. Overflow is infinity. The original 4 bytes stay available through `sample_bytes`. |
| [S-02] | Fixed point with gain data format | Format 4 raises `Format4NeedsGain` unless `gain` is `"rev0"` or `"rev2"`. It is never decoded as IEEE by default. |
| [S-03] | Tell segyio that a file has a different format than what the header says | `sample_format=` overrides the binary-header code and is stored on the volume. The header code stays visible. |
| [S-04] | Support for more numerical formats | Every code in rev 0, 1, 2.0, and 2.1 is implemented. int24 widens to int32. An unknown code is `UnknownFormat`. |
| [S-05] | Read sample interval as unsigned in python | Sample interval and the legacy sample count are unsigned. A rev 2 32-bit field overrides them when it is present. The coordinate step for bytes `0x9C40` is 40000 microseconds. |
| [X-01] | Use masked arrays while preserving int | Dead traces and gaps are the bool coordinate `trace_mask`. Integer samples stay integer. NaN is allowed only on floating formats. |
| [X-02] | converting int vars to floats when I where the enclosing ds? | `sel` and `isel` keep the sample dtype. segyrs does not route a trace pick through a `where` that needs a missing sentinel. |
| [X-03] | Variable of dtype int8 casted to float64 | Samples do not pass through a mask-and-scale decode. Format 8 stays int8. Attrs carry no fill value. |
| [X-05] | float32 instead of float64 when decoding int16 with scale_factor netcdf var using xarray | `scalco` and `scalel` apply in float64 to coordinates only. The raw integer is kept beside them. Amplitudes are not scaled. |
| [X-08] | Investigate Loading data as float32 not float64 (default). | IEEE float32 stays float32. IBM decodes to float32 with the `ibm_float32` attr. |
| [X-32] | BUG: Ingesting segy file with all sample values exactly zero fails in | An all-zero trace opens as zeros and stays live. `dead_trace` defaults to unset. |

Two lessons from issues this project does not reimplement: [X-06] ``nan`` values appearing when saving and loading from ``netCDF`` due to encoding, and [X-07] ``xr.decode_cf()`` fails to replace ``FillValue`` with ``np.nan`` with ``numpy >= 2.0`` when data array dtype is ``np.float32``. segyrs does not pack samples through a fill value. Zeros stay zeros. Optional Zarr export sets no integer fill and no scale factor, which is the X-06 lesson applied at the boundary.

On write, the opened format is the format written. `sample_format=` is the conversion ([S-35] Change segy format). The interval written is the integer microsecond count you already have, which is [S-29] Bad sample interval written due to floating point truncation.

## Endian

`endian=None` reads binary-header bytes 3297-3300 first. The constant `0x01020304` is stored in the file's own order, so the raw bytes are `01 02 03 04` (big), `04 03 02 01` (little), or `02 01 04 03` (pairwise). A zero signature is big-endian. Pairwise means each consecutive pair of bytes is swapped relative to big-endian.

Decoded samples and decoded header values are native integers and native floats. `attrs["source_endian"]` records the file. `header_bytes` and `sample_bytes` are the on-disk bytes, in file order. One accessor is native. The other is the file's own bytes.

| Id | Upstream title | Rule in segyrs |
| --- | --- | --- |
| [S-22] | SEG-Y rev2 file endianness | The signature is read before any other multi-byte field. You can still pass `endian=`. |
| [S-23] | trace header Field.buf always big endian? | Decoded values are native. Raw bytes stay in file order, so a second byteswap is not applied on top of a buffer that already lied. |
| [X-04] | Automatically decode netCDF data to native endianness | Decode to native endian in Rust, on all cores, before the array reaches Python. |
| [S-25] | Runtime Error trace count inconsistent with file size, trace lengths possibly of non-uniform | A signature that disagrees with the decoded sizes is `EndianMismatch`. |

The writer byteswaps back at the file edge from `source_endian`.

## Laziness

Open reads the textual header, the 400-byte binary header, and the trace-header columns the geometry needs. It does not read sample bytes. `sel` and `isel` read the traces in the selection. `.numpy()` materializes that selection. `attrs["chunk_shape"]` records whole traces in on-disk order, and those attrs survive `sel`.

| Id | Upstream title | Rule in segyrs |
| --- | --- | --- |
| [S-12] | How to load a SEGY file that does not fit into memory? | Open does not build a dense cube. Shape is the grid when one was inferred, otherwise `(trace, time)`. |
| [X-30] | xarray lazy indexing/loading is not sufficiently documented | The open docstring and `chunk_shape` say that samples are lazy. The first `isel` of one trace is what reads sample bytes. |
| [X-31] | Rules for propagating attrs and encoding | Revision, sample format, endian, textual header, and measurement system live on the volume and survive `sel`. |
| [X-15] | huge volume segy data loading | A sparse gather stays `(trace, time)` plus coordinates. `to_regular` is the explicit dense cube and it checks the box against available RAM. |
| [X-16] | OOM when allocating live mask | A header span whose dense box dwarfs the trace count raises `GeometryError` with min, max, and trace count. Open does not allocate that box. |
| [S-13] | mmap function | segyrs maps the file. If the operating system refuses, it reads by position. Callers do not call mmap themselves. |
| [S-10] | Time/Horizon slice mode | `sel` on one vertical index, and `slice_horizon`, read one sample window per trace. |

### What open reads

A header every trace is the limiter of a full scan. On a 1000 x 1000 x 1000 IEEE volume the trace stride is 4240 bytes, so the 240-byte headers touch almost every 4 KB page and a cold scan reads about the whole 4.24 GB file: on 2026-11-17, before the assumed-grid open, the cold full scan took 1.29 s (median) against 1.18 s for `dd` reading the file with `O_DIRECT` (the script for that run was not kept). Warm, one profiling run of the old open put most of its time in `infer`, which hashed every key on one thread; that is one run, not a benchmark. Faster column decoding cannot fix the cold case. Reading fewer headers can.

- `Scan::Auto` (default). For a reader that exposes its bytes and has more than 4096 traces, open reads traces 0 and 1, finds the end of the fastest and middle runs by exponential then binary search, and builds an arithmetic grid: key of trace `t` = origin + index * step on each axis. It then checks the last trace and 1024 headers (half evenly spaced, half from a fixed-seed generator) in parallel with `MADV_RANDOM`. If all fit and the box is not absurd, the geometry is that grid and `GeometrySource::Assumed`. The header columns become a `OnceLock`, scanned on first use.
- The assumption is checked where it matters. `read_window`, `sample_bytes`, `ops::check_traces`, and the writer's source selection check the keys (and the sample count when the binary header does not promise fixed length) of every trace they read: the header shares a page with the samples. A list of a sixteenth of the file or more, any whole-file header scan (`par_header_columns`, Python `scanned_columns`), Zarr export, and the viewer run `verify_geometry`, once per file. A mismatch is `Error::GeometryAssumption` (Python `GeometryAssumptionError`, a `GeometryError`), not a silent fallback: the `Volume` already has its shape, and changing it under a caller is worse than a clear error that says to reopen with `scan="full"`.
- Anything that does not fit (non-arithmetic numbering, one line, a broken sample, a remote reader) runs the full scan, which reports the same errors as before. The full scan now tries the same arithmetic grid on the scanned columns in one parallel pass before the hashing `infer`. The warm gain from that pass is not measured with interleaved rounds.
- `index_cache` is an opt-in sidecar for files the fast path refuses: 44 bytes per trace (all scanned columns), keyed by the canonical path, invalidated by length, mtime, and an FNV-1a fingerprint of the textual and binary headers, the first and last trace headers, and the scan parameters. An edit in the middle of the file is caught by the new mtime; an edit that also restores the size and the mtime is not.
- Sequential readahead (`MADV_SEQUENTIAL`) for the full scan was no faster cold than the default, so the scan uses `MADV_NORMAL` and selections `MADV_RANDOM`.
- Open leaves the map on `MADV_RANDOM`, so scattered selections read only their pages. Work that reads a large part of the file holds a `File::readahead()` guard, which sets `MADV_NORMAL` until it drops: the lazy header-column scan (`cols()`, so `verify_geometry` and `inline()` too), `par_header_columns`, `par_traces`, `read_window`, `decode` and `decode_indices` for a list of at least a sixteenth of the traces (`readahead_for`, the same threshold as `verify_traces`), the streaming ops (stats, rms, clip, time slice and horizon windows, crop, to_regular, resort, concat), the writer when it copies from a source, and Zarr export. Advice belongs to the mapping, not to a thread: guards nest and may live on several threads, a count under a lock switches to `MADV_NORMAL` with the first guard and back to `MADV_RANDOM` when the last one drops, and a scattered read that runs while another thread holds a guard simply gets readahead too. Without the guard, leaving `MADV_RANDOM` on made every whole-file read several times slower cold.

Measured on 2026-11-17: cold `MADV_RANDOM` for the life of the file ("before", main at `0d14654`) against the guard ("after", `ed4a535`). The script for this table was not kept; `bench_vs_segyio.py` in [`benchmarks/`](https://github.com/cratongen/segyrs/tree/main/benchmarks) times the same reads cold on current code. IEEE BE inline-sorted 1000 x 1000 x 1000 file (4.24 GB), `to_segy` on the 1000 x 1000 x 100 IEEE file (0.64 GB, output on the same NVMe, then deleted). Cold = `posix_fadvise(DONTNEED)`, then open plus the operation timed, in a fresh process. 32 cores, 5 interleaved rounds, medians with ranges, no other job running (the load average counts this benchmark's own I/O waits).

| Operation | Before | After |
| --- | --- | --- |
| `isel(time=500)` (touches every trace) | 10.17 s (7.68-10.60) | 1.63 s (1.51-1.72) |
| `headers["cdp_x"]` (lazy full header scan) | 8.23 s (7.60-10.25) | 1.53 s (1.49-1.58) |
| whole cube `numpy()` | 7.83 s (7.77-9.46) | 1.78 s (1.71-1.99) |
| `stats()` | 8.09 s (7.90-8.87) | 1.85 s (1.82-1.94) |
| `to_segy` copy, 1000 x 1000 x 100 | 1.45 s (1.43-1.48) | 1.02 s (0.97-1.05) |
| one inline (1000 contiguous traces) | 0.036 s (0.033-0.038) | 0.038 s (0.034-0.050) |
| one crossline (1000 scattered traces) | 0.041 s (0.039-0.041) | 0.043 s (0.041-0.050) |
| 1000 random single traces | 0.668 s (0.590-0.697) | 0.646 s (0.630-0.694) |

For comparison, `MADV_NORMAL` for the life of the file (measured by hand earlier on the same machine) took 0.196 s for the crossline and 0.87 s for the random traces, so the guard keeps the scattered reads on `MADV_RANDOM` timings.
- The `Arc<dyn ReadAt>` that every Python `File` holds since M5 has no measurable cost: warm `read_window` of one inline, one crossline, a 1e6-trace time slice, and 1000 random traces gave the same medians (within the run-to-run range) through `File<Storage>` and `File<Arc<dyn ReadAt>>`, 41 interleaved rounds. There is no separate local path.

Open from Python, 1000 x 1000 x 1000 files, 32 cores, NVMe, 5 interleaved rounds, medians with ranges, load average 14 to 31 from parallel builds. Cold = `posix_fadvise(DONTNEED)` first. "Before" is main at `2d5348f`.

| File | Mode | Before | `scan="auto"` | `scan="full"` |
| --- | --- | --- | --- | --- |
| IEEE BE, inline-sorted | cold | 1.381 s (1.339-2.079) | 0.040 s (0.033-0.042) | 1.292 s (1.263-1.863) |
| IEEE BE, inline-sorted | warm | 0.166 s (0.158-0.185) | 0.016 s (0.007-0.020) | 0.101 s (0.090-0.110) |
| IBM BE, crossline-sorted | cold | 1.384 s (1.327-1.830) | 0.035 s (0.034-0.036) | 1.356 s (1.261-1.475) |
| IBM BE, crossline-sorted | warm | 0.166 s (0.159-0.208) | 0.020 s (0.016-0.026) | 0.098 s (0.094-0.105) |
| int16 LE rev 2 | cold | 0.774 s (0.764-1.138) | 0.033 s (0.033-0.041) | 0.710 s (0.695-0.818) |
| int16 LE rev 2 | warm | 0.163 s (0.155-0.166) | 0.018 s (0.011-0.022) | 0.098 s (0.089-0.105) |

Cold open plus one inline, IEEE file: 1.362 s before, 0.057 s auto. The warm `scan="auto"` figure is mostly the first call in a fresh process (thread pool start, Python objects); a second open in the same process measured about 2 ms, Rust alone about 0.4 ms (single runs).

`binary_header_bytes` is the saved 400 bytes. `Editor.write_binary_header` writes them back unchanged, from [X-22] Disaster recovery of binary header.

## Parallelism

Scan and decode run with rayon in the process that called open, on all cores. The GIL is released around that work. Python receives numpy arrays of the finished columns. Buffers for whole-volume kernels are sized from available RAM. There is no compiled-in cap.

| Id | Upstream title | Rule in segyrs |
| --- | --- | --- |
| [S-07] | Question: fastest way to read headers into pandas dataframe | One parallel scan fills a column per field. There is no pandas frame and no per-trace Python loop. |
| [S-09] | TraceHeader duplication is time consuming | The bulk header array length equals the trace count. A copy is a byte-range copy of the 240-byte slots. |
| [X-33] | SEGY import spawns multiprocessing recursion | Open does not start a process pool. Child process count stays 0. |
| [S-11] | Problem to use segio in multi thread environment | A worker reopens by path (`test_s11_workers_reopen_by_path`). The library does not pickle a file handle and does not start processes. |
| [X-14] | Loading speed | Default chunks follow on-disk order: whole traces. Rechunk only when you ask, and only on optional Zarr export. |
| [X-29] | Slow performance of isel | `filter` evaluates a header predicate in Rust on the columnar table. A one-trace `isel` indexes a view of the trace grid, never a copy, on either sort order. `sel_points` and `isel_points` read scattered traces in one parallel call. |
| [S-15] | Volume support for out-of-order traces | M1 builds `(inline, crossline) -> trace` in parallel. `sel` uses the index. On-disk order is unchanged. |

## Errors

`Error` in Rust is `non_exhaustive`. PyO3 maps each variant to a Python class under `segyrs.SegyrsError`. Messages include the sizes or the key. There is no C enum ([S-39] Make the C error codes available at the python level for diagnosing data issues, [S-46] python 3.10 + segyio: SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats). Bindings are PyO3. S-46 is why: the hand-written C API is the thing this project does not grow.

| Id | Upstream title | What you catch |
| --- | --- | --- |
| [S-25] | Runtime Error trace count inconsistent with file size, trace lengths possibly of non-uniform | `EndianMismatch`, `SampleCountMismatch` (file length, expected length, remainder), and `NonUniformTraceLength` (trace index, binary `ns`, trace `ns`). `TruncatedError` is the file that ends inside a header. |
| [X-16] | OOM when allocating live mask | `GeometryError` with min, max, and trace count, before allocation. |
| [X-19] | Geometry issue/ pandas key error | `GeometryError` when you select a grid axis that was not inferred. Unstructured open itself succeeds. |
| [S-16] | Loading a segy-file that is not formed as a cube | Sparse open succeeds and `missing_pairs` lists the gaps. `DuplicateKey` names the repeated `(inline, crossline)`. |
| [X-18] | segy_loader failing to broadcast input array shape during conversion | `TraceLengthError` names the trace and both lengths. |
| [X-23] | Inconcistency in shapes (2D seismic) causes error when calling segy_loader | The binary-header `ns` is the stride. A per-trace `ns` that disagrees is `NonUniformTraceLength`, and both numbers are in the message. |
| [S-02] | Fixed point with gain data format | `Format4NeedsGain`. |
| [S-04] | Support for more numerical formats | `UnknownFormat` carries the `u16` code. |
| [S-35] | Change segy format | `FormatChangeError` on an in-place format change. The file bytes stay as they were. |
| [X-35] | combine_first of Datasets changes dtype of variable present only in one Dataset | `concat` checks format, interval, sample count, and endian, and raises `FormatChangeError` on a mismatch. |
| [S-32] | Writing time volume | A short sample vector raises `TraceLengthError` before any byte is written. |

[S-14] RuntimeError: unable to find sorting.Check iline, (189) and xline (193) in case is the open that must succeed. Unknown sort is a geometry report of `unsorted`.

## Viewer

`segyrs.view(vol)` (experimental, built from the repository, not in the 0.1.0 release) opens a native wgpu window and streams one inline slice, one crossline slice, or one time or depth slice from the map. The same renderer writes the notebook image. A picture of one slice does not read the rest of the file.

A 3D volume shows one slice: the first index of the slowest grid axis, or the current `sel` when that selection is already 2D. The image names the axis. You move the slice with `segyrs.view`. The notebook hook does not open a window. `view` runs on the main thread and returns when the window closes.

The 2D view is reduced to the pixel grid first, then the image is built once. With `vmin` and `vmax` set, only that view is read. When they are omitted, the range is the symmetric 98th percentile of the live samples in the drawn view. Masked traces stay out of the reduction. The rest of the file is not read. You do not have to run the M3 `stats` kernel before a picture.

Default axes are the 1D dimensions. A map of `cdp_x` and `cdp_y` is a separate view and reads headers only. A non-monotonic axis is sorted inside that 2D view when you ask. Otherwise the call raises `UnsortedCoordinate`.

There is no matplotlib and no second plotting stack.

| Id | Upstream title | Rule in segyrs |
| --- | --- | --- |
| [S-06] | Add quick start example with plot | The documented picture is one slice of a file under `test-data`. Samples outside that slice stay unread. |
| [X-09] | Why are there two compute calls for plot? | Downsample to the pixel grid first, then build the image once. An explicit range reads only that view. |
| [X-10] | Call .compute() in all plot methods? | A 3D volume does not upload the cube to draw one image. The image names the axis you would pass to `sel`. |
| [X-11] | Raise when pcolormesh coordinate is not sorted | `UnsortedCoordinate` unless you ask to sort that 2D view together with its samples. |
| [X-12] | when loading segy into xarray keep a running record of trace percentiles for plotting | The omitted range is the symmetric p98 of the live samples in the drawn view. Dead traces stay out. The file is not scanned to choose it. |
| [X-13] | Plotting datasets with variable coordinates | Default axes are the 1D dimensions. The map view reads `cdp_x` and `cdp_y` from headers and reads no samples. |

The window is wgpu. A machine with no display still has the notebook image, because that path writes pixels without opening a surface. Changing the clip, the polarity, or the colormap rewrites no disk. A new slice or a new zoom is what reads.
