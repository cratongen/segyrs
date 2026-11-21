# Roadmap

Milestones run from the read path and `Volume` (M1), through write and edit (M2), manipulation at scale (M3), the viewer and the notebook (M4), and interop and cloud (M5). A milestone is done when its exit criteria hold and the tests pass with the docs updated in the same change.

segyrs has no xarray dependency, no dask, and no matplotlib. The labelled volume, the streaming kernels, and the pictures belong to segyrs. There is no command-line program.

M1 to M5 are implemented in Rust and exposed in Python. What remains is the first PyPI release (its workflow is in `.github/workflows/release.yml`, see [Install and release](install-and-release.md)) and the issues still open on GitHub.

Dimension names are `inline`, `crossline`, `offset`, then `time` or `depth`, in Rust and in Python. Header columns use `inline` and `crossline`. The vertical dimension is `time` unless you pass `depth=True`. segyrs does not infer depth from the textual header.

## M1: Read path and Volume

### Goal

A clean-room Rust SEG-Y reader (rev 0, 1, 2.0, and 2.1) with a Python `Volume` on top. Open does not build a dense cube. Sample formats stay intact. Decode is native-endian and uses every core. Trace headers are columnar. Geometry is a report, so unsorted and sparse files still open. The trace index is part of this milestone, so `sel(inline=..., crossline=...)` works on an unsorted file.

### Features

- The crate exposes `File`, `OpenOptions`, `ReadAt`, `Storage`, `Layout`, `Revision`, `Endian` (including pairwise), `SampleFormat`, `Gain`, `Samples`, `HeaderSpec`, `HeaderColumns`, `Geometry`, and `Error` (S-01, S-02, S-03, S-04, S-05, S-07, S-09, S-12, S-13, S-14, S-16, S-20, S-22, S-23, S-24, S-25, X-01, X-04, X-05, X-08, X-16, X-18, X-23).
- `segyrs.open` returns a lazy `Volume`. Dimensions are `inline`, `crossline`, then `offset` when fold is constant, then `time` or `depth`. Otherwise the dimensions are `trace` and the vertical axis (S-12, S-14, S-17, S-18, X-15, X-20).
- `sel` and `isel` read only the traces they need. Attrs survive `sel`. The parallel `(inline, crossline) -> trace` index makes label `sel` hit the right trace on an out-of-order file without rewriting it (S-15, S-40, S-41, X-30, X-31).
- `.headers` is a dict of numpy arrays, one entry per trace. `.geometry` is the report. There is no pandas dependency (S-07, S-09, S-26, X-28).
- Typed Python errors mirror the Rust enum (S-39, S-46). Format 4 needs a gain convention (S-02). An absurd line span raises before allocation (X-16).
- Open takes a path, including non-ASCII paths (S-42). It does not take `strict` or `ignore_geometry` (S-38).
- Coordinate scalars are applied in float64 on coordinates only. The raw integer stays beside them (X-05, S-27). Exact zeros stay zeros (X-32). `dead_trace` defaults to unset: every stored trace is live, including an all-zero trace.
- Extended float64 offset (S-19), per-trace delay beside the 1D axis (S-27), and the optional time-stamp coordinate (S-28) land with `Volume`.
- Open uses rayon in the calling process and does not start a process pool (X-33).

### Non-goals

- No writer and no in-place edit (M2).
- No `to_regular`, no `concat`, no streaming kernels, no external `resort` (M3). The unsorted trace index is in M1.
- No window and no notebook image (M4).
- No cloud URL and no Zarr (M5).
- No second parser for file objects. `ReadAt` is the seam, and the cloud adapter is M5 (S-37).

### Exit criteria

- `mise run test` and `mise run lint` pass. The existing crate tests keep their ledger names. New `Volume` tests cover `open`, `sel`, `isel`, headers, geometry, the unsorted index, and the typed errors.
- `segyrs.open` reads no sample bytes. Decoded dtypes follow the sample format. IBM float is float32 with `sample_format` `ibm_float32` (S-01, S-04, X-08). Format 4 is not decoded unless `gain=` is passed (S-02).
- A two-trace file whose line numbers are 1 and `10**9` raises `GeometryError` and does not allocate that box (X-16).
- A permuted small grid returns, from `sel(inline=..., crossline=...)`, the trace whose headers match (S-15).

### Ledger ids

S-01, S-02, S-03, S-04, S-05, S-07, S-09, S-12, S-13, S-14, S-15, S-16, S-17, S-18, S-19, S-20, S-21, S-22, S-23, S-24, S-25, S-26, S-27, S-28, S-38, S-39, S-40, S-41, S-42, S-46, X-01, X-04, X-05, X-08, X-16, X-18, X-19, X-20, X-21, X-23, X-28, X-30, X-31, X-32, X-33.

## M2: Write and edit

Implemented. See [Writing SEG-Y](writing.md).

### Goal

Create a SEG-Y file from a `Volume`, update trace headers and samples in place when the stride does not change, and validate the file structure after every write (S-34). The validator checks sizes and codes. It does not by itself prove that another program will accept the file.

### Features

- `segyrs.create` and `Volume.to_segy` write a 2D line, a 3D grid, a pre-stack grid, or a trace list. There is no spec object (S-31, X-25).
- Sample format on write is the format recorded at open. `sample_format=` is the conversion (S-03, S-35). `create` requires `sample_format` when the volume has no `sample_format` attr. An opened IBM file stays IBM until you pass that argument.
- Binary-header sample count, interval, format code, and revision come from the `Volume` in the same write. The interval stays an integer microsecond count (S-29, S-33). Unedited bytes are the saved 400-byte header (X-22).
- Trace headers start at zero. Only the supplied overlays are written (X-26). Float coordinates are inverse-scaled into the integer word, and overflow names the field, the value, and the width (X-24).
- Batched in-place header columns (S-08) and sample replacement (S-32). A format change raises `FormatChangeError` and leaves the file untouched (S-35). A short sample vector raises `TraceLengthError` before any byte is written (S-32).
- `segyrs.validate` checks format code, sample count, file size, textual framing, and per-trace `ns` (S-34).
- `to_segy(..., rezero_time=True)` rebases a vertical window to 0 and updates the delay header (S-36).

### Non-goals

- No spec builder and no two-step uninitialized file.
- No in-place rewrite when sample format, sample count, or extended trace-header count changes. That is a new file.
- No silent float-to-integer truncation on header words.

### Exit criteria

- Round-trip tests (`open`, `create`, `open`) keep the sample payload for each supported format. An opened IBM file stays IBM unless `sample_format=` is passed.
- An in-place header column update matches a full rewrite and passes `segyrs.validate`.
- `mise run test` passes with tests named `test_s08`, `test_s29`, `test_s31` through `test_s35`, `test_x22`, `test_x24`, `test_x25`, `test_x26`.

### Ledger ids

S-08, S-29, S-31, S-32, S-33, S-34, S-35, X-22, X-24, X-25, X-26.

## M3: Manipulation at scale

Implemented. See [Manipulation at scale](manipulation.md).

### Goal

Sorting, subsetting, horizon windows, regularization, and concatenation stream through buffers sized from available RAM. They do not materialize a dense volume unless you ask for `to_regular` and the box fits. The label index for an unsorted grid already shipped in M1.

### Features

- `resort(by=..., out=...)` is the external sort, bounded by a configurable RAM budget. `sel(inline=..., crossline=...)` on an unsorted file uses the M1 index and does not rewrite the file (S-15).
- `sel` on one vertical index, and `slice_horizon`, read one sample window per trace (S-10).
- `filter` evaluates header predicates in Rust on the columnar table (X-29).
- A crop keeps coordinates, scalars, and the vertical coordinate (S-30, S-36). `rezero_time=True` on write rebases the window (S-36).
- Sparse files stay trace by sample at open (X-15, already the M1 layout). `to_regular(fill=0)` checks the dense shape against available RAM and streams the fill (X-15, X-17).
- `stats`, an amplitude limit (`clip`), and `rms` are streaming sample kernels with the same RAM rule.
- `concat` checks sample format, interval, sample count, and endian, and raises `FormatChangeError` on a mismatch (X-35).

### Non-goals

- No geophysical signal processing (deconvolution, migration, velocity analysis).
- No eager dense cube at open, including 5D gathers.

### Exit criteria

- Crop, sort, horizon extraction, and `to_regular` stream in buffers sized from available RAM. The budget is configurable. There is no compiled-in cap. Nothing here is a performance claim. It was not measured.
- A sub-volume keeps `cdp_x`, `cdp_y`, `scalco`, and the delay of the traces that remain.
- `mise run test` passes with tests named `test_s10`, `test_s30`, `test_s36`, `test_x15`, `test_x17`, `test_x29`, `test_x35`.

### Ledger ids

S-10, S-30, S-36, X-15, X-17, X-29, X-35.

## M4: Viewer and notebook

The renderer, the window and their Python calls (`Volume.plot`, the notebook image, `segyrs.view`) are in. See [viewer](viewer.md).

### Goal

A native Rust wgpu window streams one `inline` slice, one `crossline` slice, or one vertical slice from the map. The same renderer writes the notebook image. A picture of one slice must not read the rest of the file.

### Features

- `segyrs.view(vol)` opens the window. The notebook display uses that renderer on one slice and names the axis (S-06, X-09, X-10). The slice is the first index of the slowest grid axis, or the current `sel` when that selection is already 2D.
- The image is downsampled to the pixel grid first, then built once (X-09). With an explicit range set, only that view is read.
- When the range is omitted, it is the symmetric p98 of the live samples in the drawn view. Masked traces stay out of the reduction. The rest of the file is not read (X-12).
- Default axes are the 1D dimensions. A map of `cdp_x` and `cdp_y` is a separate view and reads headers only (X-13).
- A non-monotonic axis is sorted inside that 2D view when you ask, otherwise `UnsortedCoordinate` (X-11).

### Non-goals

- No matplotlib, and no second plotting stack.
- No full-volume reduction to choose the range.
- No requirement that you run the M3 `stats` kernel before a picture. The drawn view is enough (X-12).

### Exit criteria

- The quickstart on a file under `test-data/` shows one slice, and a read counter shows that samples outside that slice were not read (S-06).
- A 3D `Volume` does not upload the cube to draw one image (X-10).
- `mise run test` passes with tests named `test_s06`, `test_x09`, `test_x10`, `test_x11`, `test_x12`, `test_x13`.

### Ledger ids

S-06, X-09, X-10, X-11, X-12, X-13.

## M5: Interop and cloud

Implemented. Object-store reads, `Volume.to_zarr` and the wheel release workflow are in ([Cloud and Zarr](cloud-and-zarr.md), [Install and release](install-and-release.md)). Object-store reads and Zarr export are experimental and are in the segyrs wheel. The first wheels go to PyPI with 0.1.0; the viewer window is not in that release and is published later ([Release checklist](release-checklist.md)).

### Goal

Read SEG-Y from object storage with Rust `object_store`. A worker reopens by path. Publish wheels for the CI matrix. Offer Zarr export when you ask.

### Features

- `ReadAt` gains an `object_store` backend behind a `cloud` feature for `s3://`, `gs://`, and `az://` (S-37). A path and a memory buffer return the same samples. Zarr export is a separate function. It stays off unless you call it. There is no MDIO exporter.
- A worker reopens by path. The library does not pickle a file handle and does not start processes at import or at open (S-11, X-33).
- Optional `to_zarr`. Default chunks stay trace-aligned. Rechunk only if you ask (X-14). Samples are written with no integer fill value and no scale factor (the X-06 lesson). A missing chunk is the fill plus `trace_mask` false, not `KeyError` (X-34).
- maturin wheels for the CI matrix, `requires-python` matched to the mise pin, no sidecar DLL (S-43, S-44, S-45).

### Non-goals

- No MDIO exporter.
- No Python filesystem package as the cloud stack. `object_store` is the reader.
- No distributed scheduler inside the crate. One machine uses rayon. Anyone who wants more processes calls `open` on a path (S-11).
- Zarr is not on the open path and not on `to_segy`.

### Exit criteria

- `segyrs.open` on an object-store URL and on a memory buffer matches a local path for samples and for the line headers (S-37).
- Two worker processes each open the fixture by path and read the same first trace. A normal open does not spawn a child (S-11, X-33).
- `mise run test` passes with tests named `test_s11`, `test_s37`, `test_s43`, `test_s44`, `test_s45`, `test_x14`, `test_x34`.

### Ledger ids

S-11, S-37, S-43, S-44, S-45, X-14, X-34.

X-02, X-03, X-06, X-07, and X-27 are out of scope. They concern the labelled-array library this project does not use. The lessons are applied in `Volume.sel`, sample decode, optional Zarr export, the zero-trace rule, and `segyrs.open`.
