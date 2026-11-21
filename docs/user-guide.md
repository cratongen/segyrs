# User guide

`import segyrs`. Numpy is the only Python runtime dependency. Headers and samples come back as numpy arrays. There is no pandas, no xarray, no dask, no matplotlib, and no command-line program.

This guide covers the read path and `Volume`. Writing, manipulation, the viewer, and cloud and Zarr have their own pages: [Writing SEG-Y](writing.md), [Manipulation at scale](manipulation.md), [Viewer and notebook](viewer.md), and [Cloud and Zarr](cloud-and-zarr.md).

The examples use `survey.sgy`, the small file the [quick start](index.md#quick-start) writes: inlines 1000 to 1049, crosslines 2000 to 2059, 200 samples every 4 ms. On your own file, use its labels.

## Open

```python
import segyrs

vol = segyrs.open("survey.sgy")
```

```text
segyrs.open(
    path,                 # path-like, including non-ASCII paths, or a URL
    *,
    storage_options=None, # URLs only: object-store and read-cache keys
    sample_format=None,   # overrides the binary-header code
    endian=None,          # None detects; "big", "little", or "pairwise"
    gain=None,            # required for format 4: "rev0" or "rev2"
    inline_byte=189,      # int32
    crossline_byte=193,   # int32
    offset_byte=37,       # int32
    header_map=None,      # extra or replacement fields
    dead_trace=None,      # None: every stored trace is live
    depth=False,          # vertical dimension name
    scan="auto",          # "auto" or "full": how many trace headers open reads
    index_cache=None,     # directory for header-index sidecars (local paths)
)
```

`path` is a filesystem path or an object-store URL (`s3://`, `gs://`, `az://`, `abfs(s)://`, `http(s)://`, `file://`; experimental, in the segyrs wheel). A URL is read through Rust `object_store` on the same read interface, configured by `storage_options=` (see [Cloud and Zarr](cloud-and-zarr.md)). You do not map the file yourself. If the operating system refuses a map, segyrs reads by position.

`endian=None` reads the rev 2 signature at binary-header bytes 3297-3300 before any other multi-byte field. A zero signature is big-endian. `01 02 03 04` is big-endian, `04 03 02 01` is little-endian, and `02 01 04 03` is pairwise (each pair of bytes swapped relative to big-endian). Samples and header words are decoded to the machine's byte order. `attrs["source_endian"]` keeps `"big"`, `"little"`, or `"pairwise"`.

Format 4 raises `Format4NeedsGain` unless `gain` is set. `gain="rev0"` decodes `value * 2^gain`. `gain="rev2"` decodes `value * 2^-gain`.

`sample_format` accepts the names in [Sample formats](formats.md) (`ibm_float32`, `ieee_float32`, `int16`, and the rest). The binary-header code stays visible, so you can see that an override happened.

Opening reads the textual header, the 400-byte binary header, and the trace-header columns needed for geometry. Sample bytes stay unread until `sel`, `isel`, or an explicit decode. `attrs["chunk_shape"]` says the chunks are whole traces in on-disk order.

Binary header bytes 3505-3506 give the number of extended textual blocks. A count of -1 means the blocks end with the `((SEG: EndText))` stanza, matched in any case and spacing. A rev 2 first-trace offset (bytes 3521-3528) moves the first trace but never adds textual blocks, so one wrong header word cannot make open read the file as text. The extended blocks are decoded only after the trace layout is known to fit the file.

### How much open reads: `scan` and `index_cache`

A full header scan reads one header per trace. When a trace is about one 4 KB page long, that is almost every page of the file, so a cold full scan costs about as much as reading the whole file. `scan="auto"` (the default) avoids it for a large regular file:

- A local file with more than 4096 traces is checked for a grid whose inline, crossline, and offset numbers each step by a constant. Open reads traces 0 and 1, finds where each run of keys ends with a few dozen more headers, then checks the last trace and 1024 headers spread over the file. If they all fit, the volume opens on that grid without reading the other headers, and `vol.geometry.source` is `"assumed"`.
- Every read checks the keys (and, when the binary header does not promise a fixed length, the sample count) of the traces it touches. Those bytes sit in the same page as the samples. A selection that covers a sixteenth of the file or more checks every header once instead.
- Anything that needs every header checks them all first: `vol.headers`, `vol.coords`, `dead_trace="trace_id"`, `to_zarr`, and the viewer. The check runs once per file.
- A trace that contradicts the grid raises `GeometryAssumptionError` (a `GeometryError`), naming the trace, the key it has, and the key the grid expected. Reopen with `scan="full"` to get the file's real layout. Nothing is returned for a trace that failed the check.
- Anything else opens with the full scan, exactly as `scan="full"` does: a small file, a file whose sampled headers do not fit, a grid whose numbers are not evenly spaced, a single line, and any URL.

`scan="full"` reads every trace header at open and infers the geometry from them. Use it to validate a file up front, or when a file is known to be irregular. The scan keeps the scanned header columns in memory and releases the mapped file pages it has read as it goes, so the file itself does not stay in the process's resident set (`tests/scan_memory.rs`).

`index_cache=dir` keeps the result of a full scan. The scan stores the scanned header columns in `dir/<hash of the path>.segyrs-index`. The next `scan="auto"` open of the same file loads them instead of reading every header, and `vol.geometry.source` is `"index_cache"`. The sidecar is ignored, and replaced by the next full scan, when the file's size or modification time changed, or when its textual and binary headers or its first or last trace header changed. A missing, damaged, or stale sidecar is never an error, and deleting it is always safe. A failure to write it does not fail the open. `scan="full"` does not read it. Nothing is written unless you pass `index_cache`. URLs do not take it.

`dead_trace=None` means every stored trace is live, including a trace of exact zeros. `trace_mask` is true. Zeros stay zeros. `dead_trace="trace_id"` marks a trace dead when its trace identification code is 2. A bool array with one entry per trace, true where dead, sets the mask directly. Zeros alone never mean dead.

`header_map` adds or replaces header fields: `{"name": (byte, dtype)}`, or `{"name": byte}` for an int32. The dtype is one of `int8`, `int16`, `int32`, `int64`, `uint16`, `uint32`, or `float64`. A field that does not fit the trace header (extension blocks included) raises `ValueError` at open.

`inline_byte`, `crossline_byte`, and `offset_byte` are the byte positions of those words. The columns and the dimensions they become are named `inline`, `crossline`, and `offset`. Rust and Python use that one vocabulary.

## Volume

`Volume` is the labelled lazy array. `sel` returns a `Volume`, so file-level attrs survive the selection. `.numpy()` materializes that selection and nothing else. One sample is a 0-d numpy value, and `.item()` returns it.

```python
vol.dims        # ("inline", "crossline", "time"), with "offset", or ("trace", "time")
vol.coords
vol.attrs
vol.sel(inline=1010, crossline=2020)
vol.isel(inline=slice(0, 3))       # three positions
vol.isel(inline=0, crossline=0, time=0).item()   # one sample
vol.numpy()                        # the current selection only
```

Dimension rules:

- A complete monotonic grid with constant fold uses `(inline, crossline, time)` or `(inline, crossline, offset, time)`. `depth=True` renames the vertical dimension to `depth`. `geometry.kind` is `inline_major`, `crossline_major`, or `offset_major`. Reads follow on-disk trace order.
- Sparse, unsorted, varying fold, or not a grid: dimensions `(trace, time)` or `(trace, depth)`. `inline`, `crossline`, and `offset` are coordinates along `trace`. `geometry.missing_pairs` lists absent slots on a sparse file. `to_regular` builds the dense grid on request.
- Open builds an `(inline, crossline)` index, so `sel(inline=..., crossline=...)` hits the matching trace on an unsorted file. The file order stays as stored.
- A duplicate `(inline, crossline)` raises `DuplicateKey` and names the key. The file does not open.
- A header span whose dense box dwarfs the trace count raises `GeometryError` with the minimum, the maximum, and the trace count. Two traces numbered 1 and `10**9` raise. Open does not allocate that box.
- Unknown sort does not fail open. `sel` on a grid coordinate that was not inferred raises `GeometryError`.

`offset` is a real dimension when fold is constant, so `sel(offset=slice(None, None, 2))` keeps the first label. When a nonzero rev 2.1 float64 offset is present, that value is the coordinate and the int32 at bytes 37-40 stays `offset_raw`.

Labels include a line numbered 0. There is no one-based line generator.

`isel` takes an int (the dimension drops), a slice, an int list, or a bool mask. `sel` takes a label, an inclusive label slice, or a label list. `sel(..., method="nearest")` picks the closest label. A label slice step counts positions, so `slice(None, None, 2)` keeps the first label.

On the `(trace, time)` layout, `sel(inline=..., crossline=...)` and `sel(offset=...)` filter the traces and keep the `trace` dimension. A scalar inline and crossline pair goes through the trace index without a scan over every trace.

### Scattered traces in one call

```python
pts = vol.sel_points(inline=[1010, 1012, 1040], crossline=[2003, 2007, 2007])  # one trace per point
pts.dims                                    # ("point", "time"), plus any dim left out
pts.numpy()                                 # one parallel decode, GIL released
vol.isel_points(inline=[0, 2], crossline=[5, 1])    # the same by position
vol.isel_points(trace=[17, 3, 999])         # file trace numbers, file order (segyio's f.trace[i])
```

`sel_points` and `isel_points` pair the lists element by element: point `k` is `(inline[k], crossline[k])`. The lists must have the same length. A trace dim that is not given stays whole, after `point`: `sel_points(inline=..., crossline=...)` on a pre-stack grid gives `(point, offset, time)`. `coords` carries each point's `inline`, `crossline`, and `trace`. A label that is not on the grid raises `KeyError`, and `method="nearest"` picks the closest label per point. `trace=` takes file trace numbers, alone, on the Volume `segyrs.open` returned, whatever the sort order. On the `(trace, time)` layout, `sel(inline=..., crossline=...)` is the label filter. A read of many scattered traces is faster as one `numpy()` on the points than as one `isel(...).numpy()` per trace, which crosses into Rust once per trace.

`numpy()` reads the traces in the selection and, from each, the samples between the first and the last selected vertical position. `slice_horizon` reads a different window per trace. `vol.traces_read` counts the traces whose samples were read so far. Open leaves it at 0.

### Attributes

```python
vol.numpy().dtype                 # int8/16/32/64, uint8/16/32/64, float32, or float64
vol.attrs["sample_format"]        # for example "ibm_float32"
vol.attrs["source_endian"]        # "big", "little", or "pairwise"
vol.attrs["revision"]             # "0", "1", "2.0", "2.1"
vol.attrs["textual_header"]       # list of 80-character cards, 40 per block
vol.attrs["textual_encoding"]     # per block, "ascii" or "ebcdic"
vol.attrs["chunk_shape"]          # whole traces, on-disk order
vol.attrs["measurement_system"]   # "meters", "feet", or None, from the binary header
vol.attrs["binary_format_code"]   # the code in the file, kept when sample_format= overrides it
```

Those attrs survive `sel`.

Samples are not masked and not scaled. Attrs carry no fill value. NaN is possible only on floating formats. `sel` and `isel` keep the sample dtype. `trace_mask` is a bool coordinate, true unless `dead_trace` says otherwise. An all-zero trace opens as zeros and stays live.

### Coordinates

```python
vol.coords["time"]        # delay + i * interval, header integers, microseconds
vol.coords["cdp_x"]       # float64 after scalco
vol.coords["cdp_x_raw"]   # int32 header word beside the scaled coordinate
vol.coords["trace_mask"]  # bool
```

`apply_scalar` is the rule for `scalco` and `scalel`. A positive scalar multiplies, a negative scalar divides, and zero means 1. The scaled coordinate is float64. The raw integer stays beside it, with a `_raw` name. Elevation and depth use `scalel` the same way. The amplitude array is not scaled.

The time coordinate is `delay + i * interval` from the header integers, in microseconds. The interval stays that unsigned integer. segyrs does not rebuild it by subtracting a float axis. The delay is the delay recording time of the first trace (bytes 109-110, milliseconds) after the time scalar at bytes 215-216. If per-trace delay varies, the 1D axis stays on that value and a `delay` coordinate carries the per-trace value. The axis is int64 when every value is a whole microsecond, otherwise float64. A nonzero time stamp (bytes 157-166, plus the rev 2 nanoseconds in trace header extension 1) becomes a `datetime64[ns]` `timestamp` coordinate. All-zero time fields add no coordinate.

When a rev 2 file has a nonzero float64 offset in trace header extension 1 (bytes 129-136), that value is the `offset` coordinate and the int32 at bytes 37-40 is `offset_raw`. `headers["offset"]` stays the int32 word.

Per-trace coordinates (`cdp_x`, `source_x`, `group_elev`, `trace_mask`, `trace`, and the rest) have the shape of the non-vertical dimensions. They are built from one header scan the first time `coords` is read.

### Textual header

Cards are split on 80-byte boundaries, 40 cards per 3200-byte block. The decoder does not rewrap. A block is ASCII when the first card byte is `0x43` and EBCDIC when it is `0xC3`. A high byte does not fail open. Undecodable bytes become the replacement character. The raw 3200-byte block is kept on the volume beside the cards.

## Headers and geometry

```python
vol.headers                 # dict of numpy arrays, one entry per trace, lazy per column
vol.headers["inline"]
vol.headers["crossline"]
vol.header_table()          # the same columns packed into one structured array
vol.header_bytes(0)         # on-disk trace-header bytes of file trace 0, file order
vol.geometry
vol.binary_header           # parsed rev 2 fields, a dict
vol.textual_header_bytes    # raw 3200-byte blocks beside the decoded cards
vol.binary_header_bytes     # 400 bytes, file bytes 3200:3600
```

`len(vol.headers["inline"])` equals the trace count. On the opened volume the arrays are in file order, read-only, and shared. On a selection they hold the selected traces in the selection's order. Every field name is a key from the start, and the columns are lazy: `vol.headers["cdp_x"]` decodes that one column, on all cores with the GIL released, the first time any Volume of the file asks for it, and keeps it. A column that open already scanned (`inline`, `crossline`, `offset`, `cdp_x`, `cdp_y`, `scalco`, `trace_id`, unless `header_map` moves it) costs no file access. After an assumed-grid open, the first column read scans the headers and checks the grid. `values()`, `items()`, `copy()`, and `header_table()` decode every missing column in one parallel pass. `dict(vol.headers)` and `{**vol.headers}` work too, one scan per column. `header_table()` packs the same columns into a structured array for that call. It is not a second stored table.

`HeaderSpec` standard fields are the default map. Byte ranges outside the named fields are `(byte, dtype)` on `header_map`. A field wider than the standard word is legal when it still fits the trace header.

```python
vol.geometry.kind
# "inline_major" | "crossline_major" | "offset_major"
# | "unsorted" | "sparse" | "unstructured"
vol.geometry.inline_byte     # default 189
vol.geometry.crossline_byte  # default 193
vol.geometry.offset_byte    # default 37
vol.geometry.missing_pairs  # sparse only
vol.geometry.axes           # slowest axis first; empty when the file is not a sorted grid
vol.geometry.source         # "scan", "assumed", or "index_cache": see the open section
```

## Errors

Every class subclasses `segyrs.SegyrsError`. The message includes the sizes or the key, and says what to check. There is no C enum. A label that is not on an axis raises `KeyError` with the range of labels that exist.

A file-system failure is a `segyrs.SegyrsIOError`, which is also an `OSError`, with `errno` and `filename` set. A missing path raises `PathNotFoundError` (also a `FileNotFoundError`), a path this user may not read or write raises `PathPermissionError` (also a `PermissionError`), and a directory raises `PathIsDirectoryError` (also an `IsADirectoryError`). So `except FileNotFoundError` and `except segyrs.SegyrsError` both catch a wrong path.

| Class | When |
| --- | --- |
| `GeometryError` | The grid span is absurd, or you select `inline`, `crossline`, `offset` or `trace` on a Volume that does not have that dim. A name that is no segyrs dim at all, such as `iline`, is a `ValueError` that suggests the nearest dim |
| `SegyrsValueError` | Also a `ValueError`: an argument the Rust core cannot use, such as a header field name that does not exist, unordered `clip` bounds, or a `chunks` shape with the wrong length |
| `BudgetError` | Also a `MemoryError`: the work needs more RAM than its `budget`. The message has both numbers. Nothing was allocated |
| `GeometryAssumptionError` | A `GeometryError`: a trace header contradicts the grid that `scan="auto"` assumed. Names the trace and both keys. Reopen with `scan="full"` |
| `DuplicateKey` | Two traces share one grid slot. The file does not open |
| `EndianMismatch` | The rev 2 signature and the decoded sizes disagree |
| `SampleCountMismatch` | The file size is not the headers plus `ntr * stride`. The message has the file length, the expected length, and the remainder |
| `NonUniformTraceLength` | A per-trace `ns` disagrees with the binary-header stride. The message has both counts |
| `TraceLengthError` | One trace's byte length disagrees with the stride. On an in-place write, a sample array of the wrong length raises this before any byte is written |
| `FormatChangeError` | In-place format change, or `concat` of unlike files. The file is left untouched |
| `HeaderValueError` | A header value to write does not fit its field. Names the field, the value, and the width |
| `SampleValueError` | A sample cannot be stored in the target format without changing its value |
| `SpecError` | The description of a file to write is inconsistent, for example `sample_interval_us=0` for a file with traces, which `validate` would reject. Nothing was written |
| `ValidationError` | `validate` found structural problems. `err.problems` lists them |
| `Format4NeedsGain` | Format 4 opened with no gain convention |
| `UnknownFormat` | The binary-header format code is not a known sample format. Pass `sample_format=` for a SEG-Y file with a nonstandard code; otherwise the file is not SEG-Y |
| `TruncatedError` | The file ends before something its headers declare: the 3600-byte file header, the extended textual blocks, or the `EndText` stanza. The message names which |
| `NotSegyError` | The bytes are not SEG-Y. A Seismic Unix file (trace headers with no 3600-byte file header) is named as such |
| `UnsortedCoordinate` | A drawn coordinate axis is not monotonic and you did not ask to sort that view. The Rust draw crate raises it (`PlaneSpec::x_coord`); `plot()` draws on dims and does not take a coordinate axis |

`SampleCountMismatch`, `EndianMismatch`, and `NonUniformTraceLength` are three errors. They used to share one message upstream: ledger id S-25 in [Design](design.md).

A file with bytes after its last whole trace, such as padding a tape copy or a transfer left behind, raises `SampleCountMismatch`, because segyrs cannot tell padding from a damaged or misread trace length. There is no open option that ignores the remainder. The message gives the file length, the expected length and the remainder. When you know the extra bytes are padding, open a copy cut to the expected length: `shutil.copyfile`, then `os.truncate(copy, expected)`. A file that ends inside a trace raises the same error.

## Write

```python
import numpy as np

vol.sel(inline=slice(1010, 1020), time=slice(0, 400_000)).to_segy("crop.sgy")  # first 400 ms
vol.to_segy("ieee.sgy", sample_format="ieee_float32")   # the only conversion

cube = np.zeros((4, 5, 100), np.float32)
x = np.tile(500_000.0 + 12.5 * np.arange(5), (4, 1))    # one cdp_x per trace, in metres
segyrs.create("new.sgy", cube, sample_format="ibm_float32", sample_interval_us=4000,
              inline=[10, 11, 12, 13], crossline=[1, 2, 3, 4, 5],
              headers={"cdp_x": x, "scalco": -100})
segyrs.validate("new.sgy")                               # layout dict or ValidationError
with segyrs.edit("new.sgy") as ed:                       # in place, same layout
    ed.set_header("cdp", np.arange(ed.n_traces))
    ed.write_traces(np.ones((2, ed.n_samples), np.float32), first=10)
```

`to_segy` writes the current selection: its traces in the file's axis order (a crossline-major file stays crossline-major), each axis in selection order, each with its own header bytes, and the selected contiguous sample window. The sample format stays the source format, so an IBM file stays IBM bit for bit. `sample_format=` and `endian=` are the conversions. Each kept sample keeps its time: the delay word moves with the window, or is zero with `rezero_time=True`. The file appears complete or not at all.

`create` takes `(trace, time)`, `(inline, crossline, time)`, or `(inline, crossline, offset, time)` arrays. `sample_format` and `sample_interval_us` are required: segyrs does not pick a format, and the interval is the integer you pass: a value that is not a whole number of microseconds raises `ValueError` instead of being truncated. `headers` adds fields by standard name or `(byte, dtype)`. A float coordinate is stored with the `scalco` or `scalel` in the same mapping. A header value that does not fit raises `HeaderValueError`; a sample the format cannot hold exactly raises `SampleValueError`. Nothing is written in either case.

`segyrs.edit` changes headers, blocks, and samples in place. A change of sample format or sample count is a new file and raises `FormatChangeError`. A sample array of the wrong length raises `TraceLengthError` before any byte is written. Leaving the `with` block calls `fdatasync` once.

`validate` returns the layout, or raises `ValidationError` whose `problems` lists every problem found. The check is sizes and codes. It does not by itself prove that another program will accept the file.

The full contract is in [Writing SEG-Y](writing.md#python).

## Streaming kernels

Whole-volume work runs as rayon kernels in Rust with the GIL released. Operations that write a file stream through the writer inside `budget`: `None` is half the available RAM, an int is bytes, and a float is a fraction of the available RAM. There is no compiled-in cap. Nothing on this page is a measured timing.

```python
vol.sel(time=400_000)                           # one sample per trace, at 400 ms
horizon = np.full((50, 60), 400_000.0)          # one time per trace, microseconds
vol.slice_horizon(horizon, above=5, below=5)    # (values, valid)
vol.filter(offset=(-500, 500), cdp=[1, 2, 3])   # header predicates in Rust
vol.stats()                                     # count, nan, min, max, mean, std, rms
vol.rms()                                       # one value per trace
vol.clip(-2.0, 2.0, "clipped.sgy")              # source format kept
vol.resort("sorted.sgy", by=["crossline", "inline"])
vol.to_regular(fill=0)                          # (data, mask, coords); refuses over budget
vol.to_regular("dense.sgy")                     # the same grid, streamed to a file
vol.sel(inline=slice(1000, 1024)).to_segy("a.sgy")
vol.sel(inline=slice(1025, 1049)).to_segy("b.sgy")
segyrs.concat([segyrs.open("a.sgy"), segyrs.open("b.sgy")], "c.sgy")  # whole files
```

The crop is `sel` or `isel` followed by `to_segy`. `filter` runs on the columnar headers and returns a `(trace, time)` Volume. `concat` checks sample format, interval, sample count, byte order, and trace-header length, and raises `FormatChangeError` on a mismatch. The full contract is in [Manipulation at scale](manipulation.md#python).

`stats`, `clip`, and `rms` are separate from the view range in the next section.

## Viewer and notebook

```python
vol.sel(inline=1003)                     # Jupyter shows the picture (_repr_mimebundle_)
pic = vol.sel(inline=1003).plot(vmin=-2, vmax=2, colormap="seismic")
pic.save("inline1003.png")               # pic.png, pic.text, pic.note, pic.traces_read
vol.isel(inline=5).plot(wiggle_every=20) # wiggles on one trace in 20
```

<!-- doc-block: no-run, opens a window and waits until it is closed -->
```python
segyrs.view(vol)                         # native wgpu window (built from the repository); one slice at a time
segyrs.view("survey.sgy", vmin=-2, vmax=2)
```

`Volume.plot(...)` returns a `segyrs.Picture`: PNG bytes (`png`), a one-line description (`text`), the wiggle note (`note`) and the number of traces read (`traces_read`). It displays inline in Jupyter, and a Volume displays the same picture through `_repr_mimebundle_` (`image/png` plus `text/plain`). The plot area is 800 by 480 pixels unless you pass `width` and `height`. Options: `vmin`, `vmax`, `colormap` (`"grey"`, `"seismic"`, `"amber"`), `polarity` (`"normal"`, `"reversed"`), `gain`, `wiggle` (`"line"`, `"area"`), `wiggle_every`, `density`.

A 2D selection draws that plane: an inline or crossline section, a time slice, or a window of one made with `sel` or `isel` slices. A 3D volume draws one slice, the first index of the slowest grid axis (and of offset on a pre-stack grid). The image names that axis. It does not read the rest of the cube. Each kept dim must be a contiguous ascending run of positions; a scattered selection (a list of lines, a step) and a selection with fewer than two dims show their text and the reason instead of a picture.

Wiggles need at least 3 pixels per trace. Below that the picture shows density and says so in the image and in `note`, for example `wiggle needs >= 3 px per trace (1000 traces on 800 px); showing density; try wiggle_every=4`. `wiggle_every=N` reads and draws one trace in N (it implies `wiggle="line"`), so a dense section still shows wiggles; the density under them is decimated too.

`segyrs.view(volume_or_path, ...)` is experimental and needs the `segyrs-view` extension, which is not part of the 0.1.0 release on PyPI: build it from a checkout with `mise run build` ([Viewer](viewer.md#python)). Without it, it raises `ImportError` that says so. It takes `vmin`, `vmax`, `colormap`, `polarity`, `gain`, `wiggle`, `renderer` (`"auto"`, `"gpu"`, `"cpu"`) and `title`. It starts on the Volume's plane and shows the whole plane; zoom and step inside the window. It runs on the main thread and blocks until the window closes, with the GIL released. Ctrl-C closes it and raises `KeyboardInterrupt`. With no display it raises `SegyrsError`; use `plot()` there. It returns `"gpu"` or `"cpu"`, the presenter that ran.

The renderer downsamples the 2D view to the pixel grid first, then builds that image once. With `vmin` and `vmax` set, it reads only that view. When they are omitted, the range is the symmetric 98th percentile of the absolute value of the live samples in the drawn view. Dead traces stay out of that reduction. The rest of the file is not scanned.

Default image axes are the 1D dimensions (`inline` and `time`, or `crossline` and `time`). A 2D `cdp_x` coordinate does not become the axis. In Rust, a map view takes `cdp_x` and `cdp_y` from headers and reads no samples, and a section can take a header coordinate as its x axis: if that axis is not monotonic, you ask to sort the view with its samples, or the call raises `UnsortedCoordinate` ([Viewer](viewer.md#axes)).

There is no matplotlib.

## Cloud and optional Zarr

```python
vol.to_zarr("out.zarr")                # optional; open and to_segy do not call it
```

<!-- doc-block: no-run, needs an object store and its credentials -->
```python
remote = segyrs.open("s3://bucket/survey.sgy", storage_options={"aws_region": "eu-west-1"})
remote.to_zarr("s3://bucket/out.zarr")  # the same group, written to an object store
```

Both are experimental and are in the segyrs wheel, with no extra to install. `storage_options` takes the `object_store` keys for the URL's scheme plus the read-cache keys. Credentials also come from the environment. An unknown key raises. A selection on a remote file fetches its traces in coalesced parallel range requests before decoding. `to_zarr` exports the whole file, so call it on the Volume `segyrs.open` returned. It returns a summary dict. Details: [Cloud and Zarr](cloud-and-zarr.md).

A memory buffer and a path return the same samples. A worker reopens by path. The open file is not pickled. Every segyrs error pickles, so an error raised in a worker process reaches the parent with its class and message. Import and open do not spawn processes.

`to_zarr` keeps trace-aligned chunks unless you ask to rechunk. Sample encoding sets no integer fill value and no scale factor. A missing chunk reads as the fill and `trace_mask` false. It does not raise `KeyError`. There is no MDIO exporter.

Wheels are maturin builds for the CI matrix. `requires-python` matches the mise pin. A sidecar DLL is a bug. [Install and release](install-and-release.md) has the platform table.
