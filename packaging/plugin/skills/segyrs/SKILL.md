---
name: segyrs
description: How to write correct code with segyrs (`import segyrs`), the Rust-core Python library for SEG-Y. Open, select, read, write, edit and stream files. Use whenever code imports segyrs, or the user asks to read or write a SEG-Y file in Python and segyrs is installed or wanted.
---

# segyrs

segyrs is imported as `segyrs`. Install it with `pip install segyrs` (Python 3.11 or newer). Numpy is the only dependency.

```python
import segyrs
```

## Look it up before you write it

Do not write a segyrs call from memory. The installed package matches the version you have:

```sh
python -c "import segyrs, pathlib; print(segyrs.__version__, pathlib.Path(segyrs.__file__).with_name('_segyrs.pyi'))"
```

- `help(segyrs.open)` and `help(segyrs.Volume.sel)` print the NumPy-style docstring: every argument, its unit and its default. `Volume`, `create`, `edit` and `concat` live in the Python modules next to `segyrs.__file__`.
- `_segyrs.pyi` is the Rust extension. It is not the list of `Volume` methods.
- `segyrs.__all__` lists every public name.
- The docs are at https://cratongen.github.io/segyrs/ . The user guide is https://cratongen.github.io/segyrs/user-guide/ , writing is https://cratongen.github.io/segyrs/writing/ , and streaming operations are https://cratongen.github.io/segyrs/manipulation/ .

If a name is not in `segyrs.__all__`, it does not exist. Say so and offer the closest name that does.

## Kinds of object

Ask which kind you hold, and the calls follow. You do not construct `segyrs.Volume`, `segyrs.Geometry` or `segyrs.Picture` yourself.

| Kind | Examples | You call | Saves as |
| --- | --- | --- | --- |
| Volume | the object `segyrs.open` returns | `sel`, `isel`, `numpy`, `filter`, `to_segy`, `stats` | SEG-Y |
| Geometry | `vol.geometry` | `kind`, `axes`, `source`, `is_grid` | not a file |
| Editor | `segyrs.edit` | `set_header`, `set_headers`, `write_traces` | the same SEG-Y, in place |
| Picture | `vol.plot()` | `save` | PNG |
| Function | `segyrs.open`, `segyrs.create`, `segyrs.concat`, `segyrs.validate` | one call | a file or a dict |

`geometry.kind` is one of `inline_major`, `crossline_major`, `offset_major`, `unsorted`, `sparse`, `unstructured`. An unsorted, sparse or unstructured file still opens. On a `(trace, time)` volume, `sel(inline=..., crossline=...)` finds traces through the header index.

One run through every kind:

```python
import numpy as np
import segyrs

cube = np.random.default_rng(0).standard_normal((4, 5, 16)).astype(np.float32)
x = np.tile(500_000.0 + 25.0 * np.arange(5), (4, 1))
y = np.repeat(7_500_000.0 + 25.0 * np.arange(4), 5).reshape(4, 5)
segyrs.create(
    "survey.sgy",
    cube,
    sample_format="ieee_float32",
    sample_interval_us=4000,
    inline=np.arange(100, 104),
    crossline=np.arange(200, 205),
    headers={"cdp_x": x, "cdp_y": y, "scalco": -100},
)
vol = segyrs.open("survey.sgy")
section = vol.sel(inline=101)
samples = section.numpy()
picture = section.plot()
picture.save("inline-101.png")
with segyrs.edit("survey.sgy") as ed:
    ed.set_header("offset", np.full(ed.n_traces, 25, np.int32))
layout = segyrs.validate("survey.sgy")
```

`samples.shape` is `(5, 16)`. `vol.geometry.kind` is `inline_major`. `layout["n_traces"]` is 20.

## Rules that hold everywhere

- **Open reads headers, not samples.** `sel`, `isel`, `sel_points`, `isel_points` and `filter` return a new Volume and read no sample bytes. Assign the result: `section = vol.sel(inline=101)`. `vol.traces_read` stays 0 until something decodes samples.
- **Options are keyword-only.** `segyrs.create(path, data, "ieee_float32", 4000)` raises `TypeError`. Write `sample_format="ieee_float32"` and `sample_interval_us=4000`. The same holds for `scan`, `endian`, `budget` and the other options on `segyrs.open`, `to_segy`, `clip`, `resort` and `to_regular`.
- **The file on disk changes only through a write.** `segyrs.create`, `to_segy`, `clip`, `resort`, `segyrs.concat`, `to_regular(path)` and `to_zarr` write a new file. `segyrs.edit` changes the open file in place and keeps its layout: sample format, sample count and interval stay. A format change is `to_segy(..., sample_format=...)`.
- **Array shapes.** `create` takes `(trace, time)`, `(inline, crossline, time)` or `(inline, crossline, offset, time)`. There is no default `sample_format`.
- **Label slices include both ends.** `sel(inline=slice(100, 101))` keeps inlines 100 and 101. Positions go to `isel`. `method="nearest"` is the way to ask for the closest label.
- **Check `vol.nbytes` before `numpy()`.** It is the size of the current selection and it reads nothing. `numpy()` decodes that selection into an array you own. `np.asarray(vol)` calls `numpy()` on the current selection.
- **Time is microseconds** on `coords["time"]`, unless you opened with `depth=True`, which names the vertical dim `depth`. `sample_interval_us` is an integer.
- **Decoded dtypes follow the file.** Integer formats stay integers. IBM float comes back as float32. `to_segy` without `sample_format` copies the source format, so IBM stays IBM.
- **`segyrs.view` opens a native window** when the viewer extension is built from a checkout and a display is available. The PyPI wheel does not include that extension. `plot()` is in the wheel and returns a `segyrs.Picture` with no matplotlib.

```python
import numpy as np
import segyrs

data = np.zeros((2, 3, 4), np.float32)
try:
    segyrs.create("bad.sgy", data, "ieee_float32", 4000)
except TypeError:
    pass
segyrs.create(
    "survey.sgy",
    data,
    sample_format="ieee_float32",
    sample_interval_us=4000,
)
```

## Missing values

segyrs does not turn sentinels into NaN. A stored `-999` stays `-999`, and a stored zero stays a live trace. `segyrs.open(..., dead_trace=None)` treats every stored trace as live. `dead_trace="trace_id"` marks trace identification code 2 as dead, and `coords["trace_mask"]` is false there. You can also pass a bool array, true where the trace is dead.

IBM overflow decodes as infinity, not NaN. `stats()` counts `nan` and `infinite` and leaves both out of `min`, `max`, `mean`, `std`, `variance` and `rms`.

## Errors

segyrs errors derive from `segyrs.SegyrsError`. Read the message before changing the call. It names the field, the sizes or the key.

- `segyrs.PathNotFoundError` is also `FileNotFoundError`. `PathPermissionError` is also `PermissionError`. `PathIsDirectoryError` is also `IsADirectoryError`. All three are `segyrs.SegyrsIOError`, which is an `OSError`.
- `segyrs.SegyrsValueError` is a `ValueError`: a header name that does not exist, `clip` bounds with `lo > hi`, a `chunks` shape of the wrong length.
- `segyrs.BudgetError` is a `MemoryError`. The message has the bytes needed and the budget. Nothing was allocated.
- `segyrs.GeometryAssumptionError` means a trace header contradicts the grid `scan="auto"` assumed. Reopen with `scan="full"`.
- `segyrs.SampleValueError` means a sample does not fit the target format (a fraction into an integer format, NaN into IBM). Nothing is written.
- `segyrs.ValidationError` comes from `segyrs.validate`. `err.problems` lists what it found.

```python
import segyrs

try:
    segyrs.open("missing.sgy")
except segyrs.PathNotFoundError as err:
    assert isinstance(err, FileNotFoundError)
```

## Saving

Parameters of a new file go through `segyrs.create` or `Volume.to_segy`. `segyrs.validate(path)` returns a dict (`revision`, `sample_format`, `endian`, `n_traces`, `n_samples`, `sample_interval_us`, and the header sizes) or raises `segyrs.ValidationError`.

`segyrs.edit` writes headers and trace samples into an existing file. `set_header` and `set_headers` take one value per trace written, or one value for all of them. Float coordinates need the SEG-Y scalar in the same `set_headers` call: pass `scalco` in the values and `scalars={"cdp_x": -100}`. A negative scalar multiplies on write. The scalar word itself is not implied.

`to_zarr` writes a Zarr v3 group of the whole file `segyrs.open` returned. It is experimental, and it refuses a selection. `Picture.save` writes the PNG bytes.

There is no Parquet writer and no JSON form of a Volume.

## Data

The package ships no sample files. Build one with a seed, as the kinds example does. `np.random.default_rng(0)` makes the cube repeatable. The files under `test-data/` belong to the repository's tests. They are not installed with the wheel.

## Write it fast

Correct segyrs code still reads the whole file, or walks it in Python, when it ignores the calls below. Check this list before you write a loop.

### Select, then read

`segyrs.open` reads no sample bytes. `vol.sel(inline=101).numpy()` reads that inline. `vol.numpy()` reads every selected trace. `np.asarray(vol)` does the same. Read `vol.nbytes` first.

`headers["cdp_x"]` decodes one column, in Rust, on all cores. `header_table()`, `items()` and `values()` decode every column. Ask for the fields you use.

`section.plot()` reads the plane it draws. A 3D volume draws its first plane. Do not call `numpy()` on the cube in order to draw one inline.

### One call, not a Python loop

A `for` over traces that calls `isel(...).numpy()`, `set_header` or `write_traces` once per trace is the slow version. The benchmarks page times that loop on purpose (one `isel` per trace) and says the scattered-trace call was not measured. Use the call that takes the whole set:

| You want | Call |
| --- | --- |
| One inline, crossline or time window | `vol.sel(...)` or `vol.isel(...)`, then one `numpy()` |
| Scattered labels | `vol.sel_points(inline=ils, crossline=xls).numpy()` |
| File trace numbers | `vol.isel_points(trace=indices).numpy()` |
| Traces whose headers pass a test | `vol.filter(offset=(0, 500))` |
| One header column | `vol.headers["cdp_x"]` |
| RMS of each trace | `vol.rms()` |
| Count, min, max, mean, std | `vol.stats()` |
| Samples along a horizon | `vol.slice_horizon(horizon, above=2, below=2)` |
| A crop on disk | `vol.sel(...).to_segy(path)` |
| A copy of the file | `vol.to_segy(path)` |
| One header on every trace | `ed.set_header("offset", values)` |
| A block of trace samples | `ed.write_traces(samples, first=0)` with a 2D array |

`stats`, `rms`, `slice_horizon`, `filter`, `clip`, `resort` and `to_regular` need whole traces. A stepped or reversed time window raises `ValueError` on those calls. Select traces, and leave `time` alone, or use `sel` / `isel` for a sample window and `numpy()` for that window.

### Stream a file that does not fit in RAM

`to_segy`, `clip`, `resort`, `segyrs.concat` and `to_regular(path)` stream. `budget=None` is half the available RAM. Leave it there unless the user set one. An int is bytes. A float in `(0, 1]` is a fraction of available RAM. `segyrs.BudgetError` fires before the allocation.

`to_regular()` with no path builds the dense cube in RAM and raises `segyrs.BudgetError` when the cube does not fit. `to_regular(path)` writes the same grid as a SEG-Y file, empty cells as dead traces. That path is the core's `to_regular_file`. Use it for a sparse or unsorted file you cannot hold.

Do not `numpy()` a volume and then `segyrs.create` a copy. `to_segy` copies trace bytes and keeps the sample format. Do not `numpy()` the cube to compute a mean: `stats()` is one pass.

### prefetch

`numpy()` tells the reader which traces it is about to decode. On a local memory map that hint is ignored. On an `s3://`, `gs://`, `az://` or `http(s)://` URL the reader fetches those traces in coalesced parallel requests, then decodes them. Select every trace you need and call `numpy()` once. A loop of one trace per call asks for one range at a time. URL reads are experimental and ship in the same wheel. `storage_options` is for URLs only.

### How much open reads

`scan="auto"` is the default. On a local file of more than 4096 traces whose sampled headers fit a regular grid, open reads those headers and checks each trace when you touch it. `geometry.source` is then `"assumed"`. A contradiction raises `segyrs.GeometryAssumptionError`. `scan="full"` reads every trace header at open. Any URL is scanned in full.

`index_cache=directory` stores the scanned header columns and reloads them on the next open of the same file (size, modification time, and a header fingerprint). A stale sidecar is replaced. Pass it when you reopen an irregular file. The measured open times are on the benchmarks page.

```python
import numpy as np
import segyrs

cube = np.random.default_rng(0).standard_normal((4, 5, 16)).astype(np.float32)
segyrs.create(
    "survey.sgy",
    cube,
    sample_format="ieee_float32",
    sample_interval_us=4000,
    inline=np.arange(100, 104),
    crossline=np.arange(200, 205),
)
vol = segyrs.open("survey.sgy")
assert vol.traces_read == 0
section = vol.sel(inline=101)
assert section.nbytes < vol.nbytes
section.numpy()
points = vol.sel_points(inline=[100, 103], crossline=[200, 204])
points.numpy()
vol.stats()
vol.rms()
vol.sel(inline=slice(100, 101)).to_segy("crop.sgy")
vol.to_regular("regular.sgy")
segyrs.concat(["survey.sgy", "crop.sgy"], "joined.sgy")
```

### Do not promise a speed

Give the user a time or a speedup only after you ran the code and timed it. Otherwise say it was not measured. The measured comparisons are at https://cratongen.github.io/segyrs/benchmarks/ . Quote a number from that page, with its load and its date, or run your own timing. Do not invent one.

## Hand it to a sibling

segyrs is one of a family of standalone libraries for mining and geoscience at https://github.com/cratongen.
None imports another. They pass protocol objects (`__minetypes__`) and NumPy arrays, and tables pass through
Arrow, so pandas, polars and pyarrow work in between.

`Volume.__minetypes__(budget=None)` returns a protocol version 1 dict (`seismic_volume` on a 3D
`(inline, crossline, time)` grid, `prestack_volume` on a 4D `(inline, crossline, offset, time)` grid or a 2D
`(trace, time)` volume, with `sample_unit="us"`). It decodes the selection into `data` (a copy of size
`vol.nbytes`) after checking `budget` (`segyrs.BudgetError` before allocating). `segyrs.create(path, obj,
sample_format=...)` accepts any `seismic_volume` or `prestack_volume` protocol object or dict.

| Library | Import | What it does |
| --- | --- | --- |
| boitata | `import boitata as bt` | Geostatistics: drill holes, compositing, block models, variograms, kriging, simulation |
| segyrs | `import segyrs` | Read, write and slice SEG-Y files, including files larger than memory |
| transiograms | `import transiograms as tg` | Markov chains, hidden Markov models, transiograms and categorical simulation |
| wireframes | `import wireframes as wf` | Solids, surfaces and strings: read, write, measure and combine |

<!-- Below this line: what segyrs hands to each sibling and takes from it, with one short example per neighbour.
Only calls that exist today in both libraries, checked against ../<sibling>/python/<pkg>/*.pyi. -->

**boitata.** A time slice plus CDP coordinates is a `PointSet`: one row per trace, `coords` shaped `(n, 2)` or `(n, 3)`, the amplitude as a column. boitata does not open SEG-Y. The other way, `points.x` is a numpy array. `Editor.set_headers` writes it back into `cdp_x` when you also write `scalco`.

```python
import numpy as np
import segyrs
import boitata as bt

cube = np.random.default_rng(0).standard_normal((4, 5, 16)).astype(np.float32)
x = np.tile(500_000.0 + 25.0 * np.arange(5), (4, 1))
y = np.repeat(7_500_000.0 + 25.0 * np.arange(4), 5).reshape(4, 5)
segyrs.create(
    "survey.sgy",
    cube,
    sample_format="ieee_float32",
    sample_interval_us=4000,
    inline=np.arange(100, 104),
    crossline=np.arange(200, 205),
    headers={"cdp_x": x, "cdp_y": y, "scalco": -100},
)
vol = segyrs.open("survey.sgy")
xy = np.column_stack([vol.coords["cdp_x"].reshape(-1), vol.coords["cdp_y"].reshape(-1)])
amp = vol.isel(time=0).numpy().reshape(-1)
points = bt.PointSet(xy, {"amplitude": amp})
with segyrs.edit("survey.sgy") as ed:
    ed.set_headers(
        {"cdp_x": np.asarray(points.x), "scalco": -100},
        scalars={"cdp_x": -100},
    )
```

**wireframes.** The same coordinates, with a z in the same length unit as x and y, are a `wf.PointSet`. `wf.dtm(points)` builds a surface by Delaunay triangulation in plan. A time in microseconds is the wrong z. Put the amplitude on the attribute table. The other way, `surface.coords` and `surface.triangles` are numpy arrays. `segyrs.create` writes a 2D, 3D or 4D numeric array, not a mesh. A coordinate column goes back through `set_headers`, as in the boitata example.

```python
import numpy as np
import segyrs
import wireframes as wf

cube = np.random.default_rng(0).standard_normal((4, 5, 16)).astype(np.float32)
x = np.tile(500_000.0 + 25.0 * np.arange(5), (4, 1))
y = np.repeat(7_500_000.0 + 25.0 * np.arange(4), 5).reshape(4, 5)
segyrs.create(
    "survey.sgy",
    cube,
    sample_format="ieee_float32",
    sample_interval_us=4000,
    inline=np.arange(100, 104),
    crossline=np.arange(200, 205),
    headers={"cdp_x": x, "cdp_y": y, "scalco": -100},
)
vol = segyrs.open("survey.sgy")
xyz = np.column_stack(
    [
        vol.coords["cdp_x"].reshape(-1),
        vol.coords["cdp_y"].reshape(-1),
        np.zeros(vol.shape[0] * vol.shape[1], dtype=np.float64),
    ]
)
amp = vol.isel(time=0).numpy().reshape(-1)
points = wf.PointSet(xyz, {"amplitude": amp})
surface = wf.dtm(points)
```

**transiograms.** `tg.Grid` and the simulation calls take integer category codes. A SEG-Y sample is an amplitude. segyrs does not classify samples, and there is no handoff. Do not pass amplitudes in as codes.

## Before you report a result

Run the code. On a file you wrote, call `segyrs.validate` and report `n_traces`, `n_samples` and `sample_format`. On a volume you read, report `shape`, `geometry.kind` and `stats()` (`count`, `min`, `max`, `mean`). Time labels on `coords["time"]` are microseconds. If you did not time the run, say the time was not measured.
