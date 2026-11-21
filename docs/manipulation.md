# Manipulation at scale

Whole-volume operations live in `segyrs::ops`. They stream through buffers sized
from a RAM budget, use every core through rayon, and never build a dense cube
unless you ask for one with `to_regular` and it fits.

| Operation | Rust call | Ledger |
| --- | --- | --- |
| Time slice, one sample per trace | `ops::time_slice(&file, &traces, sample)` | S-10 |
| Horizon window, one index per trace | `ops::slice_horizon(&file, &traces, &index, above, below)` | S-10 |
| Header filter | `ops::filter(&file, &preds)`, `ops::filter_columns(&cols, &preds)` | X-29 |
| Crop (traces and samples) | `ops::crop(&file, &traces, &Crop { samples, rezero_time }, budget, &mut sink)` | S-30, S-36 |
| Resort | `ops::resort(&file, &keys, &opts, &mut sink)`, `ops::sort_order(&file, &keys)` | |
| Dense grid in RAM | `ops::to_regular(&file, &traces, fill, budget)` | X-15, X-17 |
| Dense grid streamed | `ops::to_regular_stream(&file, &traces, fill, budget, &mut sink)` | X-17 |
| Concatenate | `ops::concat(&[&a, &b], budget, &mut sink)`, `ops::check_concat` | X-35 |
| Statistics | `ops::stats(&file, &traces)` | |
| Per-trace RMS | `ops::rms(&file, &traces)` | |
| Amplitude limit | `ops::clip(&file, &traces, lo, hi, budget, &mut sink)`, `ops::clip_samples` | |

`traces` is a list of trace numbers in the order you want them. Pass `0..n` for
the whole file, or the output of `filter`.

## The RAM budget

`Budget::from_available()` is half of the RAM available at the call: Linux
`MemAvailable`, lowered to the cgroup v2 headroom when a limit is set.
`Budget::fraction_of_available(f)` picks another fraction and `Budget::bytes(n)`
sets a number. There is no compiled-in cap. On a system without
`/proc/meminfo`, `from_available` returns an `Unsupported` error and you pass
`Budget::bytes`.

An operation that cannot fit raises `Error::Io` with kind `OutOfMemory` before
it allocates. The message names the bytes needed and the budget.

Streaming operations keep two chunk buffers live: the sink writes one while the
other cores fill the next. Each is at most half the budget. One trace larger
than half the budget is still processed, one at a time.

## Output: the `TraceSink` contract

Operations that produce traces do not write files themselves. They call a
`TraceSink`:

1. `begin(&OutputSpec)` once. The spec holds the source textual header bytes
   (all blocks), the 400-byte binary header with the sample count (and, for
   `resort`, the sorting code when asked) already patched, the sample format,
   byte order, sample count, interval, trace-header length, and the trace count
   when it is known.
2. `write_trace(header, samples)` or `write_block(block, trace_len, header_len)`
   for every trace in output order. The bytes are in the source byte order and
   the source sample format. Nothing is decoded and re-encoded, so IBM stays IBM
   bit for bit. `write_block` holds whole traces back to back. Its default
   implementation calls `write_trace` per trace.
3. `finish()` once. On an error the operation returns it and does not call
   `finish`. A writer should then discard its partial output.

The sink must be `Send`, because the next chunk is read while the sink writes
the current one. The M2 writer implements this trait as
`segyrs::write::FileSink`, which publishes the stream atomically
([Writing SEG-Y](writing.md#streaming-into-a-file-filesink)).

## Windows (S-10)

`time_slice` reads one sample per trace, one positional read each, in
parallel. A counting reader in `test_s10` proves the byte count is exactly
one sample per trace. `slice_horizon` reads `above + 1 + below` samples around
a different index per trace. Samples outside the trace are zero with `valid`
false, and a negative sentinel index gives an all-invalid row.

## Filter (X-29)

A `Predicate` is `Range { field, min, max }` (inclusive, either end open) or
`In { field, values }`. Predicates are combined with AND and evaluated in
parallel on the columns. `filter` scans only the header fields the predicates
name. `inline`, `crossline`, and `offset` follow the bytes the file was opened
with. Other names come from `HeaderSpec::standard()`.

## Crop (S-30, S-36)

Trace headers are copied byte for byte, so `cdp_x`, `cdp_y`, `scalco`, line
numbers, and every other field are those of the traces that remain. Lines are
not renumbered.

A sample window `start..end` rewrites the trace sample count and moves the
delay (bytes 109-110) by `start * interval`, so each kept sample keeps its
original time. The delay is milliseconds times the time scalar (bytes
215-216). The current scalar is kept when it can express the new delay
exactly. Otherwise the first of 1, -10, -100, -1000, -10000, 10, 100, 1000,
10000 that can is written, and a delay that fits none is an error.
`rezero_time: true` sets the delay to zero instead.

## Resort

`sort_order` reads the key columns, encodes each value as an order-preserving
64-bit key (descending keys are inverted), and sorts `(keys, input position)`
in parallel. Ties keep input order. That table costs `n_traces * (8 * keys + 8)`
bytes and is charged to the budget.

The trace bytes then move by `SortStrategy`:

- `InMemory`: one parallel sequential read, a reorder in RAM. Needs twice the
  trace data inside the budget.
- `Gather`: each output chunk is read trace by trace from the input in output
  order, in parallel. No temporary file.
- `External`: run generation, then a k-way merge. Worker threads each read a
  contiguous block of input traces with one sequential read, order the block
  by output rank, and write it as a sorted run into one temporary file sized
  for all trace data. Each output chunk then takes a contiguous slice from
  every run (its ranks are known), reads all slices in parallel, and scatters
  them into place. The merge holds three chunks: the one being written, the
  staging buffer, and the one being assembled. The run file is deleted on
  success and on error.
- `Auto` (default): `Gather` when the input is already in order, `InMemory`
  when it fits, `Gather` when the file is on a non-rotational Linux block
  device, and `External` on a rotating disk or an unknown medium such as a
  network file system. The device comes from `/proc/self/mountinfo` and
  `/sys/class/block/<dev>/queue/rotational`. `SortReport.strategy` says which
  one ran.

Measured on this machine (32 threads, NVMe behind LUKS and btrfs, load average
about 35 from other builds): a 2.16 GB IEEE float32 file, a shuffled
512 x 512 grid of 2000-sample traces with random samples, sorted by inline
and crossline under a 256 MiB budget into a sink that discards bytes. Three
interleaved rounds, medians. Cold means the file's page cache was dropped with
`dd oflag=nocache conv=notrunc,fdatasync count=0` before the run (`fincore`
confirmed 0 pages resident).

| Strategy | Cold | Warm |
| --- | --- | --- |
| `External` (529 runs) | 1.82 s | 1.36 s |
| `Gather` | 0.65 s | 0.15 s |

That is why `Auto` gathers on solid-state storage. A rotating disk was not
measured. `External` is there for media where random reads are slow, and the
temporary directory needs free space equal to the trace data.

## Dense grids (X-15, X-17)

Open never builds a dense cube. A sparse or varying-fold file stays a trace
list. `to_regular` builds one on request:

- Inline and crossline labels run from the minimum to the maximum by the
  largest step that divides every gap. Offsets are the distinct values present.
  The shape is inline, crossline, offset (when there is more than one), samples.
- Cells times decoded sample bytes, plus the mask and index, are checked against
  the budget before allocation. `regular_axes` gives the labels without that cost.
- The cube is filled chunk by chunk from what is left of the budget. `mask` is
  false where no trace exists, and those samples hold `fill`.

`to_regular_stream` writes the same grid to a sink, two chunks live at most.
A cell with a trace gets that trace's bytes. An empty cell gets a dead trace
(trace id 2) with the cell's labels, `fill` samples, and `cdp_x`, `cdp_y` from
a least-squares affine fit of the existing coordinates on inline and
crossline, in the first trace's `scalco`. Delay, time scalar, coordinate
units, sample count, and interval come from the first trace.

## Concatenate (X-35)

`concat` checks sample format, interval, sample count, byte order, and
trace-header length against the first file and raises `Error::FormatChange`
on any mismatch, before the sink sees anything. Nothing is cast. The output
uses the first file's textual and binary headers.

## Kernels

`stats` returns count, NaN and infinite counts, min, max, mean, the sum of
squared deviations (`variance()`, `std()`), and `rms()`. Each trace is reduced in
two passes while it is in cache, and groups of about 1 MiB of samples are merged
with the pairwise update of Chan, Golub, and LeVeque in a fixed order, so the
result does not depend on the thread count. `rms` gives one value per trace over
its finite samples.

`clip` streams traces to a sink. A sample inside `[lo, hi]` keeps its bytes. A
sample outside becomes the nearest value of the source format inside the
bounds (floor or ceiling for integers, directed rounding for IEEE and IBM).
`clip_samples` does the same on decoded samples in RAM.

## Python

Each operation is a `Volume` method on the current selection. Every call runs
with the GIL released. Operations that write a file stream through `FileSink`,
so the output is never held in RAM. `budget` is `None` (half the available
RAM), an int (bytes), or a float in (0, 1] (a fraction of the available RAM).

| Python | Rust | Returns |
| --- | --- | --- |
| `vol.sel(time=t)`, `vol.isel(time=i)` | the read path, one sample per trace | the slice, shaped like the trace dims |
| `vol.slice_horizon(horizon, *, above=0, below=0)` | `slice_horizon` | `(values, valid)`, shaped `(*trace dims, above + 1 + below)` |
| `vol.filter(**conditions)` | `filter` | a `(trace, time)` Volume in file axis order, each axis in selection order |
| `vol.sel(...).to_segy(path, *, rezero_time=False, budget=None)` | `crop` | the crop as a new file |
| `vol.resort(path, by=("inline", "crossline"), *, descending=False, strategy="auto", sorting_code=None, tmp_dir=None, budget=None)` | `resort` | `{"strategy", "already_sorted", "runs", "spilled_bytes"}` |
| `vol.to_regular(*, fill=0.0, budget=None)` | `to_regular` | `(data, mask, coords)` |
| `vol.to_regular(path, *, fill=0.0, budget=None)` | `to_regular_stream` | `{"cells", "filled"}` |
| `segyrs.concat(volumes_or_paths, path, *, budget=None)` | `concat` | traces written |
| `vol.stats()` | `stats` | `{"count", "nan", "infinite", "min", "max", "mean", "std", "variance", "rms"}` |
| `vol.rms()` | `rms` | one value per trace, shaped like the trace dims |
| `vol.clip(lo, hi, path, *, budget=None)` | `clip` | samples changed |

`filter` conditions are `field=(lo, hi)` or `field=slice(lo, hi)` for an
inclusive range (`None` leaves an end open), or a list, array, or scalar for a
set of values. `slice_horizon` takes one `time` (or `depth`) value per trace in
the units of `coords["time"]` and uses the nearest sample. NaN means no pick
and gives an all-invalid row. `to_regular` returns `coords` with `inline`,
`crossline`, `offset` (when there is more than one), and the vertical axis.

`resort` and `concat` work on whole files, and `stats`, `rms`, `clip`, and
`to_regular` on whole traces: a selection that does not fit raises
`ValueError`. Write the selection with `to_segy` first. The sync flag `sync=True`
on every writing call flushes the data and the directory before it returns.

## Limits

- A pairwise-byte-order file with an odd sample width (int8, int24) cannot be
  cropped in samples, clipped, or regularized to a sink, because its byte pairs
  straddle samples. Windows and `to_regular` in RAM still read it.
- Format 4 (fixed point with gain) cannot be clipped or filled, because segyrs
  does not encode it.
- The sort key table and the dense-grid trace index are in RAM and charged to
  the budget. The sort has no fallback when that table alone exceeds it.
