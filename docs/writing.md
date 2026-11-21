# Writing SEG-Y

This page covers the write path in `segyrs::write`: new files, copies, in-place edits, and validation. The Python calls `segyrs.create`, `Volume.to_segy`, `segyrs.edit`, and `segyrs.validate` are built on it (see [Python](#python)).

## New files: `create` and `copy`

A file to write is one `NewFile` value. There is no two-step spec object and no uninitialized file ([S-31] Creating a file from a 3D numpy array).

| Field | Meaning |
| --- | --- |
| `text` | Raw textual blocks: the 3200-byte primary block, then each extended block. Empty means a default EBCDIC header. |
| `binary_header` | The 400 bytes to start from, or `None` to build one. |
| `n_traces`, `n_samples`, `sample_interval_us` | The layout. The interval is an integer number of microseconds. |
| `extra_trace_header_blocks` | Rev 2 extra 240-byte trace-header blocks. |
| `overlays` | Trace-header fields to write: a `HeaderField` (byte, dtype, name) and one value per trace. |
| `samples` | Trace-major samples, `n_traces * n_samples` values, in any numeric type. |
| `source` | An opened `File`, an optional list of source traces, and the first sample of the window. |

`NewFile::new(n_traces, n_samples, interval_us)` starts from nothing. `NewFile::from_file(&file)` starts from an opened file with every byte as it is. Change the fields to select traces (crop), shorten the sample window, add overlays, or replace the samples, then call `create(path, &spec, &options)`. `copy(&file, path, &options)` is `create` of an unchanged `from_file`.

```rust
use segyrs::write::{create, HeaderValues, NewFile, Overlay, SampleData, WriteOptions};
use segyrs::{HeaderDtype, HeaderField, SampleFormat};

let samples: Vec<f32> = vec![0.0; 6 * 500];
let inline = [10, 10, 11, 11, 12, 12];
let crossline = [1, 2, 1, 2, 1, 2];
let mut spec = NewFile::new(6, 500, 4000);
spec.samples = Some(SampleData::F32(&samples));
spec.overlays = vec![
    Overlay { field: HeaderField::new(189, HeaderDtype::I32, "inline"), values: HeaderValues::I32(&inline) },
    Overlay { field: HeaderField::new(193, HeaderDtype::I32, "crossline"), values: HeaderValues::I32(&crossline) },
];
let options = WriteOptions { sample_format: Some(SampleFormat::IbmFloat32), ..WriteOptions::default() };
create("out.sgy", &spec, &options)?;
```

### Sample format

The file keeps the source sample format. An IBM file stays IBM. `WriteOptions::sample_format` is the only conversion ([S-03], [S-35] Change segy format). A file with no source must name its format, because segyrs does not pick IEEE float32 on its own (roadmap decision 3).

Each format has a natural type, the type `File::decode` returns: float32 for IBM and IEEE32, float64 for IEEE64 and format 4, int32 for int32 and int24, uint32 for uint32 and uint24, and the same width for the rest. Samples of the natural type are stored exactly, apart from IBM, which rounds. Any other pairing converts:

- Float targets (IBM, IEEE32, IEEE64, format 4) round to nearest, ties to even. A finite value beyond the target range is an error. NaN is an error for IBM and format 4.
- Integer targets take only whole values in range. A fraction, NaN, infinity, or an out-of-range value is an error. Nothing is truncated or clamped.

A sample that cannot be stored raises `Error::SampleValue` with the trace, the sample index, the value, and the format. When several samples fail, the error names the first in file order, whatever the number of threads. The destination is not touched.

### IBM float encoding

`f32_to_ibm` and `f64_to_ibm` implement the IBM System/360 single-precision layout: sign bit, 7-bit excess-64 base-16 exponent, 24-bit fraction.

- Rounding is to nearest, ties to even. IBM keeps 21 to 24 significant bits, depending on the leading hex digit, so an f32 with more bits rounds. A carry out of the fraction moves to the next exponent.
- Every finite f32 is inside the IBM range. Infinity saturates to the largest IBM magnitude, `0x7FFFFFFF` with the sign, which decodes back to infinity. NaN has no IBM pattern and is an error.
- From f64, a finite value above the IBM range (about 7.2e75) is an error, not a clamp. Below the smallest normalized value (16^-65), the encoder uses exponent 0 with an unnormalized fraction and rounds to zero below 2^-280.
- Negative zero is `0x80000000`.

Decoding is `ibm_to_f32` from the read path ([S-01] IBM360Converter converts to IEEE NaN for some values). Every normalized IBM word that decodes to a normal f32 encodes back to the same word. An unnormalized word, or a zero with a nonzero exponent, decodes to the same value as its normalized form and re-encodes to that form. The raw copy keeps those words as they are.

`cargo test --release --test write_ledger -- --ignored` (`mise run test-ibm-exhaustive`) checks both claims over all 2^32 bit patterns: f32 to IBM against an independent rounding, and IBM to f32 to IBM on every normalized word.

### Binary header

The base 400 bytes are written back as they are. The writer sets a word only when its value changes: sample count, interval, format code, revision, extended textual count, extra trace-header count, and the rev 2.1 trace count and first-trace offset when those were set. An unedited copy is byte-identical ([X-22] binary header round trip), and a shorter sample window updates the count in the same write as the traces ([S-33] Copy with fewer samples fails). The interval is the integer you pass, never a step re-derived from a float axis ([S-29] Wrong sample interval written).

A new file is rev 1, or rev 2.1 when it needs rev 2 features: little-endian or pairwise byte order, more than 65535 samples or microseconds, or extra trace-header blocks. Rev 2 files get the byte-order signature at bytes 3297-3300. A rev 1 file does not, because rev 1 has no such field.

### Trace headers

Trace headers start as the source's raw bytes, or as 240 zero bytes for a new file. Only the overlays are written on top ([X-26] header fields could not be dropped). The writer owns two structural words in a new file: the sample count at bytes 115-116 and the interval at bytes 117-118, copied from the binary header. An overlay can still set them. In a copy, those words follow a changed count or interval only where they held the old value.

An overlay value that does not fit the field raises `Error::HeaderValue` with the field name and byte, the value, and the width ([X-24] header export fails on a non-integer). `HeaderValues::Scaled` stores a float coordinate as the integer that `apply_scalar` turns back into it: a negative scalar multiplies on write, a positive scalar divides, zero means 1. Coordinate 1234.5 with scalar -10 is stored as 12345. A value that the scalar cannot represent, such as 1234.56 with -10, is an error rather than a silent rounding.

### Selection, windows, and 2D lines

`Source::traces` lists the source trace for each output trace, so a crop is a selection then a write, and the headers come from the traces that remain ([X-25] export of 2D lines and crops). A 2D line needs no inline or crossline: write the `cdp` field (byte 21) as an overlay. `Source::first_sample` with a smaller `n_samples` writes a window. `create` does not change the delay word, so pass a delay overlay when the window should keep its time. `ops::crop` moves the delay with the window for you (S-36, [Crop](manipulation.md#crop-s-30-s-36)).

### Byte order

`WriteOptions::endian` writes big, little, or pairwise. A copy that changes byte order reorders every standard binary-header and trace-header word, and the sample words. Unassigned bytes are copied as they are. Rev 2 bytes 3501-3502 are two single bytes (major, minor), so they are not swapped. A copy with extra trace-header blocks keeps its byte order, because the layout of those blocks is not known.

### Publishing

`create` writes a temp file in the destination directory, preallocates the final size with `fallocate` (skipped where the filesystem does not support it), writes, calls `fdatasync`, renames the temp file over the destination, and calls `fsync` on the directory. The destination is either the old file or the complete new one. An error removes the temp file. Writing over the file that a `Source` maps is safe, because the map keeps the old inode. `WriteOptions::sync = false` skips both flushes for scratch files.

rayon threads encode buffers of whole traces while one thread writes them in order. Each buffer is 1/64 of the memory available when the write starts, and at most three are alive.

## In-place edits: `Editor`

`Editor::open(path)` opens a file for edits that never change its layout.

| Call | Does |
| --- | --- |
| `write_header_column(field, values)` | One field on every trace ([S-08] slow header writes). |
| `write_header_columns(overlays, traces)` | Several fields, on every trace or on a list of traces. |
| `write_trace_samples(trace, samples)` | One trace's samples ([S-32] scalar written as a trace). |
| `write_traces(first, samples)` | Consecutive traces, encoded in parallel. |
| `write_binary_header(bytes)` | The 400 bytes. |
| `write_textual_block(index, bytes)` | One 3200-byte block, 0 being the primary block. |
| `sync()` | One `fdatasync` after a batch. |

Every call checks all of its input first, then writes, so an error leaves the file as it was. Header values go through the same range checks as `create`. A sample vector whose length is not the sample count raises `Error::TraceLength` before any byte is written. Samples are encoded in the file's own format with the conversion rules above.

A change of sample format, sample count, or extra-header count is a new file, not an edit ([S-35]). `Editor::open_with(path, &options, &Expect { .. })` refuses a file that differs from what the caller expects with `Error::FormatChange`. A binary-header or textual-block write is first parsed by the reader with the new bytes in place, and it is refused with `FormatChange` if the format, sample count, trace count, first-trace offset, byte order, or extra-header count would change. A per-trace sample count different from the file's is refused the same way. Use `create` to write the new layout.

Header writes go out in file order. For traces up to 16 KiB, a run of consecutive traces is read, patched in memory, and written back in one `pwrite`; the sample bytes go back unchanged, so do not write samples from another process during the edit. Longer traces get one `pwrite` per trace covering the touched bytes. When the fields leave a gap, the gap is read first and written back unchanged. A multi-byte field can tear if the machine stops in the middle of a write. Call `sync` for durability.

## Validation: `validate`

`validate(path)` returns the layout or `Error::Invalid` with every problem found ([S-34] rewritten file does not open in other programs). It checks that the format code is known, that the textual blocks are whole 3200-byte blocks framed by the extended count or the `EndText` stanza, that the file length is the headers plus `n_traces * stride` exactly, that the sample count and interval are nonzero when there are traces, that the revision and fixed-length flag hold legal values, and that every trace header's sample count is zero or the binary header's. A file that passes is consistent in sizes and codes. That does not prove that every other program accepts it.

`f3.sgy` and the `multiformats/` fixtures fail the last check: they were cropped from a 462-sample file to 75 samples and their trace headers still say 462.

## Streaming into a file: `FileSink`

`FileSink::new(path, sync)` is the M2 writer behind the M3 `TraceSink` contract ([Manipulation at scale](manipulation.md#output-the-tracesink-contract)). `begin` creates a temp file beside the destination and preallocates it when the operation announces its trace count. The traces are appended as the operation hands them over, in the source byte order and sample format, so an IBM file stays IBM bit for bit. `finish` truncates to the bytes written, sets the rev 2 trace count (bytes 3513-3520) to the traces written when the source had set it, flushes when `sync` is true, renames the temp file over the destination, and flushes the directory. An operation that fails does not call `finish`, and dropping the sink removes the temp file, so the destination is the old file or the complete new one.

## Python

```text
vol.to_segy(path, *, sample_format=None, endian=None, rezero_time=False, budget=None, sync=True) -> None
segyrs.create(path, data, *, sample_format, sample_interval_us, inline=None, crossline=None,
              offset=None, headers=None, text=None, endian=None, sync=True) -> int  # bytes
segyrs.validate(path) -> dict          # revision, sample_format, n_traces, n_samples, ...
segyrs.edit(path) -> segyrs.Editor     # context manager
```

`Volume.to_segy` writes the selection: the traces flattened slowest file axis first (so a crossline-major file stays crossline-major), each axis in selection order, with their header bytes, and the selected sample window, which must be contiguous and increasing (an `isel` or `sel` slice with step 1, or one sample). Without a conversion it runs `ops::crop` into a `FileSink`, so the delay moves with the window (S-36), `rezero_time=True` sets it to zero, and memory follows `budget` (`None` is half the available RAM, an int is bytes, a float in (0, 1] is a fraction). With `sample_format=` or `endian=` it runs `create` with the volume as the `Source` and a delay overlay in the trace's own time scalar. A delay that scalar cannot express raises `HeaderValueError`. An unselected volume copies to a byte-identical file (X-22).

`segyrs.create` takes `(trace, time)`, `(inline, crossline, time)`, or `(inline, crossline, offset, time)` samples of any numeric dtype. For 3D and 4D data, `inline`, `crossline`, and `offset` are the axis labels (default `1..n`). For 2D data they are one value per trace. `headers` maps a standard field name or `(byte, dtype)` to one value per trace (in C order of the leading dims) or one for all. A float `cdp_x`, `cdp_y`, `source_*`, or `group_*` value is stored with the `scalco` in the same mapping, an elevation with `scalel`. `text` is the textual header as cards (one string with one card per line, or a list), written as EBCDIC. `sample_format` and `sample_interval_us` are required.

`segyrs.Editor` methods, each checked in full before any byte is written:

| Method | Does |
| --- | --- |
| `set_header(field, values, *, traces=None, scalars=None)` | One field, on every trace or on `traces`. `scalars` encodes float coordinates in real units. It does not write the scalar word: put `scalco` (or `scalel`) in the same `set_headers` call, knowing it scales every coordinate of those traces. |
| `set_headers(mapping, *, traces=None, scalars=None)` | Several fields in one pass. |
| `write_traces(samples, first=0)` | One trace (1D) or consecutive traces (2D), encoded in the file's format. |
| `write_binary_header(data)`, `write_textual_block(index, data)` | Raw blocks. A layout change raises `FormatChangeError`. |
| `sync()`, `close()` | One `fdatasync`; release the file. Leaving the `with` block without an error syncs, then closes. |

`n_traces`, `n_samples`, and `sample_format` describe the file. The errors map to `segyrs.HeaderValueError` (`Error::HeaderValue`), `segyrs.SampleValueError` (`Error::SampleValue`), `segyrs.SpecError` (`Error::Spec`), and `segyrs.ValidationError` (`Error::Invalid`, with the list in `err.problems`), all under `segyrs.SegyrsError`.

## Round-trip proof

`tests/write_roundtrip.rs` reads every file in `test-data/`, writes it to a temp directory twice (a raw copy, and a decode then re-encode), and compares the bytes with the original. All 81 fixtures that open are identical both ways, except the decode then re-encode of the two format 4 files: format 4 stores one value in several (gain, magnitude) pairs and the encoder picks the most precise, so the values are identical and the bytes are not. Their raw copies are identical.

These 10 fixtures do not open, so there is nothing to write. The test fails if this list goes stale.

| Fixture | Reason |
| --- | --- |
| `broken.sgy` | The end of the file is cut off. |
| `decrement.sgy`, `decrement-lsb.sgy` | Every header field is at its limit, so the per-trace sample count contradicts the binary header. |
| `increment.sgy`, `increment-lsb.sgy` | Every header field is at its minimum, with the same contradiction. |
| `long.sgy` | Declares 60000 samples, and the trace headers are missing. |
| `mapping-empty.sgy`, `mapping-invalid-value-size-scale6.sgy` | Rev 2.1 mapping fixtures whose size is not the headers plus whole traces. |
| `small.su`, `small-lsb.su` | Seismic Unix files, which have no textual or binary header. |

`mapping-shifted.sgy` opens only without geometry (its inline and crossline bytes hold an absurd span), and the test copies it that way.

Flipping `small.sgy` and `f3.sgy` to little-endian reproduces `small-lsb.sgy` and `f3-lsb.sgy` byte for byte. The `multiformats/` little-endian files differ from the flipped big-endian files in bytes 3501-3502 only, because the tool that made them swapped the revision bytes as one 16-bit word. Flipping back reproduces the big-endian originals.

## Performance

All numbers are from this machine (Intel i9-13980HX, 32 threads, Micron 2400 NVMe, btrfs with zstd:3 compression on dm-crypt) on 2026-11-17, with other jobs running: the load average was between 7 and 37. Each figure is the median of 5 interleaved rounds, and every arm includes its `fdatasync`. Samples are pseudo-random, so compression does not flatter them. Rerun with `mise run bench-write -- <dir on the disk under test>`.

New files, 300000 traces of 1500 samples (1.87 GB), `examples/write_bench`:

| Arm | Median | Throughput |
| --- | --- | --- |
| One thread writes a prepared buffer of the same size, then `fdatasync` | 0.652 s | 2.87 GB/s |
| `copy`, IBM to IBM (raw bytes) | 0.725 s | 2.58 GB/s |
| `copy`, IBM to IEEE32 | 0.739 s | 2.53 GB/s |
| `create` from an in-memory f32 array to IBM, with two overlays | 1.070 s | 1.75 GB/s |

The pipeline buffer size (1 MiB to 1 GiB, tried in one interleaved run) did not separate from the noise at this load: single rounds ranged from 0.6 s to 16 s, and the best rounds were about 0.6 s from 4 MiB to 256 MiB. The default stays at 1/64 of available memory.

In-place header column, one 4-byte field on every trace, then `fdatasync`, `examples/edit_bench` (300000 traces, stride 6240). Warm means the file is in the page cache. Cold means it was evicted with `POSIX_FADV_DONTNEED` first.

| Strategy | Warm | Cold |
| --- | --- | --- |
| One `pwrite` per trace, in file order (includes `Editor::open`) | 2.494 s | 4.876 s |
| The same `pwrite`s from rayon threads | 2.777 s | 68.127 s |
| Shared mmap, rayon stores, `msync` | 3.469 s | 3.370 s |
| Read a run of traces, patch, write the run back | 0.802 s | 1.233 s |

Through `Editor`, per-trace `pwrite` against rewriting runs (edit only, after open):

| Stride | Traces | `pwrite` warm | runs warm | `pwrite` cold | runs cold |
| --- | --- | --- | --- | --- | --- |
| 1440 | 1200000 | 1.477 s | 0.684 s | 1.447 s | 0.649 s |
| 6240 | 300000 | 2.985 s | 1.634 s | 3.478 s | 0.775 s |
| 48240 | 40000 | 0.659 s | 0.947 s | 0.739 s | 1.416 s |

So `Editor` rewrites runs when the stride is at most 16 KiB and uses one `pwrite` per trace above that. The crossover lies between the 6240 and 48240 strides measured. The exact point is not measured.
