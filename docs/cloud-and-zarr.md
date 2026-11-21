# Cloud reads and Zarr export

Both live in the `segyrs-interop` crate and are opt-in cargo features:

| Feature | What it adds |
|---------|--------------|
| `cloud` | `cloud::ObjectReader`, a `segyrs::ReadAt` over Rust `object_store` (s3, gs, az, http, local files) |
| `zarr`  | `zarr::export`, a Zarr v3 writer, and `zarr::ZarrArray`, a reader for what it writes |

Neither is on the default open path. Opening a local file still maps it.

## Reading from object storage

```rust
use segyrs_interop::cloud::ObjectReader;

let reader = ObjectReader::open(
    "s3://bucket/survey/line.sgy",
    [("aws_region", "eu-west-1"), ("block_size", "8388608")],
)?;
let file = reader.open_segy(&segyrs::OpenOptions::default())?;
let first = file.decode(0, 1)?;
```

`ObjectReader::open` accepts `s3://`, `s3a://`, `gs://`, `az://`, `abfs://`,
`abfss://`, `http(s)://` and `file://` URLs, and plain local paths (a one-letter
scheme such as `C:` is a Windows drive). Credentials come from the environment the
way `object_store` reads them (`AWS_*`, `GOOGLE_*`, `AZURE_*`), then from the
options. `ObjectReader::new(store, path, options)` takes any `object_store` store,
including `InMemory`.

### Options

One map carries both kinds of key, like an fsspec `storage_options` dict:

- Store keys go to the `object_store` builder for that scheme (`aws_region`,
  `aws_access_key_id`, `google_service_account`, `azure_storage_account_name`,
  `timeout`, `allow_http`, and the rest of that builder's keys).
- Reader keys tune the cache:

| Key | Default | Meaning |
|-----|---------|---------|
| `block_size` | 4 MiB | Size and alignment of a cache fill, and the split size of long requests |
| `readahead` | 8 | Blocks kept in flight ahead of a sequential stream |
| `coalesce_gap` | 1 MiB | `prefetch` merges ranges closer than this |
| `concurrency` | 32 | Requests in flight per reader |
| `cache_bytes` | a quarter of available RAM | Cache budget |
| `request_timeout` | 60 s | Deadline for one request attempt |
| `retries` | 3 | Extra attempts after a missed deadline |

An unknown key is an error, so a mistyped credential key is never ignored. The
defaults are starting points and their speed is not measured.

### How requests are issued

The core reads 240-byte headers and single traces from many rayon workers. The
reader turns that into few, large, parallel range requests:

- A miss fetches the aligned block around it. Other threads that need the same
  bytes wait for that one request.
- A miss right after cached bytes, or the first read of a read-ahead block, keeps
  `readahead` blocks in flight in the background.
- `ReadAt::prefetch(ranges)` (and `segyrs_interop::prefetch_traces(file, reader,
  traces)`) merges the ranges a selection needs, splits long spans at
  `block_size`, and fetches them in parallel. A crossline selection should call it
  first. A local map ignores the hint.
- A read of `block_size` bytes or more skips the cache and becomes parallel
  block-sized requests into the caller's buffer.
- Ready blocks are evicted least recently used past `cache_bytes`.

Retries are in two layers. `object_store` retries HTTP failures by status (its
`RetryConfig`, with `retry_timeout` set to `request_timeout`). The reader adds a
deadline per attempt and retries a stalled request with doubling backoff.

Opening scans every trace header. With typical trace sizes the headers sit a few
kilobytes apart, so opening a remote file reads most of its bytes once. The
requests are large and parallel, but this is not a header-only read.

### Threads and the GIL

All requests run on one shared tokio runtime (`segyrs-io` threads). A caller
blocks on a channel and never enters that runtime, so `read_at` works from rayon
workers, from plain threads, and from inside another tokio runtime. Nothing calls
back into Python, so a binding releases the GIL for the whole call.
`ObjectReader` is `Clone` (shared cache), `Send` and `Sync`. Workers reopen by
path or URL. Nothing is pickled and no process is started (S-11).

## Zarr v3 export

```rust
use segyrs_interop::zarr::{export, DirSink, ZarrOptions};

let summary = export(&file, &DirSink::new("line.zarr"), &ZarrOptions::default())?;
```

With `cloud` as well, `zarr::ObjectSink::new(store, prefix)` writes the same keys
to an object store.

The store is a group (`zarr.json`) holding:

| Array | Dims | Type |
|-------|------|------|
| `data` | `inline, crossline[, offset], time` for a grid, `trace, time` otherwise | decoded sample type: IBM float is float32, int16 stays int16 |
| `trace_mask` | grid dims | bool, true where a trace exists |
| `inline`, `crossline`, `offset` or `trace` | itself | int64 labels, on-disk order |
| `time` or `depth` | itself | float64, sample index times the interval (ms for time) |
| `cdp_x`, `cdp_y` | grid dims | float64 with the coordinate scalar applied |

A file gets grid dims when every trace has its own (inline, crossline, offset)
cell and the box holds at most twice the trace count. Otherwise it is a trace
list. `ZarrOptions { depth: true, .. }` names the vertical dim `depth`.

- **Chunks (X-14).** By default a chunk is whole traces, grouped along the
  fastest on-disk axis up to `target_chunk_bytes` (8 MiB), then along the next
  axis only once the faster one is covered. `chunks: Some(vec![..])` rechunks,
  including along samples.
- **Holes (X-34).** A chunk with no trace is not written. Zarr reads a missing
  chunk as the fill value, and `trace_mask` is false there. `ZarrArray::read_chunk`
  returns the fill and does not raise.
- **No masking or packing.** The fill value is 0 (false for the mask). No
  `_FillValue`, `scale_factor` or `add_offset` attribute is written, so a reader
  does not turn real samples into NaN.
- **Encoding.** Regular chunk grid, default chunk keys with `/`, and the `bytes`
  codec, little endian, with no compressor.

The group attributes record the on-disk sample format, revision, sample interval,
geometry kind, trace count, and the first textual header.

### Why not the zarrs crate

`zarrs` 0.23.14 needs Rust 1.91, above this crate's 1.89, and resolves to 125
crates with default features (108 with only `filesystem`). The export needs one
JSON document per node and raw little-endian chunks. That subset of the v3 core
is a few hundred lines on `serde_json`, and the `zarr` feature resolves to 21
crates. The tests check every written `zarr.json` and chunk file against the v3
core rules.

## Python API

Both are experimental and are in the segyrs wheel: `pip install segyrs` is
enough. The extension (`crates/segyrs-py`, module `segyrs._segyrs`) links
`segyrs-interop` with the `cloud` and `zarr` features. There is no extra and
no separate package.

<!-- doc-block: no-run, needs an object store and its credentials -->
```python
import segyrs

vol = segyrs.open("s3://bucket/survey.sgy", storage_options={"aws_region": "eu-west-1"})
vol.sel(crossline=2030).numpy()     # prefetched in parallel range requests, then decoded
summary = vol.to_zarr("survey.zarr", chunks=None, target_chunk_bytes=None, storage_options=None)
```

The same calls run on a local `file://` URL, which goes through the same object-store reader:

```python
from pathlib import Path

import segyrs

vol = segyrs.open(Path("survey.sgy").resolve().as_uri())
vol.sel(crossline=2030).numpy()
summary = vol.to_zarr("survey.zarr")
```

`segyrs.open(path_or_url, *, storage_options=None, ...)`:

- A string that starts with `scheme://` (two or more letters) is a URL. Anything
  else, including `C:\...` and `pathlib.Path`, is a local path and is mapped.
- `storage_options` is the one map described in [Options](#options). Values may be
  strings, numbers or booleans (`True` becomes `"true"`). An unknown key raises
  `SegyrsError`. `storage_options` on a local path raises `ValueError`.
- `http://` needs `storage_options={"allow_http": True}`. The server must answer
  `HEAD` and ranged `GET` requests.
- A missing object or a refused credential raises `SegyrsError`.

`Volume.to_zarr(path, *, chunks=None, target_chunk_bytes=None, storage_options=None)`:

- `path` is a directory or a URL. A URL writes through `object_store` with the
  same store keys as `open`. The reader cache keys (`block_size` and the rest)
  raise there, because they mean nothing for a write.
- `chunks` is the explicit `data` chunk shape, one entry per dim.
  `target_chunk_bytes` sizes the default trace-aligned chunks (8 MiB).
- A Volume opened with `depth=True` writes a `depth` dim.
- Only the Volume that `segyrs.open` returned can be exported. A selection
  raises `ValueError`, because the export always writes the whole file.
- It returns `{"dims", "shape", "chunks", "data_type", "chunks_written",
  "chunks_missing"}`.

Both calls release the GIL for every request and every write.

| Python | Rust |
|--------|------|
| `segyrs.open(url, storage_options=...)` for a URL | `ObjectReader::open(url, storage_options)`, then `File::from_reader_scan(reader, len, &opts, scan)` |
| `segyrs.open(path)` for a local path | `File::open_with(path, &opts)` (mmap), then `map_reader` to `Arc<dyn ReadAt>` |
| `Volume.numpy()` on any selection | `ReadAt::prefetch` of the selected traces' byte ranges when the reader is not a map, then the normal decode |
| `Volume.to_zarr(path)` | `zarr::export(&file, &DirSink::new(path), &opts)` on the file the Volume already holds |
| `Volume.to_zarr("s3://...")` | `cloud::store_for(url, storage_options)`, then `zarr::export(&file, &ObjectSink::new(store, prefix), &opts)` |

The binding holds every file as `File<Arc<dyn ReadAt>>`. A local map still
returns its bytes through `as_bytes`, so decoding a local file reads the map
directly. The extra indirection is one dynamic call per read; its cost is not
measured.

The viewer extension (`segyrs-view`, built from the repository and not yet
released, [Viewer](viewer.md#python)) is a separate shared library and cannot
share Rust types with segyrs, so `File.reader()` hands it the file's reader as `segyrs::bridge::RawReader`: a `#[repr(C)]`
table of `extern "C"` functions around a reference-counted `Arc<dyn ReadAt>`,
in a Python capsule named `segyrs.reader.v1`. A mapped file's bytes cross as a
pointer, so the receiving side reads the map directly. Errors cross as a
message and an I/O error kind. Both sides must come from the same segyrs
version; `segyrs-view` pins it exactly, and `RawReader::check` refuses a table
of another layout. The cost of a call through the table is not measured.

The tests (`python/tests/test_interop.py`) use no cloud credentials. They read
through `file://` URLs and an in-process HTTP server that serves byte ranges.
