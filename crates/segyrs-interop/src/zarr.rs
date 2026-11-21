//! Opt-in Zarr v3 export of a SEG-Y file, and a small reader for what it writes.
//!
//! The store is a group with these arrays:
//!
//! - `data`: samples, dims `(inline, crossline[, offset], time|depth)` for a file
//!   with a grid, else `(trace, time|depth)`. The dtype is the decoded sample
//!   type (IBM float becomes float32, int16 stays int16).
//! - `trace_mask`: bool, true where a trace exists.
//! - `inline`, `crossline`, `offset` (grid) or `trace` (no grid), and `time` or
//!   `depth`: 1D coordinates.
//! - `cdp_x`, `cdp_y`: float64 coordinates with the header scalar applied.
//!
//! Default `data` chunks follow on-disk order (X-14): whole traces along the
//! vertical axis, grouped along the fastest on-disk grid axis up to
//! `target_chunk_bytes`. Other chunk shapes only when `chunks` is passed.
//!
//! A chunk with no trace is not written. Zarr reads a missing chunk as the fill
//! value, and `trace_mask` is false there (X-34). The fill is 0 and no
//! `_FillValue`, `scale_factor` or `add_offset` attribute is written, so no
//! reader masks or rescales real samples (the X-06 lesson).
//!
//! Only the v3 core is used: the `regular` chunk grid, the `default` chunk key
//! encoding with `/`, and the `bytes` codec, little endian, with no compressor.

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use segyrs::{
    apply_scalar, Axis, Error, File, GeometryKind, ReadAt, Result, SampleFormat, Samples,
};
use serde_json::{json, Value};

/// Where the store's keys go. Keys use `/` separators, relative to the root.
pub trait Sink: Send + Sync {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;
}

/// A local directory store.
pub struct DirSink {
    root: PathBuf,
}

impl DirSink {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Sink for DirSink {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.root.join(key);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

/// An object-store prefix (`s3://bucket/out.zarr`). Writes run on the shared
/// I/O runtime.
#[cfg(feature = "cloud")]
pub struct ObjectSink {
    store: std::sync::Arc<dyn object_store::ObjectStore>,
    prefix: object_store::path::Path,
}

#[cfg(feature = "cloud")]
impl ObjectSink {
    pub fn new(
        store: std::sync::Arc<dyn object_store::ObjectStore>,
        prefix: object_store::path::Path,
    ) -> Self {
        Self { store, prefix }
    }
}

#[cfg(feature = "cloud")]
impl Sink for ObjectSink {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        use object_store::ObjectStoreExt;
        let mut path = self.prefix.clone();
        for part in key.split('/') {
            path = path.join(part);
        }
        let store = std::sync::Arc::clone(&self.store);
        let payload = object_store::PutPayload::from(bytes.to_vec());
        crate::cloud::block_on(async move { store.put(&path, payload).await })?
            .map_err(|e| Error::Io(io::Error::other(e)))?;
        Ok(())
    }
}

/// Export settings. The default is trace-aligned chunks of about 8 MiB.
#[derive(Clone, Debug)]
pub struct ZarrOptions {
    /// Explicit `data` chunk shape, one entry per dim. This is the rechunk.
    pub chunks: Option<Vec<u64>>,
    /// Size of a default chunk, in bytes. A chunk is never less than one trace.
    pub target_chunk_bytes: u64,
    /// Name the vertical dim `depth` instead of `time`.
    pub depth: bool,
}

impl Default for ZarrOptions {
    fn default() -> Self {
        Self {
            chunks: None,
            target_chunk_bytes: 8 << 20,
            depth: false,
        }
    }
}

/// What `export` wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exported {
    pub dims: Vec<String>,
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub data_type: String,
    pub chunks_written: u64,
    /// Chunks with no trace, left for the reader to fill.
    pub chunks_missing: u64,
}

/// Write `file` as a Zarr v3 group into `sink`.
pub fn export<R: ReadAt>(file: &File<R>, sink: &dyn Sink, opts: &ZarrOptions) -> Result<Exported> {
    let layout = file.layout();
    let ns = layout.n_samples as u64;
    let ntraces = layout.n_traces;
    let (data_type, itemsize) = dtype_of(file.format());
    let _ra = file.readahead();
    // The grid maps every trace by its header keys: an assumed geometry must hold.
    file.verify_geometry()?;
    let grid = Grid::of(file);
    let vertical = if opts.depth { "depth" } else { "time" };

    let mut dims: Vec<String> = grid.dims.iter().map(|d| d.to_string()).collect();
    dims.push(vertical.into());
    let mut shape = grid.shape.clone();
    shape.push(ns);
    let chunks = match &opts.chunks {
        Some(c) => {
            if c.len() != shape.len() || c.contains(&0) {
                return Err(invalid(format!(
                    "chunks {c:?} must have {} entries, each at least 1, for dims {dims:?}",
                    shape.len()
                )));
            }
            c.clone()
        }
        None => default_chunks(&grid, ns, itemsize as u64, opts.target_chunk_bytes),
    };

    // Group metadata with the SEG-Y facts a reader needs to go back.
    let text: Vec<String> = file
        .textual_header()
        .first()
        .map(|t| t.cards.clone())
        .unwrap_or_default();
    let group = json!({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "segyrs_version": env!("CARGO_PKG_VERSION"),
            "segy_sample_format": file.format().name(),
            "segy_revision": layout.revision.major(),
            "sample_interval_us": layout.sample_interval_us,
            "geometry": file.geometry().kind.as_str(),
            "n_traces": ntraces,
            "textual_header": text,
        }
    });
    put_json(sink, "zarr.json", &group)?;

    // Coordinates.
    let grid_dims: Vec<&str> = grid.dims.to_vec();
    for (name, values) in grid.dims.iter().zip(&grid.labels) {
        write_1d_i64(sink, name, values)?;
    }
    let step = f64::from(layout.sample_interval_us) / 1000.0;
    let vert: Vec<f64> = (0..ns).map(|i| i as f64 * step).collect();
    let units = if opts.depth { "" } else { "ms" };
    write_array(
        sink,
        vertical,
        &[ns],
        &[ns.max(1)],
        "float64",
        &[vertical],
        json!({ "units": units }),
        |_| Some(vert.iter().flat_map(|v| v.to_le_bytes()).collect()),
    )?;

    // Mask and scaled CDP coordinates share the grid chunking of `data`.
    let gchunks = &chunks[..grid.shape.len()];
    let mut mask = vec![0u8; grid.cells()];
    let mut cdp_x = vec![0f64; grid.cells()];
    let mut cdp_y = vec![0f64; grid.cells()];
    for (cell, trace) in grid.cell_trace.iter().enumerate() {
        if let Some(t) = *trace {
            let t = t as usize;
            mask[cell] = 1;
            cdp_x[cell] = apply_scalar(file.cdp_x()[t], file.scalco()[t]);
            cdp_y[cell] = apply_scalar(file.cdp_y()[t], file.scalco()[t]);
        }
    }
    write_grid(sink, "trace_mask", &grid, gchunks, "bool", &mask, 1)?;
    let bx: Vec<u8> = cdp_x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let by: Vec<u8> = cdp_y.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_grid(sink, "cdp_x", &grid, gchunks, "float64", &bx, 8)?;
    write_grid(sink, "cdp_y", &grid, gchunks, "float64", &by, 8)?;

    // Samples, one chunk per rayon task.
    let mut data_dims = grid_dims.clone();
    data_dims.push(vertical);
    let counts: Vec<u64> = shape
        .iter()
        .zip(&chunks)
        .map(|(n, c)| n.div_ceil(*c))
        .collect();
    let total: u64 = counts.iter().product();
    write_meta(
        sink,
        "data",
        &shape,
        &chunks,
        &data_type,
        &data_dims,
        json!({ "segy_sample_format": file.format().name() }),
    )?;
    let written: Result<Vec<bool>> = (0..total)
        .into_par_iter()
        .map(|linear| {
            let idx = unravel(linear, &counts);
            let bytes = data_chunk(file, &grid, &shape, &chunks, &idx, itemsize)?;
            match bytes {
                Some(b) => {
                    sink.put(&chunk_key("data", &idx), &b)?;
                    Ok(true)
                }
                None => Ok(false),
            }
        })
        .collect();
    let written = written?;
    let n_written = written.iter().filter(|w| **w).count() as u64;
    Ok(Exported {
        dims: data_dims.iter().map(|d| d.to_string()).collect(),
        shape,
        chunks,
        data_type,
        chunks_written: n_written,
        chunks_missing: total - n_written,
    })
}

/// Trace position of every grid cell, in C order over `dims`.
struct Grid {
    dims: Vec<&'static str>,
    shape: Vec<u64>,
    labels: Vec<Vec<i64>>,
    /// Grid dims from fastest to slowest on disk.
    fast_order: Vec<usize>,
    cell_trace: Vec<Option<u64>>,
}

impl Grid {
    fn cells(&self) -> usize {
        self.cell_trace.len()
    }

    fn of<R: ReadAt>(file: &File<R>) -> Self {
        let g = file.geometry();
        let n = file.layout().n_traces;
        let with_offset = g.offsets.len() > 1;
        let cells = (g.inlines.len() as u64)
            .saturating_mul(g.crosslines.len() as u64)
            .saturating_mul(if with_offset {
                g.offsets.len() as u64
            } else {
                1
            });
        // A grid is only used when every trace has its own cell and the box is
        // not mostly empty. Otherwise the export is a trace list.
        let gridded = g.kind != GeometryKind::Unstructured
            && g.duplicate.is_none()
            && cells > 0
            && cells <= n.saturating_mul(2);
        if !gridded {
            return Self {
                dims: vec!["trace"],
                shape: vec![n],
                labels: vec![(0..n as i64).collect()],
                fast_order: vec![0],
                cell_trace: (0..n).map(Some).collect(),
            };
        }
        let mut dims = vec!["inline", "crossline"];
        let mut labels = vec![widen(&g.inlines), widen(&g.crosslines)];
        if with_offset {
            dims.push("offset");
            labels.push(widen(&g.offsets));
        }
        let shape: Vec<u64> = labels.iter().map(|l| l.len() as u64).collect();
        let pos = |values: &[i32]| -> HashMap<i32, u64> {
            values
                .iter()
                .enumerate()
                .map(|(i, v)| (*v, i as u64))
                .collect()
        };
        let (pil, pxl, poff) = (pos(&g.inlines), pos(&g.crosslines), pos(&g.offsets));
        let mut cell_trace = vec![None; cells as usize];
        let noff = if with_offset { shape[2] } else { 1 };
        for t in 0..n as usize {
            let i = pil[&file.inline()[t]];
            let x = pxl[&file.crossline()[t]];
            let o = if with_offset {
                poff[&file.offset()[t]]
            } else {
                0
            };
            cell_trace[((i * shape[1] + x) * noff + o) as usize] = Some(t as u64);
        }
        let dim_of = |a: &Axis| match a {
            Axis::Inline => Some(0),
            Axis::Crossline => Some(1),
            Axis::Offset if with_offset => Some(2),
            Axis::Offset => None,
        };
        let mut fast_order: Vec<usize> = g.axes.iter().rev().filter_map(dim_of).collect();
        for d in (0..dims.len()).rev() {
            if !fast_order.contains(&d) {
                fast_order.push(d);
            }
        }
        Self {
            dims,
            shape,
            labels,
            fast_order,
            cell_trace,
        }
    }
}

fn widen(v: &[i32]) -> Vec<i64> {
    v.iter().map(|x| i64::from(*x)).collect()
}

/// Whole traces, grouped along the fastest on-disk axis, then the next one only
/// once the faster axis is covered.
fn default_chunks(grid: &Grid, ns: u64, itemsize: u64, target: u64) -> Vec<u64> {
    let trace_bytes = (ns * itemsize).max(1);
    let mut room = (target / trace_bytes).max(1);
    let mut chunks = vec![1u64; grid.shape.len()];
    for &d in &grid.fast_order {
        let n = grid.shape[d].max(1);
        let c = room.min(n);
        chunks[d] = c;
        room /= c;
        if c < n || room <= 1 {
            break;
        }
    }
    chunks.push(ns.max(1));
    chunks
}

/// One `data` chunk, or `None` when no trace falls in it.
fn data_chunk<R: ReadAt>(
    file: &File<R>,
    grid: &Grid,
    shape: &[u64],
    chunks: &[u64],
    idx: &[u64],
    itemsize: usize,
) -> Result<Option<Vec<u8>>> {
    let gdims = grid.shape.len();
    let ns = shape[gdims];
    // Samples are chunked too when the caller asks for it.
    let s0 = idx[gdims] * chunks[gdims];
    let s1 = (s0 + chunks[gdims]).min(ns);
    let chunk_ns = chunks[gdims] as usize;
    let cell_chunk = &chunks[..gdims];
    let n_cells: u64 = cell_chunk.iter().product();
    // (position in chunk, trace), in chunk C order.
    let mut wanted = Vec::new();
    for local in 0..n_cells {
        let lidx = unravel(local, cell_chunk);
        let mut cell = 0u64;
        let mut inside = true;
        for d in 0..gdims {
            let g = idx[d] * chunks[d] + lidx[d];
            if g >= shape[d] {
                inside = false;
                break;
            }
            cell = cell * shape[d] + g;
        }
        if inside {
            if let Some(t) = grid.cell_trace[cell as usize] {
                wanted.push((local as usize, t));
            }
        }
    }
    if wanted.is_empty() {
        return Ok(None);
    }
    let mut out = vec![0u8; n_cells as usize * chunk_ns * itemsize];
    // Consecutive positions holding consecutive traces decode as one run.
    let mut i = 0;
    while i < wanted.len() {
        let mut j = i + 1;
        while j < wanted.len()
            && wanted[j].0 == wanted[j - 1].0 + 1
            && wanted[j].1 == wanted[j - 1].1 + 1
        {
            j += 1;
        }
        let run = file.decode(wanted[i].1 as usize, j - i)?;
        let bytes = le_bytes(&run);
        let trace_bytes = ns as usize * itemsize;
        for (k, (pos, _)) in wanted[i..j].iter().enumerate() {
            let src = &bytes[k * trace_bytes..(k + 1) * trace_bytes];
            let src = &src[s0 as usize * itemsize..s1 as usize * itemsize];
            let dst = pos * chunk_ns * itemsize;
            out[dst..dst + src.len()].copy_from_slice(src);
        }
        i = j;
    }
    Ok(Some(out))
}

fn le_bytes(s: &Samples) -> Vec<u8> {
    fn le<T: Copy, const N: usize>(v: &[T], f: fn(T) -> [u8; N]) -> Vec<u8> {
        v.iter().flat_map(|x| f(*x)).collect()
    }
    match s {
        Samples::I8(v) => le(v, i8::to_le_bytes),
        Samples::U8(v) => v.clone(),
        Samples::I16(v) => le(v, i16::to_le_bytes),
        Samples::U16(v) => le(v, u16::to_le_bytes),
        Samples::I32(v) => le(v, i32::to_le_bytes),
        Samples::U32(v) => le(v, u32::to_le_bytes),
        Samples::I64(v) => le(v, i64::to_le_bytes),
        Samples::U64(v) => le(v, u64::to_le_bytes),
        Samples::F32(v) => le(v, f32::to_le_bytes),
        Samples::F64(v) => le(v, f64::to_le_bytes),
    }
}

/// Zarr data type of the decoded samples, and its size.
fn dtype_of(format: SampleFormat) -> (String, usize) {
    let name = match format {
        SampleFormat::IbmFloat32 | SampleFormat::IeeeFloat32 => "float32",
        SampleFormat::IeeeFloat64 | SampleFormat::FixedGain => "float64",
        SampleFormat::Int32 | SampleFormat::Int24 => "int32",
        SampleFormat::Uint32 | SampleFormat::Uint24 => "uint32",
        SampleFormat::Int16 => "int16",
        SampleFormat::Uint16 => "uint16",
        SampleFormat::Int64 => "int64",
        SampleFormat::Uint64 => "uint64",
        SampleFormat::Int8 => "int8",
        SampleFormat::Uint8 => "uint8",
    };
    (name.into(), itemsize(name).unwrap_or(1))
}

fn itemsize(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "bool" | "int8" | "uint8" => 1,
        "int16" | "uint16" => 2,
        "int32" | "uint32" | "float32" => 4,
        "int64" | "uint64" | "float64" => 8,
        _ => return None,
    })
}

fn fill_json(dtype: &str) -> Value {
    match dtype {
        "bool" => json!(false),
        "float32" | "float64" => json!(0.0),
        _ => json!(0),
    }
}

fn unravel(mut linear: u64, counts: &[u64]) -> Vec<u64> {
    let mut idx = vec![0; counts.len()];
    for d in (0..counts.len()).rev() {
        idx[d] = linear % counts[d];
        linear /= counts[d];
    }
    idx
}

fn chunk_key(name: &str, idx: &[u64]) -> String {
    let mut key = format!("{name}/c");
    for i in idx {
        key.push('/');
        key.push_str(&i.to_string());
    }
    key
}

fn put_json(sink: &dyn Sink, key: &str, v: &Value) -> Result<()> {
    let text = serde_json::to_vec_pretty(v).map_err(|e| Error::Io(io::Error::other(e)))?;
    sink.put(key, &text)
}

#[allow(clippy::too_many_arguments)]
fn write_meta(
    sink: &dyn Sink,
    name: &str,
    shape: &[u64],
    chunks: &[u64],
    dtype: &str,
    dims: &[&str],
    attributes: Value,
) -> Result<()> {
    let meta = json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": dtype,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunks}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_json(dtype),
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": attributes,
        "dimension_names": dims,
    });
    put_json(sink, &format!("{name}/zarr.json"), &meta)
}

/// Write a small array whose chunks are produced by `chunk(index)`.
#[allow(clippy::too_many_arguments)]
fn write_array(
    sink: &dyn Sink,
    name: &str,
    shape: &[u64],
    chunks: &[u64],
    dtype: &str,
    dims: &[&str],
    attributes: Value,
    chunk: impl Fn(&[u64]) -> Option<Vec<u8>>,
) -> Result<()> {
    write_meta(sink, name, shape, chunks, dtype, dims, attributes)?;
    let counts: Vec<u64> = shape
        .iter()
        .zip(chunks)
        .map(|(n, c)| n.div_ceil(*c))
        .collect();
    for linear in 0..counts.iter().product::<u64>() {
        let idx = unravel(linear, &counts);
        if let Some(bytes) = chunk(&idx) {
            sink.put(&chunk_key(name, &idx), &bytes)?;
        }
    }
    Ok(())
}

fn write_1d_i64(sink: &dyn Sink, name: &str, values: &[i64]) -> Result<()> {
    let n = values.len() as u64;
    write_array(
        sink,
        name,
        &[n],
        &[n.max(1)],
        "int64",
        &[name],
        json!({}),
        |_| Some(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    )
}

/// A per-cell array (`values` in C order over the grid), chunked like `data`.
/// Chunks with no trace are skipped, like the samples.
fn write_grid(
    sink: &dyn Sink,
    name: &str,
    grid: &Grid,
    chunks: &[u64],
    dtype: &str,
    values: &[u8],
    width: usize,
) -> Result<()> {
    let shape = &grid.shape;
    let n_cells: u64 = chunks.iter().product();
    write_array(
        sink,
        name,
        shape,
        chunks,
        dtype,
        &grid.dims,
        json!({}),
        |idx| {
            let mut out = vec![0u8; n_cells as usize * width];
            let mut any = false;
            for local in 0..n_cells {
                let lidx = unravel(local, chunks);
                let mut cell = 0u64;
                let mut inside = true;
                for d in 0..shape.len() {
                    let g = idx[d] * chunks[d] + lidx[d];
                    if g >= shape[d] {
                        inside = false;
                        break;
                    }
                    cell = cell * shape[d] + g;
                }
                if inside && grid.cell_trace[cell as usize].is_some() {
                    any = true;
                    let c = cell as usize * width;
                    let l = local as usize * width;
                    out[l..l + width].copy_from_slice(&values[c..c + width]);
                }
            }
            any.then_some(out)
        },
    )
}

/// A Zarr v3 array from a local directory store, limited to what `export`
/// writes: regular grid, default keys, `bytes` codec.
pub struct ZarrArray {
    dir: PathBuf,
    pub meta: Value,
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub data_type: String,
    pub dimension_names: Vec<String>,
    itemsize: usize,
    fill: Vec<u8>,
}

impl ZarrArray {
    pub fn open(root: impl AsRef<Path>, name: &str) -> Result<Self> {
        let dir = root.as_ref().join(name);
        let text = std::fs::read(dir.join("zarr.json"))?;
        let meta: Value = serde_json::from_slice(&text).map_err(|e| invalid(e.to_string()))?;
        let bad = |what: &str| invalid(format!("{}: {what}", dir.display()));
        if meta["zarr_format"] != 3 || meta["node_type"] != "array" {
            return Err(bad("not a Zarr v3 array"));
        }
        let ints =
            |v: &Value| -> Option<Vec<u64>> { v.as_array()?.iter().map(Value::as_u64).collect() };
        let shape = ints(&meta["shape"]).ok_or_else(|| bad("shape"))?;
        if meta["chunk_grid"]["name"] != "regular" {
            return Err(bad("only the regular chunk grid is read"));
        }
        let chunks = ints(&meta["chunk_grid"]["configuration"]["chunk_shape"])
            .filter(|c| c.len() == shape.len() && !c.contains(&0))
            .ok_or_else(|| bad("chunk_shape"))?;
        let key = &meta["chunk_key_encoding"];
        let sep = key["configuration"]["separator"].as_str().unwrap_or("/");
        if key["name"] != "default" || sep != "/" {
            return Err(bad("only the default chunk key encoding with '/' is read"));
        }
        let codecs = meta["codecs"].as_array().ok_or_else(|| bad("codecs"))?;
        let only_le_bytes = codecs.len() == 1
            && codecs[0]["name"] == "bytes"
            && codecs[0]["configuration"]["endian"]
                .as_str()
                .unwrap_or("little")
                == "little";
        if !only_le_bytes {
            return Err(bad("only the little-endian bytes codec is read"));
        }
        let data_type = meta["data_type"]
            .as_str()
            .ok_or_else(|| bad("data_type"))?
            .to_string();
        let itemsize = itemsize(&data_type).ok_or_else(|| bad("unsupported data_type"))?;
        let fill = fill_bytes(&data_type, &meta["fill_value"]).ok_or_else(|| bad("fill_value"))?;
        let dimension_names = meta["dimension_names"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            dir,
            meta,
            shape,
            chunks,
            data_type,
            dimension_names,
            itemsize,
            fill,
        })
    }

    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    /// One chunk's bytes. A chunk that was never written is the fill (X-34).
    pub fn read_chunk(&self, idx: &[u64]) -> Result<Vec<u8>> {
        let mut path = self.dir.join("c");
        for i in idx {
            path.push(i.to_string());
        }
        let n: u64 = self.chunks.iter().product();
        let want = n as usize * self.itemsize;
        match std::fs::read(&path) {
            Ok(bytes) if bytes.len() == want => Ok(bytes),
            Ok(bytes) => Err(invalid(format!(
                "{} has {} bytes, the chunk shape needs {want}",
                path.display(),
                bytes.len()
            ))),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(self.fill.repeat(n as usize)),
            Err(e) => Err(e.into()),
        }
    }

    /// The whole array, C order, little endian.
    pub fn read_all(&self) -> Result<Vec<u8>> {
        let total: u64 = self.shape.iter().product();
        let mut out = vec![0u8; total as usize * self.itemsize];
        let counts: Vec<u64> = self
            .shape
            .iter()
            .zip(&self.chunks)
            .map(|(n, c)| n.div_ceil(*c))
            .collect();
        let rank = self.shape.len();
        for linear in 0..counts.iter().product::<u64>() {
            let idx = unravel(linear, &counts);
            let chunk = self.read_chunk(&idx)?;
            for local in 0..self.chunks.iter().product::<u64>() {
                let lidx = unravel(local, &self.chunks);
                let mut flat = 0u64;
                let mut inside = true;
                for d in 0..rank {
                    let g = idx[d] * self.chunks[d] + lidx[d];
                    if g >= self.shape[d] {
                        inside = false;
                        break;
                    }
                    flat = flat * self.shape[d] + g;
                }
                if inside {
                    let w = self.itemsize;
                    let (s, l) = (flat as usize * w, local as usize * w);
                    out[s..s + w].copy_from_slice(&chunk[l..l + w]);
                }
            }
        }
        Ok(out)
    }
}

fn fill_bytes(dtype: &str, v: &Value) -> Option<Vec<u8>> {
    Some(match dtype {
        "bool" => vec![u8::from(v.as_bool()?)],
        "int8" => (v.as_i64()? as i8).to_le_bytes().to_vec(),
        "uint8" => vec![v.as_u64()? as u8],
        "int16" => (v.as_i64()? as i16).to_le_bytes().to_vec(),
        "uint16" => (v.as_u64()? as u16).to_le_bytes().to_vec(),
        "int32" => (v.as_i64()? as i32).to_le_bytes().to_vec(),
        "uint32" => (v.as_u64()? as u32).to_le_bytes().to_vec(),
        "int64" => v.as_i64()?.to_le_bytes().to_vec(),
        "uint64" => v.as_u64()?.to_le_bytes().to_vec(),
        "float32" => (float_fill(v)? as f32).to_le_bytes().to_vec(),
        "float64" => float_fill(v)?.to_le_bytes().to_vec(),
        _ => return None,
    })
}

fn float_fill(v: &Value) -> Option<f64> {
    match v.as_str() {
        Some("NaN") => Some(f64::NAN),
        Some("Infinity") => Some(f64::INFINITY),
        Some("-Infinity") => Some(f64::NEG_INFINITY),
        Some(_) => None,
        None => v.as_f64(),
    }
}

fn invalid(msg: String) -> Error {
    Error::Io(io::Error::new(ErrorKind::InvalidInput, msg))
}
