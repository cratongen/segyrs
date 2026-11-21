//! M5 bindings: object-store reads and Zarr v3 export (docs/cloud-and-zarr.md),
//! and the reader bridge to the viewer extension (docs/viewer.md).
//!
//! Every Python `File` holds a `segyrs::File<Arc<dyn ReadAt>>`. A local path is
//! mapped; a URL goes through `segyrs_interop::cloud::ObjectReader`. All network
//! and disk work runs with the interpreter detached. `File.reader()` exports
//! the file's reader as a capsule (`segyrs::bridge`), so the `segyrs-view`
//! extension reads the same map or the same remote cache.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use segyrs::bridge::{RawReader, CAPSULE_NAME};
use segyrs::{OpenOptions, ReadAt, ScanOptions};
use segyrs_interop::cloud::{store_for, ObjectReader};
use segyrs_interop::zarr::{export, DirSink, ObjectSink, Sink, ZarrOptions};

use crate::{to_py, File};

/// The byte source behind every Python `File`: a mapped file or an object-store reader.
pub(crate) type Source = Arc<dyn ReadAt>;

/// `Some(scheme)` for `scheme://...`. A one-letter scheme is a Windows drive.
fn url_scheme(location: &str) -> Option<&str> {
    let (scheme, _) = location.split_once("://")?;
    let valid = scheme.len() > 1
        && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then_some(scheme)
}

fn local_only(storage_options: &Option<HashMap<String, String>>, what: &str) -> PyResult<()> {
    match storage_options {
        Some(opts) if !opts.is_empty() => Err(PyValueError::new_err(format!(
            "storage_options apply to URLs only; {what} is a local path"
        ))),
        _ => Ok(()),
    }
}

/// Open a local path (mapped) or a URL (object store) for `File::new`.
pub(crate) fn open(
    py: Python<'_>,
    path: &Path,
    storage_options: Option<HashMap<String, String>>,
    options: &OpenOptions,
    scan: &ScanOptions,
) -> PyResult<segyrs::File<Source>> {
    let url = path.to_str().filter(|p| url_scheme(p).is_some());
    match url {
        Some(url) => {
            if scan.index_cache.is_some() {
                return Err(PyValueError::new_err(
                    "index_cache applies to local paths only",
                ));
            }
            let opts = storage_options.unwrap_or_default();
            py.detach(|| {
                let reader = ObjectReader::open(url, &opts)?;
                let len = reader.len();
                segyrs::File::from_reader_scan(Arc::new(reader) as Source, len, options, scan.scan)
                    .map(|f| f.with_path(url))
            })
            .map_err(to_py)
        }
        None => {
            local_only(&storage_options, "path")?;
            py.detach(|| segyrs::File::open_scan(path, options, scan))
                .map(|f| f.map_reader(|s| Arc::new(s) as Source))
                .map_err(to_py)
        }
    }
}

/// The file's reader as a capsule for the viewer extension.
pub(crate) fn export_reader<'py>(py: Python<'py>, file: &File) -> PyResult<Bound<'py, PyCapsule>> {
    let inner = &file.inner;
    let len = match inner.reader().as_bytes() {
        Some(bytes) => bytes.len() as u64,
        None => inner.layout().data_start + inner.layout().n_traces * inner.stride(),
    };
    let raw = RawReader::export(inner.reader().clone(), len);
    PyCapsule::new_with_value(py, raw, CAPSULE_NAME)
}

/// Fetch the byte ranges of `traces` ahead of a read: coalesced, parallel range
/// requests for a remote file. A local map already holds every byte, so this
/// returns at once.
#[pyfunction]
fn prefetch(py: Python<'_>, file: &File, traces: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
    let inner = &file.inner;
    if inner.reader().as_bytes().is_some() {
        return Ok(());
    }
    let n = inner.layout().n_traces;
    let start = inner.layout().data_start;
    let stride = inner.stride();
    let ranges = traces
        .as_array()
        .iter()
        .map(|&t| {
            u64::try_from(t)
                .ok()
                .filter(|&t| t < n)
                .map(|t| start + t * stride..start + (t + 1) * stride)
                .ok_or_else(|| PyValueError::new_err(format!("trace {t} is out of range 0..{n}")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| inner.reader().prefetch(&ranges))
        .map_err(to_py)
}

/// Write the whole file as a Zarr v3 group at `dest`, a directory or a URL.
/// Returns the export summary as a dict.
#[pyfunction]
#[pyo3(signature = (file, dest, *, chunks=None, target_chunk_bytes=None, depth=false, storage_options=None))]
fn zarr_export<'py>(
    py: Python<'py>,
    file: &File,
    dest: &str,
    chunks: Option<Vec<u64>>,
    target_chunk_bytes: Option<u64>,
    depth: bool,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut opts = ZarrOptions {
        chunks,
        depth,
        ..ZarrOptions::default()
    };
    if let Some(bytes) = target_chunk_bytes {
        opts.target_chunk_bytes = bytes;
    }
    let sink: Box<dyn Sink> = if url_scheme(dest).is_some() {
        let opts = storage_options.unwrap_or_default();
        let (store, prefix) = py.detach(|| store_for(dest, &opts)).map_err(to_py)?;
        Box::new(ObjectSink::new(store, prefix))
    } else {
        local_only(&storage_options, "dest")?;
        Box::new(DirSink::new(dest))
    };
    let inner = &file.inner;
    let done = py
        .detach(|| export(inner, sink.as_ref(), &opts))
        .map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("dims", done.dims)?;
    out.set_item("shape", done.shape)?;
    out.set_item("chunks", done.chunks)?;
    out.set_item("data_type", done.data_type)?;
    out.set_item("chunks_written", done.chunks_written)?;
    out.set_item("chunks_missing", done.chunks_missing)?;
    Ok(out)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prefetch, m)?)?;
    m.add_function(wrap_pyfunction!(zarr_export, m)?)?;
    Ok(())
}
