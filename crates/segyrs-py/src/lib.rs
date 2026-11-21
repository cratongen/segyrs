//! PyO3 bindings for the segyrs reader (S-46).
//!
//! Every call that touches the file runs with the interpreter detached, so
//! other Python threads keep going while rayon uses the cores. Arrays are built
//! from the `Vec` the decoder filled: numpy takes ownership, nothing is copied.

mod view;

use std::collections::HashMap;
use std::path::PathBuf;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{
    PyException, PyFileNotFoundError, PyIsADirectoryError, PyMemoryError, PyOSError,
    PyPermissionError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyTuple, PyType};

mod interop;
mod ops;
mod write;

use segyrs::{
    ColumnValues, Error, Gain, HeaderDtype, HeaderField, HeaderSpec, SampleFormat, Samples,
    ScanOptions,
};

create_exception!(
    segyrs._segyrs,
    SegyrsError,
    PyException,
    "Base class of every segyrs error."
);
create_exception!(
    segyrs._segyrs,
    GeometryError,
    SegyrsError,
    "Absurd grid span, or a grid selection where no grid was inferred."
);
create_exception!(
    segyrs._segyrs,
    GeometryAssumptionError,
    GeometryError,
    "A trace header contradicts the regular grid assumed at open. Reopen with scan=\"full\"."
);
create_exception!(
    segyrs._segyrs,
    DuplicateKey,
    SegyrsError,
    "Two traces share one (inline, crossline) slot."
);
create_exception!(
    segyrs._segyrs,
    EndianMismatch,
    SegyrsError,
    "The byte-order signature and the decoded sizes disagree."
);
create_exception!(
    segyrs._segyrs,
    SampleCountMismatch,
    SegyrsError,
    "File size is not the headers plus n_traces * stride."
);
create_exception!(
    segyrs._segyrs,
    NonUniformTraceLength,
    SegyrsError,
    "A per-trace sample count disagrees with the binary header."
);
create_exception!(
    segyrs._segyrs,
    TraceLengthError,
    SegyrsError,
    "One trace's length disagrees with the stride."
);
create_exception!(
    segyrs._segyrs,
    FormatChangeError,
    SegyrsError,
    "In-place format change, or a concat of unlike files."
);
create_exception!(
    segyrs._segyrs,
    Format4NeedsGain,
    SegyrsError,
    "Format 4 opened without a gain convention."
);
create_exception!(
    segyrs._segyrs,
    UnknownFormat,
    SegyrsError,
    "The binary-header format code is not a known sample format."
);
create_exception!(
    segyrs._segyrs,
    TruncatedError,
    SegyrsError,
    "The file is shorter than its headers say."
);
create_exception!(
    segyrs._segyrs,
    UnsortedCoordinate,
    SegyrsError,
    "A drawn axis is not monotonic (raised by the viewer)."
);

create_exception!(
    segyrs._segyrs,
    HeaderValueError,
    SegyrsError,
    "A header value does not fit its field."
);
create_exception!(
    segyrs._segyrs,
    SampleValueError,
    SegyrsError,
    "A sample cannot be stored in the target format without changing its value."
);
create_exception!(
    segyrs._segyrs,
    SpecError,
    SegyrsError,
    "The description of a file to write is inconsistent. Nothing was written."
);
create_exception!(
    segyrs._segyrs,
    ValidationError,
    SegyrsError,
    "The validator found structural problems. `problems` lists them."
);

create_exception!(
    segyrs._segyrs,
    NotSegyError,
    SegyrsError,
    "The bytes are not a SEG-Y file. The message says what they look like."
);

/// File-system errors are both a `SegyrsError` and the built-in `OSError`
/// subclass for their kind, so `except FileNotFoundError` works too.
struct OsErrors {
    io: Py<PyType>,
    not_found: Py<PyType>,
    permission: Py<PyType>,
    directory: Py<PyType>,
    value: Py<PyType>,
    budget: Py<PyType>,
}

static OS_ERRORS: PyOnceLock<OsErrors> = PyOnceLock::new();

fn os_errors(py: Python<'_>) -> PyResult<&OsErrors> {
    OS_ERRORS.get_or_try_init(py, || {
        let new_type = py.import("builtins")?.getattr("type")?;
        let make = |name: &str, bases: Bound<'_, PyTuple>, doc: &str| -> PyResult<Py<PyType>> {
            let ns = PyDict::new(py);
            ns.set_item("__module__", "segyrs")?;
            ns.set_item("__doc__", doc)?;
            Ok(new_type
                .call1((name, bases, ns))?
                .cast_into::<PyType>()?
                .unbind())
        };
        let io = make(
            "SegyrsIOError",
            PyTuple::new(
                py,
                [py.get_type::<SegyrsError>(), py.get_type::<PyOSError>()],
            )?,
            "The file system refused a read or write. Also an OSError.",
        )?;
        let sub = |name: &str, os: Bound<'_, PyType>, doc: &str| {
            make(name, PyTuple::new(py, [io.bind(py).clone(), os])?, doc)
        };
        Ok(OsErrors {
            not_found: sub(
                "PathNotFoundError",
                py.get_type::<PyFileNotFoundError>(),
                "Nothing is at the path. Also a FileNotFoundError.",
            )?,
            permission: sub(
                "PathPermissionError",
                py.get_type::<PyPermissionError>(),
                "This user may not read or write the path. Also a PermissionError.",
            )?,
            directory: sub(
                "PathIsDirectoryError",
                py.get_type::<PyIsADirectoryError>(),
                "The path is a directory, not a file. Also an IsADirectoryError.",
            )?,
            value: make(
                "SegyrsValueError",
                PyTuple::new(
                    py,
                    [py.get_type::<SegyrsError>(), py.get_type::<PyValueError>()],
                )?,
                "An argument segyrs cannot use. Also a ValueError.",
            )?,
            budget: make(
                "BudgetError",
                PyTuple::new(
                    py,
                    [py.get_type::<SegyrsError>(), py.get_type::<PyMemoryError>()],
                )?,
                "The work needs more RAM than its budget allows. Raised before \
                 allocating. Also a MemoryError.",
            )?,
            io,
        })
    })
}

fn os_error(source: &std::io::Error, msg: String, path: Option<&std::path::Path>) -> PyErr {
    Python::attach(|py| {
        let types = match os_errors(py) {
            Ok(t) => t,
            Err(e) => return e,
        };
        let ty = match source.kind() {
            std::io::ErrorKind::NotFound => &types.not_found,
            std::io::ErrorKind::PermissionDenied => &types.permission,
            std::io::ErrorKind::IsADirectory => &types.directory,
            _ => &types.io,
        };
        let errno = source.raw_os_error().unwrap_or(0);
        let ty = ty.bind(py).clone();
        match path {
            Some(p) => {
                let shown = format!(" {}", p.display());
                let msg = msg.replacen(&shown, "", 1);
                PyErr::from_type(ty, (errno, msg, p.to_string_lossy().into_owned()))
            }
            None => PyErr::from_type(ty, (errno, msg)),
        }
    })
}

fn plain_error(pick: impl Fn(&OsErrors) -> &Py<PyType>, msg: String) -> PyErr {
    Python::attach(|py| match os_errors(py) {
        Ok(types) => PyErr::from_type(pick(types).bind(py).clone(), (msg,)),
        Err(e) => e,
    })
}

fn to_py(err: Error) -> PyErr {
    let msg = err.to_string();
    match err {
        Error::Geometry { .. } => GeometryError::new_err(msg),
        Error::GeometryAssumption { .. } => GeometryAssumptionError::new_err(msg),
        Error::DuplicateKey { .. } => DuplicateKey::new_err(msg),
        Error::EndianMismatch { .. } => EndianMismatch::new_err(msg),
        Error::SampleCountMismatch { .. } => SampleCountMismatch::new_err(msg),
        Error::NonUniformTraceLength { .. } => NonUniformTraceLength::new_err(msg),
        Error::TraceLength { .. } => TraceLengthError::new_err(msg),
        Error::FormatChange => FormatChangeError::new_err(msg),
        Error::Format4NeedsGain => Format4NeedsGain::new_err(msg),
        Error::UnknownFormat(_) => UnknownFormat::new_err(msg),
        Error::Truncated { .. } => TruncatedError::new_err(msg),
        Error::NotSegy(_) => NotSegyError::new_err(msg),
        Error::Path {
            ref path,
            ref source,
            ..
        } => os_error(source, msg, Some(path)),
        Error::Io(ref source) if source.raw_os_error().is_none() => match source.kind() {
            std::io::ErrorKind::InvalidInput => plain_error(|t| &t.value, source.to_string()),
            std::io::ErrorKind::OutOfMemory => plain_error(|t| &t.budget, source.to_string()),
            _ => os_error(source, msg, None),
        },
        Error::Io(ref source) => os_error(source, msg, None),
        Error::HeaderValue { .. } => HeaderValueError::new_err(msg),
        Error::SampleValue { .. } => SampleValueError::new_err(msg),
        Error::Spec(_) => SpecError::new_err(msg),
        Error::Invalid { problems } => Python::attach(|py| {
            let e = ValidationError::new_err(msg);
            let _ = e.value(py).setattr("problems", problems);
            e
        }),
        _ => SegyrsError::new_err(msg),
    }
}

fn samples_2d(py: Python<'_>, samples: Samples, rows: usize, cols: usize) -> PyResult<Py<PyAny>> {
    macro_rules! arr {
        ($v:expr) => {
            Array2::from_shape_vec((rows, cols), $v)
                .map_err(|e| SegyrsError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
                .unbind()
        };
    }
    Ok(match samples {
        Samples::I8(v) => arr!(v),
        Samples::U8(v) => arr!(v),
        Samples::I16(v) => arr!(v),
        Samples::U16(v) => arr!(v),
        Samples::I32(v) => arr!(v),
        Samples::U32(v) => arr!(v),
        Samples::I64(v) => arr!(v),
        Samples::U64(v) => arr!(v),
        Samples::F32(v) => arr!(v),
        Samples::F64(v) => arr!(v),
    })
}

fn column(py: Python<'_>, values: ColumnValues) -> Py<PyAny> {
    match values {
        ColumnValues::I8(v) => v.into_pyarray(py).into_any().unbind(),
        ColumnValues::I16(v) => v.into_pyarray(py).into_any().unbind(),
        ColumnValues::I32(v) => v.into_pyarray(py).into_any().unbind(),
        ColumnValues::I64(v) => v.into_pyarray(py).into_any().unbind(),
        ColumnValues::U16(v) => v.into_pyarray(py).into_any().unbind(),
        ColumnValues::U32(v) => v.into_pyarray(py).into_any().unbind(),
        ColumnValues::F64(v) => v.into_pyarray(py).into_any().unbind(),
    }
}

fn header_dtype(name: &str) -> PyResult<HeaderDtype> {
    Ok(match name {
        "int8" | "i1" => HeaderDtype::I8,
        "int16" | "i2" => HeaderDtype::I16,
        "int32" | "i4" => HeaderDtype::I32,
        "int64" | "i8" => HeaderDtype::I64,
        "uint16" | "u2" => HeaderDtype::U16,
        "uint32" | "u4" => HeaderDtype::U32,
        "float64" | "f8" => HeaderDtype::F64,
        _ => {
            return Err(PyValueError::new_err(format!(
                "header dtype {name:?} is not one of int8, int16, int32, int64, uint16, uint32, float64"
            )))
        }
    })
}

fn dtype_name(dtype: HeaderDtype) -> &'static str {
    match dtype {
        HeaderDtype::I8 => "int8",
        HeaderDtype::I16 => "int16",
        HeaderDtype::I32 => "int32",
        HeaderDtype::I64 => "int64",
        HeaderDtype::U16 => "uint16",
        HeaderDtype::U32 => "uint32",
        HeaderDtype::F64 => "float64",
    }
}

fn numpy_dtype(format: SampleFormat) -> &'static str {
    match format {
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
    }
}

/// An open SEG-Y file. The Python `Volume` is the public surface; this is its engine.
#[pyclass(frozen, module = "segyrs._segyrs")]
struct File {
    inner: segyrs::File<interop::Source>,
}

#[pymethods]
impl File {
    #[new]
    #[pyo3(signature = (
        path, *, storage_options=None, sample_format=None, endian=None, gain=None,
        inline_byte=189, crossline_byte=193, offset_byte=37, scan="auto", index_cache=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        storage_options: Option<HashMap<String, String>>,
        sample_format: Option<&str>,
        endian: Option<&str>,
        gain: Option<&str>,
        inline_byte: u16,
        crossline_byte: u16,
        offset_byte: u16,
        scan: &str,
        index_cache: Option<PathBuf>,
    ) -> PyResult<Self> {
        let (options, scan) = segyrs::bridge::open_options(
            sample_format,
            endian,
            gain,
            inline_byte,
            crossline_byte,
            offset_byte,
            scan,
        )
        .map_err(PyValueError::new_err)?;
        let scan = ScanOptions { scan, index_cache };
        let inner = interop::open(py, &path, storage_options, &options, &scan)?;
        Ok(Self { inner })
    }

    /// This file's reader as a capsule for the viewer extension.
    fn reader<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        interop::export_reader(py, self)
    }

    #[getter]
    fn n_traces(&self) -> u64 {
        self.inner.layout().n_traces
    }

    #[getter]
    fn n_samples(&self) -> u32 {
        self.inner.layout().n_samples
    }

    #[getter]
    fn sample_interval_us(&self) -> u32 {
        self.inner.layout().sample_interval_us
    }

    #[getter]
    fn sample_format(&self) -> &'static str {
        self.inner.format().name()
    }

    #[getter]
    fn binary_format_code(&self) -> u16 {
        self.inner.layout().binary_format_code
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        numpy_dtype(self.inner.format())
    }

    #[getter]
    fn source_endian(&self) -> &'static str {
        self.inner.endian().as_str()
    }

    #[getter]
    fn revision(&self) -> &'static str {
        self.inner.layout().revision.as_str()
    }

    #[getter]
    fn measurement_system(&self) -> u16 {
        self.inner.layout().measurement_system
    }

    #[getter]
    fn gain(&self) -> Option<&'static str> {
        self.inner.gain().map(Gain::as_str)
    }

    #[getter]
    fn data_start(&self) -> u64 {
        self.inner.layout().data_start
    }

    #[getter]
    fn extra_trace_header_blocks(&self) -> u16 {
        self.inner.layout().extra_trace_header_blocks
    }

    /// How many traces have had sample bytes read. Open leaves it at zero.
    #[getter]
    fn samples_read(&self) -> u64 {
        self.inner.samples_read()
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path().to_path_buf()
    }

    /// `[(encoding, cards, raw_bytes), ...]`, one entry per 3200-byte block.
    fn textual_header<'py>(
        &self,
        py: Python<'py>,
    ) -> Vec<(&'static str, Vec<String>, Bound<'py, PyBytes>)> {
        self.inner
            .textual_header()
            .iter()
            .map(|b| {
                (
                    b.encoding.as_str(),
                    b.cards.clone(),
                    PyBytes::new(py, &b.raw),
                )
            })
            .collect()
    }

    fn binary_header_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.binary_header_bytes())
    }

    /// On-disk standard 240-byte trace header, file byte order (S-23).
    fn header_bytes<'py>(&self, py: Python<'py>, trace: u64) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| self.inner.header_bytes(trace))
            .map_err(to_py)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn geometry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = self.inner.geometry();
        let d = PyDict::new(py);
        d.set_item("kind", g.kind.as_str())?;
        d.set_item("inline_byte", g.inline_byte)?;
        d.set_item("crossline_byte", g.crossline_byte)?;
        d.set_item("offset_byte", g.offset_byte)?;
        d.set_item("inlines", g.inlines.clone().into_pyarray(py))?;
        d.set_item("crosslines", g.crosslines.clone().into_pyarray(py))?;
        d.set_item("offsets", g.offsets.clone().into_pyarray(py))?;
        let axes: Vec<&str> = g.axes.iter().map(|a| a.as_str()).collect();
        d.set_item("axes", axes)?;
        let flat: Vec<i32> = g.missing_pairs.iter().flat_map(|&(i, x)| [i, x]).collect();
        let pairs = Array2::from_shape_vec((g.missing_pairs.len(), 2), flat)
            .map_err(|e| SegyrsError::new_err(e.to_string()))?;
        d.set_item("missing_pairs", pairs.into_pyarray(py))?;
        d.set_item("source", self.inner.geometry_source().as_str())?;
        Ok(d)
    }

    /// The header words open scanned, as numpy arrays. After an assumed-grid
    /// open this is the full header scan, which also checks the grid.
    fn scanned_columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let f = &self.inner;
        py.detach(|| f.verify_geometry()).map_err(to_py)?;
        let d = PyDict::new(py);
        d.set_item("inline", f.inline().to_vec().into_pyarray(py))?;
        d.set_item("crossline", f.crossline().to_vec().into_pyarray(py))?;
        d.set_item("offset", f.offset().to_vec().into_pyarray(py))?;
        d.set_item("scalco", f.scalco().to_vec().into_pyarray(py))?;
        d.set_item("cdp_x", f.cdp_x().to_vec().into_pyarray(py))?;
        d.set_item("cdp_y", f.cdp_y().to_vec().into_pyarray(py))?;
        d.set_item("trace_id", f.trace_id().to_vec().into_pyarray(py))?;
        Ok(d)
    }

    /// The standard field table as `[(name, byte, dtype), ...]`.
    #[staticmethod]
    fn standard_fields() -> Vec<(String, u16, &'static str)> {
        HeaderSpec::standard()
            .fields
            .into_iter()
            .map(|f| (f.name.unwrap_or_default(), f.byte, dtype_name(f.dtype)))
            .collect()
    }

    /// One parallel columnar scan of the listed `(name, byte, dtype)` fields.
    fn header_columns<'py>(
        &self,
        py: Python<'py>,
        fields: Vec<(String, u16, String)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fields = fields
            .into_iter()
            .map(|(name, byte, dtype)| Ok(HeaderField::new(byte, header_dtype(&dtype)?, name)))
            .collect::<PyResult<Vec<_>>>()?;
        let spec = HeaderSpec { fields };
        let cols = py
            .detach(|| self.inner.par_header_columns(&spec))
            .map_err(to_py)?;
        let d = PyDict::new(py);
        for col in cols.into_columns() {
            d.set_item(col.name, column(py, col.values))?;
        }
        Ok(d)
    }

    /// Samples `start..stop` of each trace in `traces`, shape `(len(traces), stop - start)`.
    fn read(
        &self,
        py: Python<'_>,
        traces: PyReadonlyArray1<'_, i64>,
        start: usize,
        stop: usize,
    ) -> PyResult<Py<PyAny>> {
        let idx = traces.as_slice()?;
        let n = self.inner.layout().n_traces;
        let mut list = Vec::with_capacity(idx.len());
        for &t in idx {
            if t < 0 || t as u64 >= n {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "trace {t} is outside 0..{n}"
                )));
            }
            list.push(t as u64);
        }
        let rows = list.len();
        let samples = py
            .detach(|| self.inner.read_window(&list, start, stop))
            .map_err(to_py)?;
        samples_2d(py, samples, rows, stop.saturating_sub(start))
    }

    /// File trace numbers carrying `(inline, crossline)`, in file order (S-15).
    fn lookup(&self, py: Python<'_>, inline: i32, crossline: i32) -> Py<PyAny> {
        let hits: Vec<i64> = py.detach(|| {
            self.inner
                .trace_index()
                .get(inline, crossline)
                .map(|s| s.iter().map(|&t| i64::from(t)).collect())
                .unwrap_or_default()
        });
        hits.into_pyarray(py).into_any().unbind()
    }
}

#[pymodule]
fn _segyrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<File>()?;
    view::register(m)?;
    interop::register(m)?;
    m.add("SegyrsError", py.get_type::<SegyrsError>())?;
    m.add("GeometryError", py.get_type::<GeometryError>())?;
    m.add(
        "GeometryAssumptionError",
        py.get_type::<GeometryAssumptionError>(),
    )?;
    m.add("DuplicateKey", py.get_type::<DuplicateKey>())?;
    m.add("EndianMismatch", py.get_type::<EndianMismatch>())?;
    m.add("SampleCountMismatch", py.get_type::<SampleCountMismatch>())?;
    m.add(
        "NonUniformTraceLength",
        py.get_type::<NonUniformTraceLength>(),
    )?;
    m.add("TraceLengthError", py.get_type::<TraceLengthError>())?;
    m.add("FormatChangeError", py.get_type::<FormatChangeError>())?;
    m.add("Format4NeedsGain", py.get_type::<Format4NeedsGain>())?;
    m.add("UnknownFormat", py.get_type::<UnknownFormat>())?;
    m.add("TruncatedError", py.get_type::<TruncatedError>())?;
    m.add("NotSegyError", py.get_type::<NotSegyError>())?;
    let os = os_errors(py)?;
    m.add("SegyrsIOError", os.io.bind(py))?;
    m.add("PathNotFoundError", os.not_found.bind(py))?;
    m.add("PathPermissionError", os.permission.bind(py))?;
    m.add("PathIsDirectoryError", os.directory.bind(py))?;
    m.add("SegyrsValueError", os.value.bind(py))?;
    m.add("BudgetError", os.budget.bind(py))?;
    m.add("UnsortedCoordinate", py.get_type::<UnsortedCoordinate>())?;
    m.add("HeaderValueError", py.get_type::<HeaderValueError>())?;
    m.add("SampleValueError", py.get_type::<SampleValueError>())?;
    m.add("SpecError", py.get_type::<SpecError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    write::register(m)?;
    ops::register(m)?;
    Ok(())
}
