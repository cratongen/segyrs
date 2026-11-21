//! Bindings for the M2 write path: `create`, copies of a selection, `validate`,
//! and the in-place `Editor`.
//!
//! Every call that writes or reads runs with the interpreter detached. Numpy
//! arrays are borrowed as slices, never copied, for the length of the call.

use std::path::PathBuf;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use segyrs::write::{self as w, HeaderValues, NewFile, Overlay, SampleData, Scalars, WriteOptions};
use segyrs::{Endian, HeaderField, SampleFormat, TextEncoding};

use crate::{header_dtype, to_py, File};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Editor>()?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(write_from, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    Ok(())
}

pub(crate) fn parse_format(s: &str) -> PyResult<SampleFormat> {
    SampleFormat::parse(s)
        .ok_or_else(|| PyValueError::new_err(format!("unknown sample_format {s:?}")))
}

pub(crate) fn parse_endian(s: &str) -> PyResult<Endian> {
    Endian::parse(s).ok_or_else(|| {
        PyValueError::new_err(format!(
            "endian {s:?} is not \"big\", \"little\", or \"pairwise\""
        ))
    })
}

/// Trace numbers from an int64 array, checked against `n`.
pub(crate) fn trace_list(traces: &PyReadonlyArray1<'_, i64>, n: u64) -> PyResult<Vec<u64>> {
    traces
        .as_slice()?
        .iter()
        .map(|&t| {
            if t < 0 || t as u64 >= n {
                Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "trace {t} is outside 0..{n}"
                )))
            } else {
                Ok(t as u64)
            }
        })
        .collect()
}

/// One header field to write, as Python hands it over:
/// `(byte, dtype, name, values, scalars)`. `values` is int64 or float64,
/// `scalars` (int16) makes the float values coordinates in real units.
type OverlayArg<'py> = (
    u16,
    String,
    String,
    Bound<'py, PyAny>,
    Option<PyReadonlyArray1<'py, i16>>,
);

enum Vals<'py> {
    I64(PyReadonlyArray1<'py, i64>),
    F64(PyReadonlyArray1<'py, f64>),
    Scaled(PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, i16>),
}

/// An overlay whose arrays stay borrowed while the write runs.
pub(crate) struct HeldOverlay<'py> {
    field: HeaderField,
    vals: Vals<'py>,
}

impl HeldOverlay<'_> {
    fn overlay(&self) -> PyResult<Overlay<'_>> {
        let values = match &self.vals {
            Vals::I64(a) => HeaderValues::I64(a.as_slice()?),
            Vals::F64(a) => HeaderValues::F64(a.as_slice()?),
            Vals::Scaled(v, s) => HeaderValues::Scaled {
                values: v.as_slice()?,
                scalars: Scalars::PerTrace(s.as_slice()?),
            },
        };
        Ok(Overlay {
            field: self.field.clone(),
            values,
        })
    }
}

pub(crate) fn hold_overlays<'py>(args: Vec<OverlayArg<'py>>) -> PyResult<Vec<HeldOverlay<'py>>> {
    args.into_iter()
        .map(|(byte, dtype, name, values, scalars)| {
            let field = HeaderField::new(byte, header_dtype(&dtype)?, name);
            let vals = match scalars {
                Some(s) => Vals::Scaled(values.extract()?, s),
                None => match values.extract::<PyReadonlyArray1<'py, i64>>() {
                    Ok(a) => Vals::I64(a),
                    Err(_) => Vals::F64(values.extract()?),
                },
            };
            Ok(HeldOverlay { field, vals })
        })
        .collect()
}

fn overlays<'a>(held: &'a [HeldOverlay<'_>]) -> PyResult<Vec<Overlay<'a>>> {
    held.iter().map(HeldOverlay::overlay).collect()
}

/// A 2D C-contiguous sample array of any numeric dtype, borrowed.
pub(crate) enum Held2<'py> {
    I8(PyReadonlyArray2<'py, i8>),
    U8(PyReadonlyArray2<'py, u8>),
    I16(PyReadonlyArray2<'py, i16>),
    U16(PyReadonlyArray2<'py, u16>),
    I32(PyReadonlyArray2<'py, i32>),
    U32(PyReadonlyArray2<'py, u32>),
    I64(PyReadonlyArray2<'py, i64>),
    U64(PyReadonlyArray2<'py, u64>),
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

impl<'py> Held2<'py> {
    pub(crate) fn new(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        macro_rules! try_as {
            ($($v:ident: $t:ty),*) => {
                $(if let Ok(a) = obj.extract::<PyReadonlyArray2<'py, $t>>() {
                    return Ok(Self::$v(a));
                })*
            };
        }
        try_as!(F32: f32, F64: f64, I16: i16, I32: i32, I8: i8, U8: u8, U16: u16, U32: u32, I64: i64, U64: u64);
        Err(PyValueError::new_err(
            "samples must be a 2D numpy array of an integer or float dtype",
        ))
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        let s = match self {
            Self::I8(a) => a.shape(),
            Self::U8(a) => a.shape(),
            Self::I16(a) => a.shape(),
            Self::U16(a) => a.shape(),
            Self::I32(a) => a.shape(),
            Self::U32(a) => a.shape(),
            Self::I64(a) => a.shape(),
            Self::U64(a) => a.shape(),
            Self::F32(a) => a.shape(),
            Self::F64(a) => a.shape(),
        };
        (s[0], s[1])
    }

    pub(crate) fn data(&self) -> PyResult<SampleData<'_>> {
        Ok(match self {
            Self::I8(a) => SampleData::I8(a.as_slice()?),
            Self::U8(a) => SampleData::U8(a.as_slice()?),
            Self::I16(a) => SampleData::I16(a.as_slice()?),
            Self::U16(a) => SampleData::U16(a.as_slice()?),
            Self::I32(a) => SampleData::I32(a.as_slice()?),
            Self::U32(a) => SampleData::U32(a.as_slice()?),
            Self::I64(a) => SampleData::I64(a.as_slice()?),
            Self::U64(a) => SampleData::U64(a.as_slice()?),
            Self::F32(a) => SampleData::F32(a.as_slice()?),
            Self::F64(a) => SampleData::F64(a.as_slice()?),
        })
    }
}

/// Write a new file from a `(traces, samples)` array and header overlays.
#[pyfunction]
#[pyo3(signature = (
    path, samples, sample_interval_us, sample_format, *,
    endian=None, overlays=Vec::new(), text=None, sync=true,
))]
#[allow(clippy::too_many_arguments)]
fn create(
    py: Python<'_>,
    path: PathBuf,
    samples: Bound<'_, PyAny>,
    sample_interval_us: u32,
    sample_format: &str,
    endian: Option<&str>,
    overlays: Vec<OverlayArg<'_>>,
    text: Option<Vec<String>>,
    sync: bool,
) -> PyResult<u64> {
    let format = parse_format(sample_format)?;
    let endian = endian.map(parse_endian).transpose()?;
    let held = Held2::new(&samples)?;
    let (rows, cols) = held.shape();
    let ns = u32::try_from(cols)
        .map_err(|_| PyValueError::new_err(format!("{cols} samples per trace is too many")))?;
    let held_overlays = hold_overlays(overlays)?;
    let mut spec = NewFile::new(rows as u64, ns, sample_interval_us);
    if let Some(cards) = text {
        let cards: Vec<&str> = cards.iter().map(String::as_str).collect();
        spec.text = w::text_block(&cards, TextEncoding::Ebcdic).map_err(to_py)?;
    }
    spec.samples = Some(held.data()?);
    spec.overlays = self::overlays(&held_overlays)?;
    let options = WriteOptions {
        sample_format: Some(format),
        endian,
        sync,
        ..WriteOptions::default()
    };
    py.detach(|| w::create(&path, &spec, &options))
        .map_err(to_py)
}

/// Copy traces `traces` (all when `None`), samples `first_sample..first_sample + n_samples`,
/// of `file` to `path`, optionally in another sample format or byte order.
#[pyfunction]
#[pyo3(signature = (
    file, path, *, traces=None, first_sample=0, n_samples=None,
    sample_format=None, endian=None, overlays=Vec::new(), sync=true,
))]
#[allow(clippy::too_many_arguments)]
fn write_from(
    py: Python<'_>,
    file: PyRef<'_, File>,
    path: PathBuf,
    traces: Option<PyReadonlyArray1<'_, i64>>,
    first_sample: u32,
    n_samples: Option<u32>,
    sample_format: Option<&str>,
    endian: Option<&str>,
    overlays: Vec<OverlayArg<'_>>,
    sync: bool,
) -> PyResult<u64> {
    let src = &file.inner;
    let layout = src.layout();
    let list = traces
        .as_ref()
        .map(|t| trace_list(t, layout.n_traces))
        .transpose()?;
    let held_overlays = hold_overlays(overlays)?;
    let mut spec = NewFile::from_file(src).map_err(to_py)?;
    if let Some(list) = &list {
        spec.n_traces = list.len() as u64;
    }
    spec.n_samples = n_samples.unwrap_or(layout.n_samples - first_sample.min(layout.n_samples));
    if let Some(s) = spec.source.as_mut() {
        s.traces = list.as_deref();
        s.first_sample = first_sample;
    }
    spec.overlays = self::overlays(&held_overlays)?;
    let options = WriteOptions {
        sample_format: sample_format.map(parse_format).transpose()?,
        endian: endian.map(parse_endian).transpose()?,
        sync,
        ..WriteOptions::default()
    };
    py.detach(|| w::create(&path, &spec, &options))
        .map_err(to_py)
}

/// Check the file's structure. Returns its layout, or raises `ValidationError`.
#[pyfunction]
fn validate<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let layout = py.detach(|| w::validate(&path)).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("revision", layout.revision.as_str())?;
    d.set_item("sample_format", layout.sample_format.name())?;
    d.set_item("binary_format_code", layout.binary_format_code)?;
    d.set_item("endian", layout.source_endian.as_str())?;
    d.set_item("n_traces", layout.n_traces)?;
    d.set_item("n_samples", layout.n_samples)?;
    d.set_item("sample_interval_us", layout.sample_interval_us)?;
    d.set_item("extended_textual_blocks", layout.extended_textual_blocks)?;
    d.set_item(
        "extra_trace_header_blocks",
        layout.extra_trace_header_blocks,
    )?;
    d.set_item("data_start", layout.data_start)?;
    Ok(d)
}

/// In-place edits of one file. The Python `segyrs.edit` wraps it.
#[pyclass(module = "segyrs._segyrs")]
pub(crate) struct Editor {
    inner: Option<w::Editor>,
}

impl Editor {
    fn get(&mut self) -> PyResult<&mut w::Editor> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("the editor is closed"))
    }
}

#[pymethods]
impl Editor {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| w::Editor::open(&path)).map_err(to_py)?;
        Ok(Self { inner: Some(inner) })
    }

    #[getter]
    fn n_traces(&mut self) -> PyResult<u64> {
        Ok(self.get()?.layout().n_traces)
    }

    #[getter]
    fn n_samples(&mut self) -> PyResult<u32> {
        Ok(self.get()?.layout().n_samples)
    }

    #[getter]
    fn sample_format(&mut self) -> PyResult<&'static str> {
        Ok(self.get()?.layout().sample_format.name())
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_none()
    }

    /// Header fields on every trace, or on `traces` (one value per listed trace).
    #[pyo3(signature = (overlays, traces=None))]
    fn write_header_columns(
        &mut self,
        py: Python<'_>,
        overlays: Vec<OverlayArg<'_>>,
        traces: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<()> {
        let ed = self.get()?;
        let list = traces
            .as_ref()
            .map(|t| trace_list(t, ed.layout().n_traces))
            .transpose()?;
        let held = hold_overlays(overlays)?;
        let ov = self::overlays(&held)?;
        let ed = &*ed;
        py.detach(|| ed.write_header_columns(&ov, list.as_deref()))
            .map_err(to_py)
    }

    /// Samples of traces `first .. first + len(samples)`, encoded in the file's format.
    fn write_traces(
        &mut self,
        py: Python<'_>,
        first: u64,
        samples: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let ed = self.get()?;
        let held = Held2::new(&samples)?;
        let (_, cols) = held.shape();
        let ns = ed.layout().n_samples as usize;
        if cols != ns {
            return Err(crate::TraceLengthError::new_err(format!(
                "samples have {cols} columns, the file has {ns} samples per trace"
            )));
        }
        let data = held.data()?;
        let ed = &*ed;
        py.detach(|| ed.write_traces(first, data)).map_err(to_py)
    }

    fn write_binary_header(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        let ed = self.get()?;
        py.detach(|| ed.write_binary_header(data)).map_err(to_py)
    }

    fn write_textual_block(&mut self, py: Python<'_>, index: usize, data: &[u8]) -> PyResult<()> {
        let ed = self.get()?;
        py.detach(|| ed.write_textual_block(index, data))
            .map_err(to_py)
    }

    fn sync(&mut self, py: Python<'_>) -> PyResult<()> {
        let ed = self.get()?;
        py.detach(|| ed.sync()).map_err(to_py)
    }

    /// Release the file. A closed editor refuses further calls.
    fn close(&mut self) {
        self.inner = None;
    }
}
