//! Bindings for the M3 whole-volume operations.
//!
//! Operations that produce a file stream through `segyrs::write::FileSink`,
//! the M2 writer behind the `TraceSink` contract: chunks sized from the RAM
//! budget go straight to a temp file that is renamed into place at the end.
//! Nothing holds the whole output in memory.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use segyrs::ops::{self, Budget, Crop, Predicate, ResortOptions, SortKey, SortStrategy};
use segyrs::write::FileSink;

use crate::write::trace_list;
use crate::{samples_2d, to_py, File};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crop, m)?)?;
    m.add_function(wrap_pyfunction!(filter, m)?)?;
    m.add_function(wrap_pyfunction!(resort, m)?)?;
    m.add_function(wrap_pyfunction!(slice_horizon, m)?)?;
    m.add_function(wrap_pyfunction!(to_regular, m)?)?;
    m.add_function(wrap_pyfunction!(to_regular_file, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;
    m.add_function(wrap_pyfunction!(stats, m)?)?;
    m.add_function(wrap_pyfunction!(rms, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    m.add_function(wrap_pyfunction!(check_budget, m)?)?;
    Ok(())
}

/// Raise `BudgetError` when `need` bytes do not fit `budget`. Nothing is allocated.
///
/// `budget` is `None` (half the available RAM), an int of bytes, or a float in (0, 1].
#[pyfunction]
#[pyo3(signature = (need, budget=None))]
fn check_budget(need: u64, budget: Option<Bound<'_, PyAny>>) -> PyResult<()> {
    self::budget(budget.as_ref())?
        .check(u128::from(need), "the selection")
        .map_err(to_py)
}

/// `None`: half the available RAM. An int: that many bytes. A float in (0, 1]:
/// that fraction of the available RAM.
fn budget(value: Option<&Bound<'_, PyAny>>) -> PyResult<Budget> {
    let Some(v) = value else {
        return Budget::from_available().map_err(to_py);
    };
    if let Ok(bytes) = v.extract::<u64>() {
        if bytes == 0 {
            return Err(PyValueError::new_err("budget must be positive"));
        }
        return Ok(Budget::bytes(bytes));
    }
    let f: f64 = v
        .extract()
        .map_err(|_| PyValueError::new_err("budget is None, an int of bytes, or a fraction"))?;
    if !(f > 0.0 && f <= 1.0) {
        return Err(PyValueError::new_err(format!(
            "budget fraction {f} is not in (0, 1]"
        )));
    }
    Budget::fraction_of_available(f).map_err(to_py)
}

fn traces_of(file: &File, traces: Option<&PyReadonlyArray1<'_, i64>>) -> PyResult<Vec<u64>> {
    let n = file.inner.layout().n_traces;
    match traces {
        Some(t) => trace_list(t, n),
        None => Ok((0..n).collect()),
    }
}

/// Keep `traces` and samples `start..stop` of `file`, written to `path`.
#[pyfunction]
#[pyo3(signature = (file, path, traces, start=None, stop=None, *, rezero_time=false, budget=None, sync=true))]
#[allow(clippy::too_many_arguments)]
fn crop(
    py: Python<'_>,
    file: PyRef<'_, File>,
    path: PathBuf,
    traces: Option<PyReadonlyArray1<'_, i64>>,
    start: Option<usize>,
    stop: Option<usize>,
    rezero_time: bool,
    budget: Option<Bound<'_, PyAny>>,
    sync: bool,
) -> PyResult<u64> {
    let list = traces_of(&file, traces.as_ref())?;
    let ns = file.inner.layout().n_samples as usize;
    let window = match (start, stop) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0)..b.unwrap_or(ns)),
    };
    let spec = Crop {
        samples: window,
        rezero_time,
    };
    let b = self::budget(budget.as_ref())?;
    let src = &file.inner;
    py.detach(|| {
        let mut sink = FileSink::new(&path, sync);
        ops::crop(src, &list, &spec, b, &mut sink)
    })
    .map_err(to_py)
}

/// `(field, min, max, values)`.
type PredicateArg = (String, Option<f64>, Option<f64>, Option<Vec<f64>>);

/// Trace numbers whose headers pass every predicate. Each predicate is
/// `(field, min, max, values)`: a range when `values` is `None`, else a set.
#[pyfunction]
fn filter(
    py: Python<'_>,
    file: PyRef<'_, File>,
    predicates: Vec<PredicateArg>,
) -> PyResult<Py<PyAny>> {
    let preds: Vec<Predicate> = predicates
        .into_iter()
        .map(|(field, min, max, values)| match values {
            Some(v) => Predicate::is_in(field, v),
            None => Predicate::range(field, min, max),
        })
        .collect();
    let src = &file.inner;
    let hits = py.detach(|| ops::filter(src, &preds)).map_err(to_py)?;
    let hits: Vec<i64> = hits.into_iter().map(|t| t as i64).collect();
    Ok(hits.into_pyarray(py).into_any().unbind())
}

/// Sort the whole file by `keys` (`(field, descending)` pairs) into `path`.
#[pyfunction]
#[pyo3(signature = (file, path, keys, *, strategy="auto", sorting_code=None, tmp_dir=None, budget=None, sync=true))]
#[allow(clippy::too_many_arguments)]
fn resort<'py>(
    py: Python<'py>,
    file: PyRef<'py, File>,
    path: PathBuf,
    keys: Vec<(String, bool)>,
    strategy: &str,
    sorting_code: Option<i16>,
    tmp_dir: Option<PathBuf>,
    budget: Option<Bound<'py, PyAny>>,
    sync: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let strategy = match strategy {
        "auto" => SortStrategy::Auto,
        "memory" | "in_memory" => SortStrategy::InMemory,
        "gather" => SortStrategy::Gather,
        "external" => SortStrategy::External,
        other => {
            return Err(PyValueError::new_err(format!(
                "strategy {other:?} is not \"auto\", \"memory\", \"gather\", or \"external\""
            )))
        }
    };
    let keys: Vec<SortKey> = keys
        .into_iter()
        .map(|(f, desc)| {
            if desc {
                SortKey::desc(f)
            } else {
                SortKey::asc(f)
            }
        })
        .collect();
    let mut opts = ResortOptions::new(self::budget(budget.as_ref())?);
    opts.strategy = strategy;
    opts.sorting_code = sorting_code;
    opts.tmp_dir = tmp_dir;
    let src = &file.inner;
    let report = py
        .detach(|| {
            let mut sink = FileSink::new(&path, sync);
            ops::resort(src, &keys, &opts, &mut sink)
        })
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("already_sorted", report.already_sorted)?;
    d.set_item(
        "strategy",
        match report.strategy {
            SortStrategy::Auto => "auto",
            SortStrategy::InMemory => "memory",
            SortStrategy::Gather => "gather",
            SortStrategy::External => "external",
        },
    )?;
    d.set_item("runs", report.runs)?;
    d.set_item("spilled_bytes", report.spilled_bytes)?;
    Ok(d)
}

/// `above + 1 + below` samples around `index[i]` of each trace, and a validity mask.
#[pyfunction]
fn slice_horizon(
    py: Python<'_>,
    file: PyRef<'_, File>,
    traces: PyReadonlyArray1<'_, i64>,
    index: PyReadonlyArray1<'_, i64>,
    above: usize,
    below: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let list = trace_list(&traces, file.inner.layout().n_traces)?;
    let index = index.as_slice()?;
    let src = &file.inner;
    let win = py
        .detach(|| ops::slice_horizon(src, &list, index, above, below))
        .map_err(to_py)?;
    let rows = list.len();
    let data = samples_2d(py, win.samples, rows, win.width)?;
    let valid = numpy::ndarray::Array2::from_shape_vec((rows, win.width), win.valid)
        .map_err(|e| crate::SegyrsError::new_err(e.to_string()))?
        .into_pyarray(py)
        .into_any()
        .unbind();
    Ok((data, valid))
}

/// The dense grid of `traces` in RAM: `(samples, shape, mask, inlines, crosslines, offsets)`.
#[pyfunction]
#[pyo3(signature = (file, traces=None, fill=0.0, budget=None))]
#[allow(clippy::type_complexity)]
fn to_regular(
    py: Python<'_>,
    file: PyRef<'_, File>,
    traces: Option<PyReadonlyArray1<'_, i64>>,
    fill: f64,
    budget: Option<Bound<'_, PyAny>>,
) -> PyResult<(
    Py<PyAny>,
    Vec<usize>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
)> {
    let list = traces_of(&file, traces.as_ref())?;
    let b = self::budget(budget.as_ref())?;
    let src = &file.inner;
    let reg = py
        .detach(|| ops::to_regular(src, &list, fill, b))
        .map_err(to_py)?;
    let ns = *reg.shape.last().unwrap_or(&0);
    let rows = if ns == 0 { 0 } else { reg.mask.len() };
    let samples = samples_2d(py, reg.samples, rows, ns)?;
    Ok((
        samples,
        reg.shape,
        reg.mask.into_pyarray(py).into_any().unbind(),
        reg.axes.inlines.into_pyarray(py).into_any().unbind(),
        reg.axes.crosslines.into_pyarray(py).into_any().unbind(),
        reg.axes.offsets.into_pyarray(py).into_any().unbind(),
    ))
}

/// The dense grid of `traces` streamed to `path`. Returns `(cells, filled)`.
#[pyfunction]
#[pyo3(signature = (file, path, traces=None, fill=0.0, *, budget=None, sync=true))]
fn to_regular_file(
    py: Python<'_>,
    file: PyRef<'_, File>,
    path: PathBuf,
    traces: Option<PyReadonlyArray1<'_, i64>>,
    fill: f64,
    budget: Option<Bound<'_, PyAny>>,
    sync: bool,
) -> PyResult<(u64, u64)> {
    let list = traces_of(&file, traces.as_ref())?;
    let b = self::budget(budget.as_ref())?;
    let src = &file.inner;
    let report = py
        .detach(|| {
            let mut sink = FileSink::new(&path, sync);
            ops::to_regular_stream(src, &list, fill, b, &mut sink)
        })
        .map_err(to_py)?;
    Ok((report.cells, report.filled))
}

/// Concatenate whole files into `path`. Returns the number of traces.
#[pyfunction]
#[pyo3(signature = (files, path, *, budget=None, sync=true))]
fn concat(
    py: Python<'_>,
    files: Vec<PyRef<'_, File>>,
    path: PathBuf,
    budget: Option<Bound<'_, PyAny>>,
    sync: bool,
) -> PyResult<u64> {
    let b = self::budget(budget.as_ref())?;
    let srcs: Vec<&segyrs::File<crate::interop::Source>> = files.iter().map(|f| &f.inner).collect();
    py.detach(|| {
        let mut sink = FileSink::new(&path, sync);
        ops::concat(&srcs, b, &mut sink)
    })
    .map_err(to_py)
}

/// Statistics over `traces`, every sample, NaN and infinities counted apart.
#[pyfunction]
#[pyo3(signature = (file, traces=None))]
fn stats<'py>(
    py: Python<'py>,
    file: PyRef<'py, File>,
    traces: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let list = traces_of(&file, traces.as_ref())?;
    let src = &file.inner;
    let s = py.detach(|| ops::stats(src, &list)).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("count", s.count)?;
    d.set_item("nan", s.nan)?;
    d.set_item("infinite", s.infinite)?;
    d.set_item("min", s.min)?;
    d.set_item("max", s.max)?;
    d.set_item("mean", s.mean)?;
    d.set_item("std", s.std())?;
    d.set_item("variance", s.variance())?;
    d.set_item("rms", s.rms())?;
    Ok(d)
}

/// One RMS per trace over its finite samples.
#[pyfunction]
#[pyo3(signature = (file, traces=None))]
fn rms(
    py: Python<'_>,
    file: PyRef<'_, File>,
    traces: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<Py<PyAny>> {
    let list = traces_of(&file, traces.as_ref())?;
    let src = &file.inner;
    let v = py.detach(|| ops::rms(src, &list)).map_err(to_py)?;
    Ok(v.into_pyarray(py).into_any().unbind())
}

/// `traces` with samples limited to `[lo, hi]`, written to `path`. Returns the
/// number of samples changed.
#[pyfunction]
#[pyo3(signature = (file, path, lo, hi, traces=None, *, budget=None, sync=true))]
#[allow(clippy::too_many_arguments)]
fn clip(
    py: Python<'_>,
    file: PyRef<'_, File>,
    path: PathBuf,
    lo: f64,
    hi: f64,
    traces: Option<PyReadonlyArray1<'_, i64>>,
    budget: Option<Bound<'_, PyAny>>,
    sync: bool,
) -> PyResult<u64> {
    let list = traces_of(&file, traces.as_ref())?;
    let b = self::budget(budget.as_ref())?;
    let src = &file.inner;
    py.detach(|| {
        let mut sink = FileSink::new(&path, sync);
        ops::clip(src, &list, lo, hi, b, &mut sink)
    })
    .map_err(to_py)
}
