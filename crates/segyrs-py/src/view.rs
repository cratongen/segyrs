//! M4 bindings: the notebook picture (`segyrs_draw`). It reads with the
//! interpreter detached. The native window is the `segyrs-view` extension
//! (crates/segyrs-view-py).

use std::ops::Deref;
use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use segyrs_draw::{picture, resolve_plane, style_from_names, PlaneSpec, Source};

use crate::{interop, to_py, File, SegyrsError, UnsortedCoordinate};

fn draw_err(err: segyrs_draw::Error) -> PyErr {
    match err {
        segyrs_draw::Error::Core(e) => to_py(e),
        e @ segyrs_draw::Error::UnsortedCoordinate { .. } => {
            UnsortedCoordinate::new_err(e.to_string())
        }
        e => SegyrsError::new_err(e.to_string()),
    }
}

/// The core file of a Python `File`, borrowed for drawing: the header scan
/// and the trace index the Volume already has are reused, and a remote
/// (object-store) file draws through the same reader.
struct Shared(Py<File>);

impl Deref for Shared {
    type Target = segyrs::File<interop::Source>;

    fn deref(&self) -> &Self::Target {
        &self.0.get().inner
    }
}

/// A file opened for drawing. Python's `Volume` keeps one per open file.
#[pyclass(frozen, module = "segyrs._segyrs")]
pub struct Drawer {
    src: Arc<Source<interop::Source>>,
}

#[pymethods]
impl Drawer {
    /// Draw from an open `File`, sharing its header scan and trace index.
    #[new]
    #[pyo3(signature = (file, *, depth=false))]
    fn new(py: Python<'_>, file: Py<File>, depth: bool) -> PyResult<Self> {
        let shared = Shared(file);
        let src = py
            .detach(|| Source::from_file(shared))
            .map_err(draw_err)?
            .depth(depth);
        Ok(Self { src: Arc::new(src) })
    }

    /// Dim names in drawing order.
    #[getter]
    fn dims(&self) -> Vec<&'static str> {
        self.src.dims().iter().map(|d| d.as_str()).collect()
    }

    /// `(png, text, note, traces_read)` of one plane. `fixed` holds the
    /// positions of the selected-away dims and `ranges` the window of every
    /// free dim, both as positions in the opened file. A window is
    /// `(dim, start, stop, step)`: every `step`-th position from `start`.
    #[pyo3(signature = (
        fixed, ranges, *, clip=None, live=None, width=800, height=480, colormap="grey",
        polarity="normal", gain=1.0, wiggle="off", wiggle_every=1, density=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn picture<'py>(
        &self,
        py: Python<'py>,
        fixed: Vec<(String, usize)>,
        ranges: Vec<(String, usize, usize, usize)>,
        clip: Option<f32>,
        live: Option<PyReadonlyArray1<'py, bool>>,
        width: usize,
        height: usize,
        colormap: &str,
        polarity: &str,
        gain: f32,
        wiggle: &str,
        wiggle_every: usize,
        density: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, String, Option<String>, usize)> {
        let style = style_from_names(
            clip,
            colormap,
            polarity,
            gain,
            wiggle,
            wiggle_every,
            density,
        )
        .map_err(PyValueError::new_err)?;
        let (fixed, windows) =
            resolve_plane(&self.src, &fixed, &ranges).map_err(PyValueError::new_err)?;
        let mut spec = PlaneSpec::new(fixed, width, height);
        spec.live = live.map(|m| m.as_array().iter().copied().collect());
        if let [x, y] = &windows[..] {
            spec.x_range = Some(x.range.clone());
            spec.y_range = Some(y.range.clone());
            spec.x_step = x.step;
            spec.y_step = y.step;
        }
        let pic = py
            .detach(|| picture(&self.src, &spec, &style))
            .map_err(draw_err)?;
        Ok((
            PyBytes::new(py, &pic.png),
            pic.text,
            pic.note,
            pic.section.traces_read,
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Drawer>()
}
