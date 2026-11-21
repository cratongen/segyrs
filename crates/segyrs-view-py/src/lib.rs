//! The experimental native viewer window of the segyrs Python package,
//! built as the separate `segyrs-view` extension (crates/segyrs-view-py) so the
//! `segyrs` wheel carries no window or GPU stack. It is not published yet.
//!
//! `segyrs.view` passes the open file's reader as a capsule
//! (`segyrs::bridge`) with the options it was opened with. This module opens
//! its own `segyrs::File` on that reader, so it reads the same map or the
//! same remote cache. The window keeps the calling (main) thread and polls
//! for Ctrl-C with the interpreter detached.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyType};

use segyrs::bridge::{open_options, python_error, ForeignReader, RawReader, CAPSULE_NAME};
use segyrs::ReadAt;
use segyrs_draw::{resolve_plane, style_from_names, Source};
use segyrs_view::{Interrupt, Region, Renderer, ViewError, ViewOptions};

fn raise(class: &str, msg: String) -> PyErr {
    Python::attach(|py| {
        let ty = py
            .import("segyrs._segyrs")
            .and_then(|m| m.getattr(class))
            .and_then(|t| Ok(t.cast_into::<PyType>()?));
        match ty {
            Ok(ty) => PyErr::from_type(ty, (msg,)),
            Err(e) => e,
        }
    })
}

fn core_err(err: segyrs::Error) -> PyErr {
    let (class, msg) = python_error(&err);
    raise(class, msg)
}

fn draw_err(err: segyrs_draw::Error) -> PyErr {
    match err {
        segyrs_draw::Error::Core(e) => core_err(e),
        e @ segyrs_draw::Error::UnsortedCoordinate { .. } => {
            raise("UnsortedCoordinate", e.to_string())
        }
        e => raise("SegyrsError", e.to_string()),
    }
}

fn view_err(err: ViewError) -> PyErr {
    match err {
        ViewError::Draw(e) => draw_err(e),
        e => raise("SegyrsError", e.to_string()),
    }
}

fn foreign(capsule: &Bound<'_, PyCapsule>) -> PyResult<ForeignReader> {
    let ptr = capsule.pointer_checked(Some(CAPSULE_NAME))?;
    // SAFETY: a capsule named CAPSULE_NAME holds a `RawReader` exported by a
    // segyrs extension, alive while the capsule is.
    let raw = unsafe { ptr.cast::<RawReader>().as_ref() };
    raw.check().map_err(PyValueError::new_err)?;
    // SAFETY: `raw` is live and passed `check`; extension modules stay loaded.
    Ok(unsafe { ForeignReader::new(raw) })
}

/// Open the native window on one plane of the file behind `reader` and block
/// until it closes. `fixed` and `ranges` are positions in the opened file, as
/// `segyrs.Volume.plot` passes them. Returns the presenter that ran, `"gpu"`
/// or `"cpu"`. Ctrl-C closes the window and raises `KeyboardInterrupt`.
#[pyfunction]
#[pyo3(signature = (
    reader, name, fixed, ranges, *, sample_format=None, endian=None, gain_convention=None,
    inline_byte=189, crossline_byte=193, offset_byte=37, scan="auto", depth=false,
    clip=None, live=None, colormap="grey", polarity="normal", gain=1.0, wiggle="off",
    wiggle_every=1, renderer="auto", title=None,
))]
#[allow(clippy::too_many_arguments)]
fn view(
    py: Python<'_>,
    reader: Bound<'_, PyCapsule>,
    name: String,
    fixed: Vec<(String, usize)>,
    ranges: Vec<(String, usize, usize, usize)>,
    sample_format: Option<&str>,
    endian: Option<&str>,
    gain_convention: Option<&str>,
    inline_byte: u16,
    crossline_byte: u16,
    offset_byte: u16,
    scan: &str,
    depth: bool,
    clip: Option<f32>,
    live: Option<Vec<bool>>,
    colormap: &str,
    polarity: &str,
    gain: f32,
    wiggle: &str,
    wiggle_every: usize,
    renderer: &str,
    title: Option<String>,
) -> PyResult<&'static str> {
    let (options, scan) = open_options(
        sample_format,
        endian,
        gain_convention,
        inline_byte,
        crossline_byte,
        offset_byte,
        scan,
    )
    .map_err(PyValueError::new_err)?;
    let style = style_from_names(clip, colormap, polarity, gain, wiggle, wiggle_every, true)
        .map_err(PyValueError::new_err)?;
    let renderer = match renderer {
        "auto" => Renderer::Auto,
        "gpu" => Renderer::Gpu,
        "cpu" => Renderer::Cpu,
        r => {
            return Err(PyValueError::new_err(format!(
                "renderer {r:?} is not \"auto\", \"gpu\", or \"cpu\""
            )))
        }
    };
    let reader = foreign(&reader)?;
    let len = reader.len();
    let file = py
        .detach(|| {
            segyrs::File::from_reader_scan(Arc::new(reader) as Arc<dyn ReadAt>, len, &options, scan)
        })
        .map_err(core_err)?;
    let src = py
        .detach(|| Source::from_file(Box::new(file)))
        .map_err(draw_err)?
        .depth(depth);
    let (fixed, windows) = resolve_plane(&src, &fixed, &ranges).map_err(PyValueError::new_err)?;
    let region = match &windows[..] {
        [x, y] => Some(Region {
            x: x.range.clone(),
            y: y.range.clone(),
            x_step: x.step,
            y_step: y.step,
        }),
        _ => None,
    };
    let hit = Arc::new(AtomicBool::new(false));
    let flag = hit.clone();
    let options = ViewOptions {
        fixed,
        region,
        style,
        live: live.map(Arc::from),
        title: title.unwrap_or_else(|| format!("segyrs: {name}")),
        renderer,
        interrupt: Some(Interrupt(Arc::new(move || {
            let stop = Python::attach(|py| py.check_signals().is_err());
            if stop {
                flag.store(true, Ordering::Relaxed);
            }
            stop
        }))),
    };
    let src = Arc::new(src);
    let ran = py
        .detach(|| segyrs_view::run(src, options))
        .map_err(view_err)?;
    if hit.load(Ordering::Relaxed) {
        return Err(PyKeyboardInterrupt::new_err("viewer closed by Ctrl-C"));
    }
    Ok(match ran {
        Renderer::Gpu => "gpu",
        _ => "cpu",
    })
}

#[pymodule]
fn _segyrs_view(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(view, m)?)?;
    Ok(())
}
