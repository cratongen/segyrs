//! CPU renderer for segyrs sections.
//!
//! The data flow is: a plane request ([`PlaneSpec`]) names the fixed dims and
//! the pixel grid; [`Source::section`] reads only the traces of that plane and
//! only the samples in the window, on rayon, and bins them to the pixel grid
//! with max-abs; the result is a float grid ([`Section`]) with a symmetric p98
//! of its finite values; [`render`] maps it through a LUT, draws ticks and
//! (when traces fit) wiggles, and [`Image::to_png`] encodes it.
//!
//! There is no window and no GPU here. The view crate uploads the same float
//! grid and the same LUT bytes to the GPU.

mod args;
mod clip;
mod color;
mod font;
mod plane;
mod render;
mod source;

pub use args::{resolve_plane, style_from_names, Plane, Window};
pub use clip::{p98, percentile_abs};
pub use color::{lut_index, lut_rows, Colormap, Polarity, Style, Wiggle};
pub use plane::{Axis, Coord, Fixed, PlaneSpec, Section};
pub use render::{
    effective_clip, map_image, notebook, picture, render, wiggle_note, Image, Picture, MARGINS,
    NOTEBOOK_PIXELS, WIGGLE_MIN_SPACING,
};
pub use source::{Dim, FileOwner, MapReader, Source};

/// Failures while selecting, reading or drawing a plane.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Core(#[from] segyrs::Error),

    /// The selection does not leave exactly two free dims. A 3D volume does
    /// not draw itself; the message names the `sel` calls that make a plane.
    #[error("{}", not_two_d(.free, .dims))]
    NotTwoD { free: Vec<Dim>, dims: Vec<Dim> },

    #[error("no dim {dim}; the dims are ({})", join(.dims))]
    NoDim { dim: Dim, dims: Vec<Dim> },

    #[error("{0} is fixed twice")]
    Duplicate(Dim),

    #[error("{dim} index {index} is outside 0..{len}")]
    Index { dim: Dim, index: usize, len: usize },

    #[error("{dim} window {range:?} is empty or outside 0..{len}")]
    Window {
        dim: Dim,
        len: usize,
        range: std::ops::Range<usize>,
    },

    #[error("live mask has {got} entries; the file has {want} traces")]
    Mask { got: usize, want: usize },

    /// The drawn horizontal axis is not monotonic and sorting was not asked for (X-11).
    #[error("coordinate {coord} is not monotonic in this view; sort the view (sort=True) or draw against the dim")]
    UnsortedCoordinate { coord: String },

    #[error("geometry: {0}")]
    Geometry(String),

    #[error("{0}")]
    Empty(&'static str),

    #[error("png: {0}")]
    Png(String),
}

pub type Result<T> = std::result::Result<T, Error>;

fn join(dims: &[Dim]) -> String {
    dims.iter()
        .map(|d| d.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn not_two_d(free: &[Dim], dims: &[Dim]) -> String {
    let n = free.len();
    if n > 2 {
        let hint = free
            .iter()
            .map(|d| format!("sel({d}=...)"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "a {n}D selection ({}) does not draw as one image; fix {} of its dims first with {hint}",
            join(free),
            n - 2
        )
    } else {
        format!(
            "a {n}D selection ({}) of ({}) cannot be drawn; a picture needs two free dims",
            join(free),
            join(dims)
        )
    }
}
