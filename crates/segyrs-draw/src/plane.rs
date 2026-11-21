//! One 2D plane of a volume, read and binned to the pixel grid.

use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;
use segyrs::{apply_scalar, ReadAt};

use crate::clip::p98;
use crate::source::{Dim, Source};
use crate::{Error, Result};

/// Header coordinate to label the horizontal axis of a section with,
/// in place of the dim's own coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coord {
    Inline,
    Crossline,
    Offset,
    CdpX,
    CdpY,
}

impl Coord {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Crossline => "crossline",
            Self::Offset => "offset",
            Self::CdpX => "cdp_x",
            Self::CdpY => "cdp_y",
        }
    }
}

/// What to read: the fixed positions, an optional window, and the pixel grid.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneSpec {
    /// Positional index (`isel`) of every dim that is not drawn. Exactly two
    /// dims must stay free.
    pub fixed: Vec<(Dim, usize)>,
    /// Positions along the horizontal image axis. `None` is the whole axis.
    pub x_range: Option<Range<usize>>,
    /// Positions along the vertical image axis. `None` is the whole axis.
    pub y_range: Option<Range<usize>>,
    /// Pixel grid of the plot area. The float grid is never larger.
    pub width: usize,
    pub height: usize,
    /// Per-trace live flag. Masked traces are not read and draw as absent.
    /// `None`: every trace is live (roadmap decision 7).
    pub live: Option<Arc<[bool]>>,
    /// Label a section's horizontal axis with this header coordinate.
    pub x_coord: Option<Coord>,
    /// Sort a non-monotonic horizontal axis instead of raising
    /// [`Error::UnsortedCoordinate`].
    pub sort: bool,
    /// Read one position in `x_step` along the horizontal image axis, counted
    /// from the start of `x_range` (a stepped selection, or `wiggle_every`).
    /// 1 reads every position.
    pub x_step: usize,
    /// Read one position in `y_step` along the vertical image axis, counted
    /// from the start of `y_range`. On a section this decimates samples: the
    /// window is read in one piece per trace and every `y_step`-th sample kept.
    pub y_step: usize,
}

impl PlaneSpec {
    pub fn new(fixed: Vec<(Dim, usize)>, width: usize, height: usize) -> Self {
        Self {
            fixed,
            x_range: None,
            y_range: None,
            width,
            height,
            live: None,
            x_coord: None,
            sort: false,
            x_step: 1,
            y_step: 1,
        }
    }
}

/// One image axis: its name and the coordinate of each data position drawn.
#[derive(Clone, Debug)]
pub struct Axis {
    pub dim: Dim,
    /// Label printed on the axis, e.g. `crossline` or `time (ms)`.
    pub name: String,
    /// Coordinate of each data position in the window, in drawing order.
    pub labels: Vec<f64>,
    /// Number of bins (columns or rows of the float grid).
    pub bins: usize,
}

impl Axis {
    /// Data positions (indices into `labels`) that fall in bin `b`.
    pub fn bin(&self, b: usize) -> Range<usize> {
        bin_range(self.labels.len(), self.bins, b)
    }
}

pub(crate) fn bin_range(n: usize, bins: usize, b: usize) -> Range<usize> {
    let lo = (b as u128 * n as u128 / bins as u128) as usize;
    let hi = ((b as u128 + 1) * n as u128 / bins as u128) as usize;
    lo..hi
}

/// A fixed dim of the drawn plane, by position and coordinate.
#[derive(Clone, Debug)]
pub struct Fixed {
    pub dim: Dim,
    pub index: usize,
    pub coord: f64,
}

/// The float grid of one plane at one binning, kept for the cursor and for
/// every colour change (clip, polarity and colormap do not read the file).
#[derive(Clone, Debug)]
pub struct Section {
    /// Grid columns and rows. `width <= pixels.0`, `height <= pixels.1`.
    pub width: usize,
    pub height: usize,
    /// Row-major display values. NaN is an absent cell: a masked trace, a
    /// missing slot, or a bin with no finite sample.
    pub values: Vec<f32>,
    pub x: Axis,
    pub y: Axis,
    pub fixed: Vec<Fixed>,
    /// Pixel grid this was binned for.
    pub pixels: (usize, usize),
    /// Symmetric p98 of the finite values of this grid, `None` if it has none.
    pub p98: Option<f32>,
    /// Trace of each grid column when every column is one trace (a section
    /// that is not binned horizontally).
    pub column_traces: Option<Vec<Option<u32>>>,
    /// Number of distinct traces whose samples were read.
    pub traces_read: usize,
}

impl Section {
    pub fn value(&self, col: usize, row: usize) -> f32 {
        self.values[row * self.width + col]
    }

    /// Grid cell under pixel `(px, py)` of the plot area (nearest).
    pub fn cell_at(&self, px: usize, py: usize) -> (usize, usize) {
        let col = (px * self.width / self.pixels.0.max(1)).min(self.width - 1);
        let row = (py * self.height / self.pixels.1.max(1)).min(self.height - 1);
        (col, row)
    }

    /// `inline 3` style description of the fixed dims.
    pub fn title(&self) -> String {
        self.fixed
            .iter()
            .map(|f| format!("{} {}", label_name(f.dim), fmt_num(f.coord)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

pub(crate) fn label_name(dim: Dim) -> &'static str {
    match dim {
        Dim::Time => "time (ms)",
        d => d.as_str(),
    }
}

pub(crate) fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Keep the sample of larger magnitude. NaN in `acc` means empty.
#[inline]
fn max_abs(acc: f32, v: f32) -> f32 {
    if v.is_finite() && (acc.is_nan() || v.abs() > acc.abs()) {
        v
    } else {
        acc
    }
}

impl<R: ReadAt + 'static> Source<R> {
    /// The positions to fix for the notebook's default picture of this
    /// volume: the first index of the slowest grid axis (and of offset on a pre-stack
    /// grid), leaving crossline and the vertical dim free (roadmap decision 5).
    /// A `(trace, time)` volume is already 2D and fixes nothing.
    pub fn default_fixed(&self) -> Vec<(Dim, usize)> {
        let h = self.dims.len() - 1;
        if h < 2 {
            return Vec::new();
        }
        self.dims[..h]
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != 1)
            .map(|(_, d)| (*d, 0))
            .collect()
    }

    /// Read one plane, binned to the pixel grid.
    ///
    /// Only the traces of the plane are read, and only the samples inside the
    /// vertical window (one sample per trace for a time or depth slice). Bins
    /// that hold more than one sample keep the sample of largest magnitude, so
    /// a thin peak survives zooming out.
    pub fn section(&self, spec: &PlaneSpec) -> Result<Section> {
        if spec.width == 0 || spec.height == 0 {
            return Err(Error::Empty("pixel grid has no pixels"));
        }
        let fixed = self.check_fixed(&spec.fixed)?;
        let free: Vec<Dim> = self
            .dims
            .iter()
            .copied()
            .filter(|d| !spec.fixed.iter().any(|(f, _)| f == d))
            .collect();
        if free.len() != 2 {
            return Err(Error::NotTwoD {
                free,
                dims: self.dims.clone(),
            });
        }
        if let Some(live) = &spec.live {
            if live.len() as u64 != self.file().layout().n_traces {
                return Err(Error::Mask {
                    got: live.len(),
                    want: self.file().layout().n_traces as usize,
                });
            }
        }
        let (xd, yd) = (free[0], free[1]);
        if spec.x_step == 0 || spec.y_step == 0 {
            return Err(Error::Empty("x_step and y_step must be at least 1"));
        }
        if yd.is_vertical() {
            self.vertical_section(spec, fixed, xd, yd)
        } else {
            if spec.x_coord.is_some() {
                return Err(Error::Empty(
                    "x_coord labels a section; a slice has no single trace per column",
                ));
            }
            self.slice(spec, fixed, xd, yd)
        }
    }

    fn check_fixed(&self, fixed: &[(Dim, usize)]) -> Result<Vec<Fixed>> {
        let mut out = Vec::with_capacity(fixed.len());
        for (i, (dim, index)) in fixed.iter().enumerate() {
            let k = self.axis_of(*dim).ok_or(Error::NoDim {
                dim: *dim,
                dims: self.dims.clone(),
            })?;
            if fixed[..i].iter().any(|(d, _)| d == dim) {
                return Err(Error::Duplicate(*dim));
            }
            let len = self.coords[k].len();
            if *index >= len {
                return Err(Error::Index {
                    dim: *dim,
                    index: *index,
                    len,
                });
            }
            out.push(Fixed {
                dim: *dim,
                index: *index,
                coord: self.coords[k][*index],
            });
        }
        // Report fixed dims in volume order, not request order.
        out.sort_by_key(|f| self.axis_of(f.dim));
        Ok(out)
    }

    fn fixed_pos(&self, fixed: &[Fixed], dim: Dim) -> Option<usize> {
        fixed.iter().find(|f| f.dim == dim).map(|f| f.index)
    }

    fn live(&self, spec: &PlaneSpec, t: u32) -> bool {
        spec.live.as_ref().is_none_or(|m| m[t as usize])
    }

    fn window(range: &Option<Range<usize>>, len: usize, dim: Dim) -> Result<Range<usize>> {
        let r = range.clone().unwrap_or(0..len);
        if r.start >= r.end || r.end > len {
            return Err(Error::Window { dim, len, range: r });
        }
        Ok(r)
    }

    /// Header coordinate of trace `t` for `coord`, scalars applied.
    fn header_coord(&self, coord: Coord, t: u32) -> f64 {
        let f = self.file();
        let t = t as usize;
        match coord {
            Coord::Inline => f64::from(f.inline()[t]),
            Coord::Crossline => f64::from(f.crossline()[t]),
            Coord::Offset => f64::from(f.offset()[t]),
            Coord::CdpX => apply_scalar(f.cdp_x()[t], f.scalco()[t]),
            Coord::CdpY => apply_scalar(f.cdp_y()[t], f.scalco()[t]),
        }
    }

    /// Positions along a horizontal dim with every other horizontal dim fixed.
    fn traces_along(&self, fixed: &[Fixed], along: Dim) -> Vec<u32> {
        let hdims = &self.dims[..self.dims.len() - 1];
        let k = self.axis_of(along).expect("free dim exists");
        let mut pos: Vec<usize> = hdims
            .iter()
            .map(|d| self.fixed_pos(fixed, *d).unwrap_or(0))
            .collect();
        (0..self.coords[k].len())
            .map(|p| {
                pos[k] = p;
                self.trace_at(&pos)
            })
            .collect()
    }

    fn vertical_section(
        &self,
        spec: &PlaneSpec,
        fixed: Vec<Fixed>,
        xd: Dim,
        yd: Dim,
    ) -> Result<Section> {
        let kx = self.axis_of(xd).expect("free dim");
        let ky = self.axis_of(yd).expect("free dim");
        let mut traces = self.traces_along(&fixed, xd);
        let (mut labels, name) = match spec.x_coord {
            None => (self.coords[kx].to_vec(), label_name(xd).to_string()),
            Some(c) => (
                traces.iter().map(|t| self.header_coord(c, *t)).collect(),
                c.as_str().to_string(),
            ),
        };
        order_axis(&mut labels, &mut traces, spec.sort, &name)?;
        let xr = Self::window(&spec.x_range, labels.len(), xd)?;
        let yr = Self::window(&spec.y_range, self.coords[ky].len(), yd)?;
        let step = spec.x_step;
        let traces: Vec<u32> = traces[xr.clone()].iter().copied().step_by(step).collect();
        let labels: Vec<f64> = labels[xr].iter().copied().step_by(step).collect();
        let traces = &traces[..];
        // Samples kept: every y_step-th of the window, read as one span per trace.
        let ys = spec.y_step;
        let ny = yr.len().div_ceil(ys);
        let nbx = traces.len().min(spec.width);
        let nby = ny.min(spec.height);
        let d = self.decoder;
        let reader = self.reader();
        let columns: Vec<Vec<f32>> = (0..nbx)
            .into_par_iter()
            .map_init(
                || (Vec::new(), vec![0f32; yr.len()]),
                |(buf, tmp), b| -> Result<Vec<f32>> {
                    let mut col = vec![f32::NAN; nby];
                    for &t in &traces[bin_range(traces.len(), nbx, b)] {
                        if !self.live(spec, t) {
                            continue;
                        }
                        // The span up to the last kept sample, one read.
                        let span = yr.start..yr.start + (ny - 1) * ys + 1;
                        d.read(reader, t, span.clone(), buf, &mut tmp[..span.len()])?;
                        for (yb, c) in col.iter_mut().enumerate() {
                            for i in bin_range(ny, nby, yb) {
                                *c = max_abs(*c, tmp[i * ys]);
                            }
                        }
                    }
                    Ok(col)
                },
            )
            .collect::<Result<_>>()?;
        let mut values = vec![f32::NAN; nbx * nby];
        for (c, col) in columns.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                values[r * nbx + c] = *v;
            }
        }
        let traces_read = traces.iter().filter(|t| self.live(spec, **t)).count();
        let column_traces = (nbx == traces.len()).then(|| {
            traces
                .iter()
                .map(|t| self.live(spec, *t).then_some(*t))
                .collect()
        });
        Ok(Section {
            width: nbx,
            height: nby,
            p98: p98(&values),
            values,
            x: Axis {
                dim: xd,
                name,
                labels,
                bins: nbx,
            },
            y: Axis {
                dim: yd,
                name: label_name(yd).to_string(),
                labels: self.coords[ky][yr].iter().copied().step_by(ys).collect(),
                bins: nby,
            },
            fixed,
            pixels: (spec.width, spec.height),
            column_traces,
            traces_read,
        })
    }

    /// A plane with the vertical dim fixed: one sample from every trace.
    fn slice(&self, spec: &PlaneSpec, fixed: Vec<Fixed>, xd: Dim, yd: Dim) -> Result<Section> {
        let kx = self.axis_of(xd).expect("free dim");
        let ky = self.axis_of(yd).expect("free dim");
        let vd = *self.dims.last().expect("vertical dim");
        let s = self
            .fixed_pos(&fixed, vd)
            .expect("vertical fixed in a slice");
        let mut xl = self.coords[kx].to_vec();
        let mut xperm: Vec<usize> = (0..xl.len()).collect();
        order_axis(&mut xl, &mut xperm, spec.sort, label_name(xd))?;
        let mut yl = self.coords[ky].to_vec();
        let mut yperm: Vec<usize> = (0..yl.len()).collect();
        order_axis(&mut yl, &mut yperm, spec.sort, label_name(yd))?;
        let xr = Self::window(&spec.x_range, xl.len(), xd)?;
        let yr = Self::window(&spec.y_range, yl.len(), yd)?;
        // Positions kept along each axis (indices into the sorted labels).
        let xs: Vec<usize> = xr.step_by(spec.x_step).collect();
        let ys: Vec<usize> = yr.step_by(spec.y_step).collect();
        let nbx = xs.len().min(spec.width);
        let nby = ys.len().min(spec.height);
        // Trace of every (x, y) cell in the window, row-major in y.
        let hdims = &self.dims[..self.dims.len() - 1];
        let base: Vec<usize> = hdims
            .iter()
            .map(|d| self.fixed_pos(&fixed, *d).unwrap_or(0))
            .collect();
        let d = self.decoder;
        let reader = self.reader();
        let rows: Vec<(Vec<f32>, usize)> = (0..nby)
            .into_par_iter()
            .map_init(
                || (Vec::new(), base.clone()),
                |(buf, pos), b| -> Result<(Vec<f32>, usize)> {
                    let mut row = vec![f32::NAN; nbx];
                    let mut read = 0usize;
                    let mut one = [0f32];
                    for yi in bin_range(ys.len(), nby, b) {
                        pos[ky] = yperm[ys[yi]];
                        for (xb, cell) in row.iter_mut().enumerate() {
                            for xi in bin_range(xs.len(), nbx, xb) {
                                pos[kx] = xperm[xs[xi]];
                                let t = self.trace_at(pos);
                                if t == u32::MAX || !self.live(spec, t) {
                                    continue;
                                }
                                d.read(reader, t, s..s + 1, buf, &mut one)?;
                                read += 1;
                                *cell = max_abs(*cell, one[0]);
                            }
                        }
                    }
                    Ok((row, read))
                },
            )
            .collect::<Result<_>>()?;
        let traces_read = rows.iter().map(|(_, n)| n).sum();
        let values: Vec<f32> = rows.into_iter().flat_map(|(r, _)| r).collect();
        Ok(Section {
            width: nbx,
            height: nby,
            p98: p98(&values),
            values,
            x: Axis {
                dim: xd,
                name: label_name(xd).to_string(),
                labels: xs.iter().map(|i| xl[*i]).collect(),
                bins: nbx,
            },
            y: Axis {
                dim: yd,
                name: label_name(yd).to_string(),
                labels: ys.iter().map(|i| yl[*i]).collect(),
                bins: nby,
            },
            fixed,
            pixels: (spec.width, spec.height),
            column_traces: None,
            traces_read,
        })
    }
}

/// Check that `labels` is monotonic (either direction, ties allowed). If it is
/// not, sort it ascending with `items` when `sort` is set, else refuse (X-11).
fn order_axis<T: Copy>(
    labels: &mut Vec<f64>,
    items: &mut Vec<T>,
    sort: bool,
    name: &str,
) -> Result<()> {
    let inc = labels.windows(2).all(|w| w[0] <= w[1]);
    let dec = labels.windows(2).all(|w| w[0] >= w[1]);
    if inc || dec {
        return Ok(());
    }
    if !sort {
        return Err(Error::UnsortedCoordinate {
            coord: name.to_string(),
        });
    }
    let mut idx: Vec<usize> = (0..labels.len()).collect();
    idx.sort_by(|a, b| labels[*a].total_cmp(&labels[*b]));
    *labels = idx.iter().map(|i| labels[*i]).collect();
    *items = idx.iter().map(|i| items[*i]).collect();
    Ok(())
}
