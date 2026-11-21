//! Viewer state with no window: which plane, which index, zoom, style.
//! The GPU window and the CPU fallback both drive this.

use std::ops::Range;
use std::sync::Arc;

use segyrs::ReadAt;
use segyrs_draw::{Colormap, Dim, PlaneSpec, Polarity, Section, Source, Style, Wiggle};

use crate::Region;

/// A plane the toolbar can show: the two free dims and the dim the stepper moves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneKind {
    pub free: [Dim; 2],
    /// Dim the index stepper moves. `None` when the volume is already 2D.
    pub step: Option<Dim>,
}

impl PlaneKind {
    pub fn name(&self) -> String {
        match self.step {
            Some(d) if d.is_vertical() => format!("{d} slice"),
            Some(d) => format!("{d} section"),
            None => format!("{} by {}", self.free[0], self.free[1]),
        }
    }
}

/// Every plane of a volume with these dims, sections first.
pub fn planes(dims: &[Dim]) -> Vec<PlaneKind> {
    let v = *dims.last().expect("a vertical dim");
    let h: Vec<Dim> = dims[..dims.len() - 1].to_vec();
    if h.len() < 2 {
        return vec![PlaneKind {
            free: [h[0], v],
            step: None,
        }];
    }
    let mut out = Vec::new();
    // Sections: step one grid axis, the other grid axis and the vertical are free.
    for (i, d) in h.iter().enumerate().take(2) {
        out.push(PlaneKind {
            free: [h[1 - i], v],
            step: Some(*d),
        });
    }
    // Slice: the two grid axes are free, the vertical is stepped.
    out.push(PlaneKind {
        free: [h[0], h[1]],
        step: Some(v),
    });
    // Gather on a pre-stack grid: offset by vertical, step inline.
    if h.len() > 2 {
        out.push(PlaneKind {
            free: [h[2], v],
            step: Some(h[0]),
        });
    }
    out
}

/// Visible part of each image axis, as fractions of the whole axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zoom {
    pub x: (f64, f64),
    pub y: (f64, f64),
}

impl Default for Zoom {
    fn default() -> Self {
        Self {
            x: (0.0, 1.0),
            y: (0.0, 1.0),
        }
    }
}

impl Zoom {
    /// Zoom by `factor` (< 1 zooms in) around the fraction `(fx, fy)` of the view.
    pub fn zoom(&mut self, factor: f64, fx: f64, fy: f64) {
        fn one(r: &mut (f64, f64), factor: f64, f: f64) {
            let w = r.1 - r.0;
            let c = r.0 + f * w;
            let nw = (w * factor).clamp(1e-6, 1.0);
            let mut lo = c - f * nw;
            lo = lo.clamp(0.0, 1.0 - nw);
            *r = (lo, lo + nw);
        }
        one(&mut self.x, factor, fx);
        one(&mut self.y, factor, fy);
    }

    /// Pan by a fraction of the current view width and height.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        fn one(r: &mut (f64, f64), d: f64) {
            let w = r.1 - r.0;
            let lo = (r.0 + d * w).clamp(0.0, 1.0 - w);
            *r = (lo, lo + w);
        }
        one(&mut self.x, dx);
        one(&mut self.y, dy);
    }

    /// Fractions of the axis covering positions `r` of `n`.
    pub fn fractions(r: &Range<usize>, n: usize) -> (f64, f64) {
        let n = n.max(1) as f64;
        (r.start as f64 / n, r.end as f64 / n)
    }

    /// Positions of `n` inside the fractions `r`. The tolerance keeps
    /// `range(fractions(a..b, n), n) == a..b` despite rounding.
    pub fn range(r: (f64, f64), n: usize) -> Range<usize> {
        const EPS: f64 = 1e-9;
        let lo = ((r.0 * n as f64 + EPS).floor() as usize).min(n.saturating_sub(1));
        let hi = ((r.1 * n as f64 - EPS).ceil() as usize).clamp(lo + 1, n);
        lo..hi
    }
}

/// Everything the window shows, minus the window.
pub struct ViewState<R: ReadAt + 'static> {
    pub source: Arc<Source<R>>,
    pub planes: Vec<PlaneKind>,
    pub plane: usize,
    /// Current position along every dim (only the fixed ones matter).
    pub pos: Vec<usize>,
    pub zoom: Zoom,
    /// Read one position in `steps.0` along x and `steps.1` along y (a stepped
    /// selection). Back to `(1, 1)` when the plane changes.
    pub steps: (usize, usize),
    pub style: Style,
    pub live: Option<Arc<[bool]>>,
    pub section: Option<Arc<Section>>,
    pub error: Option<String>,
}

impl<R: ReadAt + 'static> ViewState<R> {
    /// Start on the plane that `fixed` describes, or the notebook default.
    pub fn new(source: Arc<Source<R>>, fixed: &[(Dim, usize)], style: Style) -> Self {
        let dims = source.dims().to_vec();
        let planes = planes(&dims);
        let fixed = if fixed.is_empty() {
            source.default_fixed()
        } else {
            fixed.to_vec()
        };
        let mut pos = vec![0; dims.len()];
        for (d, i) in &fixed {
            if let Some(k) = dims.iter().position(|x| x == d) {
                pos[k] = *i;
            }
        }
        let plane = planes
            .iter()
            .position(|p| {
                dims.iter()
                    .filter(|d| !p.free.contains(d))
                    .all(|d| fixed.iter().any(|(f, _)| f == d))
            })
            .unwrap_or(0);
        Self {
            source,
            planes,
            plane,
            pos,
            zoom: Zoom::default(),
            steps: (1, 1),
            style,
            live: None,
            section: None,
            error: None,
        }
    }

    pub fn kind(&self) -> &PlaneKind {
        &self.planes[self.plane]
    }

    pub fn dims(&self) -> &[Dim] {
        self.source.dims()
    }

    pub fn len_of(&self, dim: Dim) -> usize {
        self.source.len_of(dim).unwrap_or(1)
    }

    /// Fixed dims of the current plane with their positions.
    pub fn fixed(&self) -> Vec<(Dim, usize)> {
        let kind = self.kind();
        self.dims()
            .iter()
            .enumerate()
            .filter(|(_, d)| !kind.free.contains(d))
            .map(|(k, d)| (*d, self.pos[k]))
            .collect()
    }

    pub fn index_of(&self, dim: Dim) -> usize {
        let k = self.dims().iter().position(|d| *d == dim).unwrap_or(0);
        self.pos[k]
    }

    pub fn set_index(&mut self, dim: Dim, index: usize) {
        let len = self.len_of(dim);
        if let Some(k) = self.dims().iter().position(|d| *d == dim) {
            self.pos[k] = index.min(len.saturating_sub(1));
        }
    }

    /// Move the stepped dim by `delta`, clamped to the axis.
    pub fn step(&mut self, delta: i64) {
        if let Some(d) = self.kind().step {
            let i = self.index_of(d) as i64 + delta;
            self.set_index(d, i.max(0) as usize);
        }
    }

    pub fn set_plane(&mut self, plane: usize) {
        if plane < self.planes.len() && plane != self.plane {
            self.plane = plane;
            self.zoom = Zoom::default();
            self.steps = (1, 1);
        }
    }

    /// Zoom to `w` on the current plane and decimate by its steps.
    pub fn set_region(&mut self, w: &Region) {
        let kind = self.kind().clone();
        let nx = self.len_of(kind.free[0]);
        let ny = self.len_of(kind.free[1]);
        self.zoom = Zoom {
            x: Zoom::fractions(&w.x, nx),
            y: Zoom::fractions(&w.y, ny),
        };
        self.steps = (w.x_step.max(1), w.y_step.max(1));
    }

    /// The request for a plot area of `pixels`, inside the current zoom.
    pub fn spec(&self, pixels: (usize, usize)) -> PlaneSpec {
        let kind = self.kind();
        let mut spec = PlaneSpec::new(self.fixed(), pixels.0.max(1), pixels.1.max(1));
        let nx = self.len_of(kind.free[0]);
        let ny = self.len_of(kind.free[1]);
        spec.x_range = Some(Zoom::range(self.zoom.x, nx));
        spec.y_range = Some(Zoom::range(self.zoom.y, ny));
        spec.x_step = self.steps.0;
        spec.y_step = self.steps.1;
        // wiggle_every thins the traces of a section while a wiggle is drawn.
        if self.style.wiggle != Wiggle::Off && kind.free[1].is_vertical() {
            spec.x_step = spec.x_step.saturating_mul(self.style.wiggle_every.max(1));
        }
        spec.live = self.live.clone();
        spec
    }

    /// Read the plane for `pixels` now, on the calling thread.
    pub fn load(&mut self, pixels: (usize, usize)) {
        match self.source.section(&self.spec(pixels)) {
            Ok(s) => {
                self.section = Some(Arc::new(s));
                self.error = None;
            }
            Err(e) => self.error = Some(e.to_string()),
        }
    }

    pub fn cycle_colormap(&mut self) {
        let all = Colormap::ALL;
        let i = all
            .iter()
            .position(|c| *c == self.style.colormap)
            .unwrap_or(0);
        self.style.colormap = all[(i + 1) % all.len()];
    }

    pub fn flip_polarity(&mut self) {
        self.style.polarity = match self.style.polarity {
            Polarity::Normal => Polarity::Reversed,
            Polarity::Reversed => Polarity::Normal,
        };
    }

    pub fn cycle_wiggle(&mut self) {
        self.style.wiggle = match self.style.wiggle {
            Wiggle::Off => Wiggle::Line,
            Wiggle::Line => Wiggle::Area,
            Wiggle::Area => Wiggle::Off,
        };
    }

    /// One-line description of the cell under plot pixel `(px, py)`.
    pub fn readout(&self, px: usize, py: usize) -> Option<String> {
        let s = self.section.as_ref()?;
        let (c, r) = s.cell_at(px, py);
        let label = |a: &segyrs_draw::Axis, b: usize| {
            let p = a.bin(b);
            let (lo, hi) = (a.labels[p.start], a.labels[p.end - 1]);
            if p.len() == 1 {
                format!("{} {}", a.dim, num(lo))
            } else {
                format!("{} {}..{}", a.dim, num(lo), num(hi))
            }
        };
        let v = s.value(c, r);
        let value = if v.is_nan() {
            "absent".to_string()
        } else {
            format!("{v} (display f32)")
        };
        let mut out = format!("{}, {}: {value}", label(&s.x, c), label(&s.y, r));
        // The file value, when the cell is one sample of one trace. One
        // sample is read, so formats f32 cannot hold report exactly.
        if let (Some(cols), true) = (s.column_traces.as_ref(), s.y.bin(r).len() == 1) {
            if let Some(Some(t)) = cols.get(c) {
                if let Some(k) = self.source.coords(s.y.dim).and_then(|all| {
                    let want = s.y.labels[s.y.bin(r).start];
                    all.iter().position(|v| *v == want)
                }) {
                    if let Ok(exact) = self.source.sample_f64(*t, k) {
                        out.push_str(&format!(", file value {exact}"));
                    }
                }
            }
        }
        Some(out)
    }
}

fn num(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v:.3}")
    }
}
