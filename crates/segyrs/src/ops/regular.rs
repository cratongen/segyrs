use rayon::prelude::*;

use crate::endian::{self, Endian};
use crate::error::{Error, Result};
use crate::file::{File, ReadAt};
use crate::geometry::apply_scalar;
use crate::ops::{
    check_traces, encode_value, extend_samples, header_len, odd_pairwise, par_decode, put_i16,
    put_i32, read_window, stream, trace_offset, Budget, OutputSpec, Round, TraceSink,
};
use crate::sample::{self, Samples};

/// Labels of the dense grid. Inline and crossline run from their minimum to
/// their maximum by the largest step that divides every gap. Offsets are the
/// distinct values present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegularAxes {
    pub inlines: Vec<i32>,
    pub crosslines: Vec<i32>,
    pub offsets: Vec<i32>,
}

impl RegularAxes {
    pub fn cells(&self) -> usize {
        self.inlines.len() * self.crosslines.len() * self.offsets.len()
    }

    /// inline, crossline, offset when there is more than one, then samples.
    pub fn shape(&self, ns: usize) -> Vec<usize> {
        let mut shape = vec![self.inlines.len(), self.crosslines.len()];
        if self.offsets.len() > 1 {
            shape.push(self.offsets.len());
        }
        shape.push(ns);
        shape
    }
}

/// A dense cube built on request (X-15, X-17).
///
/// `samples` is C-ordered over `shape`. `mask` has one flag per grid cell
/// (every dim but the samples). It is false where no trace exists, and those
/// cells hold the fill value.
#[derive(Clone, Debug)]
pub struct Regular {
    pub axes: RegularAxes,
    pub shape: Vec<usize>,
    pub samples: Samples,
    pub mask: Vec<bool>,
}

struct Axis {
    min: i64,
    step: i64,
    len: u64,
}

impl Axis {
    fn of(values: &[i32], traces: &[u64]) -> Self {
        let min = traces
            .par_iter()
            .map(|&t| i64::from(values[t as usize]))
            .min()
            .unwrap_or(0);
        let max = traces
            .par_iter()
            .map(|&t| i64::from(values[t as usize]))
            .max()
            .unwrap_or(0);
        let step = traces
            .par_iter()
            .map(|&t| (i64::from(values[t as usize]) - min) as u64)
            .reduce(|| 0, gcd)
            .max(1) as i64;
        Self {
            min,
            step,
            len: ((max - min) / step + 1) as u64,
        }
    }

    fn index(&self, v: i32) -> u64 {
        ((i64::from(v) - self.min) / self.step) as u64
    }

    fn labels(&self) -> Vec<i32> {
        (0..self.len as i64)
            .map(|i| (self.min + i * self.step) as i32)
            .collect()
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

struct Plan {
    axes: RegularAxes,
    /// `(cell, trace)` ascending by cell.
    placed: Vec<(u64, u64)>,
}

/// Dense grid labels of `traces`, without building anything dense.
pub fn regular_axes<R: ReadAt>(file: &File<R>, traces: &[u64]) -> Result<RegularAxes> {
    check_traces(file, traces)?;
    Ok(RegularAxes {
        inlines: Axis::of(file.inline(), traces).labels(),
        crosslines: Axis::of(file.crossline(), traces).labels(),
        offsets: distinct_offsets(file, traces),
    })
}

fn distinct_offsets<R: ReadAt>(file: &File<R>, traces: &[u64]) -> Vec<i32> {
    let off = file.offset();
    let mut offsets: Vec<i32> = traces.par_iter().map(|&t| off[t as usize]).collect();
    offsets.par_sort_unstable();
    offsets.dedup();
    if offsets.is_empty() {
        offsets.push(0);
    }
    offsets
}

/// Check `cells * per_cell + index` against the budget before allocating the
/// labels, then place every trace in its cell.
fn plan<R: ReadAt>(
    file: &File<R>,
    traces: &[u64],
    per_cell: u128,
    budget: Budget,
) -> Result<(Plan, u64)> {
    check_traces(file, traces)?;
    let (il, xl) = (
        Axis::of(file.inline(), traces),
        Axis::of(file.crossline(), traces),
    );
    let offsets = distinct_offsets(file, traces);
    let n_off = offsets.len() as u128;
    let cells = u128::from(il.len) * u128::from(xl.len) * n_off;
    let labels = (u128::from(il.len) + u128::from(xl.len)) * 4;
    let need = cells * per_cell + traces.len() as u128 * 16 + labels;
    budget.check(need, "the dense grid")?;
    let axes = RegularAxes {
        inlines: il.labels(),
        crosslines: xl.labels(),
        offsets,
    };
    let (ils, xls, offs) = (file.inline(), file.crossline(), file.offset());
    let n_xl = xl.len;
    let mut placed: Vec<(u64, u64)> = traces
        .par_iter()
        .map(|&t| {
            let i = t as usize;
            let o = axes.offsets.binary_search(&offs[i]).unwrap_or(0) as u64;
            let cell = (il.index(ils[i]) * n_xl + xl.index(xls[i])) * n_off as u64 + o;
            (cell, t)
        })
        .collect();
    placed.par_sort_unstable();
    if let Some(pair) = placed.par_windows(2).find_first(|w| w[0].0 == w[1].0) {
        let t = pair[1].1 as usize;
        return Err(Error::DuplicateKey {
            inline: ils[t],
            crossline: xls[t],
        });
    }
    Ok((Plan { axes, placed }, (budget.get() as u128 - need) as u64))
}

impl Plan {
    fn trace_at(&self, cell: u64) -> Option<u64> {
        self.placed
            .binary_search_by_key(&cell, |p| p.0)
            .ok()
            .map(|k| self.placed[k].1)
    }
}

fn decoded_width(samples: &Samples) -> usize {
    match samples {
        Samples::I8(_) | Samples::U8(_) => 1,
        Samples::I16(_) | Samples::U16(_) => 2,
        Samples::I32(_) | Samples::U32(_) | Samples::F32(_) => 4,
        Samples::I64(_) | Samples::U64(_) | Samples::F64(_) => 8,
    }
}

fn reserve(samples: &mut Samples, n: usize) {
    match samples {
        Samples::I8(v) => v.reserve_exact(n),
        Samples::U8(v) => v.reserve_exact(n),
        Samples::I16(v) => v.reserve_exact(n),
        Samples::U16(v) => v.reserve_exact(n),
        Samples::I32(v) => v.reserve_exact(n),
        Samples::U32(v) => v.reserve_exact(n),
        Samples::I64(v) => v.reserve_exact(n),
        Samples::U64(v) => v.reserve_exact(n),
        Samples::F32(v) => v.reserve_exact(n),
        Samples::F64(v) => v.reserve_exact(n),
    }
}

/// Build the dense cube of `traces` in RAM, with `fill` where no trace exists
/// (X-15, X-17).
///
/// The size is checked against `budget` before anything dense is allocated.
/// Over budget is `io::ErrorKind::OutOfMemory`. The cube is assembled chunk
/// by chunk from what is left of the budget, and each chunk is read and decoded
/// in parallel.
pub fn to_regular<R: ReadAt>(
    file: &File<R>,
    traces: &[u64],
    fill: f64,
    budget: Budget,
) -> Result<Regular> {
    let ns = file.layout().n_samples as usize;
    let format = file.format();
    let w = format.width();
    let mut samples = sample::empty_samples(format, 0);
    let dw = decoded_width(&samples);
    let per_cell = (ns * dw + 1) as u128;
    let _ra = file.readahead_for(traces.len());
    let (plan, left) = plan(file, traces, per_cell, budget)?;
    let order = if odd_pairwise(file) {
        Endian::Big
    } else {
        file.endian()
    };
    let fill_bytes = encode_value(fill, format, order, Round::Nearest)?;
    let cells = plan.axes.cells();
    let per_chunk = (left / ((ns * (w + dw)).max(1) as u64)).max(1) as usize;
    reserve(&mut samples, cells * ns);
    let mut mask = vec![false; cells];
    let row = ns * w;
    let mut raw = vec![0u8; per_chunk.min(cells) * row];
    for c0 in (0..cells).step_by(per_chunk) {
        let c1 = (c0 + per_chunk).min(cells);
        let buf = &mut raw[..(c1 - c0) * row];
        mask[c0..c1]
            .par_iter_mut()
            .zip(buf.par_chunks_mut(row.max(1)))
            .enumerate()
            .try_for_each_init(Vec::new, |tmp, (i, (m, out))| -> Result<()> {
                match plan.trace_at((c0 + i) as u64) {
                    Some(t) => {
                        read_window(file, t, 0, ns, tmp)?;
                        out.copy_from_slice(tmp);
                        *m = true;
                    }
                    None => {
                        for s in out.chunks_exact_mut(w) {
                            s.copy_from_slice(&fill_bytes);
                        }
                    }
                }
                Ok(())
            })?;
        extend_samples(&mut samples, par_decode(buf, format, order, file.gain()));
    }
    Ok(Regular {
        shape: plan.axes.shape(ns),
        axes: plan.axes,
        samples,
        mask,
    })
}

/// What [`to_regular_stream`] wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegularReport {
    pub cells: u64,
    /// Cells with no trace, written as fill traces.
    pub filled: u64,
}

/// Stream the dense grid of `traces` into `sink` in inline, crossline, offset
/// order, one chunk at a time (X-17).
///
/// A cell with a trace gets that trace's bytes. An empty cell gets a dead
/// trace (trace id 2) whose samples are `fill`, whose line numbers and offset
/// are the cell's labels, and whose `cdp_x`, `cdp_y` are the least-squares
/// affine fit of the existing traces' coordinates on inline and crossline,
/// in the first trace's `scalco`. Delay, time scalar, coordinate units,
/// sample count, and interval are copied from the first trace. No more than
/// two chunks are ever live.
pub fn to_regular_stream<R: ReadAt, S: TraceSink + Send>(
    file: &File<R>,
    traces: &[u64],
    fill: f64,
    budget: Budget,
    sink: &mut S,
) -> Result<RegularReport> {
    let format = file.format();
    let fill_bytes = encode_value(fill, format, file.endian(), Round::Nearest)?;
    let _ra = file.readahead_for(traces.len());
    let (plan, left) = plan(file, traces, 0, budget)?;
    let mut spec = OutputSpec::from_file(file)?;
    let cells = plan.axes.cells();
    spec.n_traces = Some(cells as u64);
    let stride = spec.trace_len();
    let hl = spec.trace_header_len;
    let synth = Synth::new(file, &plan)?;
    let (n_xl, n_off) = (plan.axes.crosslines.len(), plan.axes.offsets.len());
    sink.begin(&spec)?;
    stream(cells, stride, hl, Budget::bytes(left), sink, |c, out| {
        if let Some(t) = plan.trace_at(c as u64) {
            return file.reader().read_at(trace_offset(file, t), out);
        }
        let il = plan.axes.inlines[c / (n_xl * n_off)];
        let xl = plan.axes.crosslines[(c / n_off) % n_xl];
        let off = plan.axes.offsets[c % n_off];
        synth.header(il, xl, off, &mut out[..hl]);
        for s in out[hl..].chunks_exact_mut(fill_bytes.len()) {
            s.copy_from_slice(&fill_bytes);
        }
        Ok(())
    })?;
    sink.finish()?;
    Ok(RegularReport {
        cells: cells as u64,
        filled: (cells - plan.placed.len()) as u64,
    })
}

/// Header of a fill trace.
struct Synth {
    template: Vec<u8>,
    endian: Endian,
    bytes: (usize, usize, usize),
    x: [f64; 3],
    y: [f64; 3],
    scalco: i16,
}

impl Synth {
    fn new<R: ReadAt>(file: &File<R>, plan: &Plan) -> Result<Self> {
        let endian = file.endian();
        let hl = header_len(file);
        let mut template = vec![0u8; hl];
        let mut scalco = 1i16;
        if let Some(&(_, first)) = plan.placed.first() {
            let mut head = vec![0u8; hl];
            file.reader()
                .read_at(trace_offset(file, first), &mut head)?;
            // delay, time scalar, coordinate units, scalco, ns, interval
            for (off, len) in [(108, 2), (214, 2), (88, 2), (70, 2), (114, 2), (116, 2)] {
                template[off..off + len].copy_from_slice(&head[off..off + len]);
            }
            scalco = endian::i16_at(&head, 70, endian);
        }
        put_i16(&mut template, 28, 2, endian);
        let g = file.geometry();
        let traces: Vec<usize> = plan.placed.iter().map(|p| p.1 as usize).collect();
        let (il, xl, sc) = (file.inline(), file.crossline(), file.scalco());
        let fit = |v: &[i32]| {
            affine(traces.iter().map(|&t| {
                (
                    f64::from(il[t]),
                    f64::from(xl[t]),
                    apply_scalar(v[t], sc[t]),
                )
            }))
        };
        Ok(Self {
            template,
            endian,
            bytes: (
                usize::from(g.inline_byte - 1),
                usize::from(g.crossline_byte - 1),
                usize::from(g.offset_byte - 1),
            ),
            x: fit(file.cdp_x()),
            y: fit(file.cdp_y()),
            scalco,
        })
    }

    fn header(&self, il: i32, xl: i32, off: i32, out: &mut [u8]) {
        out.copy_from_slice(&self.template);
        put_i32(out, self.bytes.0, il, self.endian);
        put_i32(out, self.bytes.1, xl, self.endian);
        put_i32(out, self.bytes.2, off, self.endian);
        let unit = apply_scalar(1, self.scalco);
        let at = |p: &[f64; 3]| {
            let v = p[0] + p[1] * f64::from(il) + p[2] * f64::from(xl);
            (v / unit).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
        };
        put_i32(out, 180, at(&self.x), self.endian);
        put_i32(out, 184, at(&self.y), self.endian);
    }
}

/// Least squares `v = a + b * il + c * xl`, falling back to fewer terms when
/// the points are on one line or one point.
fn affine(points: impl Iterator<Item = (f64, f64, f64)> + Clone) -> [f64; 3] {
    let (mut n, mut su, mut sv, mut sw) = (0.0, 0.0, 0.0, 0.0);
    for (u, v, w) in points.clone() {
        n += 1.0;
        su += u;
        sv += v;
        sw += w;
    }
    if n == 0.0 {
        return [0.0; 3];
    }
    let (mu, mv, mw) = (su / n, sv / n, sw / n);
    let (mut uu, mut vv, mut uv, mut uw, mut vw) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (u, v, w) in points {
        let (u, v, w) = (u - mu, v - mv, w - mw);
        uu += u * u;
        vv += v * v;
        uv += u * v;
        uw += u * w;
        vw += v * w;
    }
    let det = uu * vv - uv * uv;
    let (b, c) = if det > 1e-9 * uu * vv && det > 0.0 {
        ((uw * vv - vw * uv) / det, (vw * uu - uw * uv) / det)
    } else if uu > 0.0 {
        (uw / uu, 0.0)
    } else if vv > 0.0 {
        (0.0, vw / vv)
    } else {
        (0.0, 0.0)
    };
    [mw - b * mu - c * mv, b, c]
}
