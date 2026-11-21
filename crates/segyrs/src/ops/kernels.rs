use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::error::Result;
use crate::file::{File, ReadAt};
use crate::ops::{
    check_traces, encode_value, for_each_f64, header_len, invalid, stream, trace_offset, Budget,
    OutputSpec, Round, TraceSink,
};
use crate::sample::{self, Samples};

/// Moments and extremes of the finite samples. NaN and infinite samples are
/// counted apart and left out of the rest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub count: u64,
    pub nan: u64,
    pub infinite: u64,
    /// NaN when `count` is zero.
    pub min: f64,
    /// NaN when `count` is zero.
    pub max: f64,
    pub mean: f64,
    /// Sum of squared deviations from the mean.
    pub m2: f64,
    pub sum_sq: f64,
}

impl Stats {
    fn empty() -> Self {
        Self {
            count: 0,
            nan: 0,
            infinite: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            m2: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Population variance.
    pub fn variance(&self) -> f64 {
        self.m2 / self.count as f64
    }

    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Root mean square over all finite samples.
    pub fn rms(&self) -> f64 {
        (self.sum_sq / self.count as f64).sqrt()
    }

    /// Chan, Golub and LeVeque pairwise update.
    fn merge(&mut self, other: &Self) {
        self.nan += other.nan;
        self.infinite += other.infinite;
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            let (nan, inf) = (self.nan, self.infinite);
            *self = *other;
            self.nan = nan;
            self.infinite = inf;
            return;
        }
        let (na, nb) = (self.count as f64, other.count as f64);
        let n = na + nb;
        let delta = other.mean - self.mean;
        self.mean += delta * nb / n;
        self.m2 += other.m2 + delta * delta * na * nb / n;
        self.count += other.count;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum_sq += other.sum_sq;
    }

    /// Two passes over one trace that is already in cache.
    fn of_trace(samples: &Samples) -> Self {
        let mut s = Self::empty();
        let mut sum = 0.0;
        for_each_f64(samples, |_, x| {
            if x.is_nan() {
                s.nan += 1;
            } else if x.is_infinite() {
                s.infinite += 1;
            } else {
                s.count += 1;
                sum += x;
                s.sum_sq += x * x;
                s.min = s.min.min(x);
                s.max = s.max.max(x);
            }
        });
        if s.count > 0 {
            s.mean = sum / s.count as f64;
            let mean = s.mean;
            for_each_f64(samples, |_, x| {
                if x.is_finite() {
                    s.m2 += (x - mean) * (x - mean);
                }
            });
        }
        s
    }
}

/// Group size for the reductions: about one MiB of samples per group.
///
/// It depends only on the trace size, so the merge order and therefore the
/// floating-point result is the same on every machine and thread count.
fn group_of<R: ReadAt>(file: &File<R>) -> usize {
    let bytes = (file.layout().n_samples as usize * file.format().width()).max(1);
    ((1 << 20) / bytes).max(1)
}

fn decode_into<R: ReadAt>(
    file: &File<R>,
    trace: u64,
    raw: &mut Vec<u8>,
    out: &mut Samples,
) -> Result<()> {
    let ns = file.layout().n_samples as usize;
    if ns == 0 {
        return Ok(());
    }
    let order = crate::ops::read_window(file, trace, 0, ns, raw)?;
    sample::decode_trace(raw, file.format(), order, file.gain(), 0, ns, out);
    Ok(())
}

/// Streaming statistics over `traces` on every core.
///
/// Live memory is one trace per worker. The result does not depend on the
/// number of threads.
pub fn stats<R: ReadAt>(file: &File<R>, traces: &[u64]) -> Result<Stats> {
    let _ra = file.readahead_for(traces.len());
    check_traces(file, traces)?;
    let ns = file.layout().n_samples as usize;
    let parts: Vec<Stats> = traces
        .par_chunks(group_of(file))
        .map(|group| -> Result<Stats> {
            let mut raw = Vec::new();
            let mut dec = sample::empty_samples(file.format(), ns);
            let mut acc = Stats::empty();
            for &t in group {
                decode_into(file, t, &mut raw, &mut dec)?;
                acc.merge(&Stats::of_trace(&dec));
            }
            Ok(acc)
        })
        .collect::<Result<_>>()?;
    let mut total = Stats::empty();
    for p in &parts {
        total.merge(p);
    }
    if total.count == 0 {
        total.min = f64::NAN;
        total.max = f64::NAN;
        total.mean = f64::NAN;
    }
    Ok(total)
}

/// Root mean square of each trace along the vertical axis, over its finite
/// samples. NaN for a trace with none.
pub fn rms<R: ReadAt>(file: &File<R>, traces: &[u64]) -> Result<Vec<f64>> {
    let _ra = file.readahead_for(traces.len());
    check_traces(file, traces)?;
    let ns = file.layout().n_samples as usize;
    traces
        .par_iter()
        .map_init(
            || (Vec::new(), sample::empty_samples(file.format(), ns)),
            |(raw, dec), &t| -> Result<f64> {
                decode_into(file, t, raw, dec)?;
                let (mut sum, mut n) = (0.0, 0u64);
                for_each_f64(dec, |_, x| {
                    if x.is_finite() {
                        sum += x * x;
                        n += 1;
                    }
                });
                Ok(if n == 0 {
                    f64::NAN
                } else {
                    (sum / n as f64).sqrt()
                })
            },
        )
        .collect()
}

fn check_bounds(lo: f64, hi: f64) -> Result<()> {
    if lo.is_nan() || hi.is_nan() || lo > hi {
        return Err(invalid(format!("clip bounds {lo}..{hi} are not ordered")));
    }
    Ok(())
}

/// Limit decoded samples to `[lo, hi]` in place. NaN stays NaN. Integer
/// samples use the integers inside the bounds. Returns how many changed.
pub fn clip_samples(samples: &mut Samples, lo: f64, hi: f64) -> Result<u64> {
    check_bounds(lo, hi)?;
    let changed = AtomicU64::new(0);
    macro_rules! ints {
        ($v:expr, $t:ty) => {{
            let l = lo.ceil().clamp(<$t>::MIN as f64, <$t>::MAX as f64) as $t;
            let h = hi.floor().clamp(<$t>::MIN as f64, <$t>::MAX as f64) as $t;
            if l > h {
                return Err(invalid(format!("no {} lies in {lo}..{hi}", stringify!($t))));
            }
            $v.par_iter_mut().for_each(|x| {
                let c = (*x).clamp(l, h);
                if c != *x {
                    *x = c;
                    changed.fetch_add(1, Ordering::Relaxed);
                }
            });
        }};
    }
    macro_rules! floats {
        ($v:expr, $t:ty) => {{
            let (l, h) = (lo as $t, hi as $t);
            $v.par_iter_mut().for_each(|x| {
                if *x < l || *x > h {
                    *x = (*x).clamp(l, h);
                    changed.fetch_add(1, Ordering::Relaxed);
                }
            });
        }};
    }
    match samples {
        Samples::I8(v) => ints!(v, i8),
        Samples::U8(v) => ints!(v, u8),
        Samples::I16(v) => ints!(v, i16),
        Samples::U16(v) => ints!(v, u16),
        Samples::I32(v) => ints!(v, i32),
        Samples::U32(v) => ints!(v, u32),
        Samples::I64(v) => ints!(v, i64),
        Samples::U64(v) => ints!(v, u64),
        Samples::F32(v) => floats!(v, f32),
        Samples::F64(v) => floats!(v, f64),
    }
    Ok(changed.into_inner())
}

/// Stream `traces` into `sink` with every sample limited to `[lo, hi]`.
///
/// A sample inside the bounds keeps its on-disk bytes exactly. A sample below
/// `lo` becomes the smallest value of the sample format not below `lo`, and
/// one above `hi` the largest value not above `hi`. Returns how many changed.
pub fn clip<R: ReadAt, S: TraceSink + Send>(
    file: &File<R>,
    traces: &[u64],
    lo: f64,
    hi: f64,
    budget: Budget,
    sink: &mut S,
) -> Result<u64> {
    check_bounds(lo, hi)?;
    let _ra = file.readahead_for(traces.len());
    check_traces(file, traces)?;
    let (format, endian) = (file.format(), file.endian());
    let lo_bytes = encode_value(lo, format, endian, Round::Up)?;
    let hi_bytes = encode_value(hi, format, endian, Round::Down)?;
    let mut spec = OutputSpec::from_file(file)?;
    spec.n_traces = Some(traces.len() as u64);
    let hl = header_len(file);
    let ns = file.layout().n_samples as usize;
    let w = format.width();
    let changed = AtomicU64::new(0);
    sink.begin(&spec)?;
    stream(
        traces.len(),
        spec.trace_len(),
        hl,
        budget,
        sink,
        |i, out| -> Result<()> {
            file.reader().read_at(trace_offset(file, traces[i]), out)?;
            let mut dec = sample::empty_samples(format, ns);
            let body = &mut out[hl..];
            if ns > 0 {
                sample::decode_trace(body, format, endian, file.gain(), 0, ns, &mut dec);
            }
            let mut n = 0u64;
            for_each_f64(&dec, |j, x| {
                let patch = if x < lo {
                    &lo_bytes
                } else if x > hi {
                    &hi_bytes
                } else {
                    return;
                };
                body[j * w..(j + 1) * w].copy_from_slice(patch);
                n += 1;
            });
            changed.fetch_add(n, Ordering::Relaxed);
            Ok(())
        },
    )?;
    sink.finish()?;
    Ok(changed.into_inner())
}
