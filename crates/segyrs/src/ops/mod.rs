//! Whole-volume manipulation: resort, crop, header filter, sample windows,
//! regularization, concatenation, and streaming sample kernels.
//!
//! Every operation that touches more than one trace window works in buffers
//! sized from a [`Budget`]. Output goes to a [`TraceSink`] as on-disk trace
//! bytes, so the source sample format is kept bit for bit.

mod budget;
mod concat;
mod filter;
mod kernels;
mod regular;
mod sink;
mod sort;
mod subset;
mod window;

pub use budget::{available_ram, Budget};
pub use concat::{check_concat, concat};
pub use filter::{filter, filter_columns, Predicate};
pub use kernels::{clip, clip_samples, rms, stats, Stats};
pub use regular::{
    regular_axes, to_regular, to_regular_stream, Regular, RegularAxes, RegularReport,
};
pub use sink::{OutputSpec, TraceSink};
pub use sort::{resort, sort_order, ResortOptions, SortKey, SortReport, SortStrategy};
pub use subset::{crop, Crop};
pub use window::{slice_horizon, time_slice, Window};

use std::io;

use rayon::prelude::*;

use crate::endian::Endian;
use crate::error::{Error, Result};
use crate::file::{File, ReadAt};
use crate::header::HeaderColumns;
use crate::sample::{self, SampleFormat, Samples};

pub(crate) const TRACE_HEADER: usize = 240;

pub(crate) fn invalid(msg: impl Into<String>) -> Error {
    Error::Io(io::Error::new(io::ErrorKind::InvalidInput, msg.into()))
}

/// Bytes of trace header per trace, including rev 2 extension blocks.
pub(crate) fn header_len<R: ReadAt>(file: &File<R>) -> usize {
    TRACE_HEADER * (1 + usize::from(file.layout().extra_trace_header_blocks))
}

/// Refuse a trace list that reaches past the file before any byte is written.
pub(crate) fn check_traces<R: ReadAt>(file: &File<R>, traces: &[u64]) -> Result<()> {
    let n = file.layout().n_traces;
    match traces.par_iter().find_first(|&&t| t >= n) {
        Some(t) => Err(invalid(format!("trace {t} is outside 0..{n}"))),
        None => file.verify_traces(traces),
    }
}

pub(crate) fn trace_offset<R: ReadAt>(file: &File<R>, trace: u64) -> u64 {
    file.layout().data_start + trace * file.stride()
}

/// True when raw sample bytes cannot be cut or patched without re-pairing.
pub(crate) fn odd_pairwise<R: ReadAt>(file: &File<R>) -> bool {
    file.endian() == Endian::Pair && file.format().width() % 2 == 1
}

/// Read `count` samples of one trace from sample `start`.
///
/// Returns the byte order the bytes are in. A pairwise file with an odd sample
/// width is read on even file offsets and returned big-endian, because its
/// byte pairs straddle sample boundaries.
pub(crate) fn read_window<R: ReadAt>(
    file: &File<R>,
    trace: u64,
    start: usize,
    count: usize,
    buf: &mut Vec<u8>,
) -> Result<Endian> {
    let width = file.format().width() as u64;
    let lo = trace_offset(file, trace) + header_len(file) as u64 + start as u64 * width;
    let len = count as u64 * width;
    if !odd_pairwise(file) {
        buf.resize(len as usize, 0);
        file.reader().read_at(lo, buf)?;
        return Ok(file.endian());
    }
    let a0 = lo & !1;
    let a1 = (lo + len).div_ceil(2) * 2;
    let mut wide = vec![0u8; (a1 - a0) as usize];
    file.reader().read_at(a0, &mut wide)?;
    for pair in wide.chunks_exact_mut(2) {
        pair.swap(0, 1);
    }
    let skip = (lo - a0) as usize;
    buf.clear();
    buf.extend_from_slice(&wide[skip..skip + len as usize]);
    Ok(Endian::Big)
}

/// Turn big-endian value bytes into `endian` order, in place.
pub(crate) fn order_bytes(big: &mut [u8], endian: Endian) {
    match endian {
        Endian::Big => {}
        Endian::Little => big.reverse(),
        Endian::Pair => {
            for pair in big.chunks_exact_mut(2) {
                pair.swap(0, 1);
            }
        }
    }
}

pub(crate) fn put_u16(buf: &mut [u8], off: usize, value: u16, endian: Endian) {
    let mut b = value.to_be_bytes();
    order_bytes(&mut b, endian);
    buf[off..off + 2].copy_from_slice(&b);
}

pub(crate) fn put_i16(buf: &mut [u8], off: usize, value: i16, endian: Endian) {
    put_u16(buf, off, value as u16, endian);
}

pub(crate) fn put_u32(buf: &mut [u8], off: usize, value: u32, endian: Endian) {
    let mut b = value.to_be_bytes();
    order_bytes(&mut b, endian);
    buf[off..off + 4].copy_from_slice(&b);
}

pub(crate) fn put_i32(buf: &mut [u8], off: usize, value: i32, endian: Endian) {
    put_u32(buf, off, value as u32, endian);
}

/// Decode a buffer of whole samples on every core.
pub(crate) fn par_decode(
    raw: &[u8],
    format: SampleFormat,
    endian: Endian,
    gain: Option<crate::sample::Gain>,
) -> Samples {
    let width = format.width();
    let n = raw.len() / width;
    let per = (n / (rayon::current_num_threads() * 4)).max(4096);
    let per = per + per % 2;
    let parts: Vec<Samples> = raw[..n * width]
        .par_chunks(per * width)
        .map(|chunk| {
            let k = chunk.len() / width;
            let mut out = sample::empty_samples(format, k);
            if k > 0 {
                sample::decode_trace(chunk, format, endian, gain, 0, k, &mut out);
            }
            out
        })
        .collect();
    let mut acc = sample::empty_samples(format, 0);
    for part in parts {
        extend_samples(&mut acc, part);
    }
    acc
}

pub(crate) fn extend_samples(acc: &mut Samples, part: Samples) {
    match (acc, part) {
        (Samples::I8(a), Samples::I8(b)) => a.extend(b),
        (Samples::U8(a), Samples::U8(b)) => a.extend(b),
        (Samples::I16(a), Samples::I16(b)) => a.extend(b),
        (Samples::U16(a), Samples::U16(b)) => a.extend(b),
        (Samples::I32(a), Samples::I32(b)) => a.extend(b),
        (Samples::U32(a), Samples::U32(b)) => a.extend(b),
        (Samples::I64(a), Samples::I64(b)) => a.extend(b),
        (Samples::U64(a), Samples::U64(b)) => a.extend(b),
        (Samples::F32(a), Samples::F32(b)) => a.extend(b),
        (Samples::F64(a), Samples::F64(b)) => a.extend(b),
        _ => unreachable!("mixed sample buffers"),
    }
}

/// Call `f` with every decoded sample widened to f64.
pub(crate) fn for_each_f64(samples: &Samples, mut f: impl FnMut(usize, f64)) {
    macro_rules! each {
        ($v:expr) => {
            for (i, x) in $v.iter().enumerate() {
                f(i, *x as f64)
            }
        };
    }
    match samples {
        Samples::I8(v) => each!(v),
        Samples::U8(v) => each!(v),
        Samples::I16(v) => each!(v),
        Samples::U16(v) => each!(v),
        Samples::I32(v) => each!(v),
        Samples::U32(v) => each!(v),
        Samples::I64(v) => each!(v),
        Samples::U64(v) => each!(v),
        Samples::F32(v) => each!(v),
        Samples::F64(v) => each!(v),
    }
}

/// Rounding direction when a value is stored in a sample format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Round {
    /// Largest representable value not above the input.
    Down,
    /// Smallest representable value not below the input.
    Up,
    Nearest,
}

/// Encode one value in the on-disk sample format and byte order.
pub(crate) fn encode_value(
    value: f64,
    format: SampleFormat,
    endian: Endian,
    round: Round,
) -> Result<Vec<u8>> {
    if format == SampleFormat::FixedGain {
        return Err(invalid("format 4 samples cannot be encoded"));
    }
    if endian == Endian::Pair && format.width() % 2 == 1 {
        return Err(invalid(
            "pairwise byte order with an odd sample width cannot be patched per sample",
        ));
    }
    let int = |lo: f64, hi: f64| -> Result<i128> {
        if value.is_nan() {
            return Err(invalid(format!("{} cannot hold NaN", format.name())));
        }
        let r = match round {
            Round::Down => value.floor(),
            Round::Up => value.ceil(),
            Round::Nearest => value.round(),
        };
        Ok(r.clamp(lo, hi) as i128)
    };
    let mut big: Vec<u8> = match format {
        SampleFormat::IbmFloat32 => ibm_from_f64(value, round)?.to_be_bytes().to_vec(),
        SampleFormat::IeeeFloat32 => f32_toward(value, round).to_bits().to_be_bytes().to_vec(),
        SampleFormat::IeeeFloat64 => value.to_bits().to_be_bytes().to_vec(),
        SampleFormat::Int8 => (int(-128.0, 127.0)? as i8).to_be_bytes().to_vec(),
        SampleFormat::Uint8 => (int(0.0, 255.0)? as u8).to_be_bytes().to_vec(),
        SampleFormat::Int16 => (int(-32768.0, 32767.0)? as i16).to_be_bytes().to_vec(),
        SampleFormat::Uint16 => (int(0.0, 65535.0)? as u16).to_be_bytes().to_vec(),
        SampleFormat::Int24 => (int(-8_388_608.0, 8_388_607.0)? as i32).to_be_bytes()[1..].to_vec(),
        SampleFormat::Uint24 => (int(0.0, 16_777_215.0)? as u32).to_be_bytes()[1..].to_vec(),
        SampleFormat::Int32 => (int(i32::MIN as f64, i32::MAX as f64)? as i32)
            .to_be_bytes()
            .to_vec(),
        SampleFormat::Uint32 => (int(0.0, u32::MAX as f64)? as u32).to_be_bytes().to_vec(),
        SampleFormat::Int64 => (int(i64::MIN as f64, i64::MAX as f64)? as i64)
            .to_be_bytes()
            .to_vec(),
        SampleFormat::Uint64 => (int(0.0, u64::MAX as f64)? as u64).to_be_bytes().to_vec(),
        SampleFormat::FixedGain => unreachable!("refused above"),
    };
    order_bytes(&mut big, endian);
    Ok(big)
}

/// Nearest f32 in the requested direction.
fn f32_toward(value: f64, round: Round) -> f32 {
    let near = value as f32;
    if value.is_nan() || f64::from(near) == value {
        return near;
    }
    let step = |x: f32, up: bool| -> f32 {
        if x == 0.0 {
            let tiny = f32::from_bits(1);
            return if up { tiny } else { -tiny };
        }
        let bits = x.to_bits();
        let away = (x > 0.0) == up;
        f32::from_bits(if away { bits + 1 } else { bits - 1 })
    };
    match round {
        Round::Nearest => near,
        Round::Down if f64::from(near) > value => step(near, false),
        Round::Up if f64::from(near) < value => step(near, true),
        _ => near,
    }
}

/// IBM System/360 single precision: sign, excess-64 base-16 exponent, 24-bit fraction.
///
/// Values beyond the IBM range are an error. Values below it become zero.
fn ibm_from_f64(value: f64, round: Round) -> Result<u32> {
    if !value.is_finite() {
        return Err(invalid("IBM float cannot hold NaN or infinity"));
    }
    if value == 0.0 {
        return Ok(0);
    }
    let sign = if value < 0.0 { 0x8000_0000u32 } else { 0 };
    let mag = value.abs();
    // Magnitude rounding: toward zero, away from zero, or nearest.
    let up = match round {
        Round::Nearest => None,
        Round::Down => Some(value < 0.0),
        Round::Up => Some(value > 0.0),
    };
    // 16^(e-1) <= mag < 16^e, so mag / 16^e is in [1/16, 1).
    let mut e = (mag.log2() / 4.0).floor() as i32 + 1;
    loop {
        let frac = mag / 16f64.powi(e);
        if frac >= 1.0 {
            e += 1;
            continue;
        }
        if frac < 1.0 / 16.0 {
            e -= 1;
            continue;
        }
        let scaled = frac * 16_777_216.0;
        let mut f = match up {
            None => scaled.round(),
            Some(true) => scaled.ceil(),
            Some(false) => scaled.floor(),
        } as u64;
        let mut exp = e;
        if f >= 1 << 24 {
            f >>= 4;
            exp += 1;
        }
        let biased = exp + 64;
        if biased > 127 {
            return Err(invalid(format!("{value} is beyond the IBM float range")));
        }
        if biased < 0 {
            return Ok(sign);
        }
        return Ok(sign | ((biased as u32) << 24) | f as u32);
    }
}

/// One header column viewed through its stored type.
pub(crate) enum Col<'a> {
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    F64(&'a [f64]),
}

impl<'a> Col<'a> {
    pub(crate) fn find(cols: &'a HeaderColumns, name: &str) -> Result<Self> {
        if let Ok(v) = cols.i32(name) {
            return Ok(Self::I32(v));
        }
        if let Ok(v) = cols.i16(name) {
            return Ok(Self::I16(v));
        }
        if let Ok(v) = cols.u16(name) {
            return Ok(Self::U16(v));
        }
        if let Ok(v) = cols.u32(name) {
            return Ok(Self::U32(v));
        }
        if let Ok(v) = cols.i64(name) {
            return Ok(Self::I64(v));
        }
        cols.f64(name).map(Self::F64)
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub(crate) fn f64_at(&self, i: usize) -> f64 {
        match self {
            Self::I16(v) => f64::from(v[i]),
            Self::I32(v) => f64::from(v[i]),
            Self::I64(v) => v[i] as f64,
            Self::U16(v) => f64::from(v[i]),
            Self::U32(v) => f64::from(v[i]),
            Self::F64(v) => v[i],
        }
    }

    /// Order-preserving u64 key: comparing keys compares the values.
    pub(crate) fn key_at(&self, i: usize) -> u64 {
        const SIGN: u64 = 1 << 63;
        match self {
            Self::I16(v) => (i64::from(v[i]) as u64) ^ SIGN,
            Self::I32(v) => (i64::from(v[i]) as u64) ^ SIGN,
            Self::I64(v) => (v[i] as u64) ^ SIGN,
            Self::U16(v) => u64::from(v[i]),
            Self::U32(v) => u64::from(v[i]),
            Self::F64(v) => {
                let bits = v[i].to_bits();
                if bits & SIGN != 0 {
                    !bits
                } else {
                    bits | SIGN
                }
            }
        }
    }
}

/// Fill `n` output traces of `out_len` bytes with `fill` in parallel chunks and
/// hand each chunk to the sink in order.
///
/// Two chunk buffers are live: one being written to the sink while the next is
/// filled on the other cores. Together they stay within `budget`. A trace that
/// alone exceeds half the budget is still processed one at a time.
pub(crate) fn stream<S, F>(
    n: usize,
    out_len: usize,
    header_len: usize,
    budget: Budget,
    sink: &mut S,
    fill: F,
) -> Result<()>
where
    S: TraceSink + Send,
    F: Fn(usize, &mut [u8]) -> Result<()> + Sync,
{
    if n == 0 || out_len == 0 {
        return Ok(());
    }
    let per = ((budget.get() / 2) / out_len as u64).max(1) as usize;
    let per = per.min(n);
    let mut cur = vec![0u8; per * out_len];
    let mut next = vec![0u8; per * out_len];
    let fill_chunk = |buf: &mut [u8], first: usize| -> Result<()> {
        buf.par_chunks_mut(out_len)
            .enumerate()
            .map(|(i, out)| fill(first + i, out))
            .find_first(Result::is_err)
            .unwrap_or(Ok(()))
    };
    let mut start = 0usize;
    let mut len = per.min(n);
    fill_chunk(&mut cur[..len * out_len], 0)?;
    loop {
        let next_start = start + len;
        if next_start >= n {
            return sink.write_block(&cur[..len * out_len], out_len, header_len);
        }
        let next_len = per.min(n - next_start);
        let (wrote, filled) = rayon::join(
            || sink.write_block(&cur[..len * out_len], out_len, header_len),
            || fill_chunk(&mut next[..next_len * out_len], next_start),
        );
        wrote?;
        filled?;
        std::mem::swap(&mut cur, &mut next);
        start = next_start;
        len = next_len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::ibm_to_f32;

    #[test]
    fn ibm_encode_round_trips() {
        for v in [1.0f64, -1.0, 0.5, 118.625, -3.25e-5, 7.0e30, 1.0e-30] {
            let word = ibm_from_f64(v, Round::Nearest).unwrap();
            let back = f64::from(ibm_to_f32(word));
            assert!(
                (back - v).abs() <= v.abs() * 1e-6,
                "{v} -> {word:#x} -> {back}"
            );
        }
        assert_eq!(ibm_from_f64(1.0, Round::Nearest).unwrap(), 0x4110_0000);
        assert_eq!(ibm_from_f64(-1.0, Round::Nearest).unwrap(), 0xc110_0000);
        let hi = 0.1f64;
        assert!(f64::from(ibm_to_f32(ibm_from_f64(hi, Round::Down).unwrap())) <= hi);
        assert!(f64::from(ibm_to_f32(ibm_from_f64(hi, Round::Up).unwrap())) >= hi);
        assert!(f64::from(ibm_to_f32(ibm_from_f64(-hi, Round::Down).unwrap())) <= -hi);
    }

    #[test]
    fn f32_rounding_respects_direction() {
        let v = 0.1f64;
        assert!(f64::from(f32_toward(v, Round::Down)) <= v);
        assert!(f64::from(f32_toward(v, Round::Up)) >= v);
        assert!(f64::from(f32_toward(-v, Round::Down)) <= -v);
        assert!(f64::from(f32_toward(-v, Round::Up)) >= -v);
    }
}
