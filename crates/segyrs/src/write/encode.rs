//! Sample and header-word encoding: the inverse of `sample::decode_trace`.
//!
//! Every encoder writes big-endian first. Little-endian reverses each word.
//! Pairwise swaps each pair of bytes in the encoded trace, the same operation
//! the reader undoes, so odd widths (int24, int8) follow the reader's rule.

use crate::endian::Endian;
use crate::error::{Error, Result};
use crate::header::{FieldValue, HeaderDtype, HeaderField};
use crate::sample::{Gain, SampleFormat, Samples};

/// Largest IBM magnitude: fraction `0xFFFFFF`, exponent 127.
pub const IBM_MAX: u32 = 0x7fff_ffff;

/// A borrowed sample buffer, trace-major, in any numeric type.
///
/// The type that `File::decode` returns for a format is that format's natural
/// type: f32 for IBM and IEEE32, f64 for IEEE64 and format 4, i32 for int32
/// and int24, u32 for uint32 and uint24, and the matching width otherwise.
/// Any other pairing is a conversion with the rules in [`encode_samples`].
#[derive(Clone, Copy, Debug)]
pub enum SampleData<'a> {
    I8(&'a [i8]),
    U8(&'a [u8]),
    I16(&'a [i16]),
    U16(&'a [u16]),
    I32(&'a [i32]),
    U32(&'a [u32]),
    I64(&'a [i64]),
    U64(&'a [u64]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl<'a> From<&'a Samples> for SampleData<'a> {
    fn from(samples: &'a Samples) -> Self {
        match samples {
            Samples::I8(v) => Self::I8(v),
            Samples::U8(v) => Self::U8(v),
            Samples::I16(v) => Self::I16(v),
            Samples::U16(v) => Self::U16(v),
            Samples::I32(v) => Self::I32(v),
            Samples::U32(v) => Self::U32(v),
            Samples::I64(v) => Self::I64(v),
            Samples::U64(v) => Self::U64(v),
            Samples::F32(v) => Self::F32(v),
            Samples::F64(v) => Self::F64(v),
        }
    }
}

macro_rules! each {
    ($self:expr, $v:ident => $body:expr) => {
        match $self {
            SampleData::I8($v) => $body,
            SampleData::U8($v) => $body,
            SampleData::I16($v) => $body,
            SampleData::U16($v) => $body,
            SampleData::I32($v) => $body,
            SampleData::U32($v) => $body,
            SampleData::I64($v) => $body,
            SampleData::U64($v) => $body,
            SampleData::F32($v) => $body,
            SampleData::F64($v) => $body,
        }
    };
}

impl<'a> SampleData<'a> {
    pub fn len(&self) -> usize {
        each!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Samples `start..end` of the same buffer.
    pub fn slice(&self, start: usize, end: usize) -> SampleData<'a> {
        match *self {
            SampleData::I8(v) => SampleData::I8(&v[start..end]),
            SampleData::U8(v) => SampleData::U8(&v[start..end]),
            SampleData::I16(v) => SampleData::I16(&v[start..end]),
            SampleData::U16(v) => SampleData::U16(&v[start..end]),
            SampleData::I32(v) => SampleData::I32(&v[start..end]),
            SampleData::U32(v) => SampleData::U32(&v[start..end]),
            SampleData::I64(v) => SampleData::I64(&v[start..end]),
            SampleData::U64(v) => SampleData::U64(&v[start..end]),
            SampleData::F32(v) => SampleData::F32(&v[start..end]),
            SampleData::F64(v) => SampleData::F64(&v[start..end]),
        }
    }

    /// Name of the element type, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            SampleData::I8(_) => "int8",
            SampleData::U8(_) => "uint8",
            SampleData::I16(_) => "int16",
            SampleData::U16(_) => "uint16",
            SampleData::I32(_) => "int32",
            SampleData::U32(_) => "uint32",
            SampleData::I64(_) => "int64",
            SampleData::U64(_) => "uint64",
            SampleData::F32(_) => "float32",
            SampleData::F64(_) => "float64",
        }
    }
}

/// IEEE binary32 to IBM single precision, rounded to nearest, ties to even.
///
/// Every finite f32 is inside the IBM range, so only the precision changes:
/// IBM keeps 21 to 24 significant bits depending on the leading hex digit.
/// Infinity saturates to the largest IBM magnitude ([`IBM_MAX`] with the sign),
/// which `ibm_to_f32` reads back as infinity. NaN has no IBM pattern and
/// returns `None`. Negative zero is `0x80000000`.
pub fn f32_to_ibm(value: f32) -> Option<u32> {
    f64_to_ibm(f64::from(value))
}

/// IEEE binary64 to IBM single precision, rounded to nearest, ties to even.
///
/// A finite value above the IBM range (about 7.2e75) returns `None`; it is not
/// clamped. Infinity saturates as in [`f32_to_ibm`]. Values below the smallest
/// normalized IBM number use exponent 0 with an unnormalized fraction and
/// round to zero below `2^-280`.
pub fn f64_to_ibm(value: f64) -> Option<u32> {
    if value.is_nan() {
        return None;
    }
    let sign = if value.is_sign_negative() {
        0x8000_0000
    } else {
        0
    };
    if value == 0.0 {
        return Some(sign);
    }
    if value.is_infinite() {
        return Some(sign | IBM_MAX);
    }
    let bits = value.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    // |value| = m * 2^p with m an integer.
    let (m, p) = if raw_exp == 0 {
        (frac, -1074)
    } else {
        (frac | (1 << 52), raw_exp - 1075)
    };
    let len = 64 - m.leading_zeros() as i32;
    let high = p + len - 1; // |value| is in [2^high, 2^(high+1))
                            // IBM value = F * 2^q, q = 4*(E-64) - 24, F < 2^24. A normalized F has its
                            // top bit at 20..=23, so q is the multiple of 4 in [high-23, high-20].
    let mut q = (high - 23 + 3).div_euclid(4) * 4;
    let mut e = (q + 24) / 4 + 64;
    if e < 0 {
        e = 0;
        q = -280;
    }
    let mut f = round_shift(m, p - q)?;
    if f == 1 << 24 {
        f = 1 << 20;
        e += 1;
    }
    if e > 127 {
        return None;
    }
    if f == 0 {
        return Some(sign);
    }
    Some(sign | ((e as u32) << 24) | f as u32)
}

/// `m * 2^shift` rounded to nearest, ties to even. `None` if it does not fit u64.
fn round_shift(m: u64, shift: i32) -> Option<u64> {
    if shift >= 0 {
        if shift >= 64 || m.leading_zeros() < shift as u32 {
            return None;
        }
        return Some(m << shift);
    }
    let down = (-shift) as u32;
    if down > 64 {
        return Some(0);
    }
    if down == 64 {
        // m < 2^64: the result is 0 or 1 (m > 2^63, or m == 2^63 ties to even 0).
        return Some(u64::from(m > 1 << 63));
    }
    let kept = m >> down;
    let rem = m & ((1u64 << down) - 1);
    let half = 1u64 << (down - 1);
    Some(if rem > half || (rem == half && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    })
}

/// Format 4 logical word `[0, gain, sign|mag_hi, mag_lo]` for `value`.
///
/// Picks the exponent that keeps the most precision and rounds the 15-bit
/// magnitude to nearest, ties to even. `None` if the value is NaN, infinite,
/// or outside the range the convention can reach.
pub fn encode_fixed_gain(value: f64, gain: Gain) -> Option<[u8; 4]> {
    if !value.is_finite() {
        return None;
    }
    let sign: u8 = if value.is_sign_negative() { 0x80 } else { 0 };
    let mag = value.abs();
    // value = m * 2^k; rev 0 has k = g >= 0, rev 2 has k = -g <= 0.
    let (lo, hi) = match gain {
        Gain::Rev0 => (0, 255),
        Gain::Rev2 => (-255, 0),
    };
    let mut k = if mag == 0.0 {
        lo.max(0).min(hi)
    } else {
        // smallest k with mag / 2^k < 32768
        let mut k = (mag.log2().floor() as i32) - 14;
        while mag / 2f64.powi(k) >= 32767.5 {
            k += 1;
        }
        while k > lo && mag / 2f64.powi(k - 1) < 32767.5 {
            k -= 1;
        }
        k.clamp(lo, hi)
    };
    loop {
        let scaled = mag / 2f64.powi(k);
        let m = round_half_even(scaled);
        if m <= 32767.0 {
            let m = m as u16;
            let g = match gain {
                Gain::Rev0 => k,
                Gain::Rev2 => -k,
            };
            return Some([0, g as u8, sign | (m >> 8) as u8, m as u8]);
        }
        if k >= hi {
            return None;
        }
        k += 1;
    }
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// One sample that cannot be stored in the target format.
#[derive(Debug)]
pub(crate) struct BadSample {
    pub index: usize,
    pub value: String,
}

impl BadSample {
    pub(crate) fn into_error(self, trace: u64, format: SampleFormat) -> Error {
        Error::SampleValue {
            trace,
            sample: self.index as u64,
            value: self.value,
            format: format.name(),
        }
    }
}

/// Encode `data` as `format` in `endian` into `out`, which holds exactly
/// `data.len() * format.width()` bytes.
///
/// The natural type of a format (see [`SampleData`]) is stored exactly, apart
/// from IBM, which rounds an f32 to nearest. Other pairings convert:
/// - float targets (IBM, IEEE32, IEEE64, format 4) round to nearest, ties to
///   even. A finite value beyond the target range is an error, and so is NaN
///   for IBM and format 4. Infinity saturates for IBM.
/// - integer targets accept only values that are whole and in range. A
///   fraction, NaN, infinity, or an out-of-range value is an error. Nothing
///   is truncated or clamped.
pub(crate) fn encode_samples(
    data: SampleData<'_>,
    format: SampleFormat,
    endian: Endian,
    gain: Option<Gain>,
    out: &mut [u8],
) -> std::result::Result<(), BadSample> {
    let width = format.width();
    debug_assert_eq!(out.len(), data.len() * width);
    let little = endian == Endian::Little;
    match (format, data) {
        (SampleFormat::IbmFloat32, SampleData::F32(v)) => {
            for (i, (dst, &x)) in out.chunks_exact_mut(4).zip(v).enumerate() {
                let word = f32_to_ibm(x).ok_or_else(|| bad(i, x))?;
                put(dst, word.to_be_bytes(), little);
            }
        }
        (SampleFormat::IeeeFloat32, SampleData::F32(v)) => {
            for (dst, &x) in out.chunks_exact_mut(4).zip(v) {
                put(dst, x.to_bits().to_be_bytes(), little);
            }
        }
        (SampleFormat::IeeeFloat64, SampleData::F64(v)) => {
            for (dst, &x) in out.chunks_exact_mut(8).zip(v) {
                put(dst, x.to_bits().to_be_bytes(), little);
            }
        }
        (SampleFormat::Int32, SampleData::I32(v)) => {
            for (dst, &x) in out.chunks_exact_mut(4).zip(v) {
                put(dst, x.to_be_bytes(), little);
            }
        }
        (SampleFormat::Uint32, SampleData::U32(v)) => {
            for (dst, &x) in out.chunks_exact_mut(4).zip(v) {
                put(dst, x.to_be_bytes(), little);
            }
        }
        (SampleFormat::Int16, SampleData::I16(v)) => {
            for (dst, &x) in out.chunks_exact_mut(2).zip(v) {
                put(dst, x.to_be_bytes(), little);
            }
        }
        (SampleFormat::Uint16, SampleData::U16(v)) => {
            for (dst, &x) in out.chunks_exact_mut(2).zip(v) {
                put(dst, x.to_be_bytes(), little);
            }
        }
        (SampleFormat::Int64, SampleData::I64(v)) => {
            for (dst, &x) in out.chunks_exact_mut(8).zip(v) {
                put(dst, x.to_be_bytes(), little);
            }
        }
        (SampleFormat::Uint64, SampleData::U64(v)) => {
            for (dst, &x) in out.chunks_exact_mut(8).zip(v) {
                put(dst, x.to_be_bytes(), little);
            }
        }
        (SampleFormat::Int8, SampleData::I8(v)) => {
            for (dst, &x) in out.iter_mut().zip(v) {
                *dst = x as u8;
            }
        }
        (SampleFormat::Uint8, SampleData::U8(v)) => out.copy_from_slice(v),
        _ => each!(data, v => encode_converted(v, format, little, gain, out)?),
    }
    if endian == Endian::Pair {
        pair_swap(out);
    }
    Ok(())
}

fn bad(index: usize, value: impl std::fmt::Display) -> BadSample {
    BadSample {
        index,
        value: value.to_string(),
    }
}

#[inline]
fn put<const N: usize>(dst: &mut [u8], mut be: [u8; N], little: bool) {
    if little {
        be.reverse();
    }
    dst.copy_from_slice(&be);
}

/// Swap each pair of bytes. A trailing odd byte stays, as in the reader.
pub(crate) fn pair_swap(buf: &mut [u8]) {
    for pair in buf.chunks_exact_mut(2) {
        pair.swap(0, 1);
    }
}

/// A sample value before it meets the target format.
#[derive(Clone, Copy)]
pub(crate) enum Num {
    Int(i128),
    Float(f64),
}

impl std::fmt::Display for Num {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Num::Int(i) => write!(f, "{i}"),
            Num::Float(x) => write!(f, "{x}"),
        }
    }
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(x) => x,
        }
    }

    /// The exact integer, or `None` for a fraction, NaN, or infinity.
    pub(crate) fn as_int(self) -> Option<i128> {
        match self {
            Num::Int(i) => Some(i),
            Num::Float(x) => {
                if x.is_finite() && x.fract() == 0.0 && x.abs() < 1.7e38 {
                    Some(x as i128)
                } else {
                    None
                }
            }
        }
    }
}

pub(crate) trait ToNum: Copy {
    fn num(self) -> Num;
}

macro_rules! to_num_int {
    ($($t:ty),*) => {$(
        impl ToNum for $t {
            #[inline]
            fn num(self) -> Num { Num::Int(self as i128) }
        }
    )*};
}
to_num_int!(i8, u8, i16, u16, i32, u32, i64, u64);

impl ToNum for f32 {
    #[inline]
    fn num(self) -> Num {
        Num::Float(f64::from(self))
    }
}

impl ToNum for f64 {
    #[inline]
    fn num(self) -> Num {
        Num::Float(self)
    }
}

fn encode_converted<T: ToNum + std::fmt::Display>(
    v: &[T],
    format: SampleFormat,
    little: bool,
    gain: Option<Gain>,
    out: &mut [u8],
) -> std::result::Result<(), BadSample> {
    let width = format.width();
    for (i, (dst, &x)) in out.chunks_exact_mut(width).zip(v).enumerate() {
        let n = x.num();
        let fail = || bad(i, x);
        match format {
            SampleFormat::IbmFloat32 => {
                let word = f64_to_ibm(n.as_f64()).ok_or_else(fail)?;
                put(dst, word.to_be_bytes(), little);
            }
            SampleFormat::IeeeFloat32 => {
                let f = n.as_f64();
                let g = f as f32;
                if f.is_finite() && g.is_infinite() {
                    return Err(fail());
                }
                put(dst, g.to_bits().to_be_bytes(), little);
            }
            SampleFormat::IeeeFloat64 => put(dst, n.as_f64().to_bits().to_be_bytes(), little),
            SampleFormat::FixedGain => {
                let gain = gain.ok_or_else(|| BadSample {
                    index: i,
                    value: "format 4 needs a gain convention".into(),
                })?;
                let word = encode_fixed_gain(n.as_f64(), gain).ok_or_else(fail)?;
                put(dst, word, little);
            }
            _ => {
                let int = n.as_int().ok_or_else(fail)?;
                put_int(dst, int, format, little).ok_or_else(fail)?;
            }
        }
    }
    Ok(())
}

fn put_int(dst: &mut [u8], v: i128, format: SampleFormat, little: bool) -> Option<()> {
    match format {
        SampleFormat::Int8 => dst[0] = i8::try_from(v).ok()? as u8,
        SampleFormat::Uint8 => dst[0] = u8::try_from(v).ok()?,
        SampleFormat::Int16 => put(dst, i16::try_from(v).ok()?.to_be_bytes(), little),
        SampleFormat::Uint16 => put(dst, u16::try_from(v).ok()?.to_be_bytes(), little),
        SampleFormat::Int32 => put(dst, i32::try_from(v).ok()?.to_be_bytes(), little),
        SampleFormat::Uint32 => put(dst, u32::try_from(v).ok()?.to_be_bytes(), little),
        SampleFormat::Int64 => put(dst, i64::try_from(v).ok()?.to_be_bytes(), little),
        SampleFormat::Uint64 => put(dst, u64::try_from(v).ok()?.to_be_bytes(), little),
        SampleFormat::Int24 => {
            if !(-(1 << 23)..1 << 23).contains(&v) {
                return None;
            }
            let b = (v as i32).to_be_bytes();
            put(dst, [b[1], b[2], b[3]], little);
        }
        SampleFormat::Uint24 => {
            if !(0..1 << 24).contains(&v) {
                return None;
            }
            let b = (v as u32).to_be_bytes();
            put(dst, [b[1], b[2], b[3]], little);
        }
        _ => return None,
    }
    Some(())
}

/// Turn file-order sample bytes into big-endian order, or back. The operation
/// is its own inverse: reverse each word, or swap each byte pair of the trace.
pub(crate) fn reorder_samples(buf: &mut [u8], width: usize, endian: Endian) {
    match endian {
        Endian::Big => {}
        Endian::Little => {
            if width > 1 {
                for word in buf.chunks_exact_mut(width) {
                    word.reverse();
                }
            }
        }
        Endian::Pair => pair_swap(buf),
    }
}

/// Values for one trace-header field, one per trace, in any integer or float type.
#[derive(Clone, Copy, Debug)]
pub enum HeaderValues<'a> {
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    F64(&'a [f64]),
    /// Coordinates in real units, stored as the integer that `apply_scalar`
    /// turns back into the value: a negative scalar multiplies on write, a
    /// positive scalar divides, and zero means 1 (X-24).
    Scaled {
        values: &'a [f64],
        scalars: Scalars<'a>,
    },
    /// One value on every trace.
    Fill(i64),
}

/// The scalar behind [`HeaderValues::Scaled`].
#[derive(Clone, Copy, Debug)]
pub enum Scalars<'a> {
    One(i16),
    PerTrace(&'a [i16]),
}

impl HeaderValues<'_> {
    /// Number of values, `None` for [`HeaderValues::Fill`].
    pub fn count(&self) -> Option<usize> {
        Some(match self {
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Scaled { values, .. } => values.len(),
            Self::Fill(_) => return None,
        })
    }

    /// The `i`th value, with the scalar already inverted.
    fn get(&self, i: usize) -> std::result::Result<Num, String> {
        Ok(match *self {
            Self::I8(v) => Num::Int(v[i].into()),
            Self::I16(v) => Num::Int(v[i].into()),
            Self::I32(v) => Num::Int(v[i].into()),
            Self::I64(v) => Num::Int(v[i].into()),
            Self::U16(v) => Num::Int(v[i].into()),
            Self::U32(v) => Num::Int(v[i].into()),
            Self::F64(v) => Num::Float(v[i]),
            Self::Fill(x) => Num::Int(x.into()),
            Self::Scaled { values, scalars } => {
                let scalar = match scalars {
                    Scalars::One(s) => s,
                    Scalars::PerTrace(s) => s[i],
                };
                let value = values[i];
                let raw = match scalar {
                    0 => value,
                    s if s > 0 => value / f64::from(s),
                    s => value * -f64::from(s),
                };
                let whole = raw.round();
                let tol = 1e-6_f64.max(raw.abs() * 4.0 * f64::EPSILON);
                if !raw.is_finite() || (raw - whole).abs() > tol {
                    return Err(format!(
                        "{value} (scalar {scalar} gives {raw}, not a whole number; use a finer scalar)"
                    ));
                }
                Num::Float(whole)
            }
        })
    }
}

fn dtype_name(dtype: HeaderDtype) -> &'static str {
    match dtype {
        HeaderDtype::I8 => "int8",
        HeaderDtype::I16 => "int16",
        HeaderDtype::I32 => "int32",
        HeaderDtype::I64 => "int64",
        HeaderDtype::U16 => "uint16",
        HeaderDtype::U32 => "uint32",
        HeaderDtype::F64 => "float64",
    }
}

pub(crate) fn field_label(field: &HeaderField) -> String {
    match &field.name {
        Some(name) => format!("{name} (byte {})", field.byte),
        None => format!("byte {}", field.byte),
    }
}

/// Encode `values[i]` as `field.dtype`, in `endian`, into `dst` (`dtype.width()` bytes).
pub(crate) fn encode_header_value(
    values: &HeaderValues<'_>,
    i: usize,
    field: &HeaderField,
    endian: Endian,
    dst: &mut [u8],
) -> Result<()> {
    let fail = |value: String| Error::HeaderValue {
        field: field_label(field),
        value,
        dtype: dtype_name(field.dtype),
    };
    let num = values.get(i).map_err(fail)?;
    let value = match field.dtype {
        HeaderDtype::F64 => match num {
            Num::Float(x) => FieldValue::F64(x),
            Num::Int(n) => {
                let x = n as f64;
                if x as i128 != n {
                    return Err(fail(num.to_string()));
                }
                FieldValue::F64(x)
            }
        },
        dtype => {
            let n = num.as_int().ok_or_else(|| fail(num.to_string()))?;
            let out = match dtype {
                HeaderDtype::I8 => i8::try_from(n).ok().map(FieldValue::I8),
                HeaderDtype::I16 => i16::try_from(n).ok().map(FieldValue::I16),
                HeaderDtype::I32 => i32::try_from(n).ok().map(FieldValue::I32),
                HeaderDtype::I64 => i64::try_from(n).ok().map(FieldValue::I64),
                HeaderDtype::U16 => u16::try_from(n).ok().map(FieldValue::U16),
                HeaderDtype::U32 => u32::try_from(n).ok().map(FieldValue::U32),
                HeaderDtype::F64 => unreachable!(),
            };
            out.ok_or_else(|| fail(num.to_string()))?
        }
    };
    put_field(dst, value, endian);
    Ok(())
}

/// Store one header word in file order. `dst` is exactly the word.
///
/// The byte orders mirror the reader: little reverses the word, pairwise
/// swaps each byte pair of the big-endian word, and one byte is never swapped.
pub fn put_field(dst: &mut [u8], value: FieldValue, endian: Endian) {
    let little = endian == Endian::Little;
    match value {
        FieldValue::I8(x) => dst[0] = x as u8,
        FieldValue::I16(x) => put(dst, x.to_be_bytes(), little),
        FieldValue::U16(x) => put(dst, x.to_be_bytes(), little),
        FieldValue::I32(x) => put(dst, x.to_be_bytes(), little),
        FieldValue::U32(x) => put(dst, x.to_be_bytes(), little),
        FieldValue::I64(x) => put(dst, x.to_be_bytes(), little),
        FieldValue::F64(x) => put(dst, x.to_bits().to_be_bytes(), little),
    }
    if endian == Endian::Pair && dst.len() > 1 {
        pair_swap(dst);
    }
}

/// Move a word of `width` bytes from one byte order to another, in place.
pub(crate) fn reorder_word(word: &mut [u8], from: Endian, to: Endian) {
    if from == to || word.len() < 2 {
        return;
    }
    reorder_samples(word, word.len(), from);
    reorder_samples(word, word.len(), to);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::ibm_to_f32;

    #[test]
    fn ibm_known_words() {
        assert_eq!(f32_to_ibm(1.0), Some(0x4110_0000));
        assert_eq!(f32_to_ibm(-1.0), Some(0xc110_0000));
        assert_eq!(f32_to_ibm(0.5), Some(0x4080_0000));
        assert_eq!(f32_to_ibm(16.0), Some(0x4210_0000));
        assert_eq!(f32_to_ibm(-118.625), Some(0xc276_a000));
        assert_eq!(f32_to_ibm(0.0), Some(0));
        assert_eq!(f32_to_ibm(-0.0), Some(0x8000_0000));
        assert_eq!(f32_to_ibm(f32::INFINITY), Some(IBM_MAX));
        assert_eq!(f32_to_ibm(f32::NEG_INFINITY), Some(0xffff_ffff));
        assert_eq!(f32_to_ibm(f32::NAN), None);
        assert_eq!(f64_to_ibm(1e80), None);
    }

    #[test]
    fn fixed_gain_round_trip() {
        for gain in [Gain::Rev0, Gain::Rev2] {
            for v in [0.0, 1.0, -3.0, 32767.0, 40.0] {
                let w = encode_fixed_gain(v, gain).unwrap();
                assert_eq!(crate::sample::decode_fixed_gain(w, gain), v, "{v} {gain:?}");
            }
        }
        let w = encode_fixed_gain(0.625, Gain::Rev2).unwrap();
        assert_eq!(crate::sample::decode_fixed_gain(w, Gain::Rev2), 0.625);
        assert!(encode_fixed_gain(0.5, Gain::Rev0).is_some());
        assert!(encode_fixed_gain(f64::NAN, Gain::Rev0).is_none());
    }

    #[test]
    fn ibm_decode_encode_identity_on_normalized_words() {
        // Every exponent that stays a normal f32, several fractions each.
        for exp in 0u32..128 {
            for frac in [
                0x10_0000u32,
                0x10_0001,
                0x7f_ffff,
                0x80_0000,
                0xff_ffff,
                0x12_3456,
            ] {
                for sign in [0, 0x8000_0000u32] {
                    let word = sign | (exp << 24) | frac;
                    let x = ibm_to_f32(word);
                    if !x.is_normal() {
                        continue;
                    }
                    assert_eq!(f32_to_ibm(x), Some(word), "{word:#010x} -> {x}");
                }
            }
        }
    }
}
