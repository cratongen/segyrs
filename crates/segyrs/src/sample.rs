use crate::endian::Endian;

/// SEG-Y sample format codes from rev 0 through rev 2.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    IbmFloat32,
    Int32,
    Int16,
    FixedGain,
    IeeeFloat32,
    IeeeFloat64,
    Int24,
    Int8,
    Int64,
    Uint32,
    Uint16,
    Uint64,
    Uint24,
    Uint8,
}

impl SampleFormat {
    pub fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            1 => Self::IbmFloat32,
            2 => Self::Int32,
            3 => Self::Int16,
            4 => Self::FixedGain,
            5 => Self::IeeeFloat32,
            6 => Self::IeeeFloat64,
            7 => Self::Int24,
            8 => Self::Int8,
            9 => Self::Int64,
            10 => Self::Uint32,
            11 => Self::Uint16,
            12 => Self::Uint64,
            15 => Self::Uint24,
            16 => Self::Uint8,
            _ => return None,
        })
    }

    pub fn code(self) -> u16 {
        match self {
            Self::IbmFloat32 => 1,
            Self::Int32 => 2,
            Self::Int16 => 3,
            Self::FixedGain => 4,
            Self::IeeeFloat32 => 5,
            Self::IeeeFloat64 => 6,
            Self::Int24 => 7,
            Self::Int8 => 8,
            Self::Int64 => 9,
            Self::Uint32 => 10,
            Self::Uint16 => 11,
            Self::Uint64 => 12,
            Self::Uint24 => 15,
            Self::Uint8 => 16,
        }
    }

    pub fn width(self) -> usize {
        match self {
            Self::IbmFloat32 | Self::Int32 | Self::FixedGain | Self::IeeeFloat32 | Self::Uint32 => {
                4
            }
            Self::Int16 | Self::Uint16 => 2,
            Self::IeeeFloat64 | Self::Int64 | Self::Uint64 => 8,
            Self::Int24 | Self::Uint24 => 3,
            Self::Int8 | Self::Uint8 => 1,
        }
    }

    /// Name stored on the Dataset. IBM stays a float32 with this label, not a silent cast.
    pub fn name(self) -> &'static str {
        match self {
            Self::IbmFloat32 => "ibm_float32",
            Self::Int32 => "int32",
            Self::Int16 => "int16",
            Self::FixedGain => "fixed_point_gain",
            Self::IeeeFloat32 => "ieee_float32",
            Self::IeeeFloat64 => "ieee_float64",
            Self::Int24 => "int24",
            Self::Int8 => "int8",
            Self::Int64 => "int64",
            Self::Uint32 => "uint32",
            Self::Uint16 => "uint16",
            Self::Uint64 => "uint64",
            Self::Uint24 => "uint24",
            Self::Uint8 => "uint8",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        let code = match text {
            "1" | "ibm" | "ibm32" | "ibm_float32" => 1,
            "2" | "int32" => 2,
            "3" | "int16" => 3,
            "4" | "fixed_point_gain" | "gain" => 4,
            "5" | "ieee32" | "ieee_float32" | "float32" => 5,
            "6" | "ieee64" | "ieee_float64" | "float64" => 6,
            "7" | "int24" => 7,
            "8" | "int8" => 8,
            "9" | "int64" => 9,
            "10" | "uint32" => 10,
            "11" | "uint16" => 11,
            "12" | "uint64" => 12,
            "15" | "uint24" => 15,
            "16" | "uint8" => 16,
            _ => return None,
        };
        Self::from_code(code)
    }
}

/// Which power of two to apply to format 4. Rev 0 and rev 2 disagree on the sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gain {
    /// `magnitude * 2^gain`, the rev 0 / rev 1 diagram.
    Rev0,
    /// `magnitude * 2^-gain`.
    Rev2,
}

impl Gain {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "rev0" | "rev1" => Some(Self::Rev0),
            "rev2" => Some(Self::Rev2),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rev0 => "rev0",
            Self::Rev2 => "rev2",
        }
    }
}

/// IBM 32-bit float to IEEE binary32.
///
/// Overflow is infinity. A zero fraction is zero. Nothing in this mapping is a NaN.
pub fn ibm_to_f32(word: u32) -> f32 {
    let sign = word & 0x8000_0000;
    let exp = ((word >> 24) & 0x7f) as i32;
    let frac = word & 0x00ff_ffff;
    if frac == 0 {
        return f32::from_bits(sign);
    }
    // value = frac * 2^(4*(exp-64) - 24)
    let mut power = 4 * (exp - 64) - 24;
    let mut mag = frac;
    let highest = 31 - mag.leading_zeros();
    if highest < 23 {
        let shift = 23 - highest;
        mag <<= shift;
        power -= shift as i32;
    }
    let biased = power + 23 + 127;
    if biased >= 255 {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    if biased <= 0 {
        return subnormal(sign, mag, power);
    }
    f32::from_bits(sign | ((biased as u32) << 23) | (mag & 0x007f_ffff))
}

fn subnormal(sign: u32, mag: u32, power: i32) -> f32 {
    // mag * 2^power, expressed in units of 2^-149.
    let shift = power + 149;
    if shift < -31 {
        return f32::from_bits(sign);
    }
    let magnitude = if shift >= 0 {
        mag << shift
    } else {
        let down = (-shift) as u32;
        let half = 1u32 << (down - 1);
        let rem = mag & ((1u32 << down) - 1);
        let mut kept = mag >> down;
        if rem > half || (rem == half && kept & 1 == 1) {
            kept += 1;
        }
        kept
    };
    if magnitude >= 0x0080_0000 {
        return f32::from_bits(sign | (1 << 23));
    }
    f32::from_bits(sign | magnitude)
}

/// Format 4, after the four bytes have been restored to the rev 1 byte diagram.
pub fn decode_fixed_gain(logical: [u8; 4], gain: Gain) -> f64 {
    let g = i32::from(logical[1]);
    let sign = if logical[2] & 0x80 == 0 { 1.0 } else { -1.0 };
    let magnitude = (u16::from(logical[2] & 0x7f) << 8) | u16::from(logical[3]);
    let exp = match gain {
        Gain::Rev0 => g,
        Gain::Rev2 => -g,
    };
    sign * f64::from(magnitude) * 2.0_f64.powi(exp)
}

pub fn logical_word(raw: &[u8], endian: Endian) -> [u8; 4] {
    match endian {
        Endian::Big => [raw[0], raw[1], raw[2], raw[3]],
        Endian::Little => [raw[3], raw[2], raw[1], raw[0]],
        Endian::Pair => [raw[1], raw[0], raw[3], raw[2]],
    }
}

fn i24(raw: &[u8], endian: Endian) -> i32 {
    let u = match endian {
        Endian::Big => (u32::from(raw[0]) << 16) | (u32::from(raw[1]) << 8) | u32::from(raw[2]),
        Endian::Little => (u32::from(raw[2]) << 16) | (u32::from(raw[1]) << 8) | u32::from(raw[0]),
        Endian::Pair => {
            let swapped = pair_swap(raw);
            (u32::from(swapped[0]) << 16) | (u32::from(swapped[1]) << 8) | u32::from(swapped[2])
        }
    };
    if u & 0x0080_0000 == 0 {
        u as i32
    } else {
        (u | 0xff00_0000) as i32
    }
}

fn u24(raw: &[u8], endian: Endian) -> u32 {
    match endian {
        Endian::Big => (u32::from(raw[0]) << 16) | (u32::from(raw[1]) << 8) | u32::from(raw[2]),
        Endian::Little => (u32::from(raw[2]) << 16) | (u32::from(raw[1]) << 8) | u32::from(raw[0]),
        Endian::Pair => {
            let swapped = pair_swap(raw);
            (u32::from(swapped[0]) << 16) | (u32::from(swapped[1]) << 8) | u32::from(swapped[2])
        }
    }
}

fn pair_swap(raw: &[u8]) -> Vec<u8> {
    let mut out = raw.to_vec();
    let mut i = 0;
    while i + 1 < out.len() {
        out.swap(i, i + 1);
        i += 2;
    }
    out
}

#[derive(Clone, Debug)]
pub enum Samples {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub fn decode_trace(
    samples: &[u8],
    format: SampleFormat,
    endian: Endian,
    gain: Option<Gain>,
    dst_index: usize,
    ns: usize,
    out: &mut Samples,
) {
    let width = format.width();
    let bytes = &samples[..ns * width];
    let r = dst_index..dst_index + ns;
    let dst = match out {
        Samples::I8(v) => SamplesMut::I8(&mut v[r]),
        Samples::U8(v) => SamplesMut::U8(&mut v[r]),
        Samples::I16(v) => SamplesMut::I16(&mut v[r]),
        Samples::U16(v) => SamplesMut::U16(&mut v[r]),
        Samples::I32(v) => SamplesMut::I32(&mut v[r]),
        Samples::U32(v) => SamplesMut::U32(&mut v[r]),
        Samples::I64(v) => SamplesMut::I64(&mut v[r]),
        Samples::U64(v) => SamplesMut::U64(&mut v[r]),
        Samples::F32(v) => SamplesMut::F32(&mut v[r]),
        Samples::F64(v) => SamplesMut::F64(&mut v[r]),
    };
    decode_into(bytes, format, endian, gain, dst);
}

/// A borrowed output slice of one native dtype.
pub enum SamplesMut<'a> {
    I8(&'a mut [i8]),
    U8(&'a mut [u8]),
    I16(&'a mut [i16]),
    U16(&'a mut [u16]),
    I32(&'a mut [i32]),
    U32(&'a mut [u32]),
    I64(&'a mut [i64]),
    U64(&'a mut [u64]),
    F32(&'a mut [f32]),
    F64(&'a mut [f64]),
}

/// Decode `out.len()` samples from `bytes` (file order) into native values.
///
/// The byte order is matched once per call, not once per sample, so each loop
/// body is a plain load, swap, and store.
pub fn decode_into(
    bytes: &[u8],
    format: SampleFormat,
    endian: Endian,
    gain: Option<Gain>,
    out: SamplesMut<'_>,
) {
    let width = format.width();
    if endian == Endian::Pair && width % 2 == 1 {
        let swapped = pair_swap(bytes);
        decode_into(&swapped, format, Endian::Big, gain, out);
        return;
    }
    macro_rules! words {
        ($out:expr, $n:literal, $ty:ty, $map:expr) => {{
            let it = bytes.chunks_exact($n).zip($out.iter_mut());
            match endian {
                Endian::Big => {
                    for (c, o) in it {
                        *o = $map(<$ty>::from_be_bytes(c.try_into().unwrap()));
                    }
                }
                Endian::Little => {
                    for (c, o) in it {
                        *o = $map(<$ty>::from_le_bytes(c.try_into().unwrap()));
                    }
                }
                Endian::Pair => {
                    for (c, o) in it {
                        let mut w: [u8; $n] = c.try_into().unwrap();
                        for p in w.chunks_exact_mut(2) {
                            p.swap(0, 1);
                        }
                        *o = $map(<$ty>::from_be_bytes(w));
                    }
                }
            }
        }};
    }
    match (format, out) {
        (SampleFormat::IbmFloat32, SamplesMut::F32(v)) => words!(v, 4, u32, ibm_to_f32),
        (SampleFormat::IeeeFloat32, SamplesMut::F32(v)) => words!(v, 4, f32, |x| x),
        (SampleFormat::IeeeFloat64, SamplesMut::F64(v)) => words!(v, 8, f64, |x| x),
        (SampleFormat::FixedGain, SamplesMut::F64(v)) => {
            let gain = gain.expect("format 4 gain checked before decode");
            for (chunk, o) in bytes.chunks_exact(4).zip(v.iter_mut()) {
                *o = decode_fixed_gain(logical_word(chunk, endian), gain);
            }
        }
        (SampleFormat::Int32, SamplesMut::I32(v)) => words!(v, 4, i32, |x| x),
        (SampleFormat::Uint32, SamplesMut::U32(v)) => words!(v, 4, u32, |x| x),
        (SampleFormat::Int16, SamplesMut::I16(v)) => words!(v, 2, i16, |x| x),
        (SampleFormat::Uint16, SamplesMut::U16(v)) => words!(v, 2, u16, |x| x),
        (SampleFormat::Int64, SamplesMut::I64(v)) => words!(v, 8, i64, |x| x),
        (SampleFormat::Uint64, SamplesMut::U64(v)) => words!(v, 8, u64, |x| x),
        (SampleFormat::Int8, SamplesMut::I8(v)) => {
            for (b, o) in bytes.iter().zip(v.iter_mut()) {
                *o = *b as i8;
            }
        }
        (SampleFormat::Uint8, SamplesMut::U8(v)) => v.copy_from_slice(&bytes[..v.len()]),
        (SampleFormat::Int24, SamplesMut::I32(v)) => {
            for (chunk, o) in bytes.chunks_exact(3).zip(v.iter_mut()) {
                *o = i24(chunk, endian);
            }
        }
        (SampleFormat::Uint24, SamplesMut::U32(v)) => {
            for (chunk, o) in bytes.chunks_exact(3).zip(v.iter_mut()) {
                *o = u24(chunk, endian);
            }
        }
        _ => unreachable!("sample buffer type does not match the format"),
    }
}

impl Samples {
    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_i8(&self) -> Option<&[i8]> {
        match self {
            Self::I8(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i16(&self) -> Option<&[i16]> {
        match self {
            Self::I16(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Self::U16(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            Self::I32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<&[u32]> {
        match self {
            Self::U32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<&[i64]> {
        match self {
            Self::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<&[u64]> {
        match self {
            Self::U64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::F64(v) => Some(v),
            _ => None,
        }
    }
}

pub fn empty_samples(format: SampleFormat, len: usize) -> Samples {
    match format {
        SampleFormat::IbmFloat32 | SampleFormat::IeeeFloat32 => Samples::F32(vec![0.0; len]),
        SampleFormat::IeeeFloat64 | SampleFormat::FixedGain => Samples::F64(vec![0.0; len]),
        SampleFormat::Int32 | SampleFormat::Int24 => Samples::I32(vec![0; len]),
        SampleFormat::Uint32 | SampleFormat::Uint24 => Samples::U32(vec![0; len]),
        SampleFormat::Int16 => Samples::I16(vec![0; len]),
        SampleFormat::Uint16 => Samples::U16(vec![0; len]),
        SampleFormat::Int64 => Samples::I64(vec![0; len]),
        SampleFormat::Uint64 => Samples::U64(vec![0; len]),
        SampleFormat::Int8 => Samples::I8(vec![0; len]),
        SampleFormat::Uint8 => Samples::U8(vec![0; len]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s_01_ibm_overflow_is_inf_not_nan() {
        for word in [0x6110_0000, 0x6110_0001, 0x611f_ffff, 0x6120_0000] {
            let value = ibm_to_f32(word);
            assert!(
                value.is_infinite() && value.is_sign_positive(),
                "{word:#x} -> {value}"
            );
            assert!(!value.is_nan());
        }
        assert_eq!(ibm_to_f32(0), 0.0);
        assert_eq!(ibm_to_f32(0x4110_0000), 1.0);
        assert_eq!(ibm_to_f32(0xc110_0000), -1.0);
        assert!(ibm_to_f32(0x8000_0000).is_sign_negative());
    }

    #[test]
    fn s_02_fixed_gain_rev0_and_rev2() {
        // byte1 unused, byte2 gain=3, magnitude=5
        let word = [0x00, 0x03, 0x00, 0x05];
        assert!((decode_fixed_gain(word, Gain::Rev0) - 40.0).abs() < 1e-12);
        assert!((decode_fixed_gain(word, Gain::Rev2) - 0.625).abs() < 1e-12);
        let neg = [0x00, 0x01, 0x80, 0x02];
        assert!((decode_fixed_gain(neg, Gain::Rev0) - -4.0).abs() < 1e-12);
    }
}
