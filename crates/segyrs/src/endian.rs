/// Byte order of the binary header, trace headers, and samples.
///
/// Rev 2 stores the order in binary-header bytes 3297-3300. The constant
/// `0x01020304` is written in the file's own order, so the raw bytes are
/// `01 02 03 04` (big), `04 03 02 01` (little), or `02 01 04 03` (pairwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
    /// Consecutive pairs of bytes are swapped relative to big-endian.
    Pair,
}

impl Endian {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Big => "big",
            Self::Little => "little",
            Self::Pair => "pairwise",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "big" | "be" | "msb" => Some(Self::Big),
            "little" | "le" | "lsb" => Some(Self::Little),
            "pairwise" | "pair" => Some(Self::Pair),
            _ => None,
        }
    }
}

impl std::fmt::Display for Endian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn u16_at(buf: &[u8], off: usize, endian: Endian) -> u16 {
    let b = [buf[off], buf[off + 1]];
    match endian {
        Endian::Big => u16::from_be_bytes(b),
        Endian::Little | Endian::Pair => u16::from_le_bytes(b),
    }
}

pub fn i16_at(buf: &[u8], off: usize, endian: Endian) -> i16 {
    u16_at(buf, off, endian) as i16
}

pub fn u32_at(buf: &[u8], off: usize, endian: Endian) -> u32 {
    let b = [buf[off], buf[off + 1], buf[off + 2], buf[off + 3]];
    match endian {
        Endian::Big => u32::from_be_bytes(b),
        Endian::Little => u32::from_le_bytes(b),
        Endian::Pair => u32::from_be_bytes([b[1], b[0], b[3], b[2]]),
    }
}

pub fn i32_at(buf: &[u8], off: usize, endian: Endian) -> i32 {
    u32_at(buf, off, endian) as i32
}

pub fn u64_at(buf: &[u8], off: usize, endian: Endian) -> u64 {
    let b = [
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ];
    match endian {
        Endian::Big => u64::from_be_bytes(b),
        Endian::Little => u64::from_le_bytes(b),
        Endian::Pair => u64::from_be_bytes([b[1], b[0], b[3], b[2], b[5], b[4], b[7], b[6]]),
    }
}

pub fn i64_at(buf: &[u8], off: usize, endian: Endian) -> i64 {
    u64_at(buf, off, endian) as i64
}

pub fn f64_at(buf: &[u8], off: usize, endian: Endian) -> f64 {
    f64::from_bits(u64_at(buf, off, endian))
}

/// Rev 2 byte-order signature. `None` means the field is empty or not a known pattern.
pub fn signature(buf: &[u8]) -> Option<Endian> {
    if buf.len() < 3300 {
        return None;
    }
    match &buf[3296..3300] {
        [1, 2, 3, 4] => Some(Endian::Big),
        [4, 3, 2, 1] => Some(Endian::Little),
        [2, 1, 4, 3] => Some(Endian::Pair),
        _ => None,
    }
}
