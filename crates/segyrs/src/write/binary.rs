//! Binary-header and textual-header building blocks for the writer.

use crate::ebcdic;
use crate::endian::Endian;
use crate::error::{Error, Result};
use crate::file::Revision;
use crate::header::{FieldValue, HeaderSpec};
use crate::text::TextEncoding;

use super::encode::{put_field, reorder_word};

pub(crate) const TEXT_BYTES: usize = 3200;
pub(crate) const BIN_BYTES: usize = 400;
pub(crate) const TRACE_HEADER: usize = 240;

// Offsets inside the 400-byte binary header (file byte minus 3201).
pub(crate) const INTERVAL: usize = 16;
pub(crate) const NS: usize = 20;
pub(crate) const FORMAT: usize = 24;
pub(crate) const EXT_NS: usize = 68;
pub(crate) const EXT_INTERVAL: usize = 72;
pub(crate) const SIGNATURE: usize = 96;
pub(crate) const REV_MAJOR: usize = 300;
pub(crate) const REV_MINOR: usize = 301;
pub(crate) const FIXED_LENGTH: usize = 302;
pub(crate) const EXT_TEXT: usize = 304;
pub(crate) const EXTRA_HEADERS: usize = 306;
pub(crate) const N_TRACES: usize = 312;
pub(crate) const FIRST_TRACE: usize = 320;

/// Multi-byte words of the rev 2.1 binary header, as (offset, width).
/// The endian signature at 96 is rewritten, not reordered.
const BINARY_WORDS: &[(usize, usize)] = &[
    (0, 4),
    (4, 4),
    (8, 4),
    (12, 2),
    (14, 2),
    (16, 2),
    (18, 2),
    (20, 2),
    (22, 2),
    (24, 2),
    (26, 2),
    (28, 2),
    (30, 2),
    (32, 2),
    (34, 2),
    (36, 2),
    (38, 2),
    (40, 2),
    (42, 2),
    (44, 2),
    (46, 2),
    (48, 2),
    (50, 2),
    (52, 2),
    (54, 2),
    (56, 2),
    (58, 2),
    (60, 4),
    (64, 4),
    (68, 4),
    (72, 8),
    (80, 8),
    (88, 4),
    (92, 4),
    (302, 2),
    (304, 2),
    (306, 4),
    (310, 2),
    (312, 8),
    (320, 8),
    (328, 4),
];

pub(crate) fn revision_bytes(revision: Revision) -> (u8, u8) {
    match revision {
        Revision::Rev0 => (0, 0),
        Revision::Rev1 => (1, 0),
        Revision::Rev20 => (2, 0),
        Revision::Rev21 => (2, 1),
    }
}

/// Move every known binary-header word from one byte order to another.
/// Unassigned bytes are copied as they are.
pub(crate) fn reorder_binary(bin: &mut [u8], from: Endian, to: Endian) {
    if from == to {
        return;
    }
    let signed = binary_signature(bin).is_some();
    for &(off, width) in BINARY_WORDS {
        reorder_word(&mut bin[off..off + width], from, to);
    }
    if signed {
        put_signature(bin, to);
    }
}

/// The rev 2 byte-order signature inside a 400-byte binary header.
pub(crate) fn binary_signature(bin: &[u8]) -> Option<Endian> {
    match &bin[SIGNATURE..SIGNATURE + 4] {
        [1, 2, 3, 4] => Some(Endian::Big),
        [4, 3, 2, 1] => Some(Endian::Little),
        [2, 1, 4, 3] => Some(Endian::Pair),
        _ => None,
    }
}

pub(crate) fn put_signature(bin: &mut [u8], endian: Endian) {
    put_field(
        &mut bin[SIGNATURE..SIGNATURE + 4],
        FieldValue::U32(0x0102_0304),
        endian,
    );
}

/// Move every standard trace-header word (the rev 1 / rev 2.1 field list) from
/// one byte order to another. Bytes outside that list are copied as they are.
pub(crate) fn reorder_trace_header(
    header: &mut [u8],
    words: &[(usize, usize)],
    from: Endian,
    to: Endian,
) {
    if from == to {
        return;
    }
    for &(off, width) in words {
        reorder_word(&mut header[off..off + width], from, to);
    }
}

pub(crate) fn standard_words() -> Vec<(usize, usize)> {
    HeaderSpec::standard()
        .fields
        .iter()
        .map(|f| (usize::from(f.byte - 1), f.dtype.width()))
        .collect()
}

pub(crate) fn put_u16(bin: &mut [u8], off: usize, v: u16, endian: Endian) {
    put_field(&mut bin[off..off + 2], FieldValue::U16(v), endian);
}

pub(crate) fn put_i16(bin: &mut [u8], off: usize, v: i16, endian: Endian) {
    put_field(&mut bin[off..off + 2], FieldValue::I16(v), endian);
}

pub(crate) fn put_u32(bin: &mut [u8], off: usize, v: u32, endian: Endian) {
    put_field(&mut bin[off..off + 4], FieldValue::U32(v), endian);
}

pub(crate) fn put_u64(bin: &mut [u8], off: usize, v: u64, endian: Endian) {
    put_field(&mut bin[off..off + 8], FieldValue::I64(v as i64), endian);
}

pub(crate) fn put_f64(bin: &mut [u8], off: usize, v: f64, endian: Endian) {
    put_field(&mut bin[off..off + 8], FieldValue::F64(v), endian);
}

/// One 3200-byte textual block from up to 40 cards of up to 80 characters.
///
/// Each card is padded with spaces to 80 characters. Missing cards are blank.
/// EBCDIC uses code page 037. A character with no byte in the chosen encoding
/// is an error, not a silent `?`.
pub fn text_block(cards: &[&str], encoding: TextEncoding) -> Result<Vec<u8>> {
    if cards.len() > 40 {
        return Err(Error::Spec(format!(
            "a textual block holds 40 cards, got {}",
            cards.len()
        )));
    }
    let mut block = Vec::with_capacity(TEXT_BYTES);
    for (i, card) in cards.iter().enumerate() {
        let n = card.chars().count();
        if n > 80 {
            return Err(Error::Spec(format!(
                "textual card {} has {n} characters, the limit is 80",
                i + 1
            )));
        }
        for ch in card.chars() {
            block.push(encode_char(ch, encoding).ok_or_else(|| {
                Error::Spec(format!(
                    "textual card {}: {ch:?} has no {} byte",
                    i + 1,
                    encoding.as_str()
                ))
            })?);
        }
        let pad = encode_char(' ', encoding).expect("space encodes");
        block.extend(std::iter::repeat_n(pad, 80 - n));
    }
    let pad = encode_char(' ', encoding).expect("space encodes");
    block.resize(TEXT_BYTES, pad);
    Ok(block)
}

fn encode_char(ch: char, encoding: TextEncoding) -> Option<u8> {
    match encoding {
        TextEncoding::Ascii => u8::try_from(u32::from(ch)).ok().filter(|b| *b < 0x80),
        TextEncoding::Ebcdic => ebcdic::CP037
            .iter()
            .position(|&c| u32::from(c) == u32::from(ch))
            .map(|b| b as u8),
    }
}

/// The default textual header: cards `C 1` to `C40`, the revision on card 39
/// for rev 1 and later, and the end marker on card 40.
pub fn default_text(revision: Revision, encoding: TextEncoding) -> Vec<u8> {
    let mut cards: Vec<String> = (1..=40).map(|i| format!("C{i:2}")).collect();
    match revision {
        Revision::Rev0 => {}
        Revision::Rev1 => cards[38] = "C39 SEG Y REV1".into(),
        Revision::Rev20 => cards[38] = "C39 SEG-Y_REV2.0".into(),
        Revision::Rev21 => cards[38] = "C39 SEG-Y_REV2.1".into(),
    }
    cards[39] = if revision == Revision::Rev0 || revision == Revision::Rev1 {
        "C40 END EBCDIC".into()
    } else {
        "C40 END TEXTUAL HEADER".into()
    };
    let refs: Vec<&str> = cards.iter().map(String::as_str).collect();
    text_block(&refs, encoding).expect("default cards are plain ASCII")
}
