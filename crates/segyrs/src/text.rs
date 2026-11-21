use crate::ebcdic;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextEncoding {
    Ascii,
    Ebcdic,
}

impl TextEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ascii => "ascii",
            Self::Ebcdic => "ebcdic",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextHeader {
    pub encoding: TextEncoding,
    pub cards: Vec<String>,
    /// The 3200 bytes this block was split from, unchanged.
    pub raw: Vec<u8>,
}

pub fn decode_block(block: &[u8]) -> TextHeader {
    let encoding = detect(block);
    let cards = block
        .chunks(80)
        .map(|card| match encoding {
            TextEncoding::Ascii => card.iter().map(|&b| char::from(b)).collect(),
            TextEncoding::Ebcdic => ebcdic::decode(card),
        })
        .collect();
    TextHeader {
        encoding,
        cards,
        raw: block.to_vec(),
    }
}

/// Rev 1 allows ASCII. The first card byte is `C`: ASCII `0x43`, EBCDIC `0xC3`.
pub fn detect(block: &[u8]) -> TextEncoding {
    match block.first().copied() {
        Some(0x43) => TextEncoding::Ascii,
        Some(0xc3) => TextEncoding::Ebcdic,
        _ => {
            let printable = block.iter().filter(|&&b| (0x20..0x7f).contains(&b)).count();
            if block.is_empty() || printable * 2 > block.len() {
                TextEncoding::Ascii
            } else {
                TextEncoding::Ebcdic
            }
        }
    }
}

/// `((SEG: EndText))` closes the extended textual headers. Writers vary the
/// case and the blanks (`((  seg: endTEXt  ))`), so both are ignored.
pub fn is_end_text(block: &[u8]) -> bool {
    has_end_text(block.iter().copied())
        || has_end_text(
            block
                .iter()
                .map(|&b| u8::try_from(ebcdic::CP037[b as usize]).unwrap_or(0)),
        )
}

fn has_end_text(ascii: impl Iterator<Item = u8>) -> bool {
    const NEEDLE: &[u8] = b"SEG:ENDTEXT";
    let mut window = [0u8; NEEDLE.len()];
    let mut filled = 0;
    for b in ascii.filter(|b| !b.is_ascii_whitespace()) {
        window.copy_within(1.., 0);
        window[NEEDLE.len() - 1] = b.to_ascii_uppercase();
        filled += 1;
        if filled >= NEEDLE.len() && window == NEEDLE {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(text: &str) -> Vec<u8> {
        let mut b = text.as_bytes().to_vec();
        b.resize(3200, b' ');
        b
    }

    #[test]
    fn end_text_ignores_case_and_blanks_in_both_encodings() {
        for text in [
            "((SEG: EndText))",
            "((  seg: endTEXt  ))",
            "((SEG:ENDTEXT))",
            "C40 ((SEG : End Text))",
        ] {
            let ascii = block(text);
            assert!(is_end_text(&ascii), "{text}");
            assert!(
                is_end_text(&ebcdic::encode(&String::from_utf8(ascii).unwrap())),
                "{text}"
            );
        }
        for text in ["((SEG: Data))", "SEG EndText", "END TEXT", ""] {
            let ascii = block(text);
            assert!(!is_end_text(&ascii), "{text}");
            assert!(
                !is_end_text(&ebcdic::encode(&String::from_utf8(ascii).unwrap())),
                "{text}"
            );
        }
    }
}
