//! Structural validation of a SEG-Y file (S-34).

use crate::pio::FileExt;
use std::fs::File as FsFile;
use std::path::Path;

use rayon::prelude::*;

use crate::endian;
use crate::error::{Error, Result};
use crate::file::{Layout, OpenOptions, ReadAt};

use super::binary::{BIN_BYTES, FIXED_LENGTH, INTERVAL, REV_MAJOR, TEXT_BYTES, TRACE_HEADER};
use super::edit::structure;

struct Whole<'f>(&'f FsFile);

impl ReadAt for Whole<'_> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.0.read_exact_at(buf, offset)?;
        Ok(())
    }
}

/// Check the structure of the file at `path` and return its layout.
///
/// Checks: the binary-header format code is known; the textual blocks are
/// whole 3200-byte blocks and the extended count (or the `EndText` stanza)
/// frames them; the file length is the header blocks plus `n_traces * stride`
/// exactly; the binary sample count and interval are nonzero when there are
/// traces; the revision and fixed-length flag hold legal values; and every
/// trace header's sample count (bytes 115-116) is zero or equal to the binary
/// header's. All problems are collected into one [`Error::Invalid`].
///
/// A file that passes is consistent in sizes and codes. That alone does not
/// prove that every other program accepts it.
pub fn validate(path: impl AsRef<Path>) -> Result<Layout> {
    let path = path.as_ref();
    let file = FsFile::open(path).map_err(Error::path("open", path))?;
    let meta = file.metadata().map_err(Error::path("open", path))?;
    if meta.is_dir() {
        return Err(Error::path("open", path)(crate::file::is_a_directory()));
    }
    let len = meta.len();
    let parsed = match structure(Whole(&file), len, &OpenOptions::default()) {
        Ok(parsed) => parsed,
        Err(Error::Io(err)) => return Err(Error::Io(err)),
        Err(other) => {
            return Err(Error::Invalid {
                problems: vec![other.to_string()],
            })
        }
    };
    let layout = parsed.layout().clone();
    let bin = parsed.binary_header_bytes().to_vec();
    drop(parsed);
    let e = layout.source_endian;
    let mut problems = Vec::new();

    let header_len = (1 + u64::from(layout.extra_trace_header_blocks)) * TRACE_HEADER as u64;
    let stride = header_len + u64::from(layout.n_samples) * layout.sample_format.width() as u64;
    let expected = layout.data_start + layout.n_traces * stride;
    if expected != len {
        problems.push(format!(
            "file is {len} bytes; headers plus {} traces of {stride} bytes is {expected}",
            layout.n_traces
        ));
    }
    if (layout.data_start - (TEXT_BYTES + BIN_BYTES) as u64) % TEXT_BYTES as u64 != 0 {
        problems.push(format!(
            "first trace at byte {} is not after whole 3200-byte textual blocks",
            layout.data_start
        ));
    }
    if layout.n_traces > 0 && layout.n_samples == 0 {
        problems.push("binary header sample count is 0".into());
    }
    if layout.n_traces > 0
        && layout.sample_interval_us == 0
        && endian::u16_at(&bin, INTERVAL, e) == 0
    {
        problems.push("binary header sample interval is 0".into());
    }
    if bin[REV_MAJOR] > 2 {
        problems.push(format!("revision major byte is {}", bin[REV_MAJOR]));
    }
    let fixed = endian::u16_at(&bin, FIXED_LENGTH, e);
    if bin[REV_MAJOR] >= 1 && fixed > 1 {
        problems.push(format!("fixed-length flag is {fixed}, not 0 or 1"));
    }

    if layout.n_samples <= 65535 && layout.n_traces > 0 && expected == len {
        let ns16 = layout.n_samples as u16;
        let bad: Vec<(u64, u16)> = (0..layout.n_traces)
            .into_par_iter()
            .filter_map(|t| {
                let mut word = [0u8; 2];
                let at = layout.data_start + t * stride + 114;
                match file.read_exact_at(&mut word, at) {
                    Ok(()) => {
                        let ns = endian::u16_at(&word, 0, e);
                        (ns != 0 && ns != ns16).then_some((t, ns))
                    }
                    Err(_) => Some((t, 0)),
                }
            })
            .collect();
        if let Some(&(t, ns)) = bad.first() {
            problems.push(format!(
                "{} trace headers disagree with the binary sample count {}; first is trace {t} with {ns}",
                bad.len(),
                layout.n_samples
            ));
        }
    }

    if problems.is_empty() {
        Ok(layout)
    } else {
        Err(Error::Invalid { problems })
    }
}
