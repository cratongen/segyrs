//! Opt-in header-index sidecar (performance.md, Sidecar index).
//!
//! A full header scan stores its columns in `<dir>/<path hash>.segyrs-index`.
//! A later open of the same file loads them instead of reading every trace
//! header. The sidecar holds the file length, its modification time, and a
//! fingerprint of the file head, the first and last trace headers, and the
//! scan parameters. Any difference makes it stale, and a stale or unreadable
//! sidecar is ignored: the open scans and replaces it. Deleting it is always
//! safe. A failure to write it does not fail the open.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::file::TraceCols;

const MAGIC: &[u8; 8] = b"SEGYRSIX";
const VERSION: u32 = 1;
/// Magic, version, length, mtime seconds and nanoseconds, fingerprint, traces.
const HEAD: usize = 8 + 4 + 8 + 8 + 4 + 8 + 8;
/// Nine i32 columns, two i16, one u32.
const PER_TRACE: usize = 9 * 4 + 2 * 2 + 4;

/// 64-bit FNV-1a: stable across Rust versions, unlike `DefaultHasher`.
pub(crate) struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// The sidecar location and the identity of the file it describes.
pub(crate) struct IndexCache {
    sidecar: PathBuf,
    len: u64,
    mtime: (i64, u32),
}

impl IndexCache {
    /// `None` when the file's metadata cannot be read: no sidecar is used.
    pub fn for_file(dir: &Path, path: &Path, len: u64) -> Option<Self> {
        let modified = fs::metadata(path).ok()?.modified().ok()?;
        let mtime = match modified.duration_since(UNIX_EPOCH) {
            Ok(d) => (d.as_secs() as i64, d.subsec_nanos()),
            Err(e) => (
                -(e.duration().as_secs() as i64),
                e.duration().subsec_nanos(),
            ),
        };
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut hash = Fnv::default();
        hash.write(canonical.as_os_str().as_encoded_bytes());
        Some(Self {
            sidecar: dir.join(format!("{:016x}.segyrs-index", hash.finish())),
            len,
            mtime,
        })
    }

    fn head(&self, fingerprint: u64, ntraces: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEAD);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&self.mtime.0.to_le_bytes());
        out.extend_from_slice(&self.mtime.1.to_le_bytes());
        out.extend_from_slice(&fingerprint.to_le_bytes());
        out.extend_from_slice(&(ntraces as u64).to_le_bytes());
        out
    }

    /// The stored columns, when the sidecar exists and matches this file.
    pub fn load(&self, fingerprint: u64, ntraces: usize) -> Option<TraceCols> {
        let bytes = fs::read(&self.sidecar).ok()?;
        let want = self.head(fingerprint, ntraces);
        if bytes.len() != HEAD + ntraces.checked_mul(PER_TRACE)? || bytes[..HEAD] != want[..] {
            return None;
        }
        let mut at = HEAD;
        let mut take = |width: usize| {
            let part = &bytes[at..at + width * ntraces];
            at += width * ntraces;
            part
        };
        fn i32s(b: &[u8]) -> Vec<i32> {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        fn i16s(b: &[u8]) -> Vec<i16> {
            b.chunks_exact(2)
                .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        Some(TraceCols {
            inline: i32s(take(4)),
            crossline: i32s(take(4)),
            offset: i32s(take(4)),
            scalco: i16s(take(2)),
            cdp_x: i32s(take(4)),
            cdp_y: i32s(take(4)),
            source_x: i32s(take(4)),
            source_y: i32s(take(4)),
            group_x: i32s(take(4)),
            group_y: i32s(take(4)),
            trace_id: i16s(take(2)),
            declared_ns: take(4)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        })
    }

    /// Write the sidecar through a temporary file and a rename, so a reader
    /// never sees half of one. Errors are dropped: the sidecar is a cache.
    pub fn store(&self, fingerprint: u64, cols: &TraceCols) {
        let n = cols.inline.len();
        let mut out = self.head(fingerprint, n);
        out.reserve(n * PER_TRACE);
        for col in [&cols.inline, &cols.crossline, &cols.offset] {
            col.iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
        }
        cols.scalco
            .iter()
            .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
        for col in [
            &cols.cdp_x,
            &cols.cdp_y,
            &cols.source_x,
            &cols.source_y,
            &cols.group_x,
            &cols.group_y,
        ] {
            col.iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
        }
        cols.trace_id
            .iter()
            .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
        cols.declared_ns
            .iter()
            .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
        let Some(dir) = self.sidecar.parent() else {
            return;
        };
        let tmp = self
            .sidecar
            .with_extension(format!("tmp{}", std::process::id()));
        let written = fs::create_dir_all(dir)
            .and_then(|()| fs::write(&tmp, &out))
            .and_then(|()| fs::rename(&tmp, &self.sidecar));
        if written.is_err() {
            let _ = fs::remove_file(&tmp);
        }
    }
}
