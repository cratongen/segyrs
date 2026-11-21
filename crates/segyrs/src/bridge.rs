//! A [`ReadAt`] handed between separately compiled extension modules.
//!
//! The Python package builds its `segyrs` extension and the optional viewer
//! extension as separate shared libraries. They cannot share Rust
//! types, so a reader crosses between them as a [`RawReader`]: a `#[repr(C)]`
//! table of `extern "C"` functions around an opaque, reference-counted
//! context. [`RawReader::export`] builds one from any reader and
//! [`ForeignReader::new`] turns one back into a [`ReadAt`]. Both sides must
//! be built from the same segyrs version: [`RawReader::check`] compares the
//! ABI number and the table size before anything is called.
//!
//! [`python_class`] names the Python exception class of an [`Error`], so
//! every extension raises the classes of `segyrs._segyrs`.

use std::ffi::{c_void, CStr};
use std::io;
use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::{AccessHint, Endian, Error, Gain, OpenOptions, ReadAt, Result, SampleFormat, Scan};

/// Name of the Python capsule that carries a [`RawReader`].
pub const CAPSULE_NAME: &CStr = c"segyrs.reader.v1";

const ABI: u32 = 1;
const ERR_CAP: usize = 1024;

type Shared = Arc<dyn ReadAt>;

/// An error message written by the callee into the caller's buffer.
#[repr(C)]
pub struct RawError {
    kind: u8,
    len: usize,
    msg: [u8; ERR_CAP],
}

impl RawError {
    fn new() -> Self {
        Self {
            kind: 0,
            len: 0,
            msg: [0; ERR_CAP],
        }
    }

    fn set(&mut self, err: &Error) {
        self.kind = match err {
            Error::Io(e) | Error::Path { source: e, .. } => kind_code(e.kind()),
            _ => 0,
        };
        let text = err.to_string();
        let mut end = text.len().min(ERR_CAP);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.msg[..end].copy_from_slice(&text.as_bytes()[..end]);
        self.len = end;
    }

    fn into_error(self) -> Error {
        let text = String::from_utf8_lossy(&self.msg[..self.len.min(ERR_CAP)]).into_owned();
        Error::Io(io::Error::new(kind_from(self.kind), text))
    }
}

fn kind_code(kind: io::ErrorKind) -> u8 {
    match kind {
        io::ErrorKind::NotFound => 1,
        io::ErrorKind::PermissionDenied => 2,
        io::ErrorKind::InvalidInput => 3,
        io::ErrorKind::OutOfMemory => 4,
        io::ErrorKind::UnexpectedEof => 5,
        io::ErrorKind::TimedOut => 6,
        _ => 0,
    }
}

fn kind_from(code: u8) -> io::ErrorKind {
    match code {
        1 => io::ErrorKind::NotFound,
        2 => io::ErrorKind::PermissionDenied,
        3 => io::ErrorKind::InvalidInput,
        4 => io::ErrorKind::OutOfMemory,
        5 => io::ErrorKind::UnexpectedEof,
        6 => io::ErrorKind::TimedOut,
        _ => io::ErrorKind::Other,
    }
}

/// The function table and context of an exported reader. The struct that
/// holds it owns one reference to `ctx` and releases it on drop.
#[repr(C)]
pub struct RawReader {
    abi: u32,
    size: usize,
    ctx: *const c_void,
    len: u64,
    bytes: *const u8,
    bytes_len: usize,
    read_at: unsafe extern "C" fn(*const c_void, u64, *mut u8, usize, *mut RawError) -> bool,
    prefetch: unsafe extern "C" fn(*const c_void, *const u64, usize, *mut RawError) -> bool,
    release: unsafe extern "C" fn(*const c_void, u64, u64),
    access_hint: unsafe extern "C" fn(*const c_void, u8),
    clone: unsafe extern "C" fn(*const c_void) -> *const c_void,
    drop: unsafe extern "C" fn(*const c_void),
}

// SAFETY: `ctx` is a `Box<Arc<dyn ReadAt>>` and `ReadAt: Send + Sync`; `bytes`
// points into memory that reader owns and never changes while `ctx` lives.
// Every function in the table is a plain function of this module.
unsafe impl Send for RawReader {}
// SAFETY: as for `Send`: every table function takes `ctx` by shared reference.
unsafe impl Sync for RawReader {}

unsafe extern "C" fn raw_read_at(
    ctx: *const c_void,
    offset: u64,
    buf: *mut u8,
    len: usize,
    err: *mut RawError,
) -> bool {
    // SAFETY: `ctx` comes from `RawReader::export` and is alive while the
    // caller holds its reference; `buf` is the caller's `&mut [u8]` of `len`
    // bytes and `err` its exclusive `RawError`.
    let (reader, buf, err) = unsafe {
        (
            &*(ctx as *const Shared),
            std::slice::from_raw_parts_mut(buf, len),
            &mut *err,
        )
    };
    finish(
        catch_unwind(AssertUnwindSafe(|| reader.read_at(offset, buf))),
        err,
    )
}

unsafe extern "C" fn raw_prefetch(
    ctx: *const c_void,
    ranges: *const u64,
    n: usize,
    err: *mut RawError,
) -> bool {
    // SAFETY: as in `raw_read_at`; `ranges` holds `2 * n` values, start then end.
    let (reader, flat, err) = unsafe {
        (
            &*(ctx as *const Shared),
            std::slice::from_raw_parts(ranges, 2 * n),
            &mut *err,
        )
    };
    let ranges: Vec<Range<u64>> = flat.chunks_exact(2).map(|r| r[0]..r[1]).collect();
    finish(
        catch_unwind(AssertUnwindSafe(|| reader.prefetch(&ranges))),
        err,
    )
}

unsafe extern "C" fn raw_release(ctx: *const c_void, start: u64, end: u64) {
    // SAFETY: as in `raw_read_at`.
    let reader = unsafe { &*(ctx as *const Shared) };
    let _ = catch_unwind(AssertUnwindSafe(|| reader.release(start..end)));
}

unsafe extern "C" fn raw_access_hint(ctx: *const c_void, hint: u8) {
    // SAFETY: as in `raw_read_at`.
    let reader = unsafe { &*(ctx as *const Shared) };
    let hint = if hint == 0 {
        AccessHint::Random
    } else {
        AccessHint::Normal
    };
    let _ = catch_unwind(AssertUnwindSafe(|| reader.access_hint(hint)));
}

unsafe extern "C" fn raw_clone(ctx: *const c_void) -> *const c_void {
    // SAFETY: as in `raw_read_at`.
    let reader = unsafe { &*(ctx as *const Shared) };
    Box::into_raw(Box::new(reader.clone())) as *const c_void
}

unsafe extern "C" fn raw_drop(ctx: *const c_void) {
    // SAFETY: `ctx` was made by `Box::into_raw` in `export` or `raw_clone`,
    // and each reference is dropped exactly once.
    drop(unsafe { Box::from_raw(ctx as *mut Shared) });
}

fn finish(out: std::thread::Result<Result<()>>, err: &mut RawError) -> bool {
    match out {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            err.set(&e);
            false
        }
        Err(_) => {
            err.set(&Error::Io(io::Error::other("the reader panicked")));
            false
        }
    }
}

impl RawReader {
    /// Export `reader`, `len` bytes long, for another extension module.
    pub fn export(reader: Arc<dyn ReadAt>, len: u64) -> Self {
        let (bytes, bytes_len) = match reader.as_bytes() {
            Some(b) => (b.as_ptr(), b.len()),
            None => (std::ptr::null(), 0),
        };
        Self {
            abi: ABI,
            size: std::mem::size_of::<Self>(),
            ctx: Box::into_raw(Box::new(reader)) as *const c_void,
            len,
            bytes,
            bytes_len,
            read_at: raw_read_at,
            prefetch: raw_prefetch,
            release: raw_release,
            access_hint: raw_access_hint,
            clone: raw_clone,
            drop: raw_drop,
        }
    }

    /// Length of the source in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// True when the source has no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `self` was built by a module with this module's layout.
    pub fn check(&self) -> std::result::Result<(), String> {
        if self.abi == ABI && self.size == std::mem::size_of::<Self>() {
            Ok(())
        } else {
            Err(format!(
                "the reader comes from an incompatible segyrs build (abi {}, size {}; \
                 this module expects abi {ABI}, size {}): install the same version of \
                 segyrs and its extras",
                self.abi,
                self.size,
                std::mem::size_of::<Self>()
            ))
        }
    }
}

impl Drop for RawReader {
    fn drop(&mut self) {
        // SAFETY: this struct owns one reference to `ctx`.
        unsafe { (self.drop)(self.ctx) }
    }
}

/// A reader exported by another module, used through its function table.
pub struct ForeignReader {
    raw: RawReader,
}

impl ForeignReader {
    /// Take a new reference to the reader behind `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must be a live `RawReader` built by [`RawReader::export`] of a
    /// module that is still loaded, and [`RawReader::check`] must pass.
    pub unsafe fn new(raw: &RawReader) -> Self {
        Self {
            raw: RawReader {
                // SAFETY: `raw.ctx` is live (caller's contract); the clone is
                // this struct's own reference, released by `RawReader::drop`.
                ctx: unsafe { (raw.clone)(raw.ctx) },
                ..*raw
            },
        }
    }

    /// Length of the source in bytes.
    pub fn len(&self) -> u64 {
        self.raw.len
    }

    /// True when the source has no bytes.
    pub fn is_empty(&self) -> bool {
        self.raw.len == 0
    }
}

impl ReadAt for ForeignReader {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let mut err = RawError::new();
        // SAFETY: `ctx` is this struct's live reference; `buf` and `err` are
        // exclusive for the call.
        let ok = unsafe {
            (self.raw.read_at)(self.raw.ctx, offset, buf.as_mut_ptr(), buf.len(), &mut err)
        };
        if ok {
            Ok(())
        } else {
            Err(err.into_error())
        }
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        if self.raw.bytes.is_null() {
            return None;
        }
        // SAFETY: the exporter's reader owns these bytes and keeps them
        // unchanged while our reference to `ctx` lives, which outlives `&self`.
        Some(unsafe { std::slice::from_raw_parts(self.raw.bytes, self.raw.bytes_len) })
    }

    fn prefetch(&self, ranges: &[Range<u64>]) -> Result<()> {
        if ranges.is_empty() {
            return Ok(());
        }
        let flat: Vec<u64> = ranges.iter().flat_map(|r| [r.start, r.end]).collect();
        let mut err = RawError::new();
        // SAFETY: `flat` holds `2 * ranges.len()` values for the call.
        let ok =
            unsafe { (self.raw.prefetch)(self.raw.ctx, flat.as_ptr(), ranges.len(), &mut err) };
        if ok {
            Ok(())
        } else {
            Err(err.into_error())
        }
    }

    fn access_hint(&self, hint: AccessHint) {
        let code = match hint {
            AccessHint::Random => 0,
            AccessHint::Normal => 1,
        };
        // SAFETY: `ctx` is this struct's live reference.
        unsafe { (self.raw.access_hint)(self.raw.ctx, code) }
    }

    fn release(&self, range: Range<u64>) {
        // SAFETY: `ctx` is this struct's live reference.
        unsafe { (self.raw.release)(self.raw.ctx, range.start, range.end) }
    }
}

/// [`OpenOptions`] and [`Scan`] from the names the Python `segyrs.open`
/// takes. The error names the bad value.
pub fn open_options(
    sample_format: Option<&str>,
    endian: Option<&str>,
    gain: Option<&str>,
    inline_byte: u16,
    crossline_byte: u16,
    offset_byte: u16,
    scan: &str,
) -> std::result::Result<(OpenOptions, Scan), String> {
    let scan = match scan {
        "auto" => Scan::Auto,
        "full" => Scan::Full,
        _ => return Err(format!("scan {scan:?} is not \"auto\" or \"full\"")),
    };
    let sample_format = sample_format
        .map(|s| SampleFormat::parse(s).ok_or_else(|| format!("unknown sample_format {s:?}")))
        .transpose()?;
    let endian = endian
        .map(|s| {
            Endian::parse(s)
                .ok_or_else(|| format!("endian {s:?} is not \"big\", \"little\", or \"pairwise\""))
        })
        .transpose()?;
    let gain = gain
        .map(|s| Gain::parse(s).ok_or_else(|| format!("gain {s:?} is not \"rev0\" or \"rev2\"")))
        .transpose()?;
    for (name, byte) in [
        ("inline_byte", inline_byte),
        ("crossline_byte", crossline_byte),
        ("offset_byte", offset_byte),
    ] {
        if !(1..=237).contains(&byte) {
            return Err(format!(
                "{name}={byte} does not leave room for an int32 in the 240-byte header"
            ));
        }
    }
    let options = OpenOptions {
        sample_format,
        endian,
        gain,
        inline_byte,
        crossline_byte,
        offset_byte,
    };
    Ok((options, scan))
}

/// The Python exception class and message for `err`, as the core binding
/// raises them: an argument or budget mistake carried as an I/O error loses
/// the `io error:` prefix.
pub fn python_error(err: &Error) -> (&'static str, String) {
    let msg = match err {
        Error::Io(e)
            if e.raw_os_error().is_none()
                && matches!(
                    e.kind(),
                    io::ErrorKind::InvalidInput | io::ErrorKind::OutOfMemory
                ) =>
        {
            e.to_string()
        }
        _ => err.to_string(),
    };
    (python_class(err), msg)
}

/// The Python exception class, in `segyrs._segyrs`, that stands for `err`.
pub fn python_class(err: &Error) -> &'static str {
    match err {
        Error::Geometry { .. } => "GeometryError",
        Error::GeometryAssumption { .. } => "GeometryAssumptionError",
        Error::DuplicateKey { .. } => "DuplicateKey",
        Error::EndianMismatch { .. } => "EndianMismatch",
        Error::SampleCountMismatch { .. } => "SampleCountMismatch",
        Error::NonUniformTraceLength { .. } => "NonUniformTraceLength",
        Error::TraceLength { .. } => "TraceLengthError",
        Error::FormatChange => "FormatChangeError",
        Error::Format4NeedsGain => "Format4NeedsGain",
        Error::UnknownFormat(_) => "UnknownFormat",
        Error::Truncated { .. } => "TruncatedError",
        Error::NotSegy(_) => "NotSegyError",
        Error::HeaderValue { .. } => "HeaderValueError",
        Error::SampleValue { .. } => "SampleValueError",
        Error::Spec(_) => "SpecError",
        Error::Invalid { .. } => "ValidationError",
        Error::Io(e) | Error::Path { source: e, .. } => match e.kind() {
            io::ErrorKind::NotFound => "PathNotFoundError",
            io::ErrorKind::PermissionDenied => "PathPermissionError",
            io::ErrorKind::IsADirectory => "PathIsDirectoryError",
            io::ErrorKind::InvalidInput if e.raw_os_error().is_none() => "SegyrsValueError",
            io::ErrorKind::OutOfMemory if e.raw_os_error().is_none() => "BudgetError",
            _ => "SegyrsIOError",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Mem {
        data: Vec<u8>,
        prefetched: AtomicUsize,
    }

    impl ReadAt for Mem {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
            let lo = offset as usize;
            let hi = lo + buf.len();
            if hi > self.data.len() {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("read {lo}..{hi} past {}", self.data.len()),
                )));
            }
            buf.copy_from_slice(&self.data[lo..hi]);
            Ok(())
        }

        fn prefetch(&self, ranges: &[Range<u64>]) -> Result<()> {
            self.prefetched.fetch_add(ranges.len(), Ordering::Relaxed);
            Ok(())
        }
    }

    fn mem() -> Arc<Mem> {
        Arc::new(Mem {
            data: (0..=255).collect(),
            prefetched: AtomicUsize::new(0),
        })
    }

    #[test]
    fn foreign_reader_reads_prefetches_and_keeps_the_error_kind() {
        let src = mem();
        let raw = RawReader::export(src.clone(), 256);
        raw.check().unwrap();
        // SAFETY: `raw` was exported above and passed `check`.
        let foreign = unsafe { ForeignReader::new(&raw) };
        drop(raw);
        assert_eq!(foreign.len(), 256);
        assert!(foreign.as_bytes().is_none());
        let mut buf = [0u8; 4];
        foreign.read_at(10, &mut buf).unwrap();
        assert_eq!(buf, [10, 11, 12, 13]);
        foreign.prefetch(&[0..4, 8..16]).unwrap();
        assert_eq!(src.prefetched.load(Ordering::Relaxed), 2);
        let err = foreign.read_at(250, &mut [0u8; 10]).unwrap_err();
        match &err {
            Error::Io(e) => assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof),
            other => panic!("{other:?}"),
        }
        assert!(err.to_string().contains("past 256"), "{err}");
        assert_eq!(python_class(&err), "SegyrsIOError");
        let bad = Error::Io(io::Error::new(io::ErrorKind::InvalidInput, "chunks [1]"));
        assert_eq!(
            python_error(&bad),
            ("SegyrsValueError", "chunks [1]".to_string())
        );
    }

    #[test]
    fn references_are_counted_across_the_bridge() {
        let src = mem();
        let raw = RawReader::export(src.clone(), 256);
        assert_eq!(Arc::strong_count(&src), 2);
        // SAFETY: as above.
        let a = unsafe { ForeignReader::new(&raw) };
        // SAFETY: as above.
        let b = unsafe { ForeignReader::new(&raw) };
        assert_eq!(Arc::strong_count(&src), 4);
        drop(raw);
        drop(a);
        assert_eq!(Arc::strong_count(&src), 2);
        drop(b);
        assert_eq!(Arc::strong_count(&src), 1);
    }

    #[test]
    fn a_mapped_source_crosses_as_bytes() {
        let data: Arc<[u8]> = (0..64u8).collect();
        struct Bytes(Arc<[u8]>);
        impl ReadAt for Bytes {
            fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
                let lo = offset as usize;
                buf.copy_from_slice(&self.0[lo..lo + buf.len()]);
                Ok(())
            }
            fn as_bytes(&self) -> Option<&[u8]> {
                Some(&self.0)
            }
        }
        let raw = RawReader::export(Arc::new(Bytes(data.clone())), 64);
        // SAFETY: as above.
        let foreign = unsafe { ForeignReader::new(&raw) };
        drop(raw);
        assert_eq!(foreign.as_bytes(), Some(&data[..]));
    }

    #[test]
    fn an_incompatible_table_is_refused() {
        let mut raw = RawReader::export(mem(), 256);
        raw.abi += 1;
        assert!(raw.check().unwrap_err().contains("same version"));
        raw.abi -= 1;
        raw.check().unwrap();
    }

    #[test]
    fn open_options_parse_like_segyrs_open() {
        let (o, s) = open_options(
            Some("ibm_float32"),
            Some("little"),
            None,
            189,
            193,
            37,
            "full",
        )
        .unwrap();
        assert_eq!(s, Scan::Full);
        assert_eq!(o.sample_format, SampleFormat::parse("ibm_float32"));
        assert!(open_options(None, None, None, 238, 193, 37, "auto")
            .unwrap_err()
            .contains("inline_byte=238"));
        assert!(open_options(None, None, None, 189, 193, 37, "some")
            .unwrap_err()
            .contains("scan"));
        assert!(open_options(None, Some("middle"), None, 189, 193, 37, "auto").is_err());
    }

    #[test]
    fn long_messages_are_cut_on_a_character_boundary() {
        let mut err = RawError::new();
        err.set(&Error::Spec("é".repeat(ERR_CAP)));
        assert!(err.len <= ERR_CAP);
        let back = err.into_error().to_string();
        assert!(back.contains("invalid write request: é"), "{back}");
        assert!(!back.contains('\u{fffd}'), "{back}");
    }
}
