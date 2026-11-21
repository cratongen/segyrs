use std::fs::File as FsFile;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use memmap2::Mmap;
use rayon::prelude::*;

use crate::endian::{self, Endian};
use crate::error::{Error, Result};
use crate::geometry::{self, Geometry, Key, Regular, TraceIndex};
use crate::header::{ColumnValues, HeaderColumns, HeaderDtype, HeaderField, HeaderSpec};
use crate::index_cache::IndexCache;
use crate::sample::{self, Gain, SampleFormat, Samples};
use crate::text::{self, TextHeader};

const TEXT_BYTES: u64 = 3200;
const BIN_BYTES: u64 = 400;
const TRACE_HEADER: u64 = 240;
const HEAD_BYTES: usize = (TEXT_BYTES + BIN_BYTES) as usize;

/// Random access to a SEG-Y byte source. Callers do not map the file themselves.
pub trait ReadAt: Send + Sync {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// The whole source as one slice, when it is already in memory or mapped.
    ///
    /// Decoders read straight from this slice and skip the copy into a scratch
    /// buffer. `None` (the default) makes every read go through `read_at`.
    fn as_bytes(&self) -> Option<&[u8]> {
        None
    }

    /// Hint that these byte ranges are read next. A remote reader fetches them
    /// coalesced and in parallel. The default ignores the hint.
    fn prefetch(&self, _ranges: &[std::ops::Range<u64>]) -> Result<()> {
        Ok(())
    }

    /// How the next reads will walk the source. A mapped file passes it to
    /// `madvise`. The default ignores it.
    fn access_hint(&self, _hint: AccessHint) {}

    /// The bytes in `range` are not read again soon. A mapped file drops their
    /// pages from this process's resident set; the next read maps them back
    /// from the page cache. The default ignores it.
    fn release(&self, _range: std::ops::Range<u64>) {}
}

/// Read pattern announced through [`ReadAt::access_hint`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessHint {
    /// Scattered reads: no readahead. A mapped [`File`] is left on this
    /// between operations.
    Random,
    /// The kernel default readahead, for work that walks a large part of the
    /// file in order. [`File::readahead`] turns it on.
    Normal,
}

/// How much of the trace headers [`File`] reads while opening.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scan {
    /// A mapped or in-memory file with more than a few thousand traces whose
    /// first, last, and sampled headers fit a regular grid opens on that grid,
    /// marked assumed. Every trace a read touches is checked against it, and
    /// any operation that reads every header checks them all. Otherwise, the
    /// full scan.
    #[default]
    Auto,
    /// Read every trace header at open and infer the geometry from them.
    Full,
}

/// Scan policy and the optional index sidecar for [`File::open_scan`].
#[derive(Clone, Debug, Default)]
pub struct ScanOptions {
    pub scan: Scan,
    /// Directory for header-index sidecars. When set, a full scan stores the
    /// scanned columns there and a later open of the unchanged file loads
    /// them instead of reading every header. `None` writes nothing.
    pub index_cache: Option<PathBuf>,
}

/// Where the open-time geometry came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometrySource {
    /// Every trace header was read.
    Scan,
    /// A regular grid predicted from sampled headers. See [`Scan::Auto`].
    Assumed,
    /// Columns loaded from a valid index sidecar.
    IndexCache,
}

impl GeometrySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Assumed => "assumed",
            Self::IndexCache => "index_cache",
        }
    }
}

impl<T: ReadAt + ?Sized> ReadAt for Box<T> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_at(offset, buf)
    }

    fn prefetch(&self, ranges: &[std::ops::Range<u64>]) -> Result<()> {
        (**self).prefetch(ranges)
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        (**self).as_bytes()
    }

    fn access_hint(&self, hint: AccessHint) {
        (**self).access_hint(hint)
    }

    fn release(&self, range: std::ops::Range<u64>) {
        (**self).release(range)
    }
}

impl<T: ReadAt + ?Sized> ReadAt for std::sync::Arc<T> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_at(offset, buf)
    }

    fn prefetch(&self, ranges: &[std::ops::Range<u64>]) -> Result<()> {
        (**self).prefetch(ranges)
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        (**self).as_bytes()
    }

    fn access_hint(&self, hint: AccessHint) {
        (**self).access_hint(hint)
    }

    fn release(&self, range: std::ops::Range<u64>) {
        (**self).release(range)
    }
}

/// Mapped file, or a buffer filled with positional reads when mapping is refused.
pub struct Storage {
    kind: StorageKind,
}

enum StorageKind {
    Map(Mmap),
    Mem(Vec<u8>),
}

impl Storage {
    fn open(path: &Path) -> Result<Self> {
        let file = FsFile::open(path).map_err(Error::path("open", path))?;
        let meta = file.metadata().map_err(Error::path("open", path))?;
        if meta.is_dir() {
            return Err(Error::path("open", path)(is_a_directory()));
        }
        match map_file(&file) {
            Ok(map) => Ok(Self {
                kind: StorageKind::Map(map),
            }),
            Err(_) => {
                let len = file.metadata()?.len();
                let size = usize::try_from(len).map_err(|_| {
                    Error::Io(io::Error::new(
                        ErrorKind::InvalidData,
                        "file is larger than this process can address",
                    ))
                })?;
                let mut data = vec![0u8; size];
                crate::pio::FileExt::read_exact_at(&file, &mut data, 0)?;
                Ok(Self {
                    kind: StorageKind::Mem(data),
                })
            }
        }
    }

    fn byte_len(&self) -> u64 {
        match &self.kind {
            StorageKind::Map(map) => map.len() as u64,
            StorageKind::Mem(data) => data.len() as u64,
        }
    }

    fn as_slice(&self) -> &[u8] {
        match &self.kind {
            StorageKind::Map(map) => map,
            StorageKind::Mem(data) => data,
        }
    }
}

pub(crate) fn is_a_directory() -> io::Error {
    #[cfg(unix)]
    return io::Error::from_raw_os_error(libc::EISDIR);
    #[cfg(not(unix))]
    io::Error::from(ErrorKind::IsADirectory)
}

fn map_file(file: &FsFile) -> std::io::Result<Mmap> {
    // SAFETY: the file is opened read-only and this process does not write it.
    // The map is a private read-only view of those bytes.
    unsafe { Mmap::map(file) }
}

impl ReadAt for Storage {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let data = self.as_slice();
        let start = usize::try_from(offset).map_err(|_| {
            Error::Io(io::Error::new(
                ErrorKind::InvalidInput,
                "offset does not fit",
            ))
        })?;
        let end = start.checked_add(buf.len()).ok_or_else(|| {
            Error::Io(io::Error::new(
                ErrorKind::InvalidInput,
                "read range overflows",
            ))
        })?;
        if end > data.len() {
            return Err(Error::Io(io::Error::new(
                ErrorKind::UnexpectedEof,
                "read past end of file",
            )));
        }
        buf.copy_from_slice(&data[start..end]);
        Ok(())
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        Some(self.as_slice())
    }

    /// Selections touch scattered traces, so kernel readahead is wasted work
    /// there (performance.md, Decisions); a full header scan walks the file in
    /// order. The advice is a hint: a refusal is ignored.
    fn access_hint(&self, hint: AccessHint) {
        #[cfg(unix)]
        if let StorageKind::Map(map) = &self.kind {
            let advice = match hint {
                AccessHint::Random => memmap2::Advice::Random,
                AccessHint::Normal => memmap2::Advice::Normal,
            };
            let _ = map.advise(advice);
        }
        #[cfg(not(unix))]
        let _ = hint;
    }

    fn release(&self, range: std::ops::Range<u64>) {
        #[cfg(unix)]
        if let StorageKind::Map(map) = &self.kind {
            let len = map.len() as u64;
            let (start, end) = (range.start.min(len), range.end.min(len));
            if start < end {
                // SAFETY: the map is a read-only view of a file this process
                // does not write. MADV_DONTNEED only unmaps its pages; the next
                // access maps the same file bytes back, so no slice into the
                // map ever sees different contents.
                let _ = unsafe {
                    map.unchecked_advise_range(
                        memmap2::UncheckedAdvice::DontNeed,
                        start as usize,
                        (end - start) as usize,
                    )
                };
            }
        }
        #[cfg(not(unix))]
        let _ = range;
    }
}

/// Overrides applied while opening. `None` keeps the value found in the file.
#[derive(Clone, Debug)]
pub struct OpenOptions {
    pub sample_format: Option<SampleFormat>,
    pub endian: Option<Endian>,
    pub gain: Option<Gain>,
    pub inline_byte: u16,
    pub crossline_byte: u16,
    pub offset_byte: u16,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            sample_format: None,
            endian: None,
            gain: None,
            inline_byte: 189,
            crossline_byte: 193,
            offset_byte: 37,
        }
    }
}

/// SEG-Y revision recorded in binary-header bytes 3501 and 3502.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Revision {
    Rev0,
    Rev1,
    Rev20,
    Rev21,
}

impl Revision {
    pub fn from_bytes(major: u8, minor: u8) -> Self {
        match (major, minor) {
            (0, _) => Self::Rev0,
            (1, _) => Self::Rev1,
            (2, 0) => Self::Rev20,
            (2, _) => Self::Rev21,
            _ => Self::Rev21,
        }
    }

    pub fn major(self) -> u8 {
        match self {
            Self::Rev0 => 0,
            Self::Rev1 => 1,
            Self::Rev20 | Self::Rev21 => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rev0 => "0",
            Self::Rev1 => "1",
            Self::Rev20 => "2.0",
            Self::Rev21 => "2.1",
        }
    }
}

/// Sizes and coding discovered from the headers. Sample bytes are not part of this.
#[derive(Clone, Debug)]
pub struct Layout {
    pub revision: Revision,
    pub sample_format: SampleFormat,
    pub source_endian: Endian,
    pub n_traces: u64,
    pub n_samples: u32,
    pub sample_interval_us: u32,
    pub extended_textual_blocks: u16,
    pub extra_trace_header_blocks: u16,
    pub data_start: u64,
    /// Sample-format code stored in the binary header, before an override.
    pub binary_format_code: u16,
    /// Binary-header measurement system: 1 meters, 2 feet, 0 absent.
    pub measurement_system: u16,
}

/// One decoded trace, or a group when `decode` is asked for several.
#[derive(Clone, Debug)]
pub struct TracePart {
    pub index: u64,
    pub samples: Samples,
}

#[derive(Clone, Debug)]
pub(crate) struct TraceCols {
    pub(crate) inline: Vec<i32>,
    pub(crate) crossline: Vec<i32>,
    pub(crate) offset: Vec<i32>,
    pub(crate) scalco: Vec<i16>,
    pub(crate) cdp_x: Vec<i32>,
    pub(crate) cdp_y: Vec<i32>,
    pub(crate) source_x: Vec<i32>,
    pub(crate) source_y: Vec<i32>,
    pub(crate) group_x: Vec<i32>,
    pub(crate) group_y: Vec<i32>,
    pub(crate) trace_id: Vec<i16>,
    pub(crate) declared_ns: Vec<u32>,
}

/// Everything the header scan needs, kept so the scan can run after open.
#[derive(Clone, Copy, Debug)]
struct ScanPlan {
    data_start: u64,
    stride: u64,
    header_blocks: u32,
    ns: u32,
    ntraces: usize,
    endian: Endian,
    major: u8,
    /// 1-based inline, crossline, and offset bytes.
    keys: [u16; 3],
    /// Binary header says every trace has `ns` samples (bytes 3503-3504 == 1).
    fixed: bool,
}

impl ScanPlan {
    fn header_len(&self) -> usize {
        (u64::from(self.header_blocks) * TRACE_HEADER) as usize
    }

    /// The key words and the declared sample count of trace `t`.
    fn words<R: ReadAt + ?Sized>(&self, reader: &R, t: usize) -> Result<(Key, u32)> {
        let off = self.data_start + t as u64 * self.stride;
        let len = self.header_len();
        let mut buf;
        let header = match reader.as_bytes() {
            Some(all) => all
                .get(off as usize..off as usize + len)
                .ok_or_else(|| past_end("trace header"))?,
            None => {
                buf = vec![0u8; len];
                reader.read_at(off, &mut buf)?;
                &buf[..]
            }
        };
        let key = self.keys.map(|b| field_i32(header, self.endian, b));
        let ns = declared_ns(header, self.endian, self.header_blocks, self.ns, self.major);
        Ok((key, ns))
    }
}

/// The first trace that broke the assumed grid.
#[derive(Clone, Copy, Debug)]
enum Broken {
    Key {
        trace: u64,
        expected: Key,
        found: Key,
    },
    Samples {
        trace: u64,
        binary_ns: u32,
        trace_ns: u32,
    },
}

impl Broken {
    fn error(self) -> Error {
        match self {
            Self::Key {
                trace,
                expected,
                found,
            } => Error::GeometryAssumption {
                trace,
                expected: (expected[0], expected[1], expected[2]),
                found: (found[0], found[1], found[2]),
            },
            Self::Samples {
                trace,
                binary_ns,
                trace_ns,
            } => Error::NonUniformTraceLength {
                trace,
                binary_ns,
                trace_ns,
            },
        }
    }
}

/// Fewest traces for the assumed-grid open. A smaller file is scanned in full,
/// which costs little.
const FAST_MIN_TRACES: usize = 4096;
/// Headers sampled across the file before a grid is assumed, beyond the few
/// dozen the grid search reads.
const FAST_SAMPLES: usize = 1024;

/// A SEG-Y file opened for reading.
///
/// `File::open` maps the path when the operating system allows it and otherwise
/// fills a buffer with positional reads. Header bytes are read during open.
/// Sample bytes stay unread until `decode`, `par_traces`, or `sample_bytes`.
///
/// On the default [`Scan::Auto`] a large regular file opens from a sample of
/// its headers ([`GeometrySource::Assumed`]). The header columns (`inline()`,
/// `cdp_x()`, ...) are then scanned on first use, every read checks the keys
/// of the traces it touches, and [`File::verify_geometry`] checks them all.
pub struct File<R: ReadAt = Storage> {
    reader: R,
    path: PathBuf,
    format: SampleFormat,
    gain: Option<Gain>,
    ns: u32,
    ntraces: usize,
    data_start: u64,
    header_blocks: u32,
    text_blocks: Vec<TextHeader>,
    binary_header: Vec<u8>,
    geometry: Geometry,
    source: GeometrySource,
    plan: ScanPlan,
    assumed: Option<Regular>,
    /// Outcome of checking every header against `assumed`: `None` is a match.
    verified: OnceLock<Option<Broken>>,
    cols: OnceLock<TraceCols>,
    layout: Layout,
    trace_index: OnceLock<TraceIndex>,
    samples_read: AtomicU64,
    /// Live [`Readahead`] guards. The advice changes under this lock.
    readahead: Mutex<usize>,
}

/// Kernel readahead on a [`File`] while this guard lives. See [`File::readahead`].
#[must_use = "readahead lasts only while the guard is alive"]
pub struct Readahead<'a, R: ReadAt> {
    file: Option<&'a File<R>>,
}

impl<R: ReadAt> Drop for Readahead<'_, R> {
    fn drop(&mut self) {
        if let Some(file) = self.file {
            let mut live = file.readahead.lock().unwrap_or_else(|e| e.into_inner());
            *live -= 1;
            if *live == 0 {
                file.reader.access_hint(AccessHint::Random);
            }
        }
    }
}

impl File<Storage> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, &OpenOptions::default())
    }

    pub fn open_with(path: impl AsRef<Path>, options: &OpenOptions) -> Result<Self> {
        Self::open_scan(path, options, &ScanOptions::default())
    }

    /// [`File::open_with`] with a scan policy and an optional index sidecar.
    pub fn open_scan(
        path: impl AsRef<Path>,
        options: &OpenOptions,
        scan: &ScanOptions,
    ) -> Result<Self> {
        let path = path.as_ref();
        let storage = Storage::open(path)?;
        let len = storage.byte_len();
        let cache = match &scan.index_cache {
            Some(dir) => IndexCache::for_file(dir, path, len),
            None => None,
        };
        let mut file = File::open_inner(storage, len, options, scan.scan, cache.as_ref())?;
        file.path = path.to_path_buf();
        Ok(file)
    }
}

impl<R: ReadAt> File<R> {
    pub fn from_reader(reader: R, len: u64, options: &OpenOptions) -> Result<Self> {
        Self::open_inner(reader, len, options, Scan::Auto, None)
    }

    /// [`File::from_reader`] with a scan policy. [`Scan::Auto`] scans in full
    /// when the reader is not mapped or in memory.
    pub fn from_reader_scan(
        reader: R,
        len: u64,
        options: &OpenOptions,
        scan: Scan,
    ) -> Result<Self> {
        Self::open_inner(reader, len, options, scan, None)
    }

    fn open_inner(
        reader: R,
        len: u64,
        options: &OpenOptions,
        scan: Scan,
        cache: Option<&IndexCache>,
    ) -> Result<Self> {
        let mut first = [0u8; TRACE_HEADER as usize];
        let n_first = usize::try_from(len).map_or(first.len(), |l| l.min(first.len()));
        reader.read_at(0, &mut first[..n_first])?;
        Self::open_segy(reader, len, options, scan, cache).map_err(|err| match err {
            Error::Truncated { .. }
            | Error::UnknownFormat(_)
            | Error::SampleCountMismatch { .. }
            | Error::NonUniformTraceLength { .. }
            | Error::EndianMismatch { .. } => match seismic_unix(&first[..n_first], len) {
                Some(why) => Error::NotSegy(why),
                None => err,
            },
            other => other,
        })
    }

    fn open_segy(
        reader: R,
        len: u64,
        options: &OpenOptions,
        scan: Scan,
        cache: Option<&IndexCache>,
    ) -> Result<Self> {
        if len < TEXT_BYTES + BIN_BYTES {
            return Err(Error::Truncated {
                file_size: len,
                need: TEXT_BYTES + BIN_BYTES,
                context: "for the 3200-byte textual and 400-byte binary file header; \
                          the file is cut off or is not SEG-Y"
                    .into(),
            });
        }
        let mut head = vec![0u8; HEAD_BYTES];
        reader.read_at(0, &mut head)?;
        let endian = choose_endian(&head, len, options)?;
        let major = head[3500];
        let minor = head[3501];
        let revision = Revision::from_bytes(major, minor);

        let header_code = endian::u16_at(&head, 3224, endian);
        let format = match options.sample_format {
            Some(fmt) => fmt,
            None => {
                SampleFormat::from_code(header_code).ok_or(Error::UnknownFormat(header_code))?
            }
        };
        let (n_ext, data_start) = locate_text(&reader, &head, len, endian, major)?;
        if format == SampleFormat::FixedGain && options.gain.is_none() {
            return Err(Error::Format4NeedsGain);
        }
        let located = Located {
            reader: &reader,
            head: &head,
            file_len: len,
            endian,
            major,
            data_start,
        };
        let planned = plan_traces(&located, format)?;
        let text_blocks = read_text(&reader, &head, n_ext)?;
        let plan = ScanPlan {
            data_start,
            stride: stride_of(planned.header_blocks, planned.ns, planned.width),
            header_blocks: planned.header_blocks,
            ns: planned.ns,
            ntraces: planned.ntraces,
            endian,
            major,
            keys: [
                options.inline_byte,
                options.crossline_byte,
                options.offset_byte,
            ],
            fixed: endian::u16_at(&head, 3502, endian) == 1,
        };
        let assumed = if scan == Scan::Auto
            && plan.ntraces > FAST_MIN_TRACES
            && reader.as_bytes().is_some()
        {
            reader.access_hint(AccessHint::Random);
            assume_grid(&reader, &plan)?
        } else {
            None
        };
        let cols = OnceLock::new();
        let (geometry, source) = match assumed {
            Some(grid) => (
                grid.geometry(
                    options.inline_byte,
                    options.crossline_byte,
                    options.offset_byte,
                ),
                GeometrySource::Assumed,
            ),
            None => {
                let fingerprint = match cache {
                    Some(_) => Some(fingerprint(&reader, &plan, format)?),
                    None => None,
                };
                let loaded = cache
                    .zip(fingerprint)
                    .filter(|_| scan == Scan::Auto)
                    .and_then(|(c, fp)| c.load(fp, plan.ntraces));
                let source = match loaded {
                    Some(_) => GeometrySource::IndexCache,
                    None => GeometrySource::Scan,
                };
                let scanned = match loaded {
                    Some(c) => c,
                    None => {
                        reader.access_hint(AccessHint::Normal);
                        let c = read_columns(&reader, &plan)?;
                        check_declared_ns(&c, &plan)?;
                        c
                    }
                };
                let geometry = geometry_from(&scanned, options, plan.ntraces)?;
                if let (Some(c), Some(fp), GeometrySource::Scan) = (cache, fingerprint, source) {
                    c.store(fp, &scanned);
                }
                let _ = cols.set(scanned);
                (geometry, source)
            }
        };
        reader.access_hint(AccessHint::Random);
        let header_blocks = planned.header_blocks;
        let ns = planned.ns;
        let ntraces = planned.ntraces;
        let interval_us = planned.interval_us;

        let extended_textual_blocks = text_blocks
            .len()
            .saturating_sub(1)
            .min(usize::from(u16::MAX)) as u16;
        let layout = Layout {
            revision,
            sample_format: format,
            source_endian: endian,
            n_traces: ntraces as u64,
            n_samples: ns,
            sample_interval_us: interval_us,
            extended_textual_blocks,
            extra_trace_header_blocks: header_blocks.saturating_sub(1).min(u32::from(u16::MAX))
                as u16,
            data_start,
            binary_format_code: header_code,
            measurement_system: endian::u16_at(&head, 3254, endian),
        };

        Ok(Self {
            reader,
            path: PathBuf::new(),
            format,
            gain: options.gain,
            ns,
            ntraces,
            data_start,
            header_blocks,
            text_blocks,
            binary_header: head[3200..3600].to_vec(),
            geometry,
            source,
            plan,
            assumed,
            verified: OnceLock::new(),
            cols,
            layout,
            trace_index: OnceLock::new(),
            samples_read: AtomicU64::new(0),
            readahead: Mutex::new(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn textual_header(&self) -> &[TextHeader] {
        &self.text_blocks
    }

    pub fn binary_header_bytes(&self) -> &[u8] {
        &self.binary_header
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Where [`File::geometry`] came from. It does not change after open.
    pub fn geometry_source(&self) -> GeometrySource {
        self.source
    }

    /// True while the geometry is a regular grid predicted from sampled
    /// headers and not yet checked against every header.
    pub fn geometry_assumed(&self) -> bool {
        self.assumed.is_some() && !matches!(self.verified.get(), Some(None))
    }

    /// Kernel readahead on the mapped file until the guard drops.
    ///
    /// Open leaves the map on random access (`MADV_RANDOM`), so a selection of
    /// scattered traces reads only their pages. Work that walks a large part
    /// of the file in order (a header scan, a whole-cube read, a streaming
    /// operation, a copy) holds this guard for its duration: without
    /// readahead such work is several times slower from a cold cache
    /// (docs/design.md, What open reads).
    ///
    /// Advice belongs to the mapping, not to a thread, and is only a hint.
    /// Guards nest and may live on several threads at once: a count under a
    /// lock switches to readahead with the first guard and back to random
    /// access when the last one drops. While any guard lives, scattered reads
    /// of the same `File` on other threads also get readahead, which costs
    /// extra page reads but never changes a result. A source that is not
    /// mapped ignores the advice.
    pub fn readahead(&self) -> Readahead<'_, R> {
        let mut live = self.readahead.lock().unwrap_or_else(|e| e.into_inner());
        if *live == 0 {
            self.reader.access_hint(AccessHint::Normal);
        }
        *live += 1;
        Readahead { file: Some(self) }
    }

    /// [`File::readahead`] when a read of `traces` traces covers a large part
    /// of the file (at least 1/16 of it, the threshold at which
    /// [`File::verify_traces`] checks every header). Otherwise a guard that
    /// changes nothing.
    pub fn readahead_for(&self, traces: usize) -> Readahead<'_, R> {
        if traces >= self.ntraces / 16 {
            self.readahead()
        } else {
            Readahead { file: None }
        }
    }

    /// Check every trace header against the assumed grid, once, in parallel.
    ///
    /// `Ok` when the geometry was scanned, or when every key matches. Otherwise
    /// [`Error::GeometryAssumption`] for the first trace whose key differs, or
    /// [`Error::NonUniformTraceLength`] for one whose sample count differs.
    /// The outcome is kept: later calls do not rescan.
    pub fn verify_geometry(&self) -> Result<()> {
        let Some(grid) = self.assumed else {
            return Ok(());
        };
        let broken = self.verified.get_or_init(|| {
            let c = self.cols();
            if let Some(t) = grid.first_mismatch(&c.inline, &c.crossline, &c.offset) {
                return Some(Broken::Key {
                    trace: t as u64,
                    expected: grid.predict(t),
                    found: [c.inline[t], c.crossline[t], c.offset[t]],
                });
            }
            first_ns_mismatch(c, &self.plan)
        });
        broken.map_or(Ok(()), |b| Err(b.error()))
    }

    /// Check the keys of these traces against the assumed grid. Reads only
    /// their headers, which share pages with the samples a read fetches next.
    /// A list that covers a large part of the file checks every header once.
    pub fn verify_traces(&self, traces: &[u64]) -> Result<()> {
        let Some(grid) = self.assumed else {
            return Ok(());
        };
        if let Some(done) = self.verified.get() {
            return done.map_or(Ok(()), |b| Err(b.error()));
        }
        if traces.len() >= self.ntraces / 16 {
            return self.verify_geometry();
        }
        let check = |&t: &u64| -> Result<()> {
            let t = self.check_trace(t)?;
            let (found, trace_ns) = self.plan.words(&self.reader, t)?;
            let expected = grid.predict(t);
            if found != expected {
                return Err(Broken::Key {
                    trace: t as u64,
                    expected,
                    found,
                }
                .error());
            }
            if !self.plan.fixed && trace_ns != self.plan.ns {
                return Err(Broken::Samples {
                    trace: t as u64,
                    binary_ns: self.plan.ns,
                    trace_ns,
                }
                .error());
            }
            Ok(())
        };
        if traces.len() < 64 {
            traces.iter().try_for_each(check)
        } else {
            traces
                .par_iter()
                .map(check)
                .find_first(Result::is_err)
                .unwrap_or(Ok(()))
        }
    }

    /// The scanned header columns, read now if open did not read them.
    fn cols(&self) -> &TraceCols {
        self.cols.get_or_init(|| {
            // Only a mapped or in-memory source opens without the scan, and
            // every header of such a source is inside the slice.
            let _ra = self.readahead();
            read_columns(&self.reader, &self.plan).expect("trace headers of a mapped file")
        })
    }

    /// `(inline, crossline)` to file trace numbers. Built on first use, in parallel.
    pub fn trace_index(&self) -> &TraceIndex {
        self.trace_index.get_or_init(|| {
            let c = self.cols();
            TraceIndex::build(&c.inline, &c.crossline)
        })
    }

    /// How many times sample bytes have been read. Open leaves this at zero.
    pub fn samples_read(&self) -> u64 {
        self.samples_read.load(Ordering::Relaxed)
    }

    pub fn endian(&self) -> Endian {
        self.layout.source_endian
    }

    pub fn format(&self) -> SampleFormat {
        self.format
    }

    pub fn gain(&self) -> Option<Gain> {
        self.gain
    }

    /// The byte source behind this file. Streaming operations and the writer read through it.
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// The same open file over another byte source, for example a mapped file
    /// as an `Arc<dyn ReadAt>` so local and remote files share one type.
    /// Nothing is re-read: `f` must return a source with the same bytes.
    pub fn map_reader<S: ReadAt>(self, f: impl FnOnce(R) -> S) -> File<S> {
        File {
            reader: f(self.reader),
            path: self.path,
            format: self.format,
            gain: self.gain,
            ns: self.ns,
            ntraces: self.ntraces,
            data_start: self.data_start,
            header_blocks: self.header_blocks,
            text_blocks: self.text_blocks,
            binary_header: self.binary_header,
            geometry: self.geometry,
            source: self.source,
            plan: self.plan,
            assumed: self.assumed,
            verified: self.verified,
            cols: self.cols,
            layout: self.layout,
            trace_index: self.trace_index,
            samples_read: self.samples_read,
            readahead: self.readahead,
        }
    }

    /// Record where the bytes came from, such as an object-store URL. `path()`
    /// returns it. `File::open` sets it already.
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = path.into();
        self
    }

    pub fn stride(&self) -> u64 {
        stride_of(self.header_blocks, self.ns, self.format.width() as u64)
    }

    pub fn inline(&self) -> &[i32] {
        &self.cols().inline
    }

    pub fn crossline(&self) -> &[i32] {
        &self.cols().crossline
    }

    pub fn offset(&self) -> &[i32] {
        &self.cols().offset
    }

    pub fn scalco(&self) -> &[i16] {
        &self.cols().scalco
    }

    pub fn cdp_x(&self) -> &[i32] {
        &self.cols().cdp_x
    }

    pub fn cdp_y(&self) -> &[i32] {
        &self.cols().cdp_y
    }

    pub fn source_x(&self) -> &[i32] {
        &self.cols().source_x
    }

    pub fn source_y(&self) -> &[i32] {
        &self.cols().source_y
    }

    pub fn group_x(&self) -> &[i32] {
        &self.cols().group_x
    }

    pub fn group_y(&self) -> &[i32] {
        &self.cols().group_y
    }

    pub fn trace_id(&self) -> &[i16] {
        &self.cols().trace_id
    }

    /// On-disk standard trace header, file byte order.
    pub fn header_bytes(&self, trace: u64) -> Result<Vec<u8>> {
        self.read_trace_header(trace, TRACE_HEADER as usize)
    }

    /// On-disk sample bytes for one trace, file byte order.
    pub fn sample_bytes(&self, trace: u64) -> Result<Vec<u8>> {
        self.verify_traces(&[trace])?;
        let trace = self.check_trace(trace)?;
        let width = self.format.width();
        let off = self.sample_offset(trace);
        let mut buf = vec![0u8; self.ns as usize * width];
        if buf.is_empty() {
            return Ok(buf);
        }
        self.samples_read.fetch_add(1, Ordering::Relaxed);
        self.reader.read_at(off, &mut buf)?;
        Ok(buf)
    }

    /// Decode traces by file index. Indices may be sparse. Order follows `indices`.
    pub fn decode_indices(&self, indices: &[u32]) -> Result<Samples> {
        if indices.is_empty() {
            return Ok(sample::empty_samples(self.format, 0));
        }
        let _ra = self.readahead_for(indices.len());
        let parts: Result<Vec<Samples>> = indices
            .par_iter()
            .map(|&index| self.decode_one(index as usize))
            .collect();
        Ok(concat_samples(parts?))
    }

    /// Decode samples `start..stop` of each listed trace into one buffer.
    ///
    /// The result is `traces.len() * (stop - start)` values, trace-major, in the
    /// order of `traces`. Only those sample bytes are read. Traces are decoded in
    /// parallel straight into the output, with no per-trace allocation when the
    /// source is mapped. `samples_read` counts one per trace.
    pub fn read_window(&self, traces: &[u64], start: usize, stop: usize) -> Result<Samples> {
        let ns = self.ns as usize;
        if start > stop || stop > ns {
            return Err(Error::Io(io::Error::new(
                ErrorKind::InvalidInput,
                format!("sample window {start}..{stop} is outside 0..{ns}"),
            )));
        }
        for &trace in traces {
            self.check_trace(trace)?;
        }
        let _ra = self.readahead_for(traces.len());
        self.verify_traces(traces)?;
        let width = stop - start;
        let total = traces.len().checked_mul(width).ok_or_else(|| {
            Error::Io(io::Error::new(
                ErrorKind::InvalidInput,
                "selection size overflows",
            ))
        })?;
        let mut out = sample::empty_samples(self.format, total);
        if total == 0 {
            return Ok(out);
        }
        self.samples_read
            .fetch_add(traces.len() as u64, Ordering::Relaxed);
        let decode = |trace: u64, dst: sample::SamplesMut<'_>| -> Result<()> {
            let bpw = self.format.width();
            let off = self.sample_offset(trace as usize) + (start * bpw) as u64;
            let len = width * bpw;
            match self.reader.as_bytes() {
                Some(all) => {
                    let lo = off as usize;
                    let bytes = all.get(lo..lo + len).ok_or_else(|| {
                        Error::Io(io::Error::new(
                            ErrorKind::UnexpectedEof,
                            "read past end of file",
                        ))
                    })?;
                    sample::decode_into(bytes, self.format, self.endian(), self.gain, dst);
                }
                None => {
                    let mut buf = vec![0u8; len];
                    self.reader.read_at(off, &mut buf)?;
                    sample::decode_into(&buf, self.format, self.endian(), self.gain, dst);
                }
            }
            Ok(())
        };
        macro_rules! fill {
            ($v:expr, $variant:ident) => {
                $v.par_chunks_mut(width)
                    .zip(traces.par_iter())
                    .try_for_each(|(chunk, &t)| decode(t, sample::SamplesMut::$variant(chunk)))
            };
        }
        match &mut out {
            Samples::I8(v) => fill!(v, I8),
            Samples::U8(v) => fill!(v, U8),
            Samples::I16(v) => fill!(v, I16),
            Samples::U16(v) => fill!(v, U16),
            Samples::I32(v) => fill!(v, I32),
            Samples::U32(v) => fill!(v, U32),
            Samples::I64(v) => fill!(v, I64),
            Samples::U64(v) => fill!(v, U64),
            Samples::F32(v) => fill!(v, F32),
            Samples::F64(v) => fill!(v, F64),
        }?;
        Ok(out)
    }

    pub fn decode(&self, start: usize, count: usize) -> Result<Samples> {
        let end = start
            .checked_add(count)
            .ok_or_else(|| range_error(start, count))?;
        if end > self.ntraces {
            return Err(range_error(start, count));
        }
        let _ra = self.readahead_for(count);
        let parts: Result<Vec<Samples>> = (0..count)
            .into_par_iter()
            .map(|i| self.decode_one(start + i))
            .collect();
        Ok(concat_samples(parts?))
    }

    pub fn par_traces(&self) -> Result<Vec<TracePart>> {
        let _ra = self.readahead();
        (0..self.ntraces)
            .into_par_iter()
            .map(|i| {
                Ok(TracePart {
                    index: i as u64,
                    samples: self.decode_one(i)?,
                })
            })
            .collect()
    }

    /// Parallel scan of the requested header fields. One column entry per trace.
    ///
    /// Reading every header also checks an assumed geometry
    /// ([`File::verify_geometry`]), so the columns never sit beside a grid
    /// they contradict.
    pub fn par_header_columns(&self, spec: &HeaderSpec) -> Result<HeaderColumns> {
        let _ra = self.readahead();
        self.verify_geometry()?;
        let header_len = u64::from(self.header_blocks) * TRACE_HEADER;
        for field in &spec.fields {
            let width = field.dtype.width() as u64;
            if field.byte == 0 || u64::from(field.byte - 1) + width > header_len {
                return Err(Error::Io(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "header field byte {} width {} is outside the trace header",
                        field.byte, width
                    ),
                )));
            }
        }
        let endian = self.endian();
        let n = self.ntraces;
        let cols: Result<Vec<ColumnValues>> = spec
            .fields
            .par_iter()
            .map(|field| self.header_column(field, endian, n))
            .collect();
        HeaderColumns::from_columns(spec.fields.clone(), cols?)
    }

    /// One header field for every trace, read in parallel straight into its column.
    fn header_column(&self, field: &HeaderField, endian: Endian, n: usize) -> Result<ColumnValues> {
        let at = u64::from(field.byte - 1);
        let width = field.dtype.width();
        let stride = self.stride();
        let base = self.data_start + at;
        let all = self.reader.as_bytes();
        let word = |i: usize| -> Result<[u8; 8]> {
            let off = base + i as u64 * stride;
            let mut raw = [0u8; 8];
            match all {
                Some(all) => {
                    let lo = off as usize;
                    let src = all.get(lo..lo + width).ok_or_else(|| {
                        Error::Io(io::Error::new(
                            ErrorKind::UnexpectedEof,
                            "header field extends past the end of file",
                        ))
                    })?;
                    raw[..width].copy_from_slice(src);
                }
                None => self.reader.read_at(off, &mut raw[..width])?,
            }
            Ok(raw)
        };
        macro_rules! column {
            ($variant:ident, $read:expr) => {
                ColumnValues::$variant(
                    (0..n)
                        .into_par_iter()
                        .map(|i| word(i).map(|raw| $read(&raw[..])))
                        .collect::<Result<Vec<_>>>()?,
                )
            };
        }
        Ok(match field.dtype {
            HeaderDtype::I8 => column!(I8, |r: &[u8]| r[0] as i8),
            HeaderDtype::I16 => column!(I16, |r: &[u8]| endian::i16_at(r, 0, endian)),
            HeaderDtype::I32 => column!(I32, |r: &[u8]| endian::i32_at(r, 0, endian)),
            HeaderDtype::I64 => column!(I64, |r: &[u8]| endian::i64_at(r, 0, endian)),
            HeaderDtype::U16 => column!(U16, |r: &[u8]| endian::u16_at(r, 0, endian)),
            HeaderDtype::U32 => column!(U32, |r: &[u8]| endian::u32_at(r, 0, endian)),
            HeaderDtype::F64 => column!(F64, |r: &[u8]| endian::f64_at(r, 0, endian)),
        })
    }

    fn decode_one(&self, index: usize) -> Result<Samples> {
        let ns = self.ns as usize;
        let mut out = sample::empty_samples(self.format, ns);
        if ns == 0 {
            return Ok(out);
        }
        let bytes = self.sample_bytes(index as u64)?;
        sample::decode_trace(
            &bytes,
            self.format,
            self.endian(),
            self.gain,
            0,
            ns,
            &mut out,
        );
        Ok(out)
    }

    fn sample_offset(&self, trace: usize) -> u64 {
        self.data_start
            + trace as u64 * self.stride()
            + u64::from(self.header_blocks) * TRACE_HEADER
    }

    fn check_trace(&self, trace: u64) -> Result<usize> {
        if trace >= self.ntraces as u64 {
            return Err(Error::Io(io::Error::new(
                ErrorKind::InvalidInput,
                format!("trace {trace} is outside 0..{}", self.ntraces),
            )));
        }
        Ok(trace as usize)
    }

    fn read_trace_header(&self, trace: u64, len: usize) -> Result<Vec<u8>> {
        let trace = self.check_trace(trace)?;
        let off = self.data_start + trace as u64 * self.stride();
        let mut buf = vec![0u8; len];
        if len > 0 {
            self.reader.read_at(off, &mut buf)?;
        }
        Ok(buf)
    }
}

fn range_error(start: usize, count: usize) -> Error {
    Error::Io(io::Error::new(
        ErrorKind::InvalidInput,
        format!("trace range {start}..{} is outside the file", start + count),
    ))
}

struct Located<'a, R: ReadAt + ?Sized> {
    reader: &'a R,
    head: &'a [u8],
    file_len: u64,
    endian: Endian,
    major: u8,
    data_start: u64,
}

struct Planned {
    header_blocks: u32,
    ns: u32,
    ntraces: usize,
    interval_us: u32,
    width: u64,
}

fn plan_traces<R: ReadAt>(loc: &Located<'_, R>, format: SampleFormat) -> Result<Planned> {
    let width = format.width() as u64;
    let interval_us = read_interval(loc.head, loc.endian, loc.major);
    let ns16 = u32::from(endian::u16_at(loc.head, 3220, loc.endian));
    let mut ns_candidates = Vec::new();
    if loc.major >= 2 {
        let extended = endian::u32_at(loc.head, 3268, loc.endian);
        if extended > 0 {
            ns_candidates.push(extended);
        }
    }
    if !ns_candidates.contains(&ns16) {
        ns_candidates.push(ns16);
    }
    let payload = loc.file_len - loc.data_start;
    let first_ns = ns_candidates[0];
    let mut chosen = None;
    let mut last_mismatch = None;
    for binary_ns in ns_candidates {
        match choose_stride(loc, binary_ns, width, payload) {
            Ok(pair) => {
                chosen = Some(pair);
                break;
            }
            Err(err) => last_mismatch = Some(err),
        }
    }
    let Some((header_blocks, ns)) = chosen else {
        if let Some(err) = short_trace(loc, first_ns, width)? {
            return Err(err);
        }
        return Err(last_mismatch.unwrap_or(Error::SampleCountMismatch {
            file_len: loc.file_len,
            expected: loc.data_start,
            remainder: payload,
        }));
    };
    let stride = stride_of(header_blocks, ns, width);
    let ntraces = payload.checked_div(stride).unwrap_or(0) as usize;
    Ok(Planned {
        header_blocks,
        ns,
        ntraces,
        interval_us,
        width,
    })
}

/// The file size does not divide into whole traces. Walk the trace headers in
/// file order using each trace's own sample count. When those lengths add up
/// to the file size exactly, name the first trace whose length disagrees with
/// the binary-header stride, with both byte lengths (X-18). Otherwise the
/// per-trace counts do not explain the size, and the caller reports
/// `SampleCountMismatch` (S-25).
fn short_trace<R: ReadAt>(
    loc: &Located<'_, R>,
    binary_ns: u32,
    width: u64,
) -> Result<Option<Error>> {
    let header_blocks = if loc.major >= 2 {
        let extra = endian::u32_at(loc.head, 3506, loc.endian);
        1 + if extra <= 65535 { extra } else { 0 }
    } else {
        1
    };
    let header_len = u64::from(header_blocks) * TRACE_HEADER;
    let stride = stride_of(header_blocks, binary_ns, width);
    let mut word = [0u8; 2];
    let mut off = loc.data_start;
    let mut trace = 0u64;
    let mut first = None;
    while off + header_len <= loc.file_len {
        loc.reader.read_at(off + 114, &mut word)?;
        let declared = u32::from(endian::u16_at(&word, 0, loc.endian));
        let len = if declared == 0 {
            stride
        } else {
            stride_of(header_blocks, declared, width)
        };
        if len != stride && first.is_none() {
            first = Some(Error::TraceLength {
                trace,
                expected: stride,
                got: len,
            });
        }
        off += len;
        trace += 1;
    }
    Ok(if off == loc.file_len { first } else { None })
}

fn geometry_from(cols: &TraceCols, options: &OpenOptions, ntraces: usize) -> Result<Geometry> {
    if let Some((min, max)) = geometry::absurd_box(&cols.inline, &cols.crossline) {
        return Err(Error::Geometry {
            min,
            max,
            n_traces: ntraces as u64,
        });
    }
    // A regular grid is checked in one parallel pass; `infer` hashes every key.
    let (il, xl, off) = (&cols.inline, &cols.crossline, &cols.offset);
    let grid = Regular::detect(ntraces.min(il.len()), |t| {
        Ok::<_, Error>([il[t], xl[t], off[t]])
    })?;
    if let Some(grid) = grid.filter(|g| g.first_mismatch(il, xl, off).is_none()) {
        return Ok(grid.geometry(
            options.inline_byte,
            options.crossline_byte,
            options.offset_byte,
        ));
    }
    let geometry = geometry::infer(
        &cols.inline,
        &cols.crossline,
        &cols.offset,
        options.inline_byte,
        options.crossline_byte,
        options.offset_byte,
    );
    if let Some((inline, crossline, _)) = geometry.duplicate {
        return Err(Error::DuplicateKey { inline, crossline });
    }
    Ok(geometry)
}

fn stride_of(header_blocks: u32, ns: u32, width: u64) -> u64 {
    u64::from(header_blocks) * TRACE_HEADER + u64::from(ns) * width
}

fn concat_samples(mut parts: Vec<Samples>) -> Samples {
    if parts.is_empty() {
        return Samples::F32(Vec::new());
    }
    let mut acc = parts.remove(0);
    for part in parts {
        match (&mut acc, part) {
            (Samples::F32(a), Samples::F32(b)) => a.extend(b),
            (Samples::F64(a), Samples::F64(b)) => a.extend(b),
            (Samples::I32(a), Samples::I32(b)) => a.extend(b),
            (Samples::U32(a), Samples::U32(b)) => a.extend(b),
            (Samples::I16(a), Samples::I16(b)) => a.extend(b),
            (Samples::U16(a), Samples::U16(b)) => a.extend(b),
            (Samples::I64(a), Samples::I64(b)) => a.extend(b),
            (Samples::U64(a), Samples::U64(b)) => a.extend(b),
            (Samples::I8(a), Samples::I8(b)) => a.extend(b),
            (Samples::U8(a), Samples::U8(b)) => a.extend(b),
            _ => unreachable!("mixed sample buffers"),
        }
    }
    acc
}

fn choose_endian(head: &[u8], file_len: u64, options: &OpenOptions) -> Result<Endian> {
    let raw_sig = [head[3296], head[3297], head[3298], head[3299]];
    let signature = u32::from_be_bytes(raw_sig);
    let marked = endian::signature(head);
    if let Some(forced) = options.endian {
        if let Some(marked) = marked {
            if marked != forced {
                return Err(Error::EndianMismatch { signature });
            }
        } else if SampleFormat::from_code(endian::u16_at(head, 3224, forced)).is_none() {
            let other = if forced == Endian::Little {
                Endian::Big
            } else {
                Endian::Little
            };
            if SampleFormat::from_code(endian::u16_at(head, 3224, other)).is_some() {
                return Err(Error::EndianMismatch { signature });
            }
        }
        return Ok(forced);
    }
    if let Some(marked) = marked {
        return Ok(marked);
    }
    let big = score(head, file_len, Endian::Big);
    let little = score(head, file_len, Endian::Little);
    if little > big {
        Ok(Endian::Little)
    } else {
        Ok(Endian::Big)
    }
}

fn score(head: &[u8], file_len: u64, endian: Endian) -> i32 {
    let format = endian::u16_at(head, 3224, endian);
    let ns = endian::u16_at(head, 3220, endian);
    let dt = endian::u16_at(head, 3216, endian);
    let rev = head[3500];
    let ntext = endian::i16_at(head, 3504, endian);
    let mut score = 0;
    if SampleFormat::from_code(format).is_some() {
        score += 8;
    } else {
        score -= 20;
    }
    if ns > 0 {
        score += 2;
    }
    if dt > 0 {
        score += 1;
    }
    if rev <= 2 {
        score += 1;
    }
    if (-1..64).contains(&i32::from(ntext)) {
        score += 1;
    }
    if let Some(format) = SampleFormat::from_code(format) {
        if ntext >= 0 {
            let start = TEXT_BYTES + BIN_BYTES + u64::from(ntext as u16) * TEXT_BYTES;
            if start <= file_len && ns > 0 {
                let stride = TRACE_HEADER + u64::from(ns) * format.width() as u64;
                let payload = file_len - start;
                if stride > 0 && payload % stride == 0 {
                    score += 12;
                }
            }
        }
    }
    score
}

fn read_interval(head: &[u8], endian: Endian, major: u8) -> u32 {
    let legacy = u32::from(endian::u16_at(head, 3216, endian));
    if major < 2 {
        return legacy;
    }
    let raw = &head[3272..3280];
    if raw.iter().all(|b| *b == 0) {
        return legacy;
    }
    let extended = endian::f64_at(head, 3272, endian);
    let rounded = extended.round();
    if extended.is_finite() && rounded > 0.0 && rounded <= f64::from(u32::MAX) {
        rounded as u32
    } else {
        legacy
    }
}

/// The number of extended textual blocks and the offset of the first trace.
///
/// A count in bytes 3505-3506 is the number of blocks. A rev 2 first-trace
/// offset (bytes 3521-3528) moves the first trace but never adds blocks, so a
/// wrong offset cannot make open decode the file as text. A count of -1 is
/// ended by the `EndText` stanza, searched one block at a time.
fn locate_text<R: ReadAt>(
    reader: &R,
    head: &[u8],
    file_len: u64,
    endian: Endian,
    major: u8,
) -> Result<(usize, u64)> {
    let ntext = endian::i16_at(head, 3504, endian);
    let first_trace = Some(if major >= 2 {
        endian::u64_at(head, 3520, endian)
    } else {
        0
    })
    .filter(|&off| {
        off >= TEXT_BYTES + BIN_BYTES
            && off <= file_len
            && (off - TEXT_BYTES - BIN_BYTES) % TEXT_BYTES == 0
    });
    if ntext >= 0 {
        let n = ntext as u64;
        let end = TEXT_BYTES + BIN_BYTES + n * TEXT_BYTES;
        if end > file_len {
            return Err(Error::Truncated {
                file_size: file_len,
                need: end,
                context: format!(
                    "for the {n} extended textual headers that binary header bytes 3505-3506 declare"
                ),
            });
        }
        return Ok((
            n as usize,
            first_trace.filter(|&off| off >= end).unwrap_or(end),
        ));
    }
    let limit = first_trace.unwrap_or(file_len);
    let mut buf = vec![0u8; TEXT_BYTES as usize];
    let mut off = TEXT_BYTES + BIN_BYTES;
    let found = loop {
        if off + TEXT_BYTES > limit {
            break false;
        }
        reader.read_at(off, &mut buf)?;
        off += TEXT_BYTES;
        if text::is_end_text(&buf) {
            break true;
        }
    };
    reader.release(TEXT_BYTES + BIN_BYTES..off);
    let blocks = |end: u64| ((end - TEXT_BYTES - BIN_BYTES) / TEXT_BYTES) as usize;
    match (found, first_trace) {
        (true, _) => Ok((blocks(off), first_trace.unwrap_or(off))),
        (false, Some(start)) => Ok((blocks(start), start)),
        (false, None) => Err(Error::Truncated {
            file_size: file_len,
            need: off + TEXT_BYTES,
            context: "to reach the ((SEG: EndText)) stanza that ends the extended textual \
                      headers when binary header bytes 3505-3506 are -1; the file has none"
                .into(),
        }),
    }
}

/// The primary block from `head`, then `n` extended blocks. Called once the
/// trace layout is known to fit the file.
fn read_text<R: ReadAt>(reader: &R, head: &[u8], n: usize) -> Result<Vec<TextHeader>> {
    let mut out = Vec::with_capacity(n + 1);
    out.push(text::decode_block(&head[..TEXT_BYTES as usize]));
    let mut buf = vec![0u8; TEXT_BYTES as usize];
    for i in 0..n as u64 {
        reader.read_at(TEXT_BYTES + BIN_BYTES + i * TEXT_BYTES, &mut buf)?;
        out.push(text::decode_block(&buf));
    }
    Ok(out)
}

/// A Seismic Unix file is trace headers and samples with no file header. Its
/// first trace header's sample count (bytes 115-116) then divides the file
/// into whole float32 traces.
fn seismic_unix(first: &[u8], file_len: u64) -> Option<String> {
    if first.len() < TRACE_HEADER as usize {
        return None;
    }
    [Endian::Little, Endian::Big].into_iter().find_map(|e| {
        let ns = endian::u16_at(first, 114, e);
        let stride = TRACE_HEADER + 4 * u64::from(ns);
        (ns > 0 && file_len % stride == 0).then(|| {
            format!(
                "it looks like Seismic Unix (SU): {} traces of a 240-byte header and {ns} \
                 float32 samples, with no 3600-byte SEG-Y file header; convert it to SEG-Y \
                 first (for example with Seismic Unix segywrite)",
                file_len / stride
            )
        })
    })
}

fn choose_stride<R: ReadAt>(
    loc: &Located<'_, R>,
    binary_ns: u32,
    width: u64,
    payload: u64,
) -> Result<(u32, u32)> {
    let mut candidates = Vec::new();
    if loc.major >= 2 {
        let as_u32 = endian::u32_at(loc.head, 3506, loc.endian);
        let as_u16 = u32::from(endian::u16_at(loc.head, 3506, loc.endian));
        if as_u32 > 0 && as_u32 <= 65535 {
            candidates.push(as_u32);
        }
        if as_u16 > 0 && as_u16 != as_u32 {
            candidates.push(as_u16);
        }
    }
    candidates.push(0);
    candidates.dedup();

    for add0 in candidates {
        let (blocks, ns) = resolve_ns(loc, add0, binary_ns)?;
        let stride = stride_of(blocks, ns, width);
        if stride > 0 && payload % stride == 0 {
            return Ok((blocks, ns));
        }
    }
    let stride = TRACE_HEADER + u64::from(binary_ns) * width;
    let whole = payload.checked_div(stride).unwrap_or(0);
    let remainder = payload.checked_rem(stride).unwrap_or(payload);
    Err(Error::SampleCountMismatch {
        file_len: loc.file_len,
        expected: loc.data_start + whole * stride,
        remainder,
    })
}

fn resolve_ns<R: ReadAt>(
    loc: &Located<'_, R>,
    additional: u32,
    binary_ns: u32,
) -> Result<(u32, u32)> {
    if additional == 0 || loc.major < 2 {
        return Ok((1, binary_ns));
    }
    let need = TRACE_HEADER * u64::from(1 + additional.min(8));
    if loc.data_start + need > loc.file_len {
        return Ok((1 + additional, binary_ns));
    }
    let mut header = vec![0u8; need as usize];
    loc.reader.read_at(loc.data_start, &mut header)?;
    let n16 = endian::u16_at(&header, 114, loc.endian);
    let ext_ns = if header.len() >= 240 + 140 {
        endian::u32_at(&header, 240 + 136, loc.endian)
    } else {
        0
    };
    let mut ns = binary_ns;
    if ext_ns > 0 && (n16 == 0 || n16 == 32767 || binary_ns == 0 || binary_ns >= 32767) {
        ns = ext_ns;
    }
    let nthe = if header.len() >= 240 + 158 {
        endian::u16_at(&header, 240 + 156, loc.endian)
    } else {
        0
    };
    let add = if nthe > 0 {
        u32::from(nthe)
    } else {
        additional
    };
    Ok((1 + add, ns))
}

/// Predict a regular grid from a few dozen headers, then check it against the
/// last trace and [`FAST_SAMPLES`] headers spread over the file. `None` sends
/// the caller to the full scan, which then reports whatever is wrong.
fn assume_grid<R: ReadAt>(reader: &R, plan: &ScanPlan) -> Result<Option<Regular>> {
    let n = plan.ntraces;
    let words = |t: usize| plan.words(reader, t);
    let Some(grid) = Regular::detect(n, |t| words(t).map(|w| w.0))? else {
        return Ok(None);
    };
    if grid.absurd(n) {
        return Ok(None);
    }
    // Half evenly spaced, half from a fixed-seed generator so the probes do
    // not all share one position within a line.
    let mut probes: Vec<usize> = (0..FAST_SAMPLES / 2)
        .map(|i| i * (n - 1) / (FAST_SAMPLES / 2 - 1))
        .collect();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..FAST_SAMPLES / 2 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        probes.push((state >> 33) as usize % n);
    }
    probes.push(n - 1);
    let fits = probes.par_iter().try_fold(
        || true,
        |ok, &t| {
            let (key, ns) = words(t)?;
            Ok::<_, Error>(ok && key == grid.predict(t) && (plan.fixed || ns == plan.ns))
        },
    );
    let fits: Vec<bool> = fits.collect::<Result<_>>()?;
    Ok(fits.into_iter().all(|ok| ok).then_some(grid))
}

/// Identity of the bytes the index sidecar summarises: the file head, the
/// first and last trace headers, and the parameters the scan depended on.
fn fingerprint<R: ReadAt>(reader: &R, plan: &ScanPlan, format: SampleFormat) -> Result<u64> {
    let mut hash = crate::index_cache::Fnv::default();
    let mut head = vec![0u8; plan.data_start as usize];
    reader.read_at(0, &mut head)?;
    hash.write(&head);
    let len = plan.header_len();
    for t in [0, plan.ntraces.saturating_sub(1)] {
        if plan.ntraces > 0 {
            let mut header = vec![0u8; len];
            reader.read_at(plan.data_start + t as u64 * plan.stride, &mut header)?;
            hash.write(&header);
        }
    }
    let endian = match plan.endian {
        Endian::Big => 0u8,
        Endian::Little => 1,
        Endian::Pair => 2,
    };
    hash.write(&[endian, plan.major, plan.fixed as u8]);
    for v in plan.keys {
        hash.write(&v.to_le_bytes());
    }
    hash.write(&format.code().to_le_bytes());
    for v in [plan.data_start, plan.stride, plan.ntraces as u64] {
        hash.write(&v.to_le_bytes());
    }
    hash.write(&plan.ns.to_le_bytes());
    hash.write(&plan.header_blocks.to_le_bytes());
    Ok(hash.finish())
}

fn first_ns_mismatch(cols: &TraceCols, plan: &ScanPlan) -> Option<Broken> {
    if plan.fixed {
        return None;
    }
    let t = cols
        .declared_ns
        .par_iter()
        .position_first(|&ns| ns != plan.ns)?;
    Some(Broken::Samples {
        trace: t as u64,
        binary_ns: plan.ns,
        trace_ns: cols.declared_ns[t],
    })
}

fn check_declared_ns(cols: &TraceCols, plan: &ScanPlan) -> Result<()> {
    first_ns_mismatch(cols, plan).map_or(Ok(()), |b| Err(b.error()))
}

fn past_end(what: &str) -> Error {
    Error::Io(io::Error::new(
        ErrorKind::UnexpectedEof,
        format!("{what} extends past the end of file"),
    ))
}

fn read_columns<R: ReadAt + ?Sized>(reader: &R, plan: &ScanPlan) -> Result<TraceCols> {
    let header_blocks = plan.header_blocks;
    let file_ns = plan.ns;
    let ntraces = plan.ntraces;
    let stride = plan.stride;
    let header_len = plan.header_len();
    let endian = plan.endian;
    let [inline_byte, crossline_byte, offset_byte] = plan.keys;
    let all = reader.as_bytes();
    let row = |i: usize| -> Result<Row> {
        let off = plan.data_start + i as u64 * stride;
        let mut buf;
        let header = match all {
            Some(all) => all
                .get(off as usize..off as usize + header_len)
                .ok_or_else(|| past_end("trace header"))?,
            None => {
                buf = vec![0u8; header_len];
                if header_len > 0 {
                    reader.read_at(off, &mut buf)?;
                }
                &buf[..]
            }
        };
        Ok(Row {
            inline: field_i32(header, endian, inline_byte),
            crossline: field_i32(header, endian, crossline_byte),
            offset: field_i32(header, endian, offset_byte),
            scalco: word_i16(header, endian, 70),
            cdp_x: word_i32(header, endian, 180),
            cdp_y: word_i32(header, endian, 184),
            source_x: word_i32(header, endian, 72),
            source_y: word_i32(header, endian, 76),
            group_x: word_i32(header, endian, 80),
            group_y: word_i32(header, endian, 84),
            trace_id: word_i16(header, endian, 28),
            declared_ns: declared_ns(header, endian, header_blocks, file_ns, plan.major),
        })
    };
    // The mapped pages of each chunk are released once its rows are read, so
    // the file-backed resident set stays under the size of the rows themselves.
    let chunk =
        (ntraces as u64 * std::mem::size_of::<Row>() as u64 / stride.max(1)).max(1) as usize;
    let mut rows = Vec::with_capacity(ntraces);
    for first in (0..ntraces).step_by(chunk) {
        let last = (first + chunk).min(ntraces);
        let part: Result<Vec<Row>> = (first..last).into_par_iter().map(row).collect();
        rows.extend(part?);
        reader.release(
            plan.data_start + first as u64 * stride..plan.data_start + last as u64 * stride,
        );
    }
    let mut cols = TraceCols {
        inline: Vec::with_capacity(ntraces),
        crossline: Vec::with_capacity(ntraces),
        offset: Vec::with_capacity(ntraces),
        scalco: Vec::with_capacity(ntraces),
        cdp_x: Vec::with_capacity(ntraces),
        cdp_y: Vec::with_capacity(ntraces),
        source_x: Vec::with_capacity(ntraces),
        source_y: Vec::with_capacity(ntraces),
        group_x: Vec::with_capacity(ntraces),
        group_y: Vec::with_capacity(ntraces),
        trace_id: Vec::with_capacity(ntraces),
        declared_ns: Vec::with_capacity(ntraces),
    };
    for row in rows {
        cols.inline.push(row.inline);
        cols.crossline.push(row.crossline);
        cols.offset.push(row.offset);
        cols.scalco.push(row.scalco);
        cols.cdp_x.push(row.cdp_x);
        cols.cdp_y.push(row.cdp_y);
        cols.source_x.push(row.source_x);
        cols.source_y.push(row.source_y);
        cols.group_x.push(row.group_x);
        cols.group_y.push(row.group_y);
        cols.trace_id.push(row.trace_id);
        cols.declared_ns.push(row.declared_ns);
    }
    Ok(cols)
}

struct Row {
    inline: i32,
    crossline: i32,
    offset: i32,
    scalco: i16,
    cdp_x: i32,
    cdp_y: i32,
    source_x: i32,
    source_y: i32,
    group_x: i32,
    group_y: i32,
    trace_id: i16,
    declared_ns: u32,
}

fn field_i32(header: &[u8], endian: Endian, byte_1based: u16) -> i32 {
    word_i32(header, endian, usize::from(byte_1based.saturating_sub(1)))
}

fn word_i16(header: &[u8], endian: Endian, off: usize) -> i16 {
    if off + 2 > header.len() {
        return 0;
    }
    endian::i16_at(header, off, endian)
}

fn word_i32(header: &[u8], endian: Endian, off: usize) -> i32 {
    if off + 4 > header.len() {
        return 0;
    }
    endian::i32_at(header, off, endian)
}

fn declared_ns(header: &[u8], endian: Endian, header_blocks: u32, file_ns: u32, major: u8) -> u32 {
    if header.len() < 116 {
        return file_ns;
    }
    let n16 = endian::u16_at(header, 114, endian);
    if major >= 2 && header_blocks > 1 && (n16 == 0 || n16 == 32767) && header.len() >= 240 + 140 {
        let ext_ns = endian::u32_at(header, 240 + 136, endian);
        if ext_ns > 0 {
            return ext_ns;
        }
    }
    if n16 == 0 {
        file_ns
    } else {
        u32::from(n16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// An in-memory source that records the advice it is given.
    struct Logged {
        data: Vec<u8>,
        hints: Mutex<Vec<AccessHint>>,
    }

    impl ReadAt for Logged {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
            let lo = offset as usize;
            buf.copy_from_slice(&self.data[lo..lo + buf.len()]);
            Ok(())
        }

        fn access_hint(&self, hint: AccessHint) {
            self.hints.lock().unwrap().push(hint);
        }
    }

    fn logged_file() -> (File<Arc<Logged>>, Arc<Logged>) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/small.sgy");
        let data = std::fs::read(path).unwrap();
        let len = data.len() as u64;
        let src = Arc::new(Logged {
            data,
            hints: Mutex::new(Vec::new()),
        });
        let file = File::from_reader(src.clone(), len, &OpenOptions::default()).unwrap();
        src.hints.lock().unwrap().clear();
        (file, src)
    }

    fn take(src: &Logged) -> Vec<AccessHint> {
        std::mem::take(&mut *src.hints.lock().unwrap())
    }

    #[test]
    fn readahead_guards_nest_and_restore_random_on_the_last_drop() {
        use AccessHint::{Normal, Random};
        let (file, src) = logged_file();
        let outer = file.readahead();
        assert_eq!(take(&src), [Normal]);
        let inner = file.readahead();
        drop(outer);
        assert_eq!(take(&src), [], "an inner guard is still alive");
        drop(inner);
        assert_eq!(take(&src), [Random]);

        // Guards on other threads share the one count.
        std::thread::scope(|s| {
            let held: Vec<_> = (0..8).map(|_| s.spawn(|| file.readahead())).collect();
            let held: Vec<_> = held.into_iter().map(|h| h.join().unwrap()).collect();
            assert_eq!(take(&src), [Normal]);
            drop(held);
        });
        assert_eq!(take(&src), [Random]);
        assert_eq!(*file.readahead.lock().unwrap(), 0);
    }

    #[test]
    fn readahead_for_a_small_selection_changes_nothing() {
        let (file, src) = logged_file();
        let n = file.layout().n_traces as usize;
        assert!(n >= 16, "the fixture needs at least 16 traces");
        drop(file.readahead_for(n / 16 - 1));
        assert_eq!(take(&src), []);
        drop(file.readahead_for(n / 16));
        assert_eq!(take(&src), [AccessHint::Normal, AccessHint::Random]);
    }
}
