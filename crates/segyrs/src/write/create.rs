//! Create a new SEG-Y file: from arrays, from an opened file, or from a mix.

use crate::pio::FileExt;
use std::fs::File as FsFile;
use std::path::Path;
use std::sync::mpsc;

use rayon::prelude::*;

use crate::endian::{self, Endian};
use crate::error::{Error, Result};
use crate::file::{File, ReadAt, Revision, Storage};
use crate::header::HeaderField;
use crate::sample::{self, Gain, SampleFormat};
use crate::text::TextEncoding;

use super::binary::{self, *};
use super::encode::{
    encode_header_value, encode_samples, field_label, reorder_samples, HeaderValues, SampleData,
};
use super::publish::publish;

/// How to write. `None` keeps what the source file has.
#[derive(Clone, Debug)]
pub struct WriteOptions {
    /// Target sample format. `None` keeps the source format, so an IBM file
    /// stays IBM. Required when there is no source file (roadmap decision 3).
    pub sample_format: Option<SampleFormat>,
    /// Target byte order. `None` keeps the source order, big-endian otherwise.
    pub endian: Option<Endian>,
    /// Target revision. `None` keeps the source revision. A new file is rev 1
    /// unless it needs rev 2 (little-endian, more than 65535 samples or
    /// microseconds, or extra trace-header blocks), which makes it rev 2.1.
    pub revision: Option<Revision>,
    /// Gain convention for writing format 4. `None` keeps the source convention.
    pub gain: Option<Gain>,
    /// `fdatasync` the file and `fsync` its directory before returning.
    /// `false` is for scratch files that may be lost on a crash.
    pub sync: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            sample_format: None,
            endian: None,
            revision: None,
            gain: None,
            sync: true,
        }
    }
}

/// Values for one trace-header field, written over the base header of every trace.
#[derive(Clone, Debug)]
pub struct Overlay<'a> {
    pub field: HeaderField,
    pub values: HeaderValues<'a>,
}

/// Traces and samples taken from an opened file.
#[derive(Clone, Copy)]
pub struct Source<'a, R: ReadAt = Storage> {
    pub file: &'a File<R>,
    /// Source trace index for each output trace. `None` is every trace in order.
    pub traces: Option<&'a [u64]>,
    /// First source sample of the output window. The window length is
    /// [`NewFile::n_samples`].
    pub first_sample: u32,
}

/// Everything the writer needs for one new file. There is no second step: the
/// headers, the samples, and the binary-header values are written together.
///
/// Trace headers start as the source file's raw headers when there is a
/// source, and as 240 zero bytes otherwise (X-26). Overlays are written on top.
/// Samples are `samples` when given, otherwise the source samples.
pub struct NewFile<'a, R: ReadAt = Storage> {
    /// Raw textual blocks: the 3200-byte primary block, then each extended block.
    pub text: Vec<u8>,
    /// The 400 bytes to start from, in the source byte order when there is a
    /// source and in the target order otherwise. The writer changes only the
    /// words whose value differs from what these bytes already say, so an
    /// unedited header is written back byte for byte (X-22). `None` builds one.
    pub binary_header: Option<Vec<u8>>,
    pub n_traces: u64,
    /// Samples per trace (S-33).
    pub n_samples: u32,
    /// Integer microseconds, written as given and never derived from a float axis (S-29).
    pub sample_interval_us: u32,
    /// Extra 240-byte trace-header blocks after the standard one (rev 2).
    pub extra_trace_header_blocks: u16,
    pub overlays: Vec<Overlay<'a>>,
    /// Trace-major samples, `n_traces * n_samples` values.
    pub samples: Option<SampleData<'a>>,
    pub source: Option<Source<'a, R>>,
}

impl<'a> NewFile<'a, Storage> {
    /// A file built from nothing: default textual header, zero trace headers.
    /// Set `samples` and `overlays`, then call [`create`] with a `sample_format`.
    pub fn new(n_traces: u64, n_samples: u32, sample_interval_us: u32) -> Self {
        Self {
            text: Vec::new(),
            binary_header: None,
            n_traces,
            n_samples,
            sample_interval_us,
            extra_trace_header_blocks: 0,
            overlays: Vec::new(),
            samples: None,
            source: None,
        }
    }
}

impl<'a, R: ReadAt> NewFile<'a, R> {
    /// A copy of `file`: same textual blocks, binary header, trace headers, and
    /// samples. Change fields to select traces, crop samples, or add overlays.
    pub fn from_file(file: &'a File<R>) -> Result<Self> {
        let layout = file.layout();
        let mut text = vec![0u8; layout.data_start as usize - BIN_BYTES];
        file.reader().read_at(0, &mut text[..TEXT_BYTES])?;
        if text.len() > TEXT_BYTES {
            file.reader()
                .read_at((TEXT_BYTES + BIN_BYTES) as u64, &mut text[TEXT_BYTES..])?;
        }
        Ok(Self {
            text,
            binary_header: Some(file.binary_header_bytes().to_vec()),
            n_traces: layout.n_traces,
            n_samples: layout.n_samples,
            sample_interval_us: layout.sample_interval_us,
            extra_trace_header_blocks: layout.extra_trace_header_blocks,
            overlays: Vec::new(),
            samples: None,
            source: Some(Source {
                file,
                traces: None,
                first_sample: 0,
            }),
        })
    }
}

/// Write `spec` to `path` as a new file and return the number of bytes.
///
/// The file appears at `path` complete or not at all: the bytes go to a temp
/// file in the same directory, which is preallocated, flushed with
/// `fdatasync`, renamed over `path`, and then the directory is flushed.
/// Writing over the file a `Source` maps is safe: the map keeps the old inode.
pub fn create<R: ReadAt>(
    path: impl AsRef<Path>,
    spec: &NewFile<'_, R>,
    options: &WriteOptions,
) -> Result<u64> {
    let plan = Plan::new(spec, options)?;
    // Copying from a source reads most of it in order.
    let _ra = plan
        .src
        .as_ref()
        .map(|s| s.file.readahead_for(spec.n_traces as usize));
    let total = plan.total_len();
    publish(path.as_ref(), total, options.sync, |file| plan.write(file))?;
    Ok(total)
}

/// Copy `file` to `path`, keeping every byte unless `options` asks for a
/// different sample format, byte order, or revision.
pub fn copy<R: ReadAt>(
    file: &File<R>,
    path: impl AsRef<Path>,
    options: &WriteOptions,
) -> Result<u64> {
    create(path, &NewFile::from_file(file)?, options)
}

/// Values the base binary header already states. The writer leaves a word
/// alone when the target value equals the stated one.
struct Stated {
    ns: u32,
    interval: u32,
    ext_text: Option<u16>,
    extra_blocks: u16,
    revision: Revision,
}

struct Plan<'s, 'a, R: ReadAt> {
    spec: &'s NewFile<'a, R>,
    format: SampleFormat,
    endian: Endian,
    gain: Option<Gain>,
    header_len: usize,
    stride: usize,
    head: Vec<u8>,
    words: Vec<(usize, usize)>,
    src: Option<SourcePlan<'s, R>>,
}

struct SourcePlan<'s, R: ReadAt> {
    file: &'s File<R>,
    traces: Option<&'s [u64]>,
    first: u32,
    ns: u32,
    interval: u32,
    format: SampleFormat,
    endian: Endian,
    gain: Option<Gain>,
    stride: u64,
    data_start: u64,
}

fn spec_err(msg: impl Into<String>) -> Error {
    Error::Spec(msg.into())
}

impl<'s, 'a, R: ReadAt> Plan<'s, 'a, R> {
    fn new(spec: &'s NewFile<'a, R>, options: &WriteOptions) -> Result<Self> {
        let src = spec.source.as_ref().map(|s| {
            let layout = s.file.layout();
            SourcePlan {
                file: s.file,
                traces: s.traces,
                first: s.first_sample,
                ns: layout.n_samples,
                interval: layout.sample_interval_us,
                format: layout.sample_format,
                endian: layout.source_endian,
                gain: s.file.gain(),
                stride: s.file.stride(),
                data_start: layout.data_start,
            }
        });
        let format = match (options.sample_format, &src) {
            (Some(f), _) => f,
            (None, Some(s)) => s.format,
            (None, None) => {
                return Err(spec_err(
                    "sample_format is required for a file with no source",
                ))
            }
        };
        let endian = options
            .endian
            .or(src.as_ref().map(|s| s.endian))
            .unwrap_or(Endian::Big);
        let gain = options.gain.or(src.as_ref().and_then(|s| s.gain));
        if format == SampleFormat::FixedGain && gain.is_none() && spec.samples.is_some() {
            return Err(Error::Format4NeedsGain);
        }
        let ns = spec.n_samples;
        let interval = spec.sample_interval_us;
        let extra = spec.extra_trace_header_blocks;
        if interval == 0 && spec.n_traces > 0 && src.as_ref().is_none_or(|s| s.interval != 0) {
            return Err(spec_err(
                "sample_interval_us is 0; a file with traces needs a nonzero sample interval",
            ));
        }
        let base_endian = src.as_ref().map(|s| s.endian).unwrap_or(endian);
        let stated = stated(spec, src.as_ref(), base_endian);
        let needs_rev2 = endian != Endian::Big || ns > 65535 || interval > 65535 || extra > 0;
        let revision = match (options.revision, &stated) {
            (Some(r), _) => r,
            (None, Some(st)) => st.revision,
            (None, None) if needs_rev2 => Revision::Rev21,
            (None, None) => Revision::Rev1,
        };
        let rev2 = revision.major() >= 2;
        if !rev2 && (ns > 65535 || interval > 65535 || extra > 0) {
            return Err(spec_err(format!(
                "{} samples, {} us, and {} extra trace-header blocks need rev 2",
                ns, interval, extra
            )));
        }

        // Text.
        let text = if spec.text.is_empty() {
            default_text(revision, TextEncoding::Ebcdic)
        } else {
            spec.text.clone()
        };
        if text.len() % TEXT_BYTES != 0 {
            return Err(spec_err(format!(
                "textual header is {} bytes, not a multiple of 3200",
                text.len()
            )));
        }
        let n_ext = text.len() / TEXT_BYTES - 1;
        let n_ext16 = i16::try_from(n_ext)
            .map_err(|_| spec_err(format!("{n_ext} extended textual blocks")))?;

        // Traces and samples.
        let header_len = (1 + usize::from(extra)) * TRACE_HEADER;
        let stride = header_len + ns as usize * format.width();
        if let Some(s) = &src {
            let available = s
                .traces
                .map(|t| t.len() as u64)
                .unwrap_or(s.file.layout().n_traces);
            if available != spec.n_traces {
                return Err(spec_err(format!(
                    "n_traces is {} but the source selection has {available}",
                    spec.n_traces
                )));
            }
            if let Some(traces) = s.traces {
                let n = s.file.layout().n_traces;
                if let Some(bad) = traces.iter().find(|&&t| t >= n) {
                    return Err(spec_err(format!("source trace {bad} is outside 0..{n}")));
                }
                s.file.verify_traces(traces)?;
            }
            if spec.samples.is_none() && u64::from(s.first) + u64::from(ns) > u64::from(s.ns) {
                return Err(spec_err(format!(
                    "sample window {}..{} is outside the source's {} samples",
                    s.first,
                    u64::from(s.first) + u64::from(ns),
                    s.ns
                )));
            }
            if extra != s.file.layout().extra_trace_header_blocks {
                return Err(spec_err(
                    "a copy keeps the source's extra trace-header blocks",
                ));
            }
            if extra > 0 && endian != s.endian {
                return Err(spec_err(
                    "changing the byte order of extra trace-header blocks is not supported: their layout is not known",
                ));
            }
        } else if spec.samples.is_none() && ns > 0 && spec.n_traces > 0 {
            return Err(spec_err("no samples and no source file"));
        }
        if let Some(data) = &spec.samples {
            check_sample_len(data.len() as u64, spec.n_traces, ns)?;
        }
        for overlay in &spec.overlays {
            let f = &overlay.field;
            if f.byte == 0 || usize::from(f.byte - 1) + f.dtype.width() > header_len {
                return Err(spec_err(format!(
                    "header field {} is outside the {header_len}-byte trace header",
                    field_label(f)
                )));
            }
            if let Some(n) = overlay.values.count() {
                if n as u64 != spec.n_traces {
                    return Err(spec_err(format!(
                        "header field {} has {n} values for {} traces",
                        field_label(f),
                        spec.n_traces
                    )));
                }
            }
        }

        // Binary header.
        let mut bin = match &spec.binary_header {
            Some(b) if b.len() != BIN_BYTES => {
                return Err(spec_err(format!(
                    "binary header is {} bytes, not 400",
                    b.len()
                )))
            }
            Some(b) => b.clone(),
            None => vec![0u8; BIN_BYTES],
        };
        let fresh = spec.binary_header.is_none();
        binary::reorder_binary(&mut bin, base_endian, endian);
        let st = stated.unwrap_or(Stated {
            ns: u32::MAX,
            interval: u32::MAX,
            ext_text: None,
            extra_blocks: u16::MAX,
            revision: Revision::Rev0,
        });
        let data_start = (TEXT_BYTES + BIN_BYTES + n_ext * TEXT_BYTES) as u64;
        if fresh || endian::u16_at(&bin, FORMAT, endian) != format.code() {
            put_u16(&mut bin, FORMAT, format.code(), endian);
        }
        if fresh || st.ns != ns {
            put_u16(&mut bin, NS, u16::try_from(ns).unwrap_or(0), endian);
            if rev2 && (ns > 65535 || endian::u32_at(&bin, EXT_NS, endian) != 0) {
                put_u32(&mut bin, EXT_NS, ns, endian);
            }
        }
        if fresh || st.interval != interval {
            put_u16(
                &mut bin,
                INTERVAL,
                u16::try_from(interval).unwrap_or(0),
                endian,
            );
            if rev2 && (interval > 65535 || bin[EXT_INTERVAL..EXT_INTERVAL + 8] != [0; 8]) {
                put_f64(&mut bin, EXT_INTERVAL, f64::from(interval), endian);
            }
        }
        if fresh || st.revision != revision {
            let (major, minor) = revision_bytes(revision);
            bin[REV_MAJOR] = major;
            bin[REV_MINOR] = minor;
        }
        if fresh && revision != Revision::Rev0 {
            put_u16(&mut bin, FIXED_LENGTH, 1, endian);
        }
        if fresh || st.ext_text != Some(n_ext as u16) {
            put_i16(&mut bin, EXT_TEXT, n_ext16, endian);
        }
        if (fresh && extra > 0) || (!fresh && st.extra_blocks != extra) {
            put_u32(&mut bin, EXTRA_HEADERS, u32::from(extra), endian);
        }
        let signed = binary_signature(&bin).is_some();
        if rev2 && (fresh || (!signed && endian != base_endian)) {
            put_signature(&mut bin, endian);
        }
        if rev2 && !fresh {
            let stored = endian::u64_at(&bin, N_TRACES, endian);
            if stored != 0 && stored != spec.n_traces {
                put_u64(&mut bin, N_TRACES, spec.n_traces, endian);
            }
            let stored = endian::u64_at(&bin, FIRST_TRACE, endian);
            if stored != 0 && stored != data_start {
                put_u64(&mut bin, FIRST_TRACE, data_start, endian);
            }
        }

        let mut head = Vec::with_capacity(data_start as usize);
        head.extend_from_slice(&text[..TEXT_BYTES]);
        head.extend_from_slice(&bin);
        head.extend_from_slice(&text[TEXT_BYTES..]);

        let words = if src.as_ref().is_some_and(|s| s.endian != endian) {
            standard_words()
        } else {
            Vec::new()
        };
        Ok(Self {
            spec,
            format,
            endian,
            gain,
            header_len,
            stride,
            head,
            words,
            src,
        })
    }

    fn total_len(&self) -> u64 {
        self.head.len() as u64 + self.spec.n_traces * self.stride as u64
    }

    fn write(&self, file: &FsFile) -> Result<()> {
        file.write_all_at(&self.head, 0)?;
        let n = self.spec.n_traces;
        if n == 0 {
            return Ok(());
        }
        let per_chunk = chunk_traces(self.stride as u64).min(n);
        let start = self.head.len() as u64;
        let stride = self.stride;
        std::thread::scope(|scope| -> Result<()> {
            let (full_tx, full_rx) = mpsc::sync_channel::<(u64, Vec<u8>)>(1);
            let (empty_tx, empty_rx) = mpsc::channel::<Vec<u8>>();
            let writer = scope.spawn(move || -> Result<()> {
                for (offset, buf) in full_rx {
                    file.write_all_at(&buf, offset)?;
                    let _ = empty_tx.send(buf);
                }
                Ok(())
            });
            let mut first = 0u64;
            let mut result = Ok(());
            while first < n {
                let count = per_chunk.min(n - first);
                let mut buf = empty_rx.try_recv().unwrap_or_default();
                buf.resize(count as usize * stride, 0);
                let filled = buf
                    .par_chunks_mut(stride)
                    .with_min_len(1)
                    .enumerate()
                    .map(|(i, out)| self.fill(first + i as u64, out))
                    .find_first(Result::is_err)
                    .unwrap_or(Ok(()));
                if let Err(err) = filled {
                    result = Err(err);
                    break;
                }
                if full_tx.send((start + first * stride as u64, buf)).is_err() {
                    break;
                }
                first += count;
            }
            drop(full_tx);
            let written = writer.join().expect("writer thread panicked");
            result.and(written)
        })
    }

    /// Header and samples of output trace `t`, into exactly one stride.
    fn fill(&self, t: u64, out: &mut [u8]) -> Result<()> {
        let (header, samples) = out.split_at_mut(self.header_len);
        let ns = self.spec.n_samples;
        match &self.src {
            Some(s) => {
                let st = s.traces.map(|tr| tr[t as usize]).unwrap_or(t);
                let off = s.data_start + st * s.stride;
                s.file.reader().read_at(off, header)?;
                binary::reorder_trace_header(
                    &mut header[..TRACE_HEADER],
                    &self.words,
                    s.endian,
                    self.endian,
                );
                // Structural words that agreed with the source follow the new values.
                if ns != s.ns && endian::u16_at(header, 114, self.endian) == s.ns as u16 {
                    put_u16(header, 114, u16::try_from(ns).unwrap_or(0), self.endian);
                }
                let interval = self.spec.sample_interval_us;
                if interval != s.interval
                    && endian::u16_at(header, 116, self.endian) == s.interval as u16
                {
                    put_u16(
                        header,
                        116,
                        u16::try_from(interval).unwrap_or(0),
                        self.endian,
                    );
                }
                if self.spec.samples.is_none() {
                    self.copy_samples(s, st, samples, t)?;
                }
            }
            None => {
                header.fill(0);
                put_u16(header, 114, u16::try_from(ns).unwrap_or(0), self.endian);
                put_u16(
                    header,
                    116,
                    u16::try_from(self.spec.sample_interval_us).unwrap_or(0),
                    self.endian,
                );
                if ns > 65535 && self.header_len > TRACE_HEADER {
                    put_u32(header, TRACE_HEADER + 136, ns, self.endian);
                }
            }
        }
        for overlay in &self.spec.overlays {
            let at = usize::from(overlay.field.byte - 1);
            let dst = &mut header[at..at + overlay.field.dtype.width()];
            encode_header_value(
                &overlay.values,
                t as usize,
                &overlay.field,
                self.endian,
                dst,
            )?;
        }
        if let Some(data) = &self.spec.samples {
            let ns = ns as usize;
            let i = t as usize * ns;
            encode_samples(
                data.slice(i, i + ns),
                self.format,
                self.endian,
                self.gain,
                samples,
            )
            .map_err(|bad| bad.into_error(t, self.format))?;
        }
        Ok(())
    }

    fn copy_samples(&self, s: &SourcePlan<'_, R>, st: u64, out: &mut [u8], t: u64) -> Result<()> {
        let ns = self.spec.n_samples as usize;
        if ns == 0 {
            return Ok(());
        }
        let width = s.format.width();
        let base = s.data_start + st * s.stride + self.header_len as u64;
        let window = s.first as usize..s.first as usize + ns;
        let full = window.start == 0 && window.end == s.ns as usize;
        let pair_odd = width % 2 == 1 && (s.endian == Endian::Pair || self.endian == Endian::Pair);
        if s.format == self.format && (full || !pair_odd) {
            // Same format: raw bytes, reordered word by word if the byte order changes.
            s.file
                .reader()
                .read_at(base + (window.start * width) as u64, out)?;
            if s.endian != self.endian {
                reorder_samples(out, width, s.endian);
                reorder_samples(out, width, self.endian);
            }
            return Ok(());
        }
        let mut raw = vec![0u8; s.ns as usize * width];
        s.file.reader().read_at(base, &mut raw)?;
        if s.format == self.format {
            // Pairwise with an odd width: the swap runs over the whole trace.
            reorder_samples(&mut raw, width, s.endian);
            let piece = &mut raw[window.start * width..window.end * width];
            reorder_samples(piece, width, self.endian);
            out.copy_from_slice(piece);
            return Ok(());
        }
        let mut decoded = sample::empty_samples(s.format, s.ns as usize);
        sample::decode_trace(
            &raw,
            s.format,
            s.endian,
            s.gain,
            0,
            s.ns as usize,
            &mut decoded,
        );
        let data = SampleData::from(&decoded).slice(window.start, window.end);
        encode_samples(data, self.format, self.endian, self.gain, out)
            .map_err(|bad| bad.into_error(t, self.format))
    }
}

fn stated<R: ReadAt>(
    spec: &NewFile<'_, R>,
    src: Option<&SourcePlan<'_, R>>,
    base_endian: Endian,
) -> Option<Stated> {
    if let Some(s) = src {
        let layout = s.file.layout();
        return Some(Stated {
            ns: s.ns,
            interval: s.interval,
            ext_text: Some(layout.extended_textual_blocks),
            extra_blocks: layout.extra_trace_header_blocks,
            revision: layout.revision,
        });
    }
    let bin = spec
        .binary_header
        .as_ref()
        .filter(|b| b.len() == BIN_BYTES)?;
    let e = base_endian;
    let revision = Revision::from_bytes(bin[REV_MAJOR], bin[REV_MINOR]);
    let rev2 = revision.major() >= 2;
    let ext_ns = if rev2 {
        endian::u32_at(bin, EXT_NS, e)
    } else {
        0
    };
    let ns = if ext_ns > 0 {
        ext_ns
    } else {
        u32::from(endian::u16_at(bin, NS, e))
    };
    let ext_text = endian::i16_at(bin, EXT_TEXT, e);
    Some(Stated {
        ns,
        interval: u32::from(endian::u16_at(bin, INTERVAL, e)),
        ext_text: u16::try_from(ext_text).ok(),
        extra_blocks: if rev2 {
            endian::u32_at(bin, EXTRA_HEADERS, e).min(u32::from(u16::MAX)) as u16
        } else {
            0
        },
        revision,
    })
}

/// `len` samples for `n_traces` traces of `ns` samples, or `TraceLength` for
/// the first trace that would be short (S-32).
pub(crate) fn check_sample_len(len: u64, n_traces: u64, ns: u32) -> Result<()> {
    let ns = u64::from(ns);
    if len == n_traces * ns {
        return Ok(());
    }
    // The first trace whose slice is not exactly `ns` values.
    let trace = match len.checked_div(ns) {
        Some(whole) => whole.min(n_traces.saturating_sub(1)),
        None => 0,
    };
    Err(Error::TraceLength {
        trace,
        expected: ns,
        got: len - trace * ns,
    })
}

/// Traces per pipeline buffer. Up to three buffers are alive at once (one
/// filling, one queued, one writing), so the pipeline holds at most 3/64 of
/// the memory available when the write starts, and never less than one trace.
fn chunk_traces(stride: u64) -> u64 {
    let budget = available_memory() / 64;
    (budget / stride.max(1)).max(1)
}

/// `MemAvailable` from `/proc/meminfo`, or 1 GiB when it cannot be read.
pub(crate) fn available_memory() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(1 << 30)
}
