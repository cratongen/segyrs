//! In-place edits: header fields, binary and textual headers, and trace samples
//! written with `pwrite` at computed offsets. The file layout never changes.

use crate::pio::FileExt;
use std::fs::{File as FsFile, OpenOptions as FsOpenOptions};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::endian;
use crate::error::{Error, Result};
use crate::file::{File, Layout, OpenOptions, ReadAt};
use crate::header::HeaderField;
use crate::sample::{Gain, SampleFormat};

use super::binary::{BIN_BYTES, TEXT_BYTES, TRACE_HEADER};
use super::create::Overlay;
use super::encode::{encode_header_value, encode_samples, field_label, HeaderValues, SampleData};

/// Traces this short or shorter are patched by reading and rewriting whole
/// runs, which the edit benchmark measured faster than one `pwrite` per trace
/// (docs/writing.md). Longer traces get one `pwrite` each.
const COALESCE_STRIDE: u64 = 16 * 1024;

/// Positional reads from an open file, with an optional byte range replaced
/// by `patch`. The patch lets the reader's own parser judge an edit before
/// any byte of it reaches the disk.
struct Positional<'f> {
    file: &'f FsFile,
    patch: Option<(u64, &'f [u8])>,
}

impl ReadAt for Positional<'_> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.file.read_exact_at(buf, offset)?;
        if let Some((at, bytes)) = self.patch {
            let lo = offset.max(at);
            let hi = (offset + buf.len() as u64).min(at + bytes.len() as u64);
            if lo < hi {
                buf[(lo - offset) as usize..(hi - offset) as usize]
                    .copy_from_slice(&bytes[(lo - at) as usize..(hi - at) as usize]);
            }
        }
        Ok(())
    }
}

/// Parse sizes and coding only. The key bytes point past the trace header, so
/// every key reads as zero and the geometry step cannot refuse the file.
pub(crate) fn structure<R: ReadAt>(reader: R, len: u64, options: &OpenOptions) -> Result<File<R>> {
    let mut options = options.clone();
    options.inline_byte = u16::MAX;
    options.crossline_byte = u16::MAX;
    options.offset_byte = u16::MAX;
    // The gain convention changes values, not sizes.
    options.gain = options.gain.or(Some(crate::sample::Gain::Rev0));
    File::from_reader(reader, len, &options)
}

/// What the caller expects the file to be. A mismatch is a format change,
/// which is a new file, so the edit is refused before anything is written (S-35).
#[derive(Clone, Debug, Default)]
pub struct Expect {
    pub sample_format: Option<SampleFormat>,
    pub n_samples: Option<u32>,
    pub extra_trace_header_blocks: Option<u16>,
}

/// A SEG-Y file opened for in-place edits.
///
/// Every write checks its input in full, then writes. An error leaves the file
/// as it was. A multi-byte field can still tear if the machine stops in the
/// middle of a write. Call [`Editor::sync`] once after a batch for durability.
pub struct Editor {
    file: FsFile,
    path: PathBuf,
    options: OpenOptions,
    layout: Layout,
    gain: Option<Gain>,
    file_len: u64,
}

impl Editor {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, &OpenOptions::default(), &Expect::default())
    }

    /// Open with the same overrides used to read the file (S-03), and refuse
    /// with [`Error::FormatChange`] if the file differs from `expect`.
    pub fn open_with(
        path: impl AsRef<Path>,
        options: &OpenOptions,
        expect: &Expect,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = FsOpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::path("open for editing", path))?;
        let file_len = file.metadata()?.len();
        let parsed = structure(
            Positional {
                file: &file,
                patch: None,
            },
            file_len,
            options,
        )?;
        let layout = parsed.layout().clone();
        let gain = parsed.gain();
        drop(parsed);
        let changes = expect
            .sample_format
            .is_some_and(|f| f != layout.sample_format)
            || expect.n_samples.is_some_and(|n| n != layout.n_samples)
            || expect
                .extra_trace_header_blocks
                .is_some_and(|n| n != layout.extra_trace_header_blocks);
        if changes {
            return Err(Error::FormatChange);
        }
        Ok(Self {
            file,
            path: path.to_path_buf(),
            options: options.clone(),
            layout,
            gain,
            file_len,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    fn header_len(&self) -> usize {
        (1 + usize::from(self.layout.extra_trace_header_blocks)) * TRACE_HEADER
    }

    fn stride(&self) -> u64 {
        self.header_len() as u64
            + u64::from(self.layout.n_samples) * self.layout.sample_format.width() as u64
    }

    fn trace_offset(&self, trace: u64) -> u64 {
        self.layout.data_start + trace * self.stride()
    }

    fn check_trace(&self, trace: u64) -> Result<()> {
        if trace >= self.layout.n_traces {
            return Err(Error::Spec(format!(
                "trace {trace} is outside 0..{}",
                self.layout.n_traces
            )));
        }
        Ok(())
    }

    /// One field on every trace (S-08).
    pub fn write_header_column(&self, field: &HeaderField, values: HeaderValues<'_>) -> Result<()> {
        self.write_header_columns(
            &[Overlay {
                field: field.clone(),
                values,
            }],
            None,
        )
    }

    /// Several fields at once, on every trace or on the listed traces (S-08).
    ///
    /// All values are encoded and range-checked first. Then each trace gets one
    /// `pwrite` covering the touched bytes, in file order. When the fields
    /// leave gaps, those bytes are read first so they are written back as they
    /// were. Writing a per-trace sample count that differs from the file's is a
    /// format change and is refused.
    pub fn write_header_columns(
        &self,
        overlays: &[Overlay<'_>],
        traces: Option<&[u64]>,
    ) -> Result<()> {
        if overlays.is_empty() {
            return Ok(());
        }
        let n = traces
            .map(|t| t.len() as u64)
            .unwrap_or(self.layout.n_traces);
        if let Some(traces) = traces {
            for &t in traces {
                self.check_trace(t)?;
            }
        }
        let header_len = self.header_len();
        let mut lo = usize::MAX;
        let mut hi = 0;
        let mut covered = vec![false; header_len];
        for o in overlays {
            let f = &o.field;
            let at = usize::from(f.byte).wrapping_sub(1);
            if f.byte == 0 || at + f.dtype.width() > header_len {
                return Err(Error::Spec(format!(
                    "header field {} is outside the {header_len}-byte trace header",
                    field_label(f)
                )));
            }
            if let Some(len) = o.values.count() {
                if len as u64 != n {
                    return Err(Error::Spec(format!(
                        "header field {} has {len} values for {n} traces",
                        field_label(f)
                    )));
                }
            }
            lo = lo.min(at);
            hi = hi.max(at + f.dtype.width());
            covered[at..at + f.dtype.width()].fill(true);
        }
        let span = hi - lo;
        let gaps = covered[lo..hi].iter().any(|c| !c);
        let touches_ns = lo < 116 && hi > 114;
        let endian = self.layout.source_endian;
        let ns16 = u16::try_from(self.layout.n_samples).unwrap_or(0);
        let mut patches = vec![0u8; n as usize * span];
        patches
            .par_chunks_mut(span)
            .enumerate()
            .map(|(i, patch)| -> Result<()> {
                let trace = traces.map(|t| t[i]).unwrap_or(i as u64);
                if gaps || touches_ns {
                    self.file
                        .read_exact_at(patch, self.trace_offset(trace) + lo as u64)?;
                }
                for o in overlays {
                    let at = usize::from(o.field.byte - 1) - lo;
                    let dst = &mut patch[at..at + o.field.dtype.width()];
                    encode_header_value(&o.values, i, &o.field, endian, dst)?;
                }
                if touches_ns && lo <= 114 && hi >= 116 {
                    let ns = endian::u16_at(patch, 114 - lo, endian);
                    if ns != 0 && ns != ns16 {
                        return Err(Error::FormatChange);
                    }
                }
                Ok(())
            })
            .find_first(Result::is_err)
            .unwrap_or(Ok(()))?;
        // (trace, patch index) in file order.
        let mut order: Vec<(u64, usize)> = (0..n as usize)
            .map(|i| (traces.map(|t| t[i]).unwrap_or(i as u64), i))
            .collect();
        if traces.is_some() {
            order.sort_unstable();
        }
        let stride = self.stride();
        if stride <= COALESCE_STRIDE {
            self.write_runs(&order, &patches, span, lo as u64)
        } else {
            for &(trace, i) in &order {
                let patch = &patches[i * span..(i + 1) * span];
                self.file
                    .write_all_at(patch, self.trace_offset(trace) + lo as u64)?;
            }
            Ok(())
        }
    }

    /// Runs of consecutive traces: read the run, patch it in memory, write it
    /// back in one `pwrite`. The sample bytes go back unchanged.
    fn write_runs(
        &self,
        order: &[(u64, usize)],
        patches: &[u8],
        span: usize,
        lo: u64,
    ) -> Result<()> {
        let stride = self.stride();
        let per_buf = (super::create::available_memory() / 64 / stride).max(1);
        let mut buf = Vec::new();
        let mut k = 0;
        while k < order.len() {
            let first = order[k].0;
            let mut end = k + 1;
            while end < order.len()
                && order[end].0 == order[end - 1].0 + 1
                && ((end - k) as u64) < per_buf
            {
                end += 1;
            }
            if end - k == 1 {
                let (trace, i) = order[k];
                self.file.write_all_at(
                    &patches[i * span..(i + 1) * span],
                    self.trace_offset(trace) + lo,
                )?;
            } else {
                // The run ends at the last patched byte, not the last sample.
                let len = (end - k - 1) as u64 * stride + lo + span as u64;
                buf.resize(len as usize, 0);
                let at = self.trace_offset(first);
                self.file.read_exact_at(&mut buf, at)?;
                for &(trace, i) in &order[k..end] {
                    let o = ((trace - first) * stride + lo) as usize;
                    buf[o..o + span].copy_from_slice(&patches[i * span..(i + 1) * span]);
                }
                self.file.write_all_at(&buf, at)?;
            }
            k = end;
        }
        Ok(())
    }

    /// Replace one trace's samples (S-32). The length must equal the file's
    /// sample count, or `TraceLength` is returned before any byte is written.
    /// Values are encoded in the file's own format; see [`super::encode`] for
    /// the conversion rules.
    pub fn write_trace_samples(&self, trace: u64, data: SampleData<'_>) -> Result<()> {
        self.check_trace(trace)?;
        let ns = u64::from(self.layout.n_samples);
        if data.len() as u64 != ns {
            return Err(Error::TraceLength {
                trace,
                expected: ns,
                got: data.len() as u64,
            });
        }
        let format = self.layout.sample_format;
        let mut buf = vec![0u8; data.len() * format.width()];
        encode_samples(data, format, self.layout.source_endian, self.gain, &mut buf)
            .map_err(|bad| bad.into_error(trace, format))?;
        self.file
            .write_all_at(&buf, self.trace_offset(trace) + self.header_len() as u64)?;
        Ok(())
    }

    /// Replace the samples of traces `first..first + data.len() / n_samples`.
    /// Encoding runs in parallel; the writes go out in file order.
    pub fn write_traces(&self, first: u64, data: SampleData<'_>) -> Result<()> {
        let ns = self.layout.n_samples;
        let len = data.len() as u64;
        let (count, rest) = match len.checked_div(u64::from(ns)) {
            Some(count) => (count, len % u64::from(ns)),
            None => (0, len),
        };
        if rest != 0 || (len == 0 && ns > 0) {
            return Err(Error::TraceLength {
                trace: first + count,
                expected: u64::from(ns),
                got: rest,
            });
        }
        if count == 0 {
            return Ok(());
        }
        self.check_trace(first + count - 1)?;
        let format = self.layout.sample_format;
        let endian = self.layout.source_endian;
        let width = ns as usize * format.width();
        let mut buf = vec![0u8; count as usize * width];
        buf.par_chunks_mut(width)
            .enumerate()
            .map(|(i, out)| {
                let s = i * ns as usize;
                encode_samples(
                    data.slice(s, s + ns as usize),
                    format,
                    endian,
                    self.gain,
                    out,
                )
                .map_err(|bad| bad.into_error(first + i as u64, format))
            })
            .find_first(Result::is_err)
            .unwrap_or(Ok(()))?;
        let header_len = self.header_len() as u64;
        for (i, chunk) in buf.chunks_exact(width).enumerate() {
            self.file
                .write_all_at(chunk, self.trace_offset(first + i as u64) + header_len)?;
        }
        Ok(())
    }

    /// Replace the 400-byte binary header. Sample format, sample count, byte
    /// order, extra header blocks, and textual-block count must stay the same,
    /// or the edit is refused with [`Error::FormatChange`] (S-35).
    pub fn write_binary_header(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() != BIN_BYTES {
            return Err(Error::Spec(format!(
                "binary header is {} bytes, not 400",
                bytes.len()
            )));
        }
        self.patch(TEXT_BYTES as u64, bytes)
    }

    /// Replace textual block `index` (0 is the primary block). The framing
    /// must stay the same: a block whose edit moves the first trace is refused.
    pub fn write_textual_block(&mut self, index: usize, bytes: &[u8]) -> Result<()> {
        if bytes.len() != TEXT_BYTES {
            return Err(Error::Spec(format!(
                "textual block is {} bytes, not 3200",
                bytes.len()
            )));
        }
        if index > usize::from(self.layout.extended_textual_blocks) {
            return Err(Error::Spec(format!(
                "textual block {index} does not exist; the file has {}",
                1 + self.layout.extended_textual_blocks
            )));
        }
        let at = if index == 0 {
            0
        } else {
            (TEXT_BYTES + BIN_BYTES + (index - 1) * TEXT_BYTES) as u64
        };
        self.patch(at, bytes)
    }

    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<()> {
        let parsed = structure(
            Positional {
                file: &self.file,
                patch: Some((at, bytes)),
            },
            self.file_len,
            &self.options,
        )
        .map_err(|_| Error::FormatChange)?;
        let new = parsed.layout().clone();
        drop(parsed);
        let old = &self.layout;
        if new.sample_format != old.sample_format
            || new.n_samples != old.n_samples
            || new.n_traces != old.n_traces
            || new.data_start != old.data_start
            || new.source_endian != old.source_endian
            || new.extra_trace_header_blocks != old.extra_trace_header_blocks
        {
            return Err(Error::FormatChange);
        }
        self.file.write_all_at(bytes, at)?;
        self.layout = new;
        Ok(())
    }

    /// `fdatasync` once, after a batch of edits.
    pub fn sync(&self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }
}
