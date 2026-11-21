use crate::endian::Endian;
use crate::error::Result;
use crate::file::{File, ReadAt};
use crate::ops::{header_len, put_u16, put_u32};
use crate::sample::SampleFormat;

/// Everything a writer needs before the first trace.
///
/// `textual` and `binary` are the on-disk bytes of the source file, in its own
/// byte order. The operation has already patched the binary-header sample
/// count (and the trace-sorting code when asked), so a writer can copy them
/// as they are. Every trace that follows uses `sample_format`, `endian`,
/// `n_samples`, and `trace_header_len`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSpec {
    /// 3200 bytes per textual block, the standard block first, then the extended ones.
    pub textual: Vec<u8>,
    /// The 400-byte binary header.
    pub binary: Vec<u8>,
    pub sample_format: SampleFormat,
    pub endian: Endian,
    pub n_samples: u32,
    pub sample_interval_us: u32,
    /// 240 bytes per trace-header block, including rev 2 extension blocks.
    pub trace_header_len: usize,
    /// Number of traces the operation will write, when it knows it up front.
    pub n_traces: Option<u64>,
}

impl OutputSpec {
    /// The source file's headers, unchanged.
    pub fn from_file<R: ReadAt>(file: &File<R>) -> Result<Self> {
        let layout = file.layout();
        let mut textual = vec![0u8; 3200];
        file.reader().read_at(0, &mut textual)?;
        let ext = (layout.data_start - 3600) as usize;
        if ext > 0 {
            let mut more = vec![0u8; ext];
            file.reader().read_at(3600, &mut more)?;
            textual.extend_from_slice(&more);
        }
        Ok(Self {
            textual,
            binary: file.binary_header_bytes().to_vec(),
            sample_format: file.format(),
            endian: file.endian(),
            n_samples: layout.n_samples,
            sample_interval_us: layout.sample_interval_us,
            trace_header_len: header_len(file),
            n_traces: None,
        })
    }

    /// Bytes per output trace.
    pub fn trace_len(&self) -> usize {
        self.trace_header_len + self.n_samples as usize * self.sample_format.width()
    }

    /// Set the sample count, in the struct and in binary-header bytes 3221-3222
    /// (and 3269-3272 for rev 2 when that word is in use or the count needs it).
    pub fn set_samples(&mut self, ns: u32) {
        self.n_samples = ns;
        let short = u16::try_from(ns).unwrap_or(0);
        put_u16(&mut self.binary, 20, short, self.endian);
        let rev2 = self.binary[300] >= 2;
        let extended = crate::endian::u32_at(&self.binary, 68, self.endian);
        if rev2 && (extended != 0 || ns > u32::from(u16::MAX)) {
            put_u32(&mut self.binary, 68, ns, self.endian);
        }
    }

    /// Binary-header bytes 3229-3230, the trace-sorting code.
    pub fn set_sorting_code(&mut self, code: i16) {
        put_u16(&mut self.binary, 28, code as u16, self.endian);
    }
}

/// Where whole-volume operations send their traces.
///
/// The call order is `begin` once, then `write_trace` or `write_block` for each
/// trace in output order, then `finish` once. Header and sample bytes are in the
/// on-disk layout of [`OutputSpec`]: the source byte order and the source sample
/// format, never decoded and re-encoded. An operation that fails stops calling
/// the sink and returns the error without calling `finish`. A writer should then
/// discard its partial output.
pub trait TraceSink {
    fn begin(&mut self, spec: &OutputSpec) -> Result<()>;

    /// One trace: `header` is `trace_header_len` bytes, `samples` is
    /// `n_samples * width` bytes.
    fn write_trace(&mut self, header: &[u8], samples: &[u8]) -> Result<()>;

    /// Whole traces back to back, each `trace_len` bytes with the header first.
    /// A writer can override this to issue one large write.
    fn write_block(&mut self, block: &[u8], trace_len: usize, header_len: usize) -> Result<()> {
        for trace in block.chunks_exact(trace_len) {
            self.write_trace(&trace[..header_len], &trace[header_len..])?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()>;
}
