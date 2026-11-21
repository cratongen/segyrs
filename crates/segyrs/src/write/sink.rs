//! [`FileSink`]: the M2 writer behind the M3 [`TraceSink`] contract.
//!
//! An operation streams traces into the sink chunk by chunk, and the sink
//! appends them to a temp file beside the destination. `finish` flushes,
//! renames the temp file over the destination, and flushes the directory, so
//! the destination is the old file or the complete new one. An operation that
//! fails never calls `finish`, and dropping the sink removes the temp file.

use crate::pio::FileExt;
use std::fs::{self, File as FsFile, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::endian::Endian;
use crate::error::{Error, Result};
use crate::ops::{OutputSpec, TraceSink};

use super::encode::pair_swap;
use super::publish::{parent_dir, preallocate, sync_dir, temp_path, TempGuard};

/// Writes a [`TraceSink`] stream to a new SEG-Y file, published atomically.
///
/// Bytes are written as the operation hands them over: the source byte order
/// and sample format, so an IBM file stays IBM bit for bit. The only word the
/// sink changes is the rev 2 trace count (binary-header bytes 3513-3520), and
/// only when the source set it, to the number of traces actually written.
pub struct FileSink {
    dest: PathBuf,
    sync: bool,
    open: Option<Open>,
    traces: u64,
    bytes: u64,
}

struct Open {
    out: BufWriter<FsFile>,
    guard: TempGuard,
    endian: Endian,
    rev2_count: bool,
    trace_len: u64,
}

impl FileSink {
    /// A sink that publishes to `path`. `sync` flushes the data and the
    /// directory before `finish` returns (`false` is for scratch files).
    pub fn new(path: impl AsRef<Path>, sync: bool) -> Self {
        Self {
            dest: path.as_ref().to_path_buf(),
            sync,
            open: None,
            traces: 0,
            bytes: 0,
        }
    }

    /// Traces written so far.
    pub fn traces(&self) -> u64 {
        self.traces
    }

    /// Bytes written so far, headers included.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn open(&mut self) -> Result<&mut Open> {
        self.open
            .as_mut()
            .ok_or_else(|| Error::Spec("FileSink: write before begin".into()))
    }
}

impl TraceSink for FileSink {
    fn begin(&mut self, spec: &OutputSpec) -> Result<()> {
        if self.open.is_some() {
            return Err(Error::Spec("FileSink: begin called twice".into()));
        }
        let tmp = temp_path(&self.dest);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let guard = TempGuard {
            path: tmp,
            armed: true,
        };
        let head = (spec.textual.len() + spec.binary.len()) as u64;
        let trace_len = spec.trace_len() as u64;
        if let Some(n) = spec.n_traces {
            preallocate(&file, head + n * trace_len)?;
        }
        let rev2_count = spec.binary.len() >= 320
            && spec.binary[300] >= 2
            && spec.binary[312..320].iter().any(|&b| b != 0);
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&spec.textual)?;
        out.write_all(&spec.binary)?;
        self.bytes = head;
        self.traces = 0;
        self.open = Some(Open {
            out,
            guard,
            endian: spec.endian,
            rev2_count,
            trace_len,
        });
        Ok(())
    }

    fn write_trace(&mut self, header: &[u8], samples: &[u8]) -> Result<()> {
        let open = self.open()?;
        open.out.write_all(header)?;
        open.out.write_all(samples)?;
        self.traces += 1;
        self.bytes += (header.len() + samples.len()) as u64;
        Ok(())
    }

    fn write_block(&mut self, block: &[u8], trace_len: usize, _header_len: usize) -> Result<()> {
        let open = self.open()?;
        if trace_len == 0 || block.len() % trace_len != 0 || trace_len as u64 != open.trace_len {
            return Err(Error::Spec(format!(
                "FileSink: block of {} bytes is not whole traces of {}",
                block.len(),
                open.trace_len
            )));
        }
        open.out.write_all(block)?;
        self.traces += (block.len() / trace_len) as u64;
        self.bytes += block.len() as u64;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let Some(open) = self.open.take() else {
            return Err(Error::Spec("FileSink: finish before begin".into()));
        };
        let Open {
            out,
            mut guard,
            endian,
            rev2_count,
            ..
        } = open;
        let file = out.into_inner().map_err(|e| e.into_error())?;
        // Space reserved for a trace count the operation announced but did
        // not reach must not stay in the file.
        file.set_len(self.bytes)?;
        if rev2_count {
            let mut word = match endian {
                Endian::Little => self.traces.to_le_bytes(),
                _ => self.traces.to_be_bytes(),
            };
            if endian == Endian::Pair {
                pair_swap(&mut word);
            }
            file.write_all_at(&word, 3200 + 312)?;
        }
        if self.sync {
            file.sync_data()?;
        }
        drop(file);
        fs::rename(&guard.path, &self.dest)?;
        guard.armed = false;
        if self.sync {
            sync_dir(&parent_dir(&self.dest))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{self, Budget, Crop};
    use crate::File;

    fn small() -> File {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/small.sgy");
        File::open(path).unwrap()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("segyrs-sink-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn crop_through_the_sink_keeps_the_bytes() {
        let file = small();
        let dest = scratch("crop.sgy");
        let traces = [3u64, 0, 7];
        let mut sink = FileSink::new(&dest, false);
        let n = ops::crop(
            &file,
            &traces,
            &Crop::default(),
            Budget::bytes(4096),
            &mut sink,
        )
        .unwrap();
        assert_eq!(n, 3);
        assert_eq!(sink.traces(), 3);
        let out = File::open(&dest).unwrap();
        assert_eq!(out.layout().sample_format, file.layout().sample_format);
        for (k, &t) in traces.iter().enumerate() {
            assert_eq!(
                out.sample_bytes(k as u64).unwrap(),
                file.sample_bytes(t).unwrap()
            );
        }
        assert_eq!(fs::metadata(&dest).unwrap().len(), sink.bytes());
        fs::remove_file(&dest).unwrap();
    }

    #[test]
    fn a_sink_that_never_finishes_leaves_nothing() {
        let file = small();
        let dest = scratch("dropped.sgy");
        {
            let mut sink = FileSink::new(&dest, false);
            sink.begin(&OutputSpec::from_file(&file).unwrap()).unwrap();
        }
        assert!(!dest.exists());
        let dir = dest.parent().unwrap();
        let stray = fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains("dropped.sgy"));
        assert!(!stray, "temp file left behind");
    }
}
