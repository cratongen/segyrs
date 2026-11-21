use crate::error::{Error, Result};
use crate::file::{File, ReadAt};
use crate::ops::{header_len, invalid, stream, trace_offset, Budget, OutputSpec, TraceSink};

/// `Error::FormatChange` unless every file has the first file's sample format,
/// sample interval, sample count, byte order, and trace-header length (X-35).
///
/// Nothing is cast. Two unlike files are an error, never a promoted dtype.
pub fn check_concat<R: ReadAt>(files: &[&File<R>]) -> Result<()> {
    let first = files
        .first()
        .ok_or_else(|| invalid("concat needs at least one file"))?;
    let key = |f: &File<R>| {
        (
            f.format(),
            f.layout().sample_interval_us,
            f.layout().n_samples,
            f.endian(),
            header_len(f),
        )
    };
    let want = key(first);
    if files.iter().any(|f| key(f) != want) {
        return Err(Error::FormatChange);
    }
    Ok(())
}

/// Every trace of every file, in file order, into `sink` (X-35).
///
/// The textual and binary headers are the first file's. Trace bytes are copied
/// as they are on disk.
pub fn concat<R: ReadAt, S: TraceSink + Send>(
    files: &[&File<R>],
    budget: Budget,
    sink: &mut S,
) -> Result<u64> {
    check_concat(files)?;
    let mut spec = OutputSpec::from_file(files[0])?;
    let total: u64 = files.iter().map(|f| f.layout().n_traces).sum();
    spec.n_traces = Some(total);
    sink.begin(&spec)?;
    let stride = spec.trace_len();
    for file in files {
        let n = file.layout().n_traces as usize;
        let _ra = file.readahead();
        stream(n, stride, spec.trace_header_len, budget, sink, |i, out| {
            file.reader().read_at(trace_offset(file, i as u64), out)
        })?;
    }
    sink.finish()?;
    Ok(total)
}
