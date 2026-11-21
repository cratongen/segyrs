use rayon::prelude::*;

use crate::error::Result;
use crate::file::{File, ReadAt};
use crate::ops::{check_traces, invalid, par_decode, read_window};
use crate::sample::Samples;

/// Sample windows, one row of `width` samples per requested trace.
///
/// `valid` has one flag per output sample. It is false where the window runs
/// past the first or last sample of the trace; those samples hold zero.
#[derive(Clone, Debug)]
pub struct Window {
    pub samples: Samples,
    pub width: usize,
    pub valid: Vec<bool>,
}

/// One vertical sample from each trace (S-10).
///
/// Each trace costs one read of one sample, in parallel. The other samples
/// are not read.
pub fn time_slice<R: ReadAt>(file: &File<R>, traces: &[u64], sample: usize) -> Result<Samples> {
    let ns = file.layout().n_samples as usize;
    if sample >= ns {
        return Err(invalid(format!("sample {sample} is outside 0..{ns}")));
    }
    Ok(windows(file, traces, |_| sample as i64, 0, 0)?.samples)
}

/// A window of `above + 1 + below` samples around a per-trace sample index (S-10).
///
/// `index[i]` belongs to `traces[i]`. An index outside the trace, a negative
/// sentinel for a missing pick for example, gives a row with `valid` false.
pub fn slice_horizon<R: ReadAt>(
    file: &File<R>,
    traces: &[u64],
    index: &[i64],
    above: usize,
    below: usize,
) -> Result<Window> {
    if index.len() != traces.len() {
        return Err(invalid(format!(
            "{} horizon indices for {} traces",
            index.len(),
            traces.len()
        )));
    }
    windows(file, traces, |i| index[i], above, below)
}

fn windows<R: ReadAt>(
    file: &File<R>,
    traces: &[u64],
    centre: impl Fn(usize) -> i64 + Sync,
    above: usize,
    below: usize,
) -> Result<Window> {
    let _ra = file.readahead_for(traces.len());
    check_traces(file, traces)?;
    let width = above + below + 1;
    let ns = i64::from(file.layout().n_samples);
    let w = file.format().width();
    let n = traces.len();
    let mut raw = vec![0u8; n * width * w];
    let mut valid = vec![false; n * width];
    let order = if crate::ops::odd_pairwise(file) {
        crate::endian::Endian::Big
    } else {
        file.endian()
    };
    raw.par_chunks_mut(width * w)
        .zip(valid.par_chunks_mut(width))
        .enumerate()
        .try_for_each_init(Vec::new, |buf, (i, (out, ok))| -> Result<()> {
            let first = centre(i).saturating_sub(above as i64);
            let lo = first.max(0);
            let hi = first.saturating_add(width as i64).min(ns);
            if lo >= hi {
                return Ok(());
            }
            let count = (hi - lo) as usize;
            read_window(file, traces[i], lo as usize, count, buf)?;
            let dst = (lo - first) as usize;
            out[dst * w..(dst + count) * w].copy_from_slice(buf);
            ok[dst..dst + count].fill(true);
            Ok(())
        })?;
    let samples = par_decode(&raw, file.format(), order, file.gain());
    Ok(Window {
        samples,
        width,
        valid,
    })
}
