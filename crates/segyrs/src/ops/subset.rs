use std::ops::Range;

use crate::endian::{self, Endian};
use crate::error::Result;
use crate::file::{File, ReadAt};
use crate::ops::{
    check_traces, header_len, invalid, odd_pairwise, put_i16, put_u16, stream, trace_offset,
    Budget, OutputSpec, TraceSink,
};

/// Header byte offsets (0-based) this module rewrites.
const DELAY: usize = 108;
const NS: usize = 114;
const TIME_SCALAR: usize = 214;

/// What a crop keeps besides the trace list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Crop {
    /// Vertical sample window. `None` keeps every sample.
    pub samples: Option<Range<usize>>,
    /// Write the first kept sample at time zero (delay 0). Without it the
    /// delay header moves by the samples cut, so every kept sample keeps its
    /// original time (S-36).
    pub rezero_time: bool,
}

/// Stream `traces`, in the order given, cut to `crop.samples`, into `sink`
/// (S-30, S-36).
///
/// Trace headers are copied byte for byte, so coordinates, scalars, line
/// numbers, and every other field stay those of the trace that was kept. Only
/// the sample count and, when the window moves or `rezero_time` is set, the
/// delay (and its time scalar when the new delay needs one) are rewritten.
pub fn crop<R: ReadAt, S: TraceSink + Send>(
    file: &File<R>,
    traces: &[u64],
    crop: &Crop,
    budget: Budget,
    sink: &mut S,
) -> Result<u64> {
    let _ra = file.readahead_for(traces.len());
    check_traces(file, traces)?;
    let ns = file.layout().n_samples as usize;
    let window = crop.samples.clone().unwrap_or(0..ns);
    if window.start >= window.end || window.end > ns {
        return Err(invalid(format!(
            "sample window {}..{} is not inside 0..{ns}",
            window.start, window.end
        )));
    }
    let cut = window != (0..ns);
    if cut && odd_pairwise(file) {
        return Err(invalid(
            "a sample crop of a pairwise file with an odd sample width is not supported",
        ));
    }
    let hl = header_len(file);
    let w = file.format().width();
    let kept = window.end - window.start;
    let out_len = hl + kept * w;
    let endian = file.endian();
    let dt_us = i64::from(file.layout().sample_interval_us);
    let shift_us = window.start as i64 * dt_us;
    let mut spec = OutputSpec::from_file(file)?;
    spec.set_samples(kept as u32);
    spec.n_traces = Some(traces.len() as u64);
    sink.begin(&spec)?;
    stream(traces.len(), out_len, hl, budget, sink, |i, out| {
        let off = trace_offset(file, traces[i]);
        if !cut {
            file.reader().read_at(off, out)?;
        } else {
            let (head, body) = out.split_at_mut(hl);
            file.reader().read_at(off, head)?;
            let at = off + hl as u64 + (window.start * w) as u64;
            file.reader().read_at(at, body)?;
            put_u16(head, NS, u16::try_from(kept).unwrap_or(0), endian);
        }
        if cut || crop.rezero_time {
            move_delay(&mut out[..hl], endian, shift_us, crop.rezero_time)?;
        }
        Ok(())
    })?;
    sink.finish()?;
    Ok(traces.len() as u64)
}

/// Rewrite the delay (bytes 109-110) for a window that starts `shift_us` later.
///
/// The delay is in milliseconds times the time scalar (bytes 215-216: positive
/// multiplies, negative divides, zero means one). The current scalar is kept
/// when it can express the new delay exactly. Otherwise the first scalar in
/// 1, -10, -100, -1000, -10000, 10, 100, 1000, 10000 that can is written.
fn move_delay(head: &mut [u8], endian: Endian, shift_us: i64, rezero: bool) -> Result<()> {
    let raw = endian::i16_at(head, DELAY, endian);
    let scalar = endian::i16_at(head, TIME_SCALAR, endian);
    if rezero {
        put_i16(head, DELAY, 0, endian);
        return Ok(());
    }
    if shift_us == 0 {
        return Ok(());
    }
    // Delay in microseconds as num / den.
    let (mul, div) = scale(scalar);
    let num = i128::from(raw) * 1000 * mul + i128::from(shift_us) * div;
    let den = div;
    let current = if scalar == 0 { 1 } else { scalar };
    let mut candidates = vec![current];
    candidates.extend([1i16, -10, -100, -1000, -10000, 10, 100, 1000, 10000]);
    for cand in candidates {
        // One raw unit of `cand` is 1000 * m / d microseconds.
        let (m, d) = scale(cand);
        let top = num * d;
        let bottom = den * 1000 * m;
        if top % bottom != 0 {
            continue;
        }
        let Ok(value) = i16::try_from(top / bottom) else {
            continue;
        };
        put_i16(head, DELAY, value, endian);
        if cand != current {
            put_i16(head, TIME_SCALAR, cand, endian);
        }
        return Ok(());
    }
    Err(invalid(format!(
        "the cropped delay ({num}/{den} us) does not fit the 16-bit delay word with any time scalar"
    )))
}

fn scale(scalar: i16) -> (i128, i128) {
    match scalar {
        0 => (1, 1),
        s if s > 0 => (i128::from(s), 1),
        s => (1, -i128::from(s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(delay: i16, scalar: i16) -> Vec<u8> {
        let mut h = vec![0u8; 240];
        put_i16(&mut h, DELAY, delay, Endian::Big);
        put_i16(&mut h, TIME_SCALAR, scalar, Endian::Big);
        h
    }

    fn read(h: &[u8]) -> (i16, i16) {
        (
            endian::i16_at(h, DELAY, Endian::Big),
            endian::i16_at(h, TIME_SCALAR, Endian::Big),
        )
    }

    #[test]
    fn delay_moves_in_milliseconds() {
        let mut h = head(0, 0);
        move_delay(&mut h, Endian::Big, 40_000, false).unwrap();
        assert_eq!(read(&h), (40, 0));
    }

    #[test]
    fn delay_keeps_a_negative_scalar() {
        let mut h = head(10000, -10);
        move_delay(&mut h, Endian::Big, 2_000, false).unwrap();
        assert_eq!(read(&h), (10020, -10));
    }

    #[test]
    fn delay_picks_a_scalar_for_fractions() {
        let mut h = head(4, 0);
        move_delay(&mut h, Endian::Big, 500, false).unwrap();
        assert_eq!(read(&h), (45, -10));
    }

    #[test]
    fn rezero_clears_delay() {
        let mut h = head(1234, -10);
        move_delay(&mut h, Endian::Big, 4_000, true).unwrap();
        assert_eq!(read(&h), (0, -10));
    }
}
