//! Interop for segyrs: object-store reads (`cloud` feature) and opt-in Zarr v3
//! export (`zarr` feature). Neither is on the default open path.

use std::ops::Range;

use segyrs::{File, ReadAt, Result};

#[cfg(feature = "cloud")]
pub mod cloud;
#[cfg(feature = "zarr")]
pub mod zarr;

/// Byte range of each trace (header and samples) in `traces`.
pub fn trace_ranges<R: ReadAt>(
    file: &File<R>,
    traces: impl IntoIterator<Item = u64>,
) -> Vec<Range<u64>> {
    let start = file.layout().data_start;
    let stride = file.stride();
    traces
        .into_iter()
        .map(|t| start + t * stride..start + (t + 1) * stride)
        .collect()
}

/// Tell the reader which traces come next. A remote reader fetches them in
/// coalesced parallel requests. A local map ignores the hint.
pub fn prefetch_traces<R: ReadAt>(
    file: &File<R>,
    reader: &dyn ReadAt,
    traces: impl IntoIterator<Item = u64>,
) -> Result<()> {
    reader.prefetch(&trace_ranges(file, traces))
}
