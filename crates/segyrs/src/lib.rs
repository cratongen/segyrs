//! Clean-room SEG-Y reader and writer.
//!
//! Multi-byte headers and samples are decoded to native endian. Format 4 stays
//! closed unless a [`GainConvention`] is set. Sorting is a report on
//! [`Geometry`]: an unsorted, sparse, or unstructured file still opens. An
//! absurd inline/crossline span is [`Error::Geometry`] and is not allocated.

pub mod bridge;
mod ebcdic;
mod endian;
mod error;
mod file;
mod geometry;
mod header;
mod index_cache;
pub mod ops;
mod pio;
mod sample;
mod text;
pub mod write;

pub use endian::Endian;
pub use error::{Error, Result};
pub use file::{
    AccessHint, File, GeometrySource, Layout, OpenOptions, ReadAt, Readahead, Revision, Scan,
    ScanOptions, Storage, TracePart,
};
pub use geometry::{apply_scalar, Axis, Geometry, GeometryKind, TraceIndex};
pub use header::{
    ColumnValues, FieldValue, HeaderColumns, HeaderDtype, HeaderField, HeaderSpec, NamedColumn,
};
pub use sample::{decode_into, ibm_to_f32, Gain, SampleFormat, Samples, SamplesMut};
pub use text::{decode_block, TextEncoding, TextHeader};
pub use write::{
    copy, create, validate, Editor, Expect, HeaderValues, NewFile, Overlay, SampleData, Scalars,
    Source, WriteOptions,
};

/// Rev 0 multiplies by `2^gain`. Rev 2 multiplies by `2^-gain`.
pub use sample::Gain as GainConvention;
