//! The write path: new files, in-place edits, and validation.
//!
//! - [`create`] and [`copy`] publish a new file atomically. The sample format
//!   is the source's unless [`WriteOptions::sample_format`] asks for another.
//! - [`Editor`] changes header fields, header blocks, and trace samples in
//!   place with `pwrite`. It never changes the layout (S-35).
//! - [`validate`] checks a file's structure (S-34).
//! - [`FileSink`] publishes the trace stream of a [`crate::ops`] operation as
//!   a new file, so M3 output goes through the M2 writer.

mod binary;
mod create;
mod edit;
mod encode;
mod publish;
mod sink;
mod validate;

pub use binary::{default_text, text_block};
pub use create::{copy, create, NewFile, Overlay, Source, WriteOptions};
pub use edit::{Editor, Expect};
pub use encode::{
    encode_fixed_gain, f32_to_ibm, f64_to_ibm, put_field, HeaderValues, SampleData, Scalars,
    IBM_MAX,
};
pub use sink::FileSink;
pub use validate::validate;
