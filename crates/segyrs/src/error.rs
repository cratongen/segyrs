use std::io;
use std::path::{Path, PathBuf};

/// Failures from the read and write paths.
///
/// A short file, a mixed trace length, and a wrong endian used to share one
/// "trace count inconsistent" error. They are separate variants here.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("geometry span min={min} max={max} n_traces={n_traces}")]
    Geometry { min: i64, max: i64, n_traces: u64 },

    /// A trace header contradicts the regular grid that open assumed from
    /// sampled headers. Reopen with a full header scan.
    #[error(
        "trace {trace} has (inline, crossline, offset) = {found:?}, but the regular grid \
         assumed at open puts {expected:?} there; reopen with a full header scan \
         (Scan::Full, scan=\"full\" in Python)"
    )]
    GeometryAssumption {
        trace: u64,
        expected: (i32, i32, i32),
        found: (i32, i32, i32),
    },

    #[error("duplicate key inline={inline} crossline={crossline}")]
    DuplicateKey { inline: i32, crossline: i32 },

    #[error("endian mismatch: signature={signature:#010x}")]
    EndianMismatch { signature: u32 },

    #[error(
        "sample count mismatch: file_len={file_len} expected={expected} remainder={remainder}"
    )]
    SampleCountMismatch {
        file_len: u64,
        expected: u64,
        remainder: u64,
    },

    #[error("non-uniform trace length: trace {trace} binary_ns={binary_ns} trace_ns={trace_ns}")]
    NonUniformTraceLength {
        trace: u64,
        binary_ns: u32,
        trace_ns: u32,
    },

    #[error("trace length: trace {trace} expected={expected} got={got}")]
    TraceLength { trace: u64, expected: u64, got: u64 },

    #[error("format change is a new file")]
    FormatChange,

    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("format 4 (fixed-point with gain) requires a gain convention, rev0 or rev2")]
    Format4NeedsGain,

    /// The file system refused `path`. The message says what to check.
    #[error("cannot {action} {}: {source}{}", .path.display(), path_hint(.source))]
    Path {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },

    /// The bytes are not SEG-Y. The message says what they look like instead.
    #[error("not a SEG-Y file: {0}")]
    NotSegy(String),

    #[error(
        "unknown sample format code {0} in binary header bytes 3225-3226; if this is SEG-Y \
         with a nonstandard code, open it with a sample format (sample_format= in Python, \
         OpenOptions::sample_format in Rust), otherwise it is not a SEG-Y file"
    )]
    UnknownFormat(u16),

    /// The file ends before something its headers say is there. `context`
    /// names what was being read.
    #[error("truncated SEG-Y: file has {file_size} bytes, need at least {need} {context}")]
    Truncated {
        file_size: u64,
        need: u64,
        context: String,
    },

    /// A header value does not fit its field. Names the field, the value, and the width.
    #[error("header field {field}: value {value} does not fit {dtype}")]
    HeaderValue {
        field: String,
        value: String,
        dtype: &'static str,
    },

    /// A sample cannot be stored in the target format without changing its value.
    #[error("trace {trace} sample {sample}: value {value} is not representable as {format}")]
    SampleValue {
        trace: u64,
        sample: u64,
        value: String,
        format: &'static str,
    },

    /// The description of a file to write is inconsistent. Nothing was written.
    #[error("invalid write request: {0}")]
    Spec(String),

    /// The validator found structural problems in a file.
    #[error("invalid SEG-Y: {}", problems.join("; "))]
    Invalid { problems: Vec<String> },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn path(action: &'static str, path: &Path) -> impl FnOnce(io::Error) -> Self {
        let path = path.to_path_buf();
        move |source| Self::Path {
            action,
            path,
            source,
        }
    }
}

fn path_hint(err: &io::Error) -> &'static str {
    match err.kind() {
        io::ErrorKind::NotFound => {
            "; nothing is at this path: check its spelling and the working directory"
        }
        io::ErrorKind::PermissionDenied => {
            "; check the permissions: this user must be able to read the file, and write it (or its directory) to edit or create one"
        }
        io::ErrorKind::IsADirectory => "; this is a directory: give the path of a SEG-Y file",
        _ => "",
    }
}
