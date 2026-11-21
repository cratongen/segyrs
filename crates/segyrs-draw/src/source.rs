//! The open file as the renderer sees it: named dims, the core trace index,
//! and a decoder for a window of samples inside one trace.

use std::ops::{Deref, Range};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use segyrs::{apply_scalar, ibm_to_f32, Endian, File, Gain, OpenOptions, ReadAt, SampleFormat};

use crate::{Error, Result};

/// A named dimension of a drawable volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Inline,
    Crossline,
    Offset,
    /// Trace sequence number, for files that are not a complete sorted grid.
    Trace,
    Time,
    Depth,
}

impl Dim {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Crossline => "crossline",
            Self::Offset => "offset",
            Self::Trace => "trace",
            Self::Time => "time",
            Self::Depth => "depth",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "inline" => Self::Inline,
            "crossline" => Self::Crossline,
            "offset" => Self::Offset,
            "trace" => Self::Trace,
            "time" => Self::Time,
            "depth" => Self::Depth,
            _ => return None,
        })
    }

    pub fn is_vertical(self) -> bool {
        matches!(self, Self::Time | Self::Depth)
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Read-only map of a file, or the whole file in memory when mapping is refused.
pub struct MapReader {
    backing: Backing,
}

enum Backing {
    Map(Mmap),
    Mem(Vec<u8>),
}

impl MapReader {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the file is opened read-only and segyrs never writes through
        // this map. Another process truncating the file while it is mapped is
        // outside what any reader can defend against; the core makes the same
        // assumption for its own map.
        let backing = match unsafe { Mmap::map(&file) } {
            Ok(map) => Backing::Map(map),
            Err(_) => {
                let mut data = Vec::new();
                std::io::Read::read_to_end(&mut &file, &mut data)?;
                Backing::Mem(data)
            }
        };
        Ok(Self { backing })
    }

    pub fn len(&self) -> u64 {
        self.bytes().len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }

    fn bytes(&self) -> &[u8] {
        match &self.backing {
            Backing::Map(map) => map,
            Backing::Mem(data) => data,
        }
    }
}

impl ReadAt for MapReader {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> segyrs::Result<()> {
        let data = self.bytes();
        let start = usize::try_from(offset).ok();
        let end = start.and_then(|s| s.checked_add(buf.len()));
        match (start, end) {
            (Some(s), Some(e)) if e <= data.len() => {
                buf.copy_from_slice(&data[s..e]);
                Ok(())
            }
            _ => Err(segyrs::Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "read past end of file",
            ))),
        }
    }
}

/// Where trace `t`'s samples live and how to turn them into display values.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Decoder {
    pub data_start: u64,
    pub stride: u64,
    pub header_len: u64,
    pub ns: usize,
    pub format: SampleFormat,
    pub endian: Endian,
    pub gain: Option<Gain>,
}

impl Decoder {
    /// Byte range of samples `s` of trace `t`, relative to the file start.
    pub fn byte_range(&self, t: u32, s: Range<usize>) -> Range<u64> {
        let w = self.format.width() as u64;
        let base = self.data_start + u64::from(t) * self.stride + self.header_len;
        base + s.start as u64 * w..base + s.end as u64 * w
    }

    /// Read samples `s` of trace `t` and decode them to f32 into `out`.
    ///
    /// Only those bytes are read, except for pairwise byte order with an odd
    /// sample width, where the read is widened to the enclosing byte pairs.
    pub fn read<R: ReadAt + ?Sized>(
        &self,
        reader: &R,
        t: u32,
        s: Range<usize>,
        buf: &mut Vec<u8>,
        out: &mut [f32],
    ) -> Result<()> {
        debug_assert_eq!(out.len(), s.len());
        if s.is_empty() {
            return Ok(());
        }
        let w = self.format.width();
        let base = self.data_start + u64::from(t) * self.stride + self.header_len;
        let rel = s.start * w..s.end * w;
        if self.endian == Endian::Pair && w % 2 == 1 {
            // Pairs are counted from the first sample byte. A last odd byte
            // has no partner and stays where it is, as in the core decoder.
            let lo = rel.start & !1;
            let hi = if rel.end % 2 == 1 && rel.end < self.ns * w {
                rel.end + 1
            } else {
                rel.end
            };
            buf.resize(hi - lo, 0);
            reader.read_at(base + lo as u64, buf)?;
            let mut i = 0;
            while i + 1 < buf.len() {
                buf.swap(i, i + 1);
                i += 2;
            }
            let skip = rel.start - lo;
            decode(
                &buf[skip..skip + rel.len()],
                self.format,
                Endian::Big,
                self.gain,
                out,
            )
        } else {
            buf.resize(rel.len(), 0);
            reader.read_at(base + rel.start as u64, buf)?;
            decode(buf, self.format, self.endian, self.gain, out)
        }
    }
}

/// Decode raw sample bytes to f32 display values.
///
/// The view value is f32. int32, int64, uint32, uint64 and float64 samples
/// that f32 cannot hold are rounded here, and only here.
pub(crate) fn decode(
    bytes: &[u8],
    format: SampleFormat,
    endian: Endian,
    gain: Option<Gain>,
    out: &mut [f32],
) -> Result<()> {
    fn each<const N: usize>(
        bytes: &[u8],
        endian: Endian,
        out: &mut [f32],
        f: impl Fn([u8; N]) -> f32,
    ) {
        // Normalise every word to big-endian byte order, then convert.
        for (o, c) in out.iter_mut().zip(bytes.chunks_exact(N)) {
            let mut b: [u8; N] = c.try_into().expect("chunk width");
            match endian {
                Endian::Big => {}
                Endian::Little => b.reverse(),
                Endian::Pair => {
                    let mut i = 0;
                    while i + 1 < N {
                        b.swap(i, i + 1);
                        i += 2;
                    }
                }
            }
            *o = f(b);
        }
    }
    match format {
        SampleFormat::IbmFloat32 => {
            each::<4>(bytes, endian, out, |b| ibm_to_f32(u32::from_be_bytes(b)))
        }
        SampleFormat::IeeeFloat32 => each::<4>(bytes, endian, out, f32::from_be_bytes),
        SampleFormat::IeeeFloat64 => {
            each::<8>(bytes, endian, out, |b| f64::from_be_bytes(b) as f32)
        }
        SampleFormat::Int32 => each::<4>(bytes, endian, out, |b| i32::from_be_bytes(b) as f32),
        SampleFormat::Uint32 => each::<4>(bytes, endian, out, |b| u32::from_be_bytes(b) as f32),
        SampleFormat::Int16 => each::<2>(bytes, endian, out, |b| f32::from(i16::from_be_bytes(b))),
        SampleFormat::Uint16 => each::<2>(bytes, endian, out, |b| f32::from(u16::from_be_bytes(b))),
        SampleFormat::Int64 => each::<8>(bytes, endian, out, |b| i64::from_be_bytes(b) as f32),
        SampleFormat::Uint64 => each::<8>(bytes, endian, out, |b| u64::from_be_bytes(b) as f32),
        SampleFormat::Int8 => each::<1>(bytes, endian, out, |b| f32::from(b[0] as i8)),
        SampleFormat::Uint8 => each::<1>(bytes, endian, out, |b| f32::from(b[0])),
        SampleFormat::Int24 => each::<3>(bytes, endian, out, |b| {
            let u = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            ((u << 8) as i32 >> 8) as f32
        }),
        SampleFormat::Uint24 => each::<3>(bytes, endian, out, |b| {
            ((u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2])) as f32
        }),
        SampleFormat::FixedGain => {
            let gain = gain.ok_or(Error::Core(segyrs::Error::Format4NeedsGain))?;
            each::<4>(bytes, endian, out, |b| {
                let g = i32::from(b[1]);
                let exp = match gain {
                    Gain::Rev0 => g,
                    Gain::Rev2 => -g,
                };
                let mag = f64::from((u16::from(b[2] & 0x7f) << 8) | u16::from(b[3]));
                let sign = if b[2] & 0x80 == 0 { 1.0 } else { -1.0 };
                (sign * mag * 2f64.powi(exp)) as f32
            })
        }
    }
    Ok(())
}

/// Whatever owns the core [`File`] a [`Source`] draws from: a `Box`, an
/// `Arc`, or a handle another library already holds (the Python `File`).
pub type FileOwner<R> = Arc<dyn Deref<Target = File<R>> + Send + Sync>;

/// A SEG-Y file opened for drawing.
///
/// The trace headers are scanned once, by the core [`File`], and the core
/// trace index is shared with it: [`Source::from_file`] draws from a `File`
/// someone else opened without scanning again. No sample byte is read until a
/// plane is requested.
pub struct Source<R: ReadAt + 'static = MapReader> {
    file: FileOwner<R>,
    pub(crate) dims: Vec<Dim>,
    /// Coordinate of every position along each dim, in drawing units
    /// (line numbers, offsets, trace numbers, milliseconds).
    pub(crate) coords: Vec<Arc<[f64]>>,
    /// Header label at each position of each horizontal grid dim, in `dims`
    /// order. A cell's trace comes from the core's `(inline, crossline)` index
    /// ([`File::trace_index`]), the one source of truth for label lookups.
    /// Empty for a `(trace, vertical)` volume, where the trace is the position.
    grid: Vec<Vec<i32>>,
    pub(crate) decoder: Decoder,
    delay_us: i64,
    interval_us: i64,
}

impl Source<MapReader> {
    /// Map `path` and scan its headers.
    pub fn open(path: impl AsRef<Path>, options: &OpenOptions) -> Result<Self> {
        let reader = MapReader::open(path.as_ref()).map_err(segyrs::Error::Io)?;
        let len = reader.len();
        Self::from_reader(reader, len, options)
    }
}

impl<R: ReadAt + 'static> Source<R> {
    /// Use any [`ReadAt`] as the byte source (memory, object store, a read counter).
    pub fn from_reader(reader: R, len: u64, options: &OpenOptions) -> Result<Self> {
        Self::from_file(Box::new(File::from_reader(reader, len, options)?))
    }

    /// Draw from a core [`File`] that is already open, through any owner
    /// (`Box<File<R>>`, `Arc<File<R>>`, a binding's handle). Its header
    /// columns and trace index are reused; nothing is read here except trace
    /// 0's header for the recording delay.
    pub fn from_file(file: impl Deref<Target = File<R>> + Send + Sync + 'static) -> Result<Self> {
        let file: FileOwner<R> = Arc::new(file);
        let layout = file.layout().clone();
        let header_len = 240 * (1 + u64::from(layout.extra_trace_header_blocks));
        let decoder = Decoder {
            data_start: layout.data_start,
            stride: file.stride(),
            header_len,
            ns: layout.n_samples as usize,
            format: file.format(),
            endian: file.endian(),
            gain: file.gain(),
        };
        let (delay_us, interval_us) = vertical_axis(&*file)?;
        // Planes are cut by the grid labels: an assumed grid must hold for every trace.
        file.verify_geometry()?;
        let geometry = file.geometry();
        let n = layout.n_traces as usize;
        let mut dims = Vec::new();
        let mut coords: Vec<Arc<[f64]>> = Vec::new();
        let mut grid = Vec::new();
        if geometry.kind.is_grid() {
            let mut axes: Vec<(Dim, &[i32])> = vec![
                (Dim::Inline, &geometry.inlines),
                (Dim::Crossline, &geometry.crosslines),
            ];
            let plane = geometry.inlines.len() * geometry.crosslines.len();
            if geometry.offsets.len() > 1 && plane != n {
                axes.push((Dim::Offset, &geometry.offsets));
            }
            let cells: usize = axes.iter().map(|(_, l)| l.len()).product();
            if cells != n {
                return Err(Error::Geometry(format!(
                    "grid of {cells} cells for {n} traces"
                )));
            }
            // Build the core index now, on rayon, so the first plane does not pay for it.
            file.trace_index();
            for (dim, labels) in &axes {
                dims.push(*dim);
                coords.push(labels.iter().map(|v| f64::from(*v)).collect());
                grid.push(labels.to_vec());
            }
        } else {
            dims.push(Dim::Trace);
            coords.push((0..n).map(|t| t as f64).collect());
        }
        dims.push(Dim::Time);
        coords.push(
            (0..decoder.ns)
                .map(|i| (delay_us + i as i64 * interval_us) as f64 / 1000.0)
                .collect(),
        );
        Ok(Self {
            file,
            dims,
            coords,
            grid,
            decoder,
            delay_us,
            interval_us,
        })
    }

    /// Name the vertical dim `depth` instead of `time` (roadmap decision 2).
    pub fn depth(mut self, depth: bool) -> Self {
        if let Some(last) = self.dims.last_mut() {
            *last = if depth { Dim::Depth } else { Dim::Time };
        }
        self
    }

    /// The core file: layout, geometry, and header columns.
    pub fn file(&self) -> &File<R> {
        &self.file
    }

    /// The byte source sample windows are read from (the core file's reader).
    pub(crate) fn reader(&self) -> &R {
        self.file.reader()
    }

    /// Dims in order: grid axes (or `trace`), then the vertical dim.
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    pub fn shape(&self) -> Vec<usize> {
        self.coords.iter().map(|c| c.len()).collect()
    }

    pub fn len_of(&self, dim: Dim) -> Option<usize> {
        self.axis_of(dim).map(|k| self.coords[k].len())
    }

    /// Coordinate values along `dim`: line numbers, offsets, trace numbers,
    /// or milliseconds (delay plus index times the header sample interval).
    pub fn coords(&self, dim: Dim) -> Option<&[f64]> {
        self.axis_of(dim).map(|k| &*self.coords[k])
    }

    /// Recording delay of trace 0 in microseconds, after the time scalar.
    pub fn delay_us(&self) -> i64 {
        self.delay_us
    }

    pub fn interval_us(&self) -> i64 {
        self.interval_us
    }

    pub(crate) fn axis_of(&self, dim: Dim) -> Option<usize> {
        self.dims.iter().position(|d| *d == dim)
    }

    /// Live mask from trace identification code 2 (dead). Off by default
    /// (roadmap decision 7); pass it as `PlaneSpec::live` to use it.
    pub fn trace_id_mask(&self) -> Arc<[bool]> {
        self.file.trace_id().iter().map(|id| *id != 2).collect()
    }

    /// Trace at horizontal positions `pos` (one per horizontal dim, in order),
    /// or `u32::MAX` for an empty slot. A grid cell is looked up in the core's
    /// `(inline, crossline)` index; on a pre-stack grid the offset picks one of
    /// that pair's traces.
    pub(crate) fn trace_at(&self, pos: &[usize]) -> u32 {
        if self.grid.is_empty() {
            return pos[0] as u32;
        }
        let file = &self.file;
        let Some(hits) = file
            .trace_index()
            .get(self.grid[0][pos[0]], self.grid[1][pos[1]])
        else {
            return u32::MAX;
        };
        match self.grid.get(2) {
            None => hits.first().copied().unwrap_or(u32::MAX),
            Some(offsets) => {
                let want = offsets[pos[2]];
                hits.iter()
                    .copied()
                    .find(|t| file.offset()[*t as usize] == want)
                    .unwrap_or(u32::MAX)
            }
        }
    }

    /// Exact file value of one sample, for a cursor readout of formats that
    /// f32 cannot hold. Reads one sample.
    pub fn sample_f64(&self, trace: u32, sample: usize) -> Result<f64> {
        let d = &self.decoder;
        if trace as u64 >= self.file.layout().n_traces || sample >= d.ns {
            return Err(Error::Index {
                dim: Dim::Trace,
                index: trace as usize,
                len: self.file.layout().n_traces as usize,
            });
        }
        let w = d.format.width();
        let mut raw = vec![0u8; w];
        let r = d.byte_range(trace, sample..sample + 1);
        let exact = matches!(
            d.format,
            SampleFormat::Int32
                | SampleFormat::Uint32
                | SampleFormat::Int64
                | SampleFormat::Uint64
                | SampleFormat::IeeeFloat64
        ) && d.endian != Endian::Pair;
        if !exact {
            let mut buf = Vec::new();
            let mut out = [0f32];
            d.read(self.reader(), trace, sample..sample + 1, &mut buf, &mut out)?;
            return Ok(f64::from(out[0]));
        }
        self.reader().read_at(r.start, &mut raw)?;
        if d.endian == Endian::Little {
            raw.reverse();
        }
        Ok(match d.format {
            SampleFormat::Int32 => f64::from(i32::from_be_bytes(raw[..4].try_into().unwrap())),
            SampleFormat::Uint32 => f64::from(u32::from_be_bytes(raw[..4].try_into().unwrap())),
            SampleFormat::Int64 => i64::from_be_bytes(raw[..8].try_into().unwrap()) as f64,
            SampleFormat::Uint64 => u64::from_be_bytes(raw[..8].try_into().unwrap()) as f64,
            _ => f64::from_be_bytes(raw[..8].try_into().unwrap()),
        })
    }
}

/// Delay and interval of the vertical axis, in microseconds.
///
/// The delay is trace 0's recording delay (bytes 109-110, milliseconds) with
/// the time scalar (bytes 215-216) applied when it is not zero. The interval
/// is the binary-header interval the core resolved.
fn vertical_axis<R: ReadAt>(file: &File<R>) -> Result<(i64, i64)> {
    let interval = i64::from(file.layout().sample_interval_us);
    if file.layout().n_traces == 0 {
        return Ok((0, interval));
    }
    let header = file.header_bytes(0)?;
    let word = |off: usize| {
        let b = [header[off], header[off + 1]];
        match file.endian() {
            Endian::Big => i16::from_be_bytes(b),
            Endian::Little | Endian::Pair => i16::from_le_bytes(b),
        }
    };
    let delay_ms = word(108);
    let scalar = word(214);
    let ms = apply_scalar(i32::from(delay_ms), scalar);
    Ok(((ms * 1000.0).round() as i64, interval))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mem(Vec<u8>);

    impl ReadAt for Mem {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> segyrs::Result<()> {
            let s = offset as usize;
            buf.copy_from_slice(&self.0[s..s + buf.len()]);
            Ok(())
        }
    }

    fn pair_swapped(big: &[u8]) -> Vec<u8> {
        let mut v = big.to_vec();
        let mut i = 0;
        while i + 1 < v.len() {
            v.swap(i, i + 1);
            i += 2;
        }
        v
    }

    /// Pairwise order with an odd sample width: every window, at every odd or
    /// even byte start, decodes like the whole trace does.
    #[test]
    fn pairwise_odd_width_windows_match_whole_trace() {
        for (format, w) in [(SampleFormat::Int8, 1usize), (SampleFormat::Int24, 3)] {
            let ns = 7usize;
            let mut big = Vec::new();
            let step = if w == 1 { 20 } else { 100_000 };
            let want: Vec<f32> = (0..ns as i32).map(|i| ((i - 3) * step) as f32).collect();
            for v in &want {
                let v = *v as i32;
                let b = v.to_be_bytes();
                big.extend_from_slice(&b[4 - w..]);
            }
            let header = vec![0u8; 240];
            let mut file = header.clone();
            file.extend(pair_swapped(&big));
            let d = Decoder {
                data_start: 0,
                stride: 240 + (ns * w) as u64,
                header_len: 240,
                ns,
                format,
                endian: Endian::Pair,
                gain: None,
            };
            let reader = Mem(file);
            let mut buf = Vec::new();
            for a in 0..ns {
                for b in a + 1..=ns {
                    let mut out = vec![0f32; b - a];
                    d.read(&reader, 0, a..b, &mut buf, &mut out).unwrap();
                    assert_eq!(out, want[a..b], "{format:?} window {a}..{b}");
                }
            }
        }
    }

    #[test]
    fn fixed_gain_follows_the_convention() {
        // gain byte 3, magnitude 5, negative.
        let raw = [0u8, 3, 0x80, 5];
        let mut out = [0f32];
        decode(
            &raw,
            SampleFormat::FixedGain,
            Endian::Big,
            Some(Gain::Rev0),
            &mut out,
        )
        .unwrap();
        assert_eq!(out[0], -40.0);
        decode(
            &raw,
            SampleFormat::FixedGain,
            Endian::Big,
            Some(Gain::Rev2),
            &mut out,
        )
        .unwrap();
        assert_eq!(out[0], -0.625);
    }
}
