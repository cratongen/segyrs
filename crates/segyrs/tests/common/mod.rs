//! Shared helpers for the ops tests: fixture paths, a counting reader, an
//! in-memory sink, a minimal SEG-Y test writer, and a synthetic file builder.
//! The writer exists only here. The library's writer is M2's.
#![allow(dead_code)]

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use segyrs::ops::{OutputSpec, TraceSink};
use segyrs::{Error, File, ReadAt, Result};

pub fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name)
}

pub fn open(name: &str) -> File {
    File::open(data(name)).unwrap_or_else(|err| panic!("open {name}: {err}"))
}

pub fn all(file: &File<impl ReadAt>) -> Vec<u64> {
    (0..file.layout().n_traces).collect()
}

/// A fresh directory under cargo's per-target scratch area.
pub fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Bytes in memory with a counter of the sample bytes read through it.
pub struct Counting {
    pub data: Vec<u8>,
    pub bytes: Arc<AtomicU64>,
    pub calls: Arc<AtomicU64>,
}

impl ReadAt for Counting {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.bytes.fetch_add(buf.len() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        let start = offset as usize;
        let src = self.data.get(start..start + buf.len()).ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read",
            ))
        })?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

/// Collects everything in RAM and records the largest block it was handed.
#[derive(Default)]
pub struct MemSink {
    pub spec: Option<OutputSpec>,
    pub headers: Vec<Vec<u8>>,
    pub samples: Vec<Vec<u8>>,
    pub max_block: usize,
    pub finished: bool,
}

impl TraceSink for MemSink {
    fn begin(&mut self, spec: &OutputSpec) -> Result<()> {
        self.spec = Some(spec.clone());
        Ok(())
    }

    fn write_trace(&mut self, header: &[u8], samples: &[u8]) -> Result<()> {
        self.headers.push(header.to_vec());
        self.samples.push(samples.to_vec());
        Ok(())
    }

    fn write_block(&mut self, block: &[u8], trace_len: usize, header_len: usize) -> Result<()> {
        self.max_block = self.max_block.max(block.len());
        for trace in block.chunks_exact(trace_len) {
            self.write_trace(&trace[..header_len], &trace[header_len..])?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.finished = true;
        Ok(())
    }
}

/// The smallest writer that turns the sink stream into a SEG-Y file.
pub struct FileSink {
    out: BufWriter<fs::File>,
}

impl FileSink {
    pub fn create(path: &Path) -> Self {
        Self {
            out: BufWriter::new(fs::File::create(path).unwrap()),
        }
    }
}

impl TraceSink for FileSink {
    fn begin(&mut self, spec: &OutputSpec) -> Result<()> {
        self.out.write_all(&spec.textual)?;
        self.out.write_all(&spec.binary)?;
        Ok(())
    }

    fn write_trace(&mut self, header: &[u8], samples: &[u8]) -> Result<()> {
        self.out.write_all(header)?;
        self.out.write_all(samples)?;
        Ok(())
    }

    fn write_block(&mut self, block: &[u8], _: usize, _: usize) -> Result<()> {
        self.out.write_all(block)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }
}

/// One synthetic trace: labels, coordinates, and its samples.
#[derive(Clone, Debug)]
pub struct Synth {
    pub inline: i32,
    pub crossline: i32,
    pub offset: i32,
    pub cdp_x: i32,
    pub cdp_y: i32,
    pub samples: Vec<f32>,
}

/// Rev 1, big-endian, IEEE float32 file with `ns` samples at 4 ms.
pub fn write_synth(path: &Path, ns: usize, traces: &[Synth]) {
    let mut out = BufWriter::new(fs::File::create(path).unwrap());
    let mut text = vec![b' '; 3200];
    text[..20].copy_from_slice(b"C 1 segyrs synthetic");
    out.write_all(&text).unwrap();
    let mut bin = vec![0u8; 400];
    bin[16..18].copy_from_slice(&4000u16.to_be_bytes());
    bin[20..22].copy_from_slice(&(ns as u16).to_be_bytes());
    bin[24..26].copy_from_slice(&5u16.to_be_bytes());
    bin[300] = 1;
    bin[302..304].copy_from_slice(&1u16.to_be_bytes());
    out.write_all(&bin).unwrap();
    for (i, t) in traces.iter().enumerate() {
        let mut h = vec![0u8; 240];
        h[0..4].copy_from_slice(&(i as i32 + 1).to_be_bytes());
        h[28..30].copy_from_slice(&1i16.to_be_bytes());
        h[36..40].copy_from_slice(&t.offset.to_be_bytes());
        h[70..72].copy_from_slice(&(-10i16).to_be_bytes());
        h[114..116].copy_from_slice(&(ns as u16).to_be_bytes());
        h[116..118].copy_from_slice(&4000u16.to_be_bytes());
        h[180..184].copy_from_slice(&t.cdp_x.to_be_bytes());
        h[184..188].copy_from_slice(&t.cdp_y.to_be_bytes());
        h[188..192].copy_from_slice(&t.inline.to_be_bytes());
        h[192..196].copy_from_slice(&t.crossline.to_be_bytes());
        out.write_all(&h).unwrap();
        assert_eq!(t.samples.len(), ns);
        for s in &t.samples {
            out.write_all(&s.to_be_bytes()).unwrap();
        }
    }
    out.flush().unwrap();
}

/// A sample value that names its trace, so moved samples can be traced back.
pub fn tag(inline: i32, crossline: i32, offset: i32, sample: usize) -> f32 {
    (inline * 10_000 + crossline * 10 + offset) as f32 + sample as f32 / 1024.0
}

/// A grid trace whose coordinates are an exact affine function of its labels.
pub fn grid_trace(inline: i32, crossline: i32, offset: i32, ns: usize) -> Synth {
    Synth {
        inline,
        crossline,
        offset,
        cdp_x: 1_000 + 250 * inline + 120 * crossline,
        cdp_y: 5_000 - 80 * inline + 300 * crossline,
        samples: (0..ns).map(|s| tag(inline, crossline, offset, s)).collect(),
    }
}

/// Deterministic shuffle (xorshift), so tests do not need a random crate.
pub fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut x = seed | 1;
    for i in (1..items.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        items.swap(i, (x % (i as u64 + 1)) as usize);
    }
}
