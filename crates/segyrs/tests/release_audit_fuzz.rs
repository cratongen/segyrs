//! Release audit: seeded mutations of small valid fixtures.
//!
//! Every mutated file must open to a typed error or to a file whose every read
//! is a typed error or a valid result. No case may panic, run past its time
//! limit, or allocate more than a bound proportional to its size. A counting
//! global allocator measures the peak, so this binary holds no other tests that
//! run concurrently with the fuzz loop (they take `SERIAL`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

use segyrs::{ops, Error, File, HeaderSpec, OpenOptions, ReadAt, Result, Scan};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to `System` with the same layout; the counters
// are only bookkeeping and never change what is allocated.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new) };
        if !p.is_null() {
            if new >= layout.size() {
                let now =
                    LIVE.fetch_add(new - layout.size(), Ordering::Relaxed) + new - layout.size();
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static SERIAL: Mutex<()> = Mutex::new(());

/// Bytes in memory. `mapped` exposes them as a slice, like an mmap, which
/// takes the zero-copy and assumed-grid paths; otherwise every read is
/// positional, like a remote reader.
struct Mem {
    data: Vec<u8>,
    mapped: bool,
}

impl ReadAt for Mem {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let start = usize::try_from(offset).map_err(|_| eof())?;
        let src = start
            .checked_add(buf.len())
            .and_then(|end| self.data.get(start..end))
            .ok_or_else(eof)?;
        buf.copy_from_slice(src);
        Ok(())
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        self.mapped.then_some(&self.data[..])
    }
}

fn eof() -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "read past end",
    ))
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name)
}

const SEEDS: &[&str] = &[
    "small.sgy",
    "small-lsb.sgy",
    "small-ps.sgy",
    "f3.sgy",
    "f3-lsb.sgy",
    "multi-text.sgy",
    "stanzas-known-count.sgy",
    "rotated-small-rev2.sgy",
    "trace-header-extensions.sgy",
    "mapping-all-types.sgy",
    "shot-gather.sgy",
    "segyrs/delay-scalar.sgy",
    "multiformats/Format1lsb.sgy",
    "multiformats/Format6lsb.sgy",
    "multiformats/Format7msb.sgy",
    "multiformats/Format9lsb.sgy",
    "multiformats/Format15msb.sgy",
    "multiformats/Format16lsb.sgy",
];

/// Binary-header words that steer the layout: interval, ns, format, units,
/// extended ns, extended interval, revision, fixed-length flag, extended text
/// count, extra trace headers, first-trace offset, byte-order signature.
const BIN_WORDS: &[(usize, usize)] = &[
    (3216, 2),
    (3220, 2),
    (3224, 2),
    (3254, 2),
    (3268, 4),
    (3272, 8),
    (3296, 4),
    (3500, 1),
    (3501, 1),
    (3502, 2),
    (3504, 2),
    (3506, 4),
    (3520, 8),
];

/// Trace-header words read at open, relative to the first trace header.
const TRACE_WORDS: &[(usize, usize)] = &[
    (28, 2),
    (36, 4),
    (70, 2),
    (114, 2),
    (180, 4),
    (188, 4),
    (192, 4),
    (240 + 136, 4),
    (240 + 156, 2),
];

fn extreme(rng: &mut Rng, width: usize) -> Vec<u8> {
    let picks: [u64; 8] = [0, 1, 2, 0x7f, 0x80, 0x7fff, 0x8000, u64::MAX];
    let v = match rng.below(3) {
        0 => picks[rng.below(picks.len())],
        1 => rng.next(),
        _ => rng.below(70000) as u64,
    };
    let bytes = if rng.below(2) == 0 {
        v.to_be_bytes()
    } else {
        v.to_le_bytes()
    };
    bytes[8 - width..].to_vec()
}

fn mutate(src: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut out = src.to_vec();
    let rounds = 1 + rng.below(4);
    for _ in 0..rounds {
        match rng.below(7) {
            0 => {
                for _ in 0..1 + rng.below(8) {
                    let at = rng.below(out.len());
                    out[at] ^= 1 << rng.below(8);
                }
            }
            1 => {
                let (at, w) = BIN_WORDS[rng.below(BIN_WORDS.len())];
                if at + w <= out.len() {
                    let v = extreme(rng, w);
                    out[at..at + w].copy_from_slice(&v);
                }
            }
            2 => {
                let (rel, w) = TRACE_WORDS[rng.below(TRACE_WORDS.len())];
                let trace = rng.below(4);
                let at = 3600 + trace * 240 * 2 + rel;
                if at + w <= out.len() {
                    let v = extreme(rng, w);
                    out[at..at + w].copy_from_slice(&v);
                }
            }
            3 => {
                let keep = rng.below(out.len() + 1);
                out.truncate(keep);
                if out.is_empty() {
                    return out;
                }
            }
            4 => {
                for _ in 0..1 + rng.below(5000) {
                    out.push(rng.next() as u8);
                }
            }
            5 => {
                let at = rng.below(out.len());
                let n = rng.below(64).min(out.len() - at);
                let fill = rng.next() as u8;
                out[at..at + n].fill(fill);
            }
            _ => {
                let at = 3200 + rng.below(400);
                if at < out.len() {
                    out[at] = rng.next() as u8;
                }
            }
        }
    }
    out
}

/// Open and then exercise every read path. Only typed errors may come back.
fn exercise(bytes: Vec<u8>, mapped: bool, scan: Scan) -> std::result::Result<(), Error> {
    let len = bytes.len() as u64;
    let file = File::from_reader_scan(
        Mem {
            data: bytes,
            mapped,
        },
        len,
        &OpenOptions::default(),
        scan,
    )?;
    let lay = file.layout().clone();
    let n = lay.n_traces;
    let ns = lay.n_samples as usize;
    let _ = file.textual_header().len();
    let _ = file.geometry();
    let _ = file.trace_index();
    file.verify_geometry()?;
    if n > 0 {
        file.header_bytes(0)?;
        file.header_bytes(n - 1)?;
        file.sample_bytes(n - 1)?;
    }
    let all: Vec<u64> = (0..n).collect();
    let decoded = file.decode(0, n as usize)?;
    assert_eq!(decoded.len(), n as usize * ns, "decode length");
    let window = file.read_window(&all, ns / 3, ns)?;
    assert_eq!(window.len(), n as usize * (ns - ns / 3), "window length");
    let cols = file.par_header_columns(&HeaderSpec::standard())?;
    assert_eq!(cols.len(), n as usize, "header column length");
    ops::stats(&file, &all)?;
    ops::rms(&file, &all)?;
    Ok(())
}

enum Outcome {
    Ok,
    Typed,
    Panic(String),
}

#[test]
fn seeded_mutations_return_typed_errors_or_valid_results() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cases_per_seed: usize = std::env::var("SEGYRS_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);
    let mut rng = Rng(0x5eed_5e9d_a0d1_7001);
    let mut failures = Vec::new();
    let (mut ok, mut typed) = (0usize, 0usize);
    for name in SEEDS {
        let src = std::fs::read(data(name)).unwrap();
        for case in 0..cases_per_seed {
            let bytes = mutate(&src, &mut rng);
            let mapped = rng.below(2) == 0;
            let scan = if rng.below(2) == 0 {
                Scan::Auto
            } else {
                Scan::Full
            };
            let size = bytes.len();
            let limit = 32 * size + (8 << 20);
            let (tx, rx) = mpsc::channel();
            let base = LIVE.load(Ordering::Relaxed);
            PEAK.store(base, Ordering::Relaxed);
            std::thread::spawn(move || {
                let r = catch_unwind(AssertUnwindSafe(|| exercise(bytes, mapped, scan)));
                let outcome = match r {
                    Ok(Ok(())) => Outcome::Ok,
                    Ok(Err(_)) => Outcome::Typed,
                    Err(p) => Outcome::Panic(
                        p.downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default(),
                    ),
                };
                let _ = tx.send(outcome);
            });
            let label = format!("{name} case {case} (mapped={mapped}, {scan:?}, {size} bytes)");
            match rx.recv_timeout(Duration::from_secs(20)) {
                Ok(Outcome::Ok) => ok += 1,
                Ok(Outcome::Typed) => typed += 1,
                Ok(Outcome::Panic(msg)) => failures.push(format!("{label}: panic: {msg}")),
                Err(_) => {
                    failures.push(format!("{label}: no result after 20 s"));
                    break;
                }
            }
            let used = PEAK.load(Ordering::Relaxed).saturating_sub(base);
            if used > limit {
                failures.push(format!("{label}: peak allocation {used} > {limit}"));
            }
        }
    }
    eprintln!(
        "fuzz: {ok} valid, {typed} typed errors, {} failures",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A virtual file of `len` bytes: a valid rev 1 head, then zeros.
struct Virtual {
    head: Vec<u8>,
    len: u64,
}

impl ReadAt for Virtual {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset + buf.len() as u64 > self.len {
            return Err(eof());
        }
        for (i, b) in buf.iter_mut().enumerate() {
            let at = offset as usize + i;
            *b = self.head.get(at).copied().unwrap_or(0);
        }
        Ok(())
    }
}

fn virtual_head(rev: u8, ntext: i16) -> Vec<u8> {
    let mut head = std::fs::read(data("small.sgy")).unwrap();
    head.truncate(3600);
    head[3500] = rev;
    head[3504..3506].copy_from_slice(&ntext.to_be_bytes());
    head
}

fn peak_during(f: impl FnOnce()) -> usize {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    f();
    PEAK.load(Ordering::Relaxed).saturating_sub(base)
}

/// An extended textual count of -1 with no EndText stanza makes open decode
/// every 3200-byte block to the end of the file and keep them all, so a large
/// file with one wrong header byte costs about twice its size in memory.
#[test]
fn unknown_text_count_without_end_text_is_bounded() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let len = 256u64 << 20;
    let reader = Virtual {
        head: virtual_head(1, -1),
        len,
    };
    let peak = peak_during(|| {
        let r = File::from_reader(reader, len, &OpenOptions::default());
        assert!(r.is_err());
    });
    assert!(
        peak < 64 << 20,
        "open of a {len}-byte file peaked at {peak} bytes"
    );
}

/// The rev 2 first-trace offset (bytes 3521-3528) is trusted as the end of
/// the extended textual headers even when bytes 3505-3506 say there are none,
/// so a wrong offset makes open read and keep every block up to it.
#[test]
fn rev2_first_trace_offset_is_bounded() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let len = 256u64 << 20;
    let mut head = virtual_head(2, 0);
    let off = 3600 + 3200 * ((len - 3600) / 3200 - 1);
    head[3520..3528].copy_from_slice(&off.to_be_bytes());
    let reader = Virtual { head, len };
    let peak = peak_during(|| {
        let _ = File::from_reader(reader, len, &OpenOptions::default());
    });
    assert!(
        peak < 64 << 20,
        "open of a {len}-byte file peaked at {peak} bytes"
    );
}
