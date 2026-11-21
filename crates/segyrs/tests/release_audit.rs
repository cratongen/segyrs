//! Release audit: decoding against an independent decoder, thread-count
//! independence, and the reductions of each failure found.
//!
//! The large-file checks live in python/tests/test_release_audit_data.py.

use std::path::PathBuf;

use segyrs::ops::{self, Budget, OutputSpec, ResortOptions, SortKey, SortStrategy, TraceSink};
use segyrs::write::{copy, create, NewFile, SampleData, WriteOptions};
use segyrs::{
    Endian, File, HeaderSpec, OpenOptions, Result, SampleFormat, Samples, Scan, ScanOptions,
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name)
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("release_audit");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn bits(s: &Samples) -> Vec<u8> {
    macro_rules! le {
        ($v:expr) => {
            $v.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
    }
    match s {
        Samples::I8(v) => le!(v),
        Samples::U8(v) => le!(v),
        Samples::I16(v) => le!(v),
        Samples::U16(v) => le!(v),
        Samples::I32(v) => le!(v),
        Samples::U32(v) => le!(v),
        Samples::I64(v) => le!(v),
        Samples::U64(v) => le!(v),
        Samples::F32(v) => le!(v),
        Samples::F64(v) => le!(v),
    }
}

/// IBM hexadecimal float to the nearest float32, written from the format
/// definition: sign, base-16 exponent biased by 64, 24-bit fraction.
fn ibm(word: u32) -> f32 {
    let sign = if word >> 31 == 1 { -1.0 } else { 1.0 };
    let exp = ((word >> 24) & 0x7f) as i32 - 64;
    let frac = f64::from(word & 0x00ff_ffff) / f64::from(1u32 << 24);
    (sign * frac * 16f64.powi(exp)) as f32
}

/// Decode `raw` (one or more whole traces in file order) into little-endian
/// bytes of the natural numpy type, without any segyrs code.
fn oracle(raw: &[u8], code: u16, little: bool) -> Option<Vec<u8>> {
    let w = match code {
        1 | 2 | 5 | 10 => 4,
        3 | 11 => 2,
        6 | 9 | 12 => 8,
        7 | 15 => 3,
        8 | 16 => 1,
        _ => return None,
    };
    let mut out = Vec::new();
    for chunk in raw.chunks_exact(w) {
        let mut v = chunk.to_vec();
        if !little {
            v.reverse();
        }
        match code {
            1 => {
                let word = u32::from_le_bytes(v[..].try_into().unwrap());
                out.extend(ibm(word).to_le_bytes());
            }
            7 => {
                let sign = if v[2] & 0x80 != 0 { 0xff } else { 0 };
                out.extend([v[0], v[1], v[2], sign]);
            }
            15 => out.extend([v[0], v[1], v[2], 0]),
            _ => out.extend(v),
        }
    }
    Some(out)
}

const FIXTURES: &[&str] = &[
    "small.sgy",
    "small-lsb.sgy",
    "small-ps.sgy",
    "f3.sgy",
    "f3-lsb.sgy",
    "shot-gather.sgy",
    "segyrs/delay-scalar.sgy",
    "multi-text.sgy",
    "stanzas-known-count.sgy",
    "rotated-small-rev2.sgy",
    "trace-header-extensions.sgy",
    "mapping-all-types.sgy",
    "mapping-all-types-lsb.sgy",
    "multiformats/Format1lsb.sgy",
    "multiformats/Format1msb.sgy",
    "multiformats/Format2lsb.sgy",
    "multiformats/Format2msb.sgy",
    "multiformats/Format3lsb.sgy",
    "multiformats/Format3msb.sgy",
    "multiformats/Format5lsb.sgy",
    "multiformats/Format5msb.sgy",
    "multiformats/Format6lsb.sgy",
    "multiformats/Format6msb.sgy",
    "multiformats/Format7lsb.sgy",
    "multiformats/Format7msb.sgy",
    "multiformats/Format8lsb.sgy",
    "multiformats/Format8msb.sgy",
    "multiformats/Format9lsb.sgy",
    "multiformats/Format9msb.sgy",
    "multiformats/Format10lsb.sgy",
    "multiformats/Format10msb.sgy",
    "multiformats/Format11lsb.sgy",
    "multiformats/Format11msb.sgy",
    "multiformats/Format12lsb.sgy",
    "multiformats/Format12msb.sgy",
    "multiformats/Format15lsb.sgy",
    "multiformats/Format15msb.sgy",
    "multiformats/Format16lsb.sgy",
    "multiformats/Format16msb.sgy",
];

#[test]
fn fixtures_decode_like_an_independent_decoder() {
    for name in FIXTURES {
        let raw = std::fs::read(data(name)).unwrap();
        let file = File::open(data(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let lay = file.layout();
        let little = name.contains("lsb");
        assert_eq!(
            file.endian() == Endian::Little,
            little,
            "{name}: byte order"
        );
        let code = lay.binary_format_code;
        let n = lay.n_traces as usize;
        let stride = file.stride() as usize;
        let hl = stride - lay.n_samples as usize * file.format().width();
        let mut want = Vec::new();
        for t in 0..n {
            let at = lay.data_start as usize + t * stride;
            want.extend(oracle(&raw[at + hl..at + stride], code, little).unwrap());
            assert_eq!(
                file.header_bytes(t as u64).unwrap(),
                raw[at..at + 240],
                "{name}: header {t}"
            );
        }
        assert_eq!(
            lay.data_start as usize + n * stride,
            raw.len(),
            "{name}: size"
        );
        let got = bits(&file.decode(0, n).unwrap());
        assert!(got == want, "{name}: decoded samples differ");
    }
}

/// `((SEG: EndText))` is a stanza header. The fixture spells it
/// `((  seg: endTEXt  ))`, and test-data/README.md says EndText is present.
#[test]
fn end_text_stanza_is_found_in_any_case_and_spacing() {
    let file = File::open(data("stanzas-unknown-count.sgy")).unwrap();
    assert_eq!(file.textual_header().len(), 4);
    assert_eq!(file.layout().data_start, 3600 + 3 * 3200);
    assert_eq!(file.layout().n_traces, 6);
}

/// A regular grid big enough for the assumed-grid open: inline-major,
/// IEEE float32, big-endian, rev 1.
fn grid_file(path: &std::path::Path, il: i32, xl: i32, ns: u16) {
    let mut out = vec![b' '; 3200];
    let mut bin = vec![0u8; 400];
    bin[16..18].copy_from_slice(&4000u16.to_be_bytes());
    bin[20..22].copy_from_slice(&ns.to_be_bytes());
    bin[24..26].copy_from_slice(&5u16.to_be_bytes());
    bin[300] = 1;
    bin[302..304].copy_from_slice(&1u16.to_be_bytes());
    out.extend(bin);
    let mut state = 0x1234_5678_9abc_def1u64;
    for i in 0..il {
        for x in 0..xl {
            let mut h = [0u8; 240];
            h[70..72].copy_from_slice(&(-100i16).to_be_bytes());
            h[114..116].copy_from_slice(&ns.to_be_bytes());
            h[180..184].copy_from_slice(&(1_000_000 + 1250 * x).to_be_bytes());
            h[184..188].copy_from_slice(&(2_000_000 + 1250 * i).to_be_bytes());
            h[188..192].copy_from_slice(&(100 + i).to_be_bytes());
            h[192..196].copy_from_slice(&(500 + 2 * x).to_be_bytes());
            out.extend(h);
            for _ in 0..ns {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let v = (state >> 40) as f32 / 1e3 - 8000.0;
                out.extend(v.to_be_bytes());
            }
        }
    }
    std::fs::write(path, out).unwrap();
}

#[derive(Default)]
struct VecSink(Vec<u8>);

impl TraceSink for VecSink {
    fn begin(&mut self, spec: &OutputSpec) -> Result<()> {
        self.0.extend(&spec.textual);
        self.0.extend(&spec.binary);
        Ok(())
    }

    fn write_trace(&mut self, header: &[u8], samples: &[u8]) -> Result<()> {
        self.0.extend(header);
        self.0.extend(samples);
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Everything observable from one file, as bytes.
fn everything(path: &std::path::Path, tag: &str) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for scan in [Scan::Auto, Scan::Full] {
        let file = File::open_scan(
            path,
            &OpenOptions::default(),
            &ScanOptions {
                scan,
                index_cache: None,
            },
        )
        .unwrap();
        let n = file.layout().n_traces;
        let all: Vec<u64> = (0..n).collect();
        out.push(format!("{:?} {:?}", file.geometry(), file.geometry_source()).into_bytes());
        out.push(bits(&file.decode(0, n as usize).unwrap()));
        let ns = file.layout().n_samples as usize;
        let picks: Vec<u64> = (0..n).rev().step_by(7).collect();
        out.push(bits(&file.read_window(&picks, ns / 4, ns / 2).unwrap()));
        let cols = file.par_header_columns(&HeaderSpec::standard()).unwrap();
        out.push(format!("{:?}", cols.into_columns()).into_bytes());
        out.push(format!("{:?}", file.trace_index().get(110, 520)).into_bytes());
        out.push(format!("{:?}", ops::stats(&file, &all).unwrap()).into_bytes());
        let rms = ops::rms(&file, &all).unwrap();
        out.push(rms.iter().flat_map(|x| x.to_le_bytes()).collect());
        out.push(bits(&ops::time_slice(&file, &all, ns / 2).unwrap()));

        let fast = WriteOptions {
            sync: false,
            ..WriteOptions::default()
        };
        let dst = scratch(&format!("{tag}-copy.sgy"));
        copy(&file, &dst, &fast).unwrap();
        out.push(std::fs::read(&dst).unwrap());
        let decoded = file.decode(0, n as usize).unwrap();
        let mut spec = NewFile::from_file(&file).unwrap();
        spec.samples = Some(SampleData::from(&decoded));
        let ibm = WriteOptions {
            sample_format: Some(SampleFormat::IbmFloat32),
            endian: Some(Endian::Little),
            sync: false,
            ..WriteOptions::default()
        };
        create(&dst, &spec, &ibm).unwrap();
        out.push(std::fs::read(&dst).unwrap());

        let keys = [SortKey::asc("crossline"), SortKey::desc("cdp_y")];
        for strategy in [SortStrategy::External, SortStrategy::Gather] {
            let mut opts = ResortOptions::new(Budget::bytes(n * 64 + 5 * file.stride()));
            opts.strategy = strategy;
            opts.tmp_dir = Some(scratch("").parent().unwrap().to_path_buf());
            let mut sink = VecSink::default();
            let report = ops::resort(&file, &keys, &opts, &mut sink).unwrap();
            if strategy == SortStrategy::External {
                assert!(report.runs > 1, "{tag}: the external sort did not spill");
            }
            out.push(sink.0);
        }
        let order = ops::sort_order(&file, &[SortKey::asc("offset")]).unwrap();
        out.push(order.iter().flat_map(|x| x.to_le_bytes()).collect());
    }
    out
}

#[test]
fn one_thread_and_many_threads_give_identical_bytes() {
    let grid = scratch("grid-90x70.sgy");
    grid_file(&grid, 90, 70, 48);
    let opened = File::open(&grid).unwrap();
    assert_eq!(opened.geometry_source(), segyrs::GeometrySource::Assumed);
    drop(opened);
    let files = [
        grid,
        data("small-ps.sgy"),
        data("multiformats/Format1lsb.sgy"),
    ];
    for (k, path) in files.iter().enumerate() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| everything(path, &format!("t{threads}-{k}")))
        };
        let one = run(1);
        for threads in [3, 8] {
            let many = run(threads);
            assert_eq!(one.len(), many.len());
            for (i, (a, b)) in one.iter().zip(&many).enumerate() {
                assert!(
                    a == b,
                    "{}: item {i} differs at {threads} threads",
                    path.display()
                );
            }
        }
    }
}

/// A wrong path names the path and the kind of failure, and keeps the
/// `io::Error` underneath so callers can match on its kind.
#[test]
fn a_path_the_file_system_refuses_names_the_path_and_what_to_check() {
    let missing = scratch("surey.sgy");
    let _ = std::fs::remove_file(&missing);
    let err = File::open(&missing).err().unwrap();
    let msg = err.to_string();
    assert!(msg.contains("surey.sgy"), "{msg}");
    assert!(msg.contains("check its spelling"), "{msg}");
    match err {
        segyrs::Error::Path { source, .. } => {
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound)
        }
        other => panic!("{other:?}"),
    }

    let dir = scratch("a-directory");
    std::fs::create_dir_all(&dir).unwrap();
    let msg = File::open(&dir).err().unwrap().to_string();
    assert!(
        msg.contains("a-directory") && msg.contains("is a directory"),
        "{msg}"
    );
    let msg = segyrs::write::validate(&dir).err().unwrap().to_string();
    assert!(msg.contains("is a directory"), "{msg}");
}

/// Files that are not SEG-Y say so instead of reporting a truncated SEG-Y.
#[test]
fn files_that_are_not_segy_say_what_they_look_like() {
    for name in ["small.su", "small-lsb.su"] {
        let err = File::open(data(name)).err().unwrap();
        assert!(matches!(err, segyrs::Error::NotSegy(_)), "{name}: {err:?}");
        assert!(err.to_string().contains("Seismic Unix"), "{name}: {err}");
    }

    let text = scratch("notes.txt");
    std::fs::write(&text, b"hello world\n").unwrap();
    let msg = File::open(&text).err().unwrap().to_string();
    assert!(msg.contains("file has 12 bytes"), "{msg}");
    assert!(msg.contains("is not SEG-Y"), "{msg}");

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let random: Vec<u8> = (0..5000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    let path = scratch("random.bin");
    std::fs::write(&path, &random).unwrap();
    let msg = File::open(&path).err().unwrap().to_string();
    assert!(
        msg.contains("not a SEG-Y file") || msg.contains("not SEG-Y"),
        "{msg}"
    );
}

/// The validator rejects a zero sample interval, so create refuses to write one.
#[test]
fn create_refuses_a_zero_sample_interval() {
    let path = scratch("zero-interval.sgy");
    let _ = std::fs::remove_file(&path);
    let samples = vec![0f32; 6];
    let mut spec = NewFile::new(2, 3, 0);
    spec.samples = Some(SampleData::F32(&samples));
    let options = WriteOptions {
        sample_format: Some(SampleFormat::IeeeFloat32),
        sync: false,
        ..WriteOptions::default()
    };
    let err = create(&path, &spec, &options).err().unwrap();
    assert!(matches!(err, segyrs::Error::Spec(_)), "{err:?}");
    assert!(!path.exists());

    spec.sample_interval_us = 4000;
    create(&path, &spec, &options).unwrap();
    segyrs::write::validate(&path).unwrap();
}
