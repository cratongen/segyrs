//! Every fixture in test-data/ is read, written to a temp dir, and compared
//! byte for byte. The files that cannot be opened are listed in `MALFORMED`
//! with the reason, and the test fails if that list goes stale.

use std::path::{Path, PathBuf};

use segyrs::write::{copy, create, NewFile, SampleData, WriteOptions};
use segyrs::{Endian, Error, File, GainConvention, OpenOptions, SampleFormat};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-data")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("write_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn fast() -> WriteOptions {
    WriteOptions {
        sync: false,
        ..WriteOptions::default()
    }
}

/// Fixtures that do not open, and why. None of them is a writer problem: the
/// reader refuses them before there is anything to write.
const MALFORMED: &[(&str, &str)] = &[
    ("broken.sgy", "the end of the file is cut off"),
    (
        "decrement.sgy",
        "every header field is set to its limit, so per-trace ns contradicts the binary header",
    ),
    ("decrement-lsb.sgy", "little-endian copy of decrement.sgy"),
    (
        "increment.sgy",
        "every header field is set to its minimum, so per-trace ns contradicts the binary header",
    ),
    ("increment-lsb.sgy", "little-endian copy of increment.sgy"),
    (
        "long.sgy",
        "declares 60000 samples but the trace headers are missing (test-data/README.md)",
    ),
    (
        "mapping-empty.sgy",
        "rev 2.1 mapping fixture whose size is not headers plus whole traces",
    ),
    (
        "mapping-invalid-value-size-scale6.sgy",
        "rev 2.1 mapping fixture whose size is not headers plus whole traces",
    ),
    ("small.su", "Seismic Unix: no textual or binary header"),
    ("small-lsb.su", "Seismic Unix: no textual or binary header"),
];

/// Decode then re-encode is not byte-identical here: format 4 stores one value
/// in several (gain, magnitude) pairs and the encoder picks the most precise.
/// The values are identical, and the raw copy is byte-identical.
const DECODED_VALUE_ONLY: &[&str] = &["multiformats/Format4msb.sgy", "multiformats/Format4lsb.sgy"];

fn fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in [root(), root().join("multiformats"), root().join("segyrs")] {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() && path.extension().is_some_and(|e| e == "sgy" || e == "su") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn rel(path: &Path) -> String {
    path.strip_prefix(root()).unwrap().display().to_string()
}

/// Open for copying. Format 4 needs a gain convention to open; which one does
/// not matter to a byte copy. A geometry refusal is retried with the key bytes
/// pointed past the trace header, because a copy does not need the grid.
fn open_for_copy(path: &Path) -> segyrs::Result<File> {
    let options = OpenOptions {
        gain: Some(GainConvention::Rev0),
        ..OpenOptions::default()
    };
    match File::open_with(path, &options) {
        Err(Error::Geometry { .. }) | Err(Error::DuplicateKey { .. }) => {
            let keyless = OpenOptions {
                inline_byte: u16::MAX,
                crossline_byte: u16::MAX,
                offset_byte: u16::MAX,
                ..options
            };
            File::open_with(path, &keyless)
        }
        other => other,
    }
}

#[test]
fn every_fixture_round_trips_byte_for_byte() {
    let mut refused = Vec::new();
    let mut copied = 0;
    for path in fixtures() {
        let name = rel(&path);
        let file = match open_for_copy(&path) {
            Ok(file) => file,
            Err(err) => {
                refused.push(name.clone());
                assert!(
                    MALFORMED.iter().any(|(m, _)| *m == name),
                    "{name} does not open and is not listed as malformed: {err}"
                );
                continue;
            }
        };
        let original = std::fs::read(&path).unwrap();

        // Raw path: headers and sample bytes copied, source format kept.
        let out = scratch("raw.sgy");
        let written = copy(&file, &out, &fast()).unwrap();
        assert_eq!(written, original.len() as u64, "{name}");
        assert!(
            std::fs::read(&out).unwrap() == original,
            "{name}: raw copy differs"
        );

        // Decoded path: every sample decoded to its natural type, then encoded again.
        let n = file.layout().n_traces as usize;
        let decoded = file.decode(0, n).unwrap();
        let mut spec = NewFile::from_file(&file).unwrap();
        spec.samples = Some(SampleData::from(&decoded));
        create(&out, &spec, &fast()).unwrap();
        let again = std::fs::read(&out).unwrap();
        if DECODED_VALUE_ONLY.contains(&name.as_str()) {
            let reread = File::open_with(
                &out,
                &OpenOptions {
                    gain: Some(GainConvention::Rev0),
                    ..OpenOptions::default()
                },
            )
            .unwrap();
            let values = reread.decode(0, n).unwrap();
            assert_eq!(values.as_f64(), decoded.as_f64(), "{name}: values differ");
            assert_eq!(again[..3600], original[..3600], "{name}");
        } else {
            assert!(again == original, "{name}: decode and re-encode differs");
        }
        copied += 1;
    }
    let mut listed: Vec<String> = MALFORMED.iter().map(|(m, _)| m.to_string()).collect();
    listed.sort();
    refused.sort();
    assert_eq!(refused, listed, "the malformed list is stale");
    assert!(copied >= 80, "only {copied} fixtures were copied");
}

#[test]
fn endian_flip_reproduces_the_lsb_fixtures() {
    // small and f3 were flipped field by field, so the flip is exact.
    for (msb, lsb) in [("small.sgy", "small-lsb.sgy"), ("f3.sgy", "f3-lsb.sgy")] {
        let file = File::open(root().join(msb)).unwrap();
        let out = scratch("le.sgy");
        copy(
            &file,
            &out,
            &WriteOptions {
                endian: Some(Endian::Little),
                ..fast()
            },
        )
        .unwrap();
        assert!(
            std::fs::read(&out).unwrap() == std::fs::read(root().join(lsb)).unwrap(),
            "{msb} flipped is not {lsb}"
        );
    }
    // The multiformat lsb files also swapped bytes 3501-3502 as one 16-bit
    // word. Rev 2 defines them as two single bytes (major, minor), which have
    // no byte order, so the writer leaves them. Everything else matches, and
    // flipping back gives the original.
    for code in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16] {
        let msb = root().join(format!("multiformats/Format{code}msb.sgy"));
        let lsb = root().join(format!("multiformats/Format{code}lsb.sgy"));
        let file = open_for_copy(&msb).unwrap();
        let out = scratch("le.sgy");
        copy(
            &file,
            &out,
            &WriteOptions {
                endian: Some(Endian::Little),
                ..fast()
            },
        )
        .unwrap();
        let got = std::fs::read(&out).unwrap();
        let want = std::fs::read(&lsb).unwrap();
        assert_eq!(got.len(), want.len());
        let diff: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        assert_eq!(diff, vec![3500, 3501], "format {code}");

        let le = open_for_copy(&out).unwrap();
        let back = scratch("be.sgy");
        copy(
            &le,
            &back,
            &WriteOptions {
                endian: Some(Endian::Big),
                ..fast()
            },
        )
        .unwrap();
        assert!(
            std::fs::read(&back).unwrap() == std::fs::read(&msb).unwrap(),
            "format {code}"
        );
    }
}

#[test]
fn pairwise_round_trip_for_every_format() {
    for code in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16] {
        let msb = root().join(format!("multiformats/Format{code}msb.sgy"));
        let file = open_for_copy(&msb).unwrap();
        let out = scratch("pair.sgy");
        copy(
            &file,
            &out,
            &WriteOptions {
                endian: Some(Endian::Pair),
                revision: Some(segyrs::Revision::Rev21),
                ..fast()
            },
        )
        .unwrap();
        let pair = open_for_copy(&out).unwrap();
        assert_eq!(pair.endian(), Endian::Pair);
        let n = file.layout().n_traces as usize;
        let a = file.decode(0, n).unwrap();
        let b = pair.decode(0, n).unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "format {code}");
    }
}

#[test]
fn every_format_converts_to_every_float_target() {
    // Integer and float fixtures to IEEE64: exact for every source.
    for code in [1, 2, 3, 5, 6, 7, 8, 10, 11, 15, 16] {
        let src = root().join(format!("multiformats/Format{code}msb.sgy"));
        let file = File::open(&src).unwrap();
        let n = file.layout().n_traces as usize;
        let out = scratch("f64.sgy");
        create(
            &out,
            &NewFile::from_file(&file).unwrap(),
            &WriteOptions {
                sample_format: Some(SampleFormat::IeeeFloat64),
                ..fast()
            },
        )
        .unwrap();
        let converted = File::open(&out).unwrap();
        assert_eq!(converted.format(), SampleFormat::IeeeFloat64);
        let want: Vec<f64> = as_f64(&file.decode(0, n).unwrap());
        assert_eq!(
            converted.decode(0, n).unwrap().as_f64().unwrap(),
            &want[..],
            "format {code}"
        );
    }
}

fn as_f64(s: &segyrs::Samples) -> Vec<f64> {
    use segyrs::Samples::*;
    match s {
        I8(v) => v.iter().map(|&x| x.into()).collect(),
        U8(v) => v.iter().map(|&x| x.into()).collect(),
        I16(v) => v.iter().map(|&x| x.into()).collect(),
        U16(v) => v.iter().map(|&x| x.into()).collect(),
        I32(v) => v.iter().map(|&x| x.into()).collect(),
        U32(v) => v.iter().map(|&x| x.into()).collect(),
        I64(v) => v.iter().map(|&x| x as f64).collect(),
        U64(v) => v.iter().map(|&x| x as f64).collect(),
        F32(v) => v.iter().map(|&x| x.into()).collect(),
        F64(v) => v.clone(),
    }
}
