//! Write-path tests named after their ledger ids (.ai/research/upstream-issues.md).

use std::path::PathBuf;

use segyrs::write::{
    copy, create, f32_to_ibm, f64_to_ibm, validate, Editor, Expect, HeaderValues, NewFile, Overlay,
    SampleData, Scalars, Source, WriteOptions, IBM_MAX,
};
use segyrs::{
    ibm_to_f32, Endian, Error, File, GeometryKind, HeaderDtype, HeaderField, HeaderSpec, Revision,
    SampleFormat,
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name)
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("write_ledger");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A private copy of a fixture, so in-place edits never touch test-data/.
fn working_copy(fixture: &str, name: &str) -> PathBuf {
    let out = scratch(name);
    std::fs::copy(data(fixture), &out).unwrap();
    out
}

fn fast() -> WriteOptions {
    WriteOptions {
        sync: false,
        ..WriteOptions::default()
    }
}

fn field(byte: u16, dtype: HeaderDtype, name: &str) -> HeaderField {
    HeaderField::new(byte, dtype, name)
}

fn be_i32(bytes: &[u8], at: usize) -> i32 {
    i32::from_be_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn be_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_be_bytes(bytes[at..at + 2].try_into().unwrap())
}

#[test]
fn ibm_encode_table() {
    // (f64 input, IBM word): exact values, rounding to nearest with ties to
    // even, a carry into the next hex digit, underflow, and the range edges.
    let one_ulp = 2f64.powi(-20); // IBM spacing in [1, 2)
    let table: &[(f64, u32)] = &[
        (0.0, 0x0000_0000),
        (-0.0, 0x8000_0000),
        (1.0, 0x4110_0000),
        (-1.0, 0xc110_0000),
        (0.15625, 0x4028_0000),
        (-118.625, 0xc276_a000),
        (1.0 + one_ulp / 4.0, 0x4110_0000), // below half: down
        (1.0 + one_ulp / 2.0, 0x4110_0000), // tie, even fraction stays
        (1.0 + 1.5 * one_ulp, 0x4110_0002), // tie, odd fraction rounds up
        (1.0 + one_ulp / 2.0 + 1e-12, 0x4110_0001), // above half: up
        (16.0 - 2f64.powi(-22), 0x4210_0000), // carry: 0xFFFFFF + 1 -> 16
        (2f64.powi(-260), 0x0010_0000),     // smallest normalized
        (2f64.powi(-280), 0x0000_0001),     // smallest unnormalized
        (2f64.powi(-281), 0x0000_0000),     // tie with zero: even
        (2f64.powi(-300), 0x0000_0000),     // underflow to zero
        (f64::INFINITY, IBM_MAX),
        (f64::NEG_INFINITY, 0x8000_0000 | IBM_MAX),
    ];
    for &(value, word) in table {
        assert_eq!(f64_to_ibm(value), Some(word), "{value:e} -> {word:#010x}");
    }
    assert_eq!(f64_to_ibm(f64::NAN), None);
    assert_eq!(
        f64_to_ibm(1e76),
        None,
        "finite overflow is an error, not a clamp"
    );
    assert!(f64_to_ibm(7.2e75).is_some());
    // f32 inputs: every finite f32 is in range.
    assert_eq!(f32_to_ibm(f32::MAX).map(ibm_to_f32), Some(f32::MAX));
    assert_eq!(
        f32_to_ibm(f32::MIN_POSITIVE).map(ibm_to_f32),
        Some(f32::MIN_POSITIVE)
    );
    assert_eq!(
        f32_to_ibm(f32::INFINITY).map(ibm_to_f32),
        Some(f32::INFINITY)
    );
}

#[test]
fn ibm_decode_table() {
    let table: &[(u32, f32)] = &[
        (0x4110_0000, 1.0),
        (0xc110_0000, -1.0),
        (0x4028_0000, 0.15625),
        (0xc276_a000, -118.625),
        (0x4210_0000, 16.0),
        (0x0000_0000, 0.0),
    ];
    for &(word, value) in table {
        assert_eq!(ibm_to_f32(word), value, "{word:#010x}");
        assert_eq!(f32_to_ibm(value), Some(word), "{value}");
    }
}

/// Reference IBM rounding built a different way: scale by an exact power of
/// two in f64, then `round_ties_even`.
fn reference_ibm(x: f32) -> u32 {
    let sign = if x.is_sign_negative() {
        0x8000_0000u32
    } else {
        0
    };
    let a = f64::from(x).abs();
    if a == 0.0 {
        return sign;
    }
    // smallest e with a < 16^(e-64)
    let mut e: i32 = 64;
    while a >= 16f64.powi(e - 64) {
        e += 1;
    }
    while a < 16f64.powi(e - 65) {
        e -= 1;
    }
    let mut frac = (a * 2f64.powi(24 - 4 * (e - 64))).round_ties_even();
    if frac >= 16_777_216.0 {
        frac /= 16.0;
        e += 1;
    }
    sign | ((e as u32) << 24) | frac as u32
}

fn check_f32_rounding(step: usize) {
    use rayon::prelude::*;
    let count = (u64::from(u32::MAX) / step as u64) + 1;
    let bad = (0..count)
        .into_par_iter()
        .map(|i| (i * step as u64) as u32)
        .filter(|&bits| {
            let x = f32::from_bits(bits);
            if !x.is_finite() {
                return false;
            }
            f32_to_ibm(x) != Some(reference_ibm(x))
        })
        .count();
    assert_eq!(bad, 0);
}

#[test]
fn ibm_encode_matches_reference_rounding_sampled() {
    check_f32_rounding(4099);
}

/// Every one of the 2^32 f32 bit patterns. Run with `--ignored --release`.
#[test]
#[ignore]
fn ibm_encode_matches_reference_rounding_exhaustive() {
    check_f32_rounding(1);
}

/// Every normalized IBM word that decodes to a normal f32 encodes back to
/// itself. Run with `--ignored --release`.
#[test]
#[ignore]
fn ibm_decode_encode_identity_exhaustive() {
    use rayon::prelude::*;
    let bad = (0..=u32::MAX)
        .into_par_iter()
        .filter(|&w| w & 0x00f0_0000 != 0) // leading hex digit nonzero
        .filter(|&w| {
            let x = ibm_to_f32(w);
            x.is_normal() && f32_to_ibm(x) != Some(w)
        })
        .count();
    assert_eq!(bad, 0);
}

#[test]
fn s08_batched_header_column_matches_full_rewrite() {
    let path = working_copy("small.sgy", "s08.sgy");
    let file = File::open(&path).unwrap();
    let n = file.layout().n_traces as usize;
    let cdp = field(21, HeaderDtype::I32, "cdp");
    let values: Vec<i32> = (0..n as i32).map(|i| 1000 + 7 * i).collect();

    // Full rewrite with the same column, for comparison.
    let mut spec = NewFile::from_file(&file).unwrap();
    spec.overlays.push(Overlay {
        field: cdp.clone(),
        values: HeaderValues::I32(&values),
    });
    let rewrite = scratch("s08-rewrite.sgy");
    create(&rewrite, &spec, &fast()).unwrap();
    drop(file);

    let editor = Editor::open(&path).unwrap();
    editor
        .write_header_column(&cdp, HeaderValues::I32(&values))
        .unwrap();
    editor.sync().unwrap();
    assert!(std::fs::read(&path).unwrap() == std::fs::read(&rewrite).unwrap());
    validate(&path).unwrap();

    let reread = File::open(&path).unwrap();
    let spec = HeaderSpec { fields: vec![cdp] };
    assert_eq!(
        reread
            .par_header_columns(&spec)
            .unwrap()
            .i32("cdp")
            .unwrap(),
        &values[..]
    );
}

#[test]
fn s08_several_fields_on_selected_traces_keep_the_gap_bytes() {
    let path = working_copy("small.sgy", "s08-gap.sgy");
    let before = std::fs::read(&path).unwrap();
    let editor = Editor::open(&path).unwrap();
    // 3 and 4 form a run (rewritten together); 0 and 7 are single pwrites.
    let traces = [3u64, 0, 7, 4];
    // Bytes 9-12 and 21-24: the 8 bytes between them must survive.
    editor
        .write_header_columns(
            &[
                Overlay {
                    field: field(9, HeaderDtype::I32, "field_record"),
                    values: HeaderValues::Fill(-5),
                },
                Overlay {
                    field: field(21, HeaderDtype::I32, "cdp"),
                    values: HeaderValues::I64(&[30, 0, 70, 40]),
                },
            ],
            Some(&traces),
        )
        .unwrap();
    let after = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let stride = file.stride() as usize;
    for (i, t) in traces.iter().enumerate() {
        let h = 3600 + *t as usize * stride;
        assert_eq!(be_i32(&after, h + 8), -5);
        assert_eq!(be_i32(&after, h + 20), [30, 0, 70, 40][i]);
        assert_eq!(after[h + 12..h + 20], before[h + 12..h + 20]);
    }
    let changed: Vec<usize> = (0..after.len())
        .filter(|&i| after[i] != before[i])
        .collect();
    assert!(changed.iter().all(|&i| traces.iter().any(|&t| {
        let h = 3600 + t as usize * stride;
        (h + 8..h + 24).contains(&i)
    })));
}

#[test]
fn s08_long_traces_take_one_pwrite_each_and_match_a_rewrite() {
    // 240 + 5000 * 4 bytes per trace: above the run-rewrite threshold.
    let samples: Vec<f32> = (0..4 * 5000).map(|i| i as f32).collect();
    let mut spec = NewFile::new(4, 5000, 1000);
    spec.samples = Some(SampleData::F32(&samples));
    let options = WriteOptions {
        sample_format: Some(SampleFormat::IeeeFloat32),
        ..fast()
    };
    let path = scratch("s08-long.sgy");
    create(&path, &spec, &options).unwrap();
    let cdp = field(21, HeaderDtype::I32, "cdp");
    let values = [9, 8, 7, 6];
    spec.overlays.push(Overlay {
        field: cdp.clone(),
        values: HeaderValues::I32(&values),
    });
    let rewrite = scratch("s08-long-rewrite.sgy");
    create(&rewrite, &spec, &options).unwrap();
    Editor::open(&path)
        .unwrap()
        .write_header_column(&cdp, HeaderValues::I32(&values))
        .unwrap();
    assert!(std::fs::read(&path).unwrap() == std::fs::read(&rewrite).unwrap());
}

#[test]
fn s29_interval_is_the_integer_given() {
    // A step taken from a float axis truncates: 0.2 / 0.01 is 19.999999999999996.
    let derived = ((0.3f64 - 0.1) / 0.01) as u32;
    assert_eq!(derived, 19);
    // The writer takes the integer the caller gives and never re-derives it.
    let samples = vec![0f32; 2 * 5];
    let mut spec = NewFile::new(2, 5, 20);
    spec.samples = Some(SampleData::F32(&samples));
    let out = scratch("s29.sgy");
    create(
        &out,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            ..fast()
        },
    )
    .unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(be_u16(&bytes, 3216), 20);
    assert_eq!(be_u16(&bytes, 3600 + 116), 20, "trace header interval");
    assert_eq!(File::open(&out).unwrap().layout().sample_interval_us, 20);
}

#[test]
fn s31_create_from_arrays_with_no_spec_object() {
    // 3 inlines x 2 crosslines x 4 samples, IEEE float, big-endian.
    let (nil, nxl, ns) = (3usize, 2usize, 4u32);
    let n = nil * nxl;
    let samples: Vec<f32> = (0..n * ns as usize).map(|i| i as f32 * 0.5 - 3.0).collect();
    let inline: Vec<i32> = (0..n).map(|t| 10 + (t / nxl) as i32).collect();
    let crossline: Vec<i32> = (0..n).map(|t| 20 + (t % nxl) as i32).collect();
    let mut spec = NewFile::new(n as u64, ns, 4000);
    spec.samples = Some(SampleData::F32(&samples));
    spec.overlays = vec![
        Overlay {
            field: field(189, HeaderDtype::I32, "inline"),
            values: HeaderValues::I32(&inline),
        },
        Overlay {
            field: field(193, HeaderDtype::I32, "crossline"),
            values: HeaderValues::I32(&crossline),
        },
    ];
    let out = scratch("s31.sgy");
    create(
        &out,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            ..fast()
        },
    )
    .unwrap();
    let file = File::open(&out).unwrap();
    assert_eq!(file.format(), SampleFormat::IeeeFloat32);
    assert_eq!(file.decode(0, n).unwrap().as_f32().unwrap(), &samples[..]);
    assert_eq!(file.inline(), &inline[..]);
    assert_eq!(file.crossline(), &crossline[..]);
    assert_eq!(file.geometry().kind, GeometryKind::InlineMajor);
    validate(&out).unwrap();

    // A new file with no source needs an explicit sample format (decision 3).
    let err = create(scratch("s31-none.sgy"), &spec, &fast()).unwrap_err();
    assert!(matches!(err, Error::Spec(_)), "{err}");
}

#[test]
fn s31_every_format_and_endian_from_arrays() {
    let ints: Vec<i32> = vec![-3, 0, 1, 2, 100, -100];
    let unsigned: Vec<u32> = vec![0, 1, 2, 3, 100, 200];
    for format in [
        SampleFormat::IbmFloat32,
        SampleFormat::Int32,
        SampleFormat::Int16,
        SampleFormat::IeeeFloat32,
        SampleFormat::IeeeFloat64,
        SampleFormat::Int24,
        SampleFormat::Int8,
        SampleFormat::Int64,
        SampleFormat::Uint32,
        SampleFormat::Uint16,
        SampleFormat::Uint64,
        SampleFormat::Uint24,
        SampleFormat::Uint8,
    ] {
        for endian in [Endian::Big, Endian::Little, Endian::Pair] {
            let signed = !matches!(
                format,
                SampleFormat::Uint32
                    | SampleFormat::Uint16
                    | SampleFormat::Uint64
                    | SampleFormat::Uint24
                    | SampleFormat::Uint8
            );
            let mut spec = NewFile::new(2, 3, 2000);
            spec.samples = Some(if signed {
                SampleData::I32(&ints)
            } else {
                SampleData::U32(&unsigned)
            });
            let out = scratch("s31-any.sgy");
            create(
                &out,
                &spec,
                &WriteOptions {
                    sample_format: Some(format),
                    endian: Some(endian),
                    ..fast()
                },
            )
            .unwrap();
            let file = File::open(&out).unwrap_or_else(|e| panic!("{format:?} {endian:?}: {e}"));
            assert_eq!(file.endian(), endian);
            assert_eq!(file.format(), format);
            let got: Vec<f64> = match file.decode(0, 2).unwrap() {
                segyrs::Samples::I8(v) => v.iter().map(|&x| x.into()).collect(),
                segyrs::Samples::U8(v) => v.iter().map(|&x| x.into()).collect(),
                segyrs::Samples::I16(v) => v.iter().map(|&x| x.into()).collect(),
                segyrs::Samples::U16(v) => v.iter().map(|&x| x.into()).collect(),
                segyrs::Samples::I32(v) => v.iter().map(|&x| x.into()).collect(),
                segyrs::Samples::U32(v) => v.iter().map(|&x| x.into()).collect(),
                segyrs::Samples::I64(v) => v.iter().map(|&x| x as f64).collect(),
                segyrs::Samples::U64(v) => v.iter().map(|&x| x as f64).collect(),
                segyrs::Samples::F32(v) => v.iter().map(|&x| x.into()).collect(),
                segyrs::Samples::F64(v) => v.clone(),
            };
            let want: Vec<f64> = if signed {
                ints.iter().map(|&x| x.into()).collect()
            } else {
                unsigned.iter().map(|&x| x.into()).collect()
            };
            assert_eq!(got, want, "{format:?} {endian:?}");
            validate(&out).unwrap();
        }
    }
}

#[test]
fn s32_short_sample_vector_raises_before_any_write() {
    let path = working_copy("small.sgy", "s32.sgy");
    let before = std::fs::read(&path).unwrap();
    let editor = Editor::open(&path).unwrap();
    let ns = editor.layout().n_samples as u64;
    // A scalar where a trace belongs.
    let err = editor
        .write_trace_samples(2, SampleData::F32(&[375.0]))
        .unwrap_err();
    assert!(
        matches!(err, Error::TraceLength { trace: 2, expected, got: 1 } if expected == ns),
        "{err}"
    );
    // A batch with a partial last trace.
    let partial = vec![1.0f32; ns as usize + 3];
    let err = editor
        .write_traces(4, SampleData::F32(&partial))
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::TraceLength {
                trace: 5,
                got: 3,
                ..
            }
        ),
        "{err}"
    );
    assert!(std::fs::read(&path).unwrap() == before, "file changed");

    // The same rule on create: the destination is never created.
    let file = File::open(&path).unwrap();
    let mut spec = NewFile::from_file(&file).unwrap();
    spec.samples = Some(SampleData::F32(&partial));
    let dest = scratch("s32-new.sgy");
    let _ = std::fs::remove_file(&dest);
    assert!(matches!(
        create(&dest, &spec, &fast()).unwrap_err(),
        Error::TraceLength { .. }
    ));
    assert!(!dest.exists());

    // A full trace goes through, in the file's own format.
    let trace: Vec<f32> = (0..ns).map(|i| i as f32).collect();
    editor
        .write_trace_samples(2, SampleData::F32(&trace))
        .unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.format(), SampleFormat::IbmFloat32);
    assert_eq!(file.decode(2, 1).unwrap().as_f32().unwrap(), &trace[..]);
}

#[test]
fn s33_shorter_sample_axis_updates_binary_and_trace_headers() {
    let src = File::open(data("f3.sgy")).unwrap();
    let ns = src.layout().n_samples;
    let mut spec = NewFile::from_file(&src).unwrap();
    spec.n_samples = ns - 50;
    spec.source.as_mut().unwrap().first_sample = 10;
    let out = scratch("s33.sgy");
    create(&out, &spec, &fast()).unwrap();
    let file = File::open(&out).unwrap();
    assert_eq!(file.layout().n_samples, ns - 50);
    let len = std::fs::metadata(&out).unwrap().len();
    assert_eq!(
        len,
        3600 + src.layout().n_traces * (240 + u64::from(ns - 50) * 2)
    );
    let n = src.layout().n_traces as usize;
    let a = src.decode(0, n).unwrap();
    let b = file.decode(0, n).unwrap();
    let (a, b) = (a.as_i16().unwrap(), b.as_i16().unwrap());
    for t in 0..n {
        let s = t * ns as usize + 10;
        assert_eq!(
            &b[t * (ns as usize - 50)..(t + 1) * (ns as usize - 50)],
            &a[s..s + ns as usize - 50]
        );
    }

    // A file that stores ns in every trace header: those words follow the change.
    let samples: Vec<f32> = (0..3 * 60).map(|i| i as f32).collect();
    let mut spec = NewFile::new(3, 60, 2000);
    spec.samples = Some(SampleData::F32(&samples));
    let full = scratch("s33-full.sgy");
    create(
        &full,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            ..fast()
        },
    )
    .unwrap();
    let full = File::open(&full).unwrap();
    let mut spec = NewFile::from_file(&full).unwrap();
    spec.n_samples = 10;
    let out = scratch("s33-short.sgy");
    create(&out, &spec, &fast()).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(be_u16(&bytes, 3220), 10);
    for t in 0..3 {
        assert_eq!(be_u16(&bytes, 3600 + t * (240 + 40) + 114), 10);
    }
    validate(&out).unwrap();
}

#[test]
fn s34_validator_accepts_fixtures_and_rejects_a_truncated_copy() {
    for good in [
        "small.sgy",
        "small-lsb.sgy",
        "multi-text.sgy",
        "stanzas-known-count.sgy",
        "text.sgy",
        "trace-header-extensions.sgy",
    ] {
        validate(data(good)).unwrap_or_else(|e| panic!("{good}: {e}"));
    }
    let bytes = std::fs::read(data("small.sgy")).unwrap();
    let cut = scratch("s34-cut.sgy");
    std::fs::write(&cut, &bytes[..bytes.len() - 100]).unwrap();
    assert!(matches!(validate(&cut).unwrap_err(), Error::Invalid { .. }));

    // A per-trace ns that disagrees with the binary header is reported.
    let mut bad = bytes.clone();
    bad[3600 + 114..3600 + 116].copy_from_slice(&7u16.to_be_bytes());
    let path = scratch("s34-ns.sgy");
    std::fs::write(&path, &bad).unwrap();
    let err = validate(&path).unwrap_err();
    assert!(
        matches!(err, Error::Invalid { .. }) && err.to_string().contains("trace_ns=7"),
        "{err}"
    );

    // An IEEE fixture written back is identical from byte 3600 on.
    let src = File::open(data("multiformats/Format5msb.sgy")).unwrap();
    let out = scratch("s34-ieee.sgy");
    copy(&src, &out, &fast()).unwrap();
    assert!(
        std::fs::read(&out).unwrap()[3600..]
            == std::fs::read(data("multiformats/Format5msb.sgy")).unwrap()[3600..]
    );
}

#[test]
fn s35_in_place_format_change_is_refused_and_the_file_is_untouched() {
    let path = working_copy("small.sgy", "s35.sgy");
    let before = std::fs::read(&path).unwrap();
    let expect = Expect {
        sample_format: Some(SampleFormat::IeeeFloat32),
        ..Expect::default()
    };
    let err = Editor::open_with(&path, &Default::default(), &expect)
        .err()
        .unwrap();
    assert!(matches!(err, Error::FormatChange), "{err}");

    // Changing the format code, the sample count, or the textual framing
    // through a header write is refused too.
    let mut editor = Editor::open(&path).unwrap();
    let mut bin = before[3200..3600].to_vec();
    bin[24..26].copy_from_slice(&5u16.to_be_bytes());
    assert!(matches!(
        editor.write_binary_header(&bin),
        Err(Error::FormatChange)
    ));
    let mut bin = before[3200..3600].to_vec();
    bin[20..22].copy_from_slice(&49u16.to_be_bytes());
    assert!(matches!(
        editor.write_binary_header(&bin),
        Err(Error::FormatChange)
    ));
    let ns_field = field(115, HeaderDtype::U16, "ns");
    assert!(matches!(
        editor.write_header_column(&ns_field, HeaderValues::Fill(49)),
        Err(Error::FormatChange)
    ));
    assert!(std::fs::read(&path).unwrap() == before, "file changed");

    // A value-only binary edit (the interval) goes through.
    let mut bin = before[3200..3600].to_vec();
    bin[16..18].copy_from_slice(&2000u16.to_be_bytes());
    editor.write_binary_header(&bin).unwrap();
    assert_eq!(editor.layout().sample_interval_us, 2000);

    // A format change is a new file, written beside and renamed over.
    let file = File::open(&path).unwrap();
    let n = file.layout().n_traces as usize;
    let want = file.decode(0, n).unwrap();
    copy(
        &file,
        &path,
        &WriteOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            ..fast()
        },
    )
    .unwrap();
    let converted = File::open(&path).unwrap();
    assert_eq!(converted.format(), SampleFormat::IeeeFloat32);
    assert_eq!(converted.decode(0, n).unwrap().as_f32(), want.as_f32());
}

#[test]
fn s35_ibm_stays_ibm_unless_asked() {
    let src = File::open(data("small.sgy")).unwrap();
    assert_eq!(src.format(), SampleFormat::IbmFloat32);
    let n = src.layout().n_traces as usize;
    let decoded = src.decode(0, n).unwrap();
    let mut spec = NewFile::from_file(&src).unwrap();
    spec.samples = Some(SampleData::from(&decoded));
    let out = scratch("s35-ibm.sgy");
    create(&out, &spec, &fast()).unwrap();
    assert_eq!(File::open(&out).unwrap().format(), SampleFormat::IbmFloat32);

    // IEEE to IBM, explicitly: values within IBM rounding, NaN refused.
    let ieee = File::open(data("multiformats/Format5msb.sgy")).unwrap();
    let out = scratch("s35-to-ibm.sgy");
    copy(
        &ieee,
        &out,
        &WriteOptions {
            sample_format: Some(SampleFormat::IbmFloat32),
            ..fast()
        },
    )
    .unwrap();
    let ibm = File::open(&out).unwrap();
    assert_eq!(ibm.format(), SampleFormat::IbmFloat32);
    let n = ieee.layout().n_traces as usize;
    let a = ieee.decode(0, n).unwrap();
    let b = ibm.decode(0, n).unwrap();
    for (x, y) in a.as_f32().unwrap().iter().zip(b.as_f32().unwrap()) {
        assert!((x - y).abs() <= x.abs() * 2f32.powi(-20), "{x} {y}");
    }
    let nan = [1.0f32, f32::NAN];
    let mut spec = NewFile::new(1, 2, 1000);
    spec.samples = Some(SampleData::F32(&nan));
    let err = create(
        scratch("s35-nan.sgy"),
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IbmFloat32),
            ..fast()
        },
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            Error::SampleValue {
                trace: 0,
                sample: 1,
                ..
            }
        ),
        "{err}"
    );

    // An integer target refuses a fraction rather than truncating it.
    let frac = [1.0f32, 2.5];
    spec.samples = Some(SampleData::F32(&frac));
    let err = create(
        scratch("s35-int.sgy"),
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::Int16),
            ..fast()
        },
    )
    .unwrap_err();
    assert!(matches!(err, Error::SampleValue { sample: 1, .. }), "{err}");
}

#[test]
fn x22_unedited_binary_header_is_written_back_byte_for_byte() {
    let src = File::open(data("rotated-small-rev2.sgy")).unwrap();
    let original = std::fs::read(data("rotated-small-rev2.sgy")).unwrap();
    assert_eq!(src.binary_header_bytes(), &original[3200..3600]);
    let out = scratch("x22.sgy");
    copy(&src, &out, &fast()).unwrap();
    assert_eq!(
        std::fs::read(&out).unwrap()[3200..3600],
        original[3200..3600]
    );

    // Editing one value changes only that word.
    let mut spec = NewFile::from_file(&src).unwrap();
    spec.sample_interval_us = src.layout().sample_interval_us + 1000;
    create(&out, &spec, &fast()).unwrap();
    let written = std::fs::read(&out).unwrap();
    let changed: Vec<usize> = (3200..3600)
        .filter(|&i| written[i] != original[i])
        .collect();
    assert!(
        changed
            .iter()
            .all(|&i| (3216..3218).contains(&i) || (3272..3280).contains(&i)),
        "{changed:?}"
    );
    assert_eq!(
        File::open(&out).unwrap().layout().sample_interval_us,
        src.layout().sample_interval_us + 1000
    );
}

#[test]
fn x24_float_coordinates_are_inverse_scaled_and_overflow_names_the_field() {
    let samples = [0f32; 3];
    let cdp_x = [1234.5, -0.3, 2_000_000.0];
    let scalco = [-10i16; 3];
    let mut spec = NewFile::new(3, 1, 1000);
    spec.samples = Some(SampleData::F32(&samples));
    spec.overlays = vec![
        Overlay {
            field: field(71, HeaderDtype::I16, "scalco"),
            values: HeaderValues::I16(&scalco),
        },
        Overlay {
            field: field(181, HeaderDtype::I32, "cdp_x"),
            values: HeaderValues::Scaled {
                values: &cdp_x,
                scalars: Scalars::PerTrace(&scalco),
            },
        },
    ];
    let options = WriteOptions {
        sample_format: Some(SampleFormat::IeeeFloat32),
        ..fast()
    };
    let out = scratch("x24.sgy");
    create(&out, &spec, &options).unwrap();
    let file = File::open(&out).unwrap();
    assert_eq!(file.cdp_x(), &[12345, -3, 20_000_000]);
    for (raw, want) in file.cdp_x().iter().zip(cdp_x) {
        assert!((segyrs::apply_scalar(*raw, -10) - want).abs() < 1e-9);
    }

    // Above int32 after scaling: the error names the field, value, and width.
    let too_big = [1234.5, 0.0, 3e8];
    spec.overlays[1].values = HeaderValues::Scaled {
        values: &too_big,
        scalars: Scalars::One(-10),
    };
    let err = create(scratch("x24-big.sgy"), &spec, &options).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, Error::HeaderValue { .. }), "{msg}");
    assert!(
        msg.contains("cdp_x") && msg.contains("3000000000") && msg.contains("int32"),
        "{msg}"
    );

    // A value the scalar cannot represent is refused, not rounded away.
    let fine = [1234.56, 0.0, 0.0];
    spec.overlays[1].values = HeaderValues::Scaled {
        values: &fine,
        scalars: Scalars::One(-10),
    };
    let err = create(scratch("x24-frac.sgy"), &spec, &options).unwrap_err();
    assert!(err.to_string().contains("1234.56"), "{err}");
}

#[test]
fn x25_two_dimensional_line_round_trips_the_cdp_coordinate() {
    // dims (cdp, time): no inline or crossline at all.
    let cdp: Vec<i32> = (100..110).collect();
    let samples: Vec<f32> = (0..10 * 8).map(|i| (i as f32).sin()).collect();
    let mut spec = NewFile::new(10, 8, 4000);
    spec.samples = Some(SampleData::F32(&samples));
    spec.overlays.push(Overlay {
        field: field(21, HeaderDtype::I32, "cdp"),
        values: HeaderValues::I32(&cdp),
    });
    let out = scratch("x25.sgy");
    create(
        &out,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IbmFloat32),
            ..fast()
        },
    )
    .unwrap();
    let file = File::open(&out).unwrap();
    assert_eq!(file.geometry().kind, GeometryKind::Unstructured);
    let spec_cdp = HeaderSpec {
        fields: vec![field(21, HeaderDtype::I32, "cdp")],
    };
    assert_eq!(
        file.par_header_columns(&spec_cdp)
            .unwrap()
            .i32("cdp")
            .unwrap(),
        &cdp[..]
    );
    let rounded: Vec<f32> = samples
        .iter()
        .map(|&x| ibm_to_f32(f32_to_ibm(x).unwrap()))
        .collect();
    assert_eq!(file.decode(0, 10).unwrap().as_f32().unwrap(), &rounded[..]);

    // Crop a 3D volume (sel), then write: headers come from the kept traces.
    let cube = File::open(data("small.sgy")).unwrap();
    let keep: Vec<u64> = (0..cube.layout().n_traces)
        .filter(|&t| {
            (2..=3).contains(&cube.inline()[t as usize]) && cube.crossline()[t as usize] >= 22
        })
        .collect();
    let mut spec = NewFile::from_file(&cube).unwrap();
    spec.n_traces = keep.len() as u64;
    spec.source = Some(Source {
        file: &cube,
        traces: Some(&keep),
        first_sample: 0,
    });
    let out = scratch("x25-crop.sgy");
    create(&out, &spec, &fast()).unwrap();
    let crop = File::open(&out).unwrap();
    assert_eq!(crop.geometry().kind, GeometryKind::InlineMajor);
    assert_eq!(crop.geometry().inlines, vec![2, 3]);
    assert_eq!(crop.geometry().crosslines, vec![22, 23, 24]);
    for (i, &t) in keep.iter().enumerate() {
        assert_eq!(
            crop.header_bytes(i as u64).unwrap(),
            cube.header_bytes(t).unwrap()
        );
        assert_eq!(
            crop.sample_bytes(i as u64).unwrap(),
            cube.sample_bytes(t).unwrap()
        );
    }
}

#[test]
fn x26_only_supplied_overlays_are_written() {
    let samples = [0f32; 4];
    let mut spec = NewFile::new(2, 2, 4000);
    spec.samples = Some(SampleData::F32(&samples));
    spec.overlays = vec![
        Overlay {
            field: field(189, HeaderDtype::I32, "inline"),
            values: HeaderValues::I32(&[5, 5]),
        },
        Overlay {
            field: field(193, HeaderDtype::I32, "crossline"),
            values: HeaderValues::I32(&[1, 2]),
        },
    ];
    let out = scratch("x26.sgy");
    create(
        &out,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            ..fast()
        },
    )
    .unwrap();
    let file = File::open(&out).unwrap();
    assert_eq!(file.inline(), &[5, 5]);
    assert_eq!(file.crossline(), &[1, 2]);
    for t in 0..2 {
        let h = file.header_bytes(t).unwrap();
        for (i, b) in h.iter().enumerate() {
            // The writer owns two structural words: ns (115-116) and interval (117-118).
            let owned = (114..118).contains(&i) || (188..196).contains(&i);
            if !owned {
                assert_eq!(*b, 0, "trace {t} byte {}", i + 1);
            }
        }
        assert_eq!(be_u16(&h, 114), 2);
        assert_eq!(be_u16(&h, 116), 4000);
    }
}

#[test]
fn new_little_endian_file_is_rev2_with_a_signature() {
    let samples = [1.5f32, -2.0];
    let mut spec = NewFile::new(1, 2, 1000);
    spec.samples = Some(SampleData::F32(&samples));
    let out = scratch("le.sgy");
    create(
        &out,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            endian: Some(Endian::Little),
            ..fast()
        },
    )
    .unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(&bytes[3296..3300], &[4, 3, 2, 1]);
    assert_eq!((bytes[3500], bytes[3501]), (2, 1));
    let file = File::open(&out).unwrap();
    assert_eq!(file.layout().revision, Revision::Rev21);
    assert_eq!(file.decode(0, 1).unwrap().as_f32().unwrap(), &samples);
}

#[test]
fn publish_leaves_no_temp_file_and_keeps_the_old_file_on_error() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("write_publish");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("out.sgy");
    std::fs::write(&dest, b"old").unwrap();
    let samples = [f32::NAN];
    let mut spec = NewFile::new(1, 1, 1000);
    spec.samples = Some(SampleData::F32(&samples));
    let err = create(
        &dest,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IbmFloat32),
            ..WriteOptions::default()
        },
    );
    assert!(err.is_err());
    assert_eq!(std::fs::read(&dest).unwrap(), b"old");
    let names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");

    // With sync (the default) the full sequence runs: fallocate, fdatasync, rename, dir fsync.
    let ok = [1.0f32];
    spec.samples = Some(SampleData::F32(&ok));
    create(
        &dest,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IbmFloat32),
            ..WriteOptions::default()
        },
    )
    .unwrap();
    validate(&dest).unwrap();
    let names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
}

#[test]
fn sample_value_error_names_the_first_bad_trace_on_any_thread_count() {
    let (n, ns) = (4096usize, 64usize);
    let mut frac = vec![0f32; n * ns];
    frac[1000 * ns] = 0.5;
    frac[2048 * ns..].fill(0.5);
    let mut spec = NewFile::new(n as u64, ns as u32, 1000);
    spec.samples = Some(SampleData::F32(&frac));
    let options = WriteOptions {
        sample_format: Some(SampleFormat::Int16),
        ..fast()
    };
    let path = scratch("first-bad-int16.sgy");
    let mut zeros = NewFile::new(n as u64, ns as u32, 1000);
    let zero = vec![0f32; n * ns];
    zeros.samples = Some(SampleData::F32(&zero));
    create(&path, &zeros, &options).unwrap();
    for threads in [1, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for _ in 0..5 {
            let err = pool
                .install(|| create(scratch("first-bad-out.sgy"), &spec, &options))
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::SampleValue {
                        trace: 1000,
                        sample: 0,
                        ..
                    }
                ),
                "{threads} threads: {err}"
            );
            let editor = Editor::open(&path).unwrap();
            let err = pool
                .install(|| editor.write_traces(0, SampleData::F32(&frac)))
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::SampleValue {
                        trace: 1000,
                        sample: 0,
                        ..
                    }
                ),
                "{threads} threads: {err}"
            );
        }
    }
}
