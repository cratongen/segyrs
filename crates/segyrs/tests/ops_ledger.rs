//! M3 ledger tests. Each test is named after the ledger id it proves.

mod common;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use common::*;
use segyrs::ops::{self, Budget, Crop, Predicate};
use segyrs::{Error, File, HeaderSpec, OpenOptions, SampleFormat, Samples};

fn budget() -> Budget {
    Budget::bytes(64 << 20)
}

fn column_i16(file: &File, name: &str) -> Vec<i16> {
    file.par_header_columns(&HeaderSpec::standard())
        .unwrap()
        .i16(name)
        .unwrap()
        .to_vec()
}

fn column_u16(file: &File, name: &str) -> Vec<u16> {
    file.par_header_columns(&HeaderSpec::standard())
        .unwrap()
        .u16(name)
        .unwrap()
        .to_vec()
}

fn f32s(samples: &Samples) -> &[f32] {
    samples.as_f32().unwrap()
}

#[test]
fn test_s10() {
    let bytes = std::fs::read(data("small.sgy")).unwrap();
    let reference = open("small.sgy");
    let n = reference.layout().n_traces as usize;
    let ns = reference.layout().n_samples as usize;
    let full = reference.decode(0, n).unwrap();
    let full = f32s(&full);

    let counter = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicU64::new(0));
    let reader = Counting {
        data: bytes.clone(),
        bytes: counter.clone(),
        calls: calls.clone(),
    };
    let file = File::from_reader(reader, bytes.len() as u64, &OpenOptions::default()).unwrap();
    let traces = all(&file);

    // One sample index: one 4-byte read per trace, nothing else.
    counter.store(0, Ordering::SeqCst);
    calls.store(0, Ordering::SeqCst);
    let slice = ops::time_slice(&file, &traces, 10).unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), (n * 4) as u64);
    assert_eq!(calls.load(Ordering::SeqCst), n as u64);
    let want: Vec<f32> = (0..n).map(|t| full[t * ns + 10]).collect();
    assert_eq!(f32s(&slice), &want[..]);

    // A horizon: a different index per trace, a window of 2 above and 1 below.
    let index: Vec<i64> = (0..n as i64)
        .map(|t| if t == 3 { -100 } else { (t * 2) % 50 })
        .collect();
    counter.store(0, Ordering::SeqCst);
    let win = ops::slice_horizon(&file, &traces, &index, 2, 1).unwrap();
    assert!(counter.load(Ordering::SeqCst) <= (n * 4 * 4) as u64);
    assert_eq!(win.width, 4);
    let got = f32s(&win.samples);
    for t in 0..n {
        for k in 0..4 {
            let s = index[t] - 2 + k as i64;
            let ok = (0..ns as i64).contains(&s);
            assert_eq!(win.valid[t * 4 + k], ok, "trace {t} k {k}");
            let expect = if ok { full[t * ns + s as usize] } else { 0.0 };
            assert_eq!(got[t * 4 + k], expect, "trace {t} k {k}");
        }
    }
    assert!(win.valid[3 * 4..4 * 4].iter().all(|v| !v));

    assert!(ops::time_slice(&file, &traces, ns).is_err());
}

#[test]
fn test_s30() {
    let file = open("acute-small.sgy");
    // Keep two inlines.
    let keep = ops::filter(&file, &[Predicate::is_in("inline", [2.0, 4.0])]).unwrap();
    assert_eq!(keep.len(), 10);

    let dir = scratch("test_s30");
    let path = dir.join("crop.sgy");
    let mut sink = FileSink::create(&path);
    ops::crop(&file, &keep, &Crop::default(), budget(), &mut sink).unwrap();
    let out = File::open(&path).unwrap();
    assert_eq!(out.layout().n_traces, 10);
    assert_eq!(out.format(), SampleFormat::IbmFloat32);
    let pick = |v: &[i32]| keep.iter().map(|&t| v[t as usize]).collect::<Vec<_>>();
    assert_eq!(out.cdp_x(), &pick(file.cdp_x())[..]);
    assert_eq!(out.cdp_y(), &pick(file.cdp_y())[..]);
    assert_eq!(out.inline(), &pick(file.inline())[..]);
    assert_eq!(out.crossline(), &pick(file.crossline())[..]);
    let scalco: Vec<i16> = keep.iter().map(|&t| file.scalco()[t as usize]).collect();
    assert_eq!(out.scalco(), &scalco[..]);
    assert!(out.cdp_y().iter().any(|&y| y != 0));
    for (i, &t) in keep.iter().enumerate() {
        assert_eq!(
            out.header_bytes(i as u64).unwrap(),
            file.header_bytes(t).unwrap()
        );
        assert_eq!(
            out.sample_bytes(i as u64).unwrap(),
            file.sample_bytes(t).unwrap()
        );
    }
}

#[test]
fn test_s36() {
    let dir = scratch("test_s36");
    let file = open("small.sgy");
    let traces = all(&file);
    let dt_ms = file.layout().sample_interval_us as i16 / 1000;

    // Cut samples 10..30: the first kept sample is still at 10 * dt.
    let path = dir.join("window.sgy");
    let crop = Crop {
        samples: Some(10..30),
        rezero_time: false,
    };
    ops::crop(
        &file,
        &traces,
        &crop,
        budget(),
        &mut FileSink::create(&path),
    )
    .unwrap();
    let out = File::open(&path).unwrap();
    assert_eq!(out.layout().n_samples, 20);
    assert!(column_i16(&out, "delay").iter().all(|&d| d == 10 * dt_ms));
    assert!(column_u16(&out, "ns").iter().all(|&n| n == 20));
    for t in 0..out.layout().n_traces {
        assert_eq!(
            out.sample_bytes(t).unwrap(),
            file.sample_bytes(t).unwrap()[10 * 4..30 * 4]
        );
    }

    // Rezero is explicit: delay becomes 0, samples are the same.
    let path = dir.join("rezero.sgy");
    let crop = Crop {
        samples: Some(10..30),
        rezero_time: true,
    };
    ops::crop(
        &file,
        &traces,
        &crop,
        budget(),
        &mut FileSink::create(&path),
    )
    .unwrap();
    let rezeroed = File::open(&path).unwrap();
    assert!(column_i16(&rezeroed, "delay").iter().all(|&d| d == 0));
    assert_eq!(
        rezeroed.sample_bytes(0).unwrap(),
        out.sample_bytes(0).unwrap()
    );

    // A delay with a time scalar: 10000 * 10^-1 ms = 1000 ms, plus 5 * 4 ms.
    let file = open("segyrs/delay-scalar.sgy");
    let path = dir.join("scalar.sgy");
    let crop = Crop {
        samples: Some(5..100),
        rezero_time: false,
    };
    ops::crop(
        &file,
        &all(&file),
        &crop,
        budget(),
        &mut FileSink::create(&path),
    )
    .unwrap();
    let out = File::open(&path).unwrap();
    assert_eq!(column_i16(&out, "delay"), vec![10200]);
    assert_eq!(column_i16(&out, "time_scalar"), vec![-10]);
    assert_eq!(out.cdp_x(), file.cdp_x());
    assert_eq!(out.scalco(), file.scalco());
}

/// A prestack file with varying fold, and a 3-trace file whose dense box is huge.
fn x15_files(dir: &std::path::Path) -> (File, File) {
    let ns = 8;
    let mut traces = Vec::new();
    for il in 1..=2 {
        for xl in 1..=2 {
            let fold = if (il, xl) == (2, 2) { 3 } else { 2 };
            for off in 1..=fold {
                traces.push(grid_trace(il, xl, off * 100, ns));
            }
        }
    }
    let varying = dir.join("varying.sgy");
    write_synth(&varying, ns, &traces);
    let wide = dir.join("wide.sgy");
    write_synth(
        &wide,
        1000,
        &[
            grid_trace(1, 1, 0, 1000),
            grid_trace(2, 2, 0, 1000),
            grid_trace(1000, 1000, 0, 1000),
        ],
    );
    (File::open(&varying).unwrap(), File::open(&wide).unwrap())
}

#[test]
fn test_x15() {
    let dir = scratch("test_x15");
    let (varying, wide) = x15_files(&dir);
    // Open keeps a varying-fold file as a trace list. Nothing dense is built.
    assert!(!varying.geometry().kind.is_grid());

    // to_regular on a crop (inline 2) returns a mask over (crossline, offset).
    let crop = ops::filter(&varying, &[Predicate::is_in("inline", [2.0])]).unwrap();
    let reg = ops::to_regular(&varying, &crop, 0.0, budget()).unwrap();
    assert_eq!(reg.shape, vec![1, 2, 3, 8]);
    assert_eq!(reg.axes.offsets, vec![100, 200, 300]);
    assert_eq!(reg.mask, vec![true, true, false, true, true, true]);
    let values = f32s(&reg.samples);
    assert!(values[2 * 8..3 * 8].iter().all(|&v| v == 0.0));
    assert_eq!(values[5 * 8], tag(2, 2, 300, 0));

    // Three traces at (1, 1), (2, 2) and (1000, 1000): 10^6 cells of 1000 samples.
    // The budget refuses before anything dense is allocated.
    let err = ops::to_regular(&wide, &all(&wide), 0.0, budget()).unwrap_err();
    match err {
        Error::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::OutOfMemory),
        other => panic!("expected OutOfMemory, got {other:?}"),
    }
    let axes = ops::regular_axes(&wide, &all(&wide)).unwrap();
    assert_eq!(axes.cells(), 1_000_000);
}

#[test]
fn test_x17() {
    let dir = scratch("test_x17");
    let ns = 16;
    let mut traces = Vec::new();
    for il in [10, 12, 14] {
        for xl in [3, 4, 5] {
            if (il, xl) != (12, 4) {
                traces.push(grid_trace(il, xl, 0, ns));
            }
        }
    }
    let path = dir.join("gap.sgy");
    write_synth(&path, ns, &traces);
    let file = File::open(&path).unwrap();
    let all = all(&file);

    let reg = ops::to_regular(&file, &all, 0.0, budget()).unwrap();
    assert_eq!(reg.axes.inlines, vec![10, 12, 14]);
    assert_eq!(reg.axes.crosslines, vec![3, 4, 5]);
    assert_eq!(reg.shape, vec![3, 3, ns]);
    let gap = 4;
    for (c, &m) in reg.mask.iter().enumerate() {
        assert_eq!(m, c != gap, "cell {c}");
    }
    let values = f32s(&reg.samples);
    assert!(values[gap * ns..(gap + 1) * ns].iter().all(|&v| v == 0.0));
    assert_eq!(values[0], tag(10, 3, 0, 0));
    assert_eq!(values[8 * ns + 5], tag(14, 5, 0, 5));

    // Streamed: the sink never sees more than one trace at a time when the
    // budget leaves room for two chunks of one trace each.
    let stride = 240 + ns * 4;
    let index = all.len() * 16 + 6 * 4;
    let mut sink = MemSink::default();
    let report = ops::to_regular_stream(
        &file,
        &all,
        0.0,
        Budget::bytes((index + 2 * stride) as u64),
        &mut sink,
    )
    .unwrap();
    assert_eq!(report.cells, 9);
    assert_eq!(report.filled, 1);
    assert_eq!(sink.max_block, stride);
    assert_eq!(sink.headers.len(), 9);

    // Through the test writer: the gap is a dead trace with fitted coordinates.
    let out_path = dir.join("regular.sgy");
    ops::to_regular_stream(&file, &all, 0.0, budget(), &mut FileSink::create(&out_path)).unwrap();
    let out = File::open(&out_path).unwrap();
    assert_eq!(out.layout().n_traces, 9);
    assert_eq!(out.trace_id()[gap], 2);
    assert_eq!((out.inline()[gap], out.crossline()[gap]), (12, 4));
    let expect = grid_trace(12, 4, 0, ns);
    assert_eq!(
        (out.cdp_x()[gap], out.cdp_y()[gap]),
        (expect.cdp_x, expect.cdp_y)
    );
    assert_eq!(out.scalco()[gap], -10);
    assert!(out
        .sample_bytes(gap as u64)
        .unwrap()
        .iter()
        .all(|&b| b == 0));
    assert_eq!(out.sample_bytes(0).unwrap(), file.sample_bytes(0).unwrap());
    assert_eq!(out.geometry().kind, segyrs::GeometryKind::InlineMajor);
}

#[test]
fn test_x29() {
    let file = open("small.sgy");
    let preds = [
        Predicate::is_in("inline", [3.0]),
        Predicate::range("crossline", Some(20.0), Some(22.0)),
    ];
    let keep = ops::filter(&file, &preds).unwrap();
    assert_eq!(keep, vec![10, 11, 12]);
    for &t in &keep {
        assert_eq!(file.inline()[t as usize], 3);
    }

    // The same predicates on a table the caller already holds.
    let cols = file.par_header_columns(&HeaderSpec::standard()).unwrap();
    assert_eq!(ops::filter_columns(&cols, &preds).unwrap(), keep);
    assert!(ops::filter(&file, &[Predicate::is_in("no_such_field", [1.0])]).is_err());
    assert_eq!(ops::filter(&file, &[]).unwrap().len(), 25);
}

#[test]
fn test_x35() {
    let dir = scratch("test_x35");
    let ns = 12;
    let part = |name: &str, lines: [i32; 2]| {
        let traces: Vec<_> = lines
            .iter()
            .flat_map(|&il| (1..=3).map(move |xl| grid_trace(il, xl, 0, ns)))
            .collect();
        let path = dir.join(name);
        write_synth(&path, ns, &traces);
        File::open(&path).unwrap()
    };
    let a = part("a.sgy", [1, 2]);
    let b = part("b.sgy", [3, 4]);
    let path = dir.join("joined.sgy");
    let n = ops::concat(&[&a, &b], budget(), &mut FileSink::create(&path)).unwrap();
    assert_eq!(n, 12);
    let joined = File::open(&path).unwrap();
    assert_eq!(joined.format(), SampleFormat::IeeeFloat32);
    assert_eq!(joined.layout().n_traces, 12);
    assert_eq!(joined.geometry().kind, segyrs::GeometryKind::InlineMajor);
    assert_eq!(joined.sample_bytes(7).unwrap(), b.sample_bytes(1).unwrap());

    // IBM float32 next to int16, the same format in the other byte order,
    // and a different sample count: refused before anything is written.
    let small = open("small.sgy");
    for other in ["f3.sgy", "small-lsb.sgy", "tiny.sgy"] {
        let c = open(other);
        let mut sink = MemSink::default();
        let err = ops::concat(&[&small, &c], budget(), &mut sink).unwrap_err();
        assert!(matches!(err, Error::FormatChange), "{other}: {err:?}");
        assert!(sink.spec.is_none(), "{other}: nothing is written");
    }
    let err = ops::concat(&[&a, &small], budget(), &mut MemSink::default()).unwrap_err();
    assert!(matches!(err, Error::FormatChange));
}
