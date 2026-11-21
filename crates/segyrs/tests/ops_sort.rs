//! External resort: order, strategies, and the out-of-core path.

mod common;

use std::collections::HashSet;

use common::*;
use segyrs::ops::{self, Budget, ResortOptions, SortKey, SortStrategy};
use segyrs::{File, GeometryKind};

fn keys(names: &[&str]) -> Vec<SortKey> {
    names.iter().map(|n| SortKey::asc(*n)).collect()
}

#[test]
fn resort_permuted_prestack_fixtures_come_out_sorted() {
    let sorted = open("small-ps.sgy");
    for name in [
        "small-ps-dec-il-xl-off.sgy",
        "small-ps-dec-xl-off-inc-il.sgy",
        "small-ps-dec-il-off-inc-xl.sgy",
        "small-ps-dec-off-inc-il-xl.sgy",
    ] {
        let file = open(name);
        let dir = scratch(&format!("resort_{name}"));
        let path = dir.join("out.sgy");
        let opts = ResortOptions::new(Budget::bytes(1 << 20));
        let by = keys(&["inline", "crossline", "offset"]);
        let report = ops::resort(&file, &by, &opts, &mut FileSink::create(&path)).unwrap();
        assert!(!report.already_sorted, "{name}");
        let out = File::open(&path).unwrap();
        let n = out.layout().n_traces as usize;
        let key = |f: &File, t: usize| (f.inline()[t], f.crossline()[t], f.offset()[t]);
        for t in 1..n {
            assert!(key(&out, t - 1) < key(&out, t), "{name} {t}");
        }
        // Samples travel with their headers.
        for t in 0..n {
            let src = (0..n).find(|&s| key(&file, s) == key(&out, t)).unwrap();
            assert_eq!(
                out.sample_bytes(t as u64).unwrap(),
                file.sample_bytes(src as u64).unwrap()
            );
        }
        // This fixture holds the same samples as the sorted one.
        if name == "small-ps-dec-il-xl-off.sgy" {
            for t in 0..n as u64 {
                assert_eq!(
                    out.sample_bytes(t).unwrap(),
                    sorted.sample_bytes(t).unwrap()
                );
            }
        }
    }
}

#[test]
fn resort_detects_sorted_input_and_reverses_with_descending_keys() {
    let file = open("small.sgy");
    let opts = ResortOptions::new(Budget::bytes(1 << 20));
    let mut sink = MemSink::default();
    let report = ops::resort(&file, &keys(&["inline", "crossline"]), &opts, &mut sink).unwrap();
    assert!(report.already_sorted);
    assert_eq!(report.runs, 0);

    let desc = [SortKey::desc("inline"), SortKey::desc("crossline")];
    let mut sink = MemSink::default();
    ops::resort(&file, &desc, &opts, &mut sink).unwrap();
    let n = file.layout().n_traces;
    for (j, samples) in sink.samples.iter().enumerate() {
        assert_eq!(samples, &file.sample_bytes(n - 1 - j as u64).unwrap());
    }
}

#[test]
fn resort_to_crossline_major_reopens_as_crossline_major() {
    let file = open("small.sgy");
    let dir = scratch("resort_xl");
    let path = dir.join("xl.sgy");
    let opts = ResortOptions::new(Budget::bytes(1 << 20));
    ops::resort(
        &file,
        &keys(&["crossline", "inline"]),
        &opts,
        &mut FileSink::create(&path),
    )
    .unwrap();
    let out = File::open(&path).unwrap();
    assert_eq!(out.geometry().kind, GeometryKind::CrosslineMajor);
    for t in 0..out.layout().n_traces as usize {
        let (il, xl) = (out.inline()[t], out.crossline()[t]);
        let src = ((il - 1) * 5 + (xl - 20)) as u64;
        assert_eq!(
            out.sample_bytes(t as u64).unwrap(),
            file.sample_bytes(src).unwrap()
        );
    }
}

/// A shuffled 64 x 64 grid of 500-sample traces, about 9 MB of trace data.
fn shuffled_grid(dir: &std::path::Path) -> (File, usize) {
    let ns = 500;
    let mut traces: Vec<_> = (1..=64)
        .flat_map(|il| (1..=64).map(move |xl| grid_trace(il, xl, 0, ns)))
        .collect();
    shuffle(&mut traces, 0x5e9a);
    let path = dir.join("shuffled.sgy");
    write_synth(&path, ns, &traces);
    (File::open(&path).unwrap(), ns)
}

#[test]
fn resort_spills_when_the_data_exceeds_the_budget() {
    let dir = scratch("resort_spill");
    let runs_dir = dir.join("runs");
    std::fs::create_dir_all(&runs_dir).unwrap();
    let (file, ns) = shuffled_grid(&dir);
    let n = file.layout().n_traces as usize;
    let data = n as u64 * file.stride();
    let budget = 1 << 20;
    assert!(data > 8 * budget, "the file must dwarf the budget");

    let mut opts = ResortOptions::new(Budget::bytes(budget));
    opts.tmp_dir = Some(runs_dir.clone());
    opts.strategy = SortStrategy::External;
    opts.sorting_code = Some(4);
    let path = dir.join("sorted.sgy");
    let report = ops::resort(
        &file,
        &keys(&["inline", "crossline"]),
        &opts,
        &mut FileSink::create(&path),
    )
    .unwrap();
    assert_eq!(report.strategy, SortStrategy::External);
    assert!(report.runs > 1, "{report:?}");
    assert_eq!(report.spilled_bytes, data);
    assert_eq!(
        std::fs::read_dir(&runs_dir).unwrap().count(),
        0,
        "run file removed"
    );

    // Balance: every input trace comes out exactly once, with its own samples.
    let out = File::open(&path).unwrap();
    assert_eq!(out.layout().n_traces as usize, n);
    assert_eq!(out.geometry().kind, GeometryKind::InlineMajor);
    let mut seen = HashSet::new();
    let decoded = out.decode(0, n).unwrap();
    let values = decoded.as_f32().unwrap();
    for t in 0..n {
        let (il, xl) = (out.inline()[t], out.crossline()[t]);
        assert!(seen.insert((il, xl)));
        for s in [0, ns / 2, ns - 1] {
            assert_eq!(values[t * ns + s], tag(il, xl, 0, s), "trace {t}");
        }
    }
    assert_eq!(seen.len(), 64 * 64);
    let input_sum: f64 = file
        .decode(0, n)
        .unwrap()
        .as_f32()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum();
    let output_sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
    assert!((input_sum - output_sum).abs() <= input_sum.abs() * 1e-12);
}

#[test]
fn resort_strategies_write_the_same_bytes() {
    let dir = scratch("resort_strategies");
    let (file, _) = shuffled_grid(&dir);
    let by = keys(&["crossline", "inline"]);
    let run = |strategy, budget| {
        let mut opts = ResortOptions::new(Budget::bytes(budget));
        opts.strategy = strategy;
        opts.tmp_dir = Some(dir.clone());
        let mut sink = MemSink::default();
        let report = ops::resort(&file, &by, &opts, &mut sink).unwrap();
        (report, sink)
    };
    let (external, a) = run(SortStrategy::External, 300_000);
    let (_, b) = run(SortStrategy::Gather, 300_000);
    let (auto, c) = run(SortStrategy::Auto, 1 << 30);
    assert!(external.runs > 1);
    assert_eq!(auto.strategy, SortStrategy::InMemory);
    assert_eq!(auto.spilled_bytes, 0);
    let mut opts = ResortOptions::new(Budget::bytes(300_000));
    opts.strategy = SortStrategy::InMemory;
    let err = ops::resort(&file, &by, &opts, &mut MemSink::default()).unwrap_err();
    assert!(matches!(err, segyrs::Error::Io(e) if e.kind() == std::io::ErrorKind::OutOfMemory));
    assert!(a.max_block <= 300_000);
    assert_eq!(a.headers, b.headers);
    assert_eq!(a.samples, b.samples);
    assert_eq!(a.headers, c.headers);
    assert_eq!(a.samples, c.samples);
}

#[test]
fn resort_refuses_a_budget_below_the_key_table() {
    let file = open("small.sgy");
    let opts = ResortOptions::new(Budget::bytes(16));
    let err = ops::resort(&file, &keys(&["inline"]), &opts, &mut MemSink::default()).unwrap_err();
    assert!(matches!(err, segyrs::Error::Io(e) if e.kind() == std::io::ErrorKind::OutOfMemory));
}
