//! Streaming sample kernels: stats, rms, clip, and chunked to_regular.

mod common;

use common::*;
use segyrs::ops::{self, Budget};
use segyrs::{File, Samples};

fn as_f64(s: &Samples) -> Vec<f64> {
    match s {
        Samples::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        Samples::I16(v) => v.iter().map(|&x| f64::from(x)).collect(),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn stats_match_a_two_pass_reference_and_ignore_threads() {
    for name in ["small.sgy", "f3.sgy"] {
        let file = open(name);
        let n = file.layout().n_traces as usize;
        let all_values = as_f64(&file.decode(0, n).unwrap());
        let count = all_values.len() as f64;
        let mean = all_values.iter().sum::<f64>() / count;
        let var = all_values
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / count;
        let min = all_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let st = ops::stats(&file, &all(&file)).unwrap();
        assert_eq!(st.count as usize, all_values.len());
        assert_eq!((st.min, st.max), (min, max));
        assert!(
            (st.mean - mean).abs() <= 1e-9 * mean.abs().max(1.0),
            "{name}"
        );
        assert!((st.variance() - var).abs() <= 1e-9 * var.max(1.0), "{name}");

        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let single = one.install(|| ops::stats(&file, &all(&file)).unwrap());
        assert_eq!(single.mean.to_bits(), st.mean.to_bits(), "{name}");
        assert_eq!(single.m2.to_bits(), st.m2.to_bits(), "{name}");
    }
}

#[test]
fn rms_is_per_trace() {
    let file = open("small.sgy");
    let ns = file.layout().n_samples as usize;
    let got = ops::rms(&file, &[0, 7, 24]).unwrap();
    for (k, &t) in [0usize, 7, 24].iter().enumerate() {
        let v = as_f64(&file.decode(t, 1).unwrap());
        let want = (v.iter().map(|x| x * x).sum::<f64>() / ns as f64).sqrt();
        assert!((got[k] - want).abs() <= 1e-12 * want, "trace {t}");
    }
}

#[test]
fn clip_keeps_inside_samples_bit_exact() {
    for (name, lo, hi) in [("small.sgy", 1.2, 3.25), ("f3.sgy", -1000.5, 1200.7)] {
        let file = open(name);
        let n = file.layout().n_traces as usize;
        let before = as_f64(&file.decode(0, n).unwrap());
        let dir = scratch(&format!("clip_{name}"));
        let path = dir.join("clip.sgy");
        let changed = ops::clip(
            &file,
            &all(&file),
            lo,
            hi,
            Budget::bytes(4096),
            &mut FileSink::create(&path),
        )
        .unwrap();
        let want = before.iter().filter(|&&x| x < lo || x > hi).count() as u64;
        assert!(want > 0, "{name}: the bounds must cut something");
        assert_eq!(changed, want, "{name}");

        let out = File::open(&path).unwrap();
        assert_eq!(out.format(), file.format(), "{name}: format kept");
        let after = as_f64(&out.decode(0, n).unwrap());
        let width = file.format().width();
        for t in 0..n {
            let (a, b) = (
                file.sample_bytes(t as u64).unwrap(),
                out.sample_bytes(t as u64).unwrap(),
            );
            let ns = a.len() / width;
            for s in 0..ns {
                let i = t * ns + s;
                assert!(
                    after[i] >= lo && after[i] <= hi,
                    "{name} {t} {s}: {}",
                    after[i]
                );
                if before[i] >= lo && before[i] <= hi {
                    assert_eq!(a[s * width..(s + 1) * width], b[s * width..(s + 1) * width]);
                }
            }
        }

        let mut decoded = file.decode(0, n).unwrap();
        assert_eq!(ops::clip_samples(&mut decoded, lo, hi).unwrap(), want);
    }
}

#[test]
fn to_regular_in_small_chunks_matches_one_chunk() {
    let file = open("small.sgy");
    let traces = all(&file);
    let big = ops::to_regular(&file, &traces, 0.0, Budget::bytes(1 << 30)).unwrap();
    // Dense result plus mask, plus room for about one cell of scratch.
    let ns = file.layout().n_samples as u64;
    let tight = 25 * (ns * 4 + 1) + 25 * 16 + 40 + ns * 8;
    let small = ops::to_regular(&file, &traces, 0.0, Budget::bytes(tight)).unwrap();
    assert_eq!(big.shape, vec![5, 5, ns as usize]);
    assert!(big.mask.iter().all(|&m| m));
    assert_eq!(big.samples.as_f32(), small.samples.as_f32());
    assert_eq!(
        big.samples.as_f32().unwrap(),
        file.decode(0, 25).unwrap().as_f32().unwrap()
    );
}
