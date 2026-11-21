use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use segyrs::ibm_to_f32;

fn ibm(c: &mut Criterion) {
    let words: Vec<u32> = (0..4096u32)
        .map(|i| 0x4000_0000u32.wrapping_add(i.wrapping_mul(64)))
        .collect();
    c.bench_function("ibm_to_f32", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for w in &words {
                acc += ibm_to_f32(black_box(*w));
            }
            black_box(acc)
        });
    });
}

criterion_group!(benches, ibm);
criterion_main!(benches);
