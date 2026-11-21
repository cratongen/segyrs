//! In-place header-column benchmark: strategies for writing one 4-byte field
//! on every trace, then `fdatasync`. Interleaved rounds, medians.
//!
//! `edit_bench <dir> [traces] [samples] [rounds]`. Each arm runs with the file
//! in the page cache (warm) and after `POSIX_FADV_DONTNEED` evicted it (cold).
//! - `editor`: the shipped `Editor` (runs rewritten when the stride is short).
//! - `pwrite_par`: the same `pwrite`s spread over rayon threads.
//! - `mmap_par`: `MAP_SHARED` map, rayon stores, `msync`.
//! - `rmw_chunks`: read a run of traces, patch in memory, write the run back.

use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use segyrs::write::{create, Editor, HeaderValues, NewFile, SampleData, WriteOptions};
use segyrs::{HeaderDtype, HeaderField, SampleFormat};

fn evict(path: &Path) {
    let f = std::fs::File::open(path).unwrap();
    f.sync_all().unwrap();
    // SAFETY: valid descriptor for the duration of the call; advice only.
    unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let traces: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(300_000);
    let ns: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1500);
    let rounds: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(5);
    std::fs::create_dir_all(dir).unwrap();
    let mut s = 0x1234_5678u64;
    let samples: Vec<f32> = (0..traces as usize * ns as usize)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32
        })
        .collect();
    let mut spec = NewFile::new(traces, ns, 2000);
    spec.samples = Some(SampleData::F32(&samples));
    let path = dir.join("edit.sgy");
    create(
        &path,
        &spec,
        &WriteOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            ..WriteOptions::default()
        },
    )
    .unwrap();
    drop(samples);
    let stride = 240 + u64::from(ns) * 4;
    let at = |t: u64| 3600 + t * stride + 20;
    let field = HeaderField::new(21, HeaderDtype::I32, "cdp");

    let arms = ["editor", "pwrite_par", "mmap_par", "rmw_chunks"];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len() * 2];
    for round in 0..rounds {
        for (a, arm) in arms.iter().enumerate() {
            for cold in [false, true] {
                let value = (round * 10 + a) as i32 * if cold { -1 } else { 1 };
                let vals: Vec<i32> = (0..traces as i32).map(|t| t ^ value).collect();
                if cold {
                    evict(&path);
                } else {
                    // warm the whole file
                    let _ = std::fs::read(&path).unwrap().len();
                }
                let t0 = Instant::now();
                match *arm {
                    "editor" => {
                        let e = Editor::open(&path).unwrap();
                        e.write_header_column(&field, HeaderValues::I32(&vals))
                            .unwrap();
                        e.sync().unwrap();
                    }
                    "pwrite_par" => {
                        let f = OpenOptions::new().write(true).open(&path).unwrap();
                        (0..traces).into_par_iter().for_each(|t| {
                            f.write_all_at(&vals[t as usize].to_be_bytes(), at(t))
                                .unwrap();
                        });
                        f.sync_data().unwrap();
                    }
                    "mmap_par" => {
                        let f = OpenOptions::new()
                            .read(true)
                            .write(true)
                            .open(&path)
                            .unwrap();
                        // SAFETY: this process is the only writer during the arm.
                        let mut map = unsafe { memmap2::MmapMut::map_mut(&f).unwrap() };
                        let base = map.as_mut_ptr() as usize;
                        (0..traces).into_par_iter().for_each(|t| {
                            let p = (base + at(t) as usize) as *mut [u8; 4];
                            // SAFETY: each trace writes 4 distinct in-bounds bytes.
                            unsafe { p.write_unaligned(vals[t as usize].to_be_bytes()) };
                        });
                        map.flush().unwrap();
                        f.sync_data().unwrap();
                    }
                    "rmw_chunks" => {
                        let f = OpenOptions::new()
                            .read(true)
                            .write(true)
                            .open(&path)
                            .unwrap();
                        let per = ((64u64 << 20) / stride).max(1);
                        let mut buf = Vec::new();
                        let mut t = 0;
                        while t < traces {
                            let n = per.min(traces - t);
                            buf.resize((n * stride) as usize, 0);
                            let off = 3600 + t * stride;
                            f.read_exact_at(&mut buf, off).unwrap();
                            for i in 0..n {
                                let o = (i * stride + 20) as usize;
                                buf[o..o + 4]
                                    .copy_from_slice(&vals[(t + i) as usize].to_be_bytes());
                            }
                            f.write_all_at(&buf, off).unwrap();
                            t += n;
                        }
                        f.sync_data().unwrap();
                    }
                    _ => unreachable!(),
                }
                times[a * 2 + cold as usize].push(t0.elapsed().as_secs_f64());
                // check
                let f = std::fs::File::open(&path).unwrap();
                let mut w = [0u8; 4];
                f.read_exact_at(&mut w, at(traces - 1)).unwrap();
                assert_eq!(i32::from_be_bytes(w), vals[traces as usize - 1]);
            }
        }
        eprintln!("round {round}");
    }
    println!("{traces} traces x {ns} samples (stride {stride}), rounds {rounds}");
    for (a, arm) in arms.iter().enumerate() {
        for cold in [false, true] {
            let v = &mut times[a * 2 + cold as usize];
            let raw = v.clone();
            println!(
                "{arm:12} {:5} median {:.3} s  rounds {raw:.3?}",
                if cold { "cold" } else { "warm" },
                median(v)
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}
