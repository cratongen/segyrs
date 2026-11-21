//! Write-path benchmark: interleaved rounds, medians, every arm flushed.
//!
//! `write_bench <dir> [traces] [samples] [rounds]`. The directory must be on
//! the filesystem under test, not tmpfs. Arms per round, in this order:
//! - `ceiling`: one thread writes a prepared random buffer of the same size,
//!   then `fdatasync`. No encoding. The device and filesystem limit here.
//! - `copy_raw`: `segyrs::copy`, IBM to IBM, raw sample bytes.
//! - `copy_ibm_to_ieee`: `segyrs::copy` with `sample_format = IEEE32`.
//! - `create_f32_to_ibm`: `segyrs::create` from an in-memory f32 array.
//! - `edit_header_column`: `Editor::write_header_column` on every trace + `fdatasync`.

use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use segyrs::write::{copy, create, Editor, HeaderValues, NewFile, SampleData, WriteOptions};
use segyrs::{File, HeaderDtype, HeaderField, SampleFormat};

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2000.0
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(
        args.get(1)
            .expect("usage: write_bench <dir> [traces] [samples] [rounds]"),
    );
    let traces: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(300_000);
    let ns: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1500);
    let rounds: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(5);
    std::fs::create_dir_all(dir).unwrap();

    let samples = random_f32(traces as usize * ns as usize, 0x5eed);
    let inline: Vec<i32> = (0..traces as i32).map(|t| t / 500).collect();
    let crossline: Vec<i32> = (0..traces as i32).map(|t| t % 500).collect();
    let mut spec = NewFile::new(traces, ns, 2000);
    spec.samples = Some(SampleData::F32(&samples));
    spec.overlays = vec![
        segyrs::write::Overlay {
            field: HeaderField::new(189, HeaderDtype::I32, "inline"),
            values: HeaderValues::I32(&inline),
        },
        segyrs::write::Overlay {
            field: HeaderField::new(193, HeaderDtype::I32, "crossline"),
            values: HeaderValues::I32(&crossline),
        },
    ];
    let ibm = WriteOptions {
        sample_format: Some(SampleFormat::IbmFloat32),
        ..WriteOptions::default()
    };
    let src_path = dir.join("src.sgy");
    let bytes = create(&src_path, &spec, &ibm).unwrap();
    let src = File::open(&src_path).unwrap();
    let ceiling_buf: Vec<u8> = random_f32(bytes as usize / 4 + 1, 7)
        .iter()
        .flat_map(|x| x.to_be_bytes())
        .take(bytes as usize)
        .collect();
    let out = dir.join("out.sgy");
    let edit_path = dir.join("edit.sgy");
    std::fs::copy(&src_path, &edit_path).unwrap();
    let cdp: Vec<i32> = (0..traces as i32).collect();

    let mut ceiling = Vec::new();
    let mut copy_raw = Vec::new();
    let mut convert = Vec::new();
    let mut from_f32 = Vec::new();
    let mut edit = Vec::new();
    for round in 0..rounds {
        let _ = std::fs::remove_file(&out);
        let t = Instant::now();
        {
            let mut f = std::fs::File::create(&out).unwrap();
            f.write_all(&ceiling_buf).unwrap();
            f.sync_data().unwrap();
        }
        ceiling.push(t.elapsed().as_secs_f64());

        let _ = std::fs::remove_file(&out);
        let t = Instant::now();
        copy(&src, &out, &WriteOptions::default()).unwrap();
        copy_raw.push(t.elapsed().as_secs_f64());

        let _ = std::fs::remove_file(&out);
        let t = Instant::now();
        copy(
            &src,
            &out,
            &WriteOptions {
                sample_format: Some(SampleFormat::IeeeFloat32),
                ..WriteOptions::default()
            },
        )
        .unwrap();
        convert.push(t.elapsed().as_secs_f64());

        let _ = std::fs::remove_file(&out);
        let t = Instant::now();
        create(&out, &spec, &ibm).unwrap();
        from_f32.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        let editor = Editor::open(&edit_path).unwrap();
        let t_open = t.elapsed().as_secs_f64();
        editor
            .write_header_column(
                &HeaderField::new(21, HeaderDtype::I32, "cdp"),
                HeaderValues::I32(&cdp),
            )
            .unwrap();
        let t_write = t.elapsed().as_secs_f64();
        editor.sync().unwrap();
        edit.push(t.elapsed().as_secs_f64());
        eprintln!(
            "edit split: open {t_open:.3} write {:.3} sync {:.3}",
            t_write - t_open,
            t.elapsed().as_secs_f64() - t_write
        );
        eprintln!("round {round} done");
    }
    let _ = std::fs::remove_file(&out);
    let gb = bytes as f64 / 1e9;
    println!("file: {traces} traces x {ns} samples, {bytes} bytes, rounds {rounds}");
    for (name, v) in [
        ("ceiling", &mut ceiling),
        ("copy_raw", &mut copy_raw),
        ("copy_ibm_to_ieee", &mut convert),
        ("create_f32_to_ibm", &mut from_f32),
    ] {
        let raw = v.clone();
        let m = median(v);
        println!(
            "{name:20} median {m:.3} s  {:.2} GB/s  rounds {raw:.3?}",
            gb / m
        );
    }
    let raw = edit.clone();
    println!(
        "{:20} median {:.3} s  ({traces} traces)  rounds {raw:.3?}",
        "edit_header_column",
        median(&mut edit)
    );
    let mut probe = [0u8; 4];
    std::fs::File::open(&edit_path)
        .unwrap()
        .read_exact_at(&mut probe, 3600 + 20)
        .unwrap();
    let _ = std::fs::remove_file(&edit_path);
    let _ = std::fs::remove_file(&src_path);
}
