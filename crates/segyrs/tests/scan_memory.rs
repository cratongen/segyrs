//! A full header scan must not leave the mapped file resident. Alone in its
//! own test binary so no other test maps files while it measures.

#![cfg(target_os = "linux")]

use std::path::PathBuf;

use segyrs::{File, OpenOptions, Scan, ScanOptions};

fn rss_file_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    status
        .lines()
        .find_map(|l| l.strip_prefix("RssFile:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap()
}

/// Unsorted keys, one int8 sample per trace: a 241-byte stride, so every
/// page of the file holds trace headers.
fn unsorted_file(path: &std::path::Path, n: u32) {
    let mut out = vec![b' '; 3200];
    let mut bin = vec![0u8; 400];
    bin[16..18].copy_from_slice(&4000u16.to_be_bytes());
    bin[20..22].copy_from_slice(&1u16.to_be_bytes());
    bin[24..26].copy_from_slice(&8u16.to_be_bytes());
    bin[300] = 1;
    bin[302..304].copy_from_slice(&1u16.to_be_bytes());
    out.extend(bin);
    for t in 0..n {
        let key = (u64::from(t) * 7919 % u64::from(n)) as u32;
        let mut h = [0u8; 240];
        h[114..116].copy_from_slice(&1u16.to_be_bytes());
        h[188..192].copy_from_slice(&(key / 1000).to_be_bytes());
        h[192..196].copy_from_slice(&(key % 1000).to_be_bytes());
        out.extend(h);
        out.push(t as u8);
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn full_scan_releases_the_mapped_pages() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("scan_memory");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("unsorted.sgy");
    unsorted_file(&path, 200_000);
    let file_kib = std::fs::metadata(&path).unwrap().len() / 1024;

    let before = rss_file_kib();
    let scan = ScanOptions {
        scan: Scan::Full,
        ..ScanOptions::default()
    };
    let file = File::open_scan(&path, &OpenOptions::default(), &scan).unwrap();
    let grown = rss_file_kib().saturating_sub(before);
    assert_eq!(file.layout().n_traces, 200_000);
    assert!(
        grown < file_kib / 4,
        "RssFile grew {grown} KiB over a full scan of a {file_kib} KiB file"
    );

    let last = file.decode(199_999, 1).unwrap();
    assert!(matches!(last, segyrs::Samples::I8(v) if v == [199_999u32 as u8 as i8]));
}
