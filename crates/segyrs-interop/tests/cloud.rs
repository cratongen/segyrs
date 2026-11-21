//! Remote reads through object_store, with no network: the in-memory and local
//! file system backends stand in for s3, gs, az and http.
#![cfg(feature = "cloud")]

use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path as StorePath;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use segyrs::{File, OpenOptions, ReadAt};
use segyrs_interop::cloud::{coalesce, ObjectReader, ReaderOptions};
use segyrs_interop::{prefetch_traces, trace_ranges};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-data")
}

fn fixtures() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|e| e == "sgy" || e == "su") {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    walk(&data_dir(), &mut out);
    out.sort();
    assert!(out.len() > 50, "test-data has {} files", out.len());
    out
}

/// Small blocks and a small cache, so every fixture crosses block edges,
/// triggers read-ahead and evicts.
fn tight() -> ReaderOptions {
    ReaderOptions {
        block_size: 1024,
        readahead: 3,
        coalesce_gap: 256,
        concurrency: 8,
        cache_bytes: Some(8 * 1024),
        ..ReaderOptions::default()
    }
}

fn in_memory(bytes: &[u8], opts: ReaderOptions) -> (Arc<Counting>, ObjectReader) {
    let store = Arc::new(Counting::new(bytes));
    let reader = ObjectReader::new(store.clone(), StorePath::from("f.sgy"), opts).unwrap();
    (store, reader)
}

/// Everything the mmap path exposes must come back identical through `R`.
fn assert_same<R: ReadAt>(name: &str, local: &File, remote: &File<R>) {
    assert_eq!(
        format!("{:?}", local.layout()),
        format!("{:?}", remote.layout()),
        "{name}"
    );
    assert_eq!(
        format!("{:?}", local.geometry()),
        format!("{:?}", remote.geometry()),
        "{name}"
    );
    assert_eq!(
        local.binary_header_bytes(),
        remote.binary_header_bytes(),
        "{name}"
    );
    let cards = |f: &[segyrs::TextHeader]| f.iter().map(|t| t.cards.clone()).collect::<Vec<_>>();
    assert_eq!(
        cards(local.textual_header()),
        cards(remote.textual_header()),
        "{name}"
    );
    assert_eq!(local.inline(), remote.inline(), "{name}");
    assert_eq!(local.crossline(), remote.crossline(), "{name}");
    assert_eq!(local.offset(), remote.offset(), "{name}");
    assert_eq!(local.cdp_x(), remote.cdp_x(), "{name}");
    assert_eq!(local.cdp_y(), remote.cdp_y(), "{name}");
    assert_eq!(local.scalco(), remote.scalco(), "{name}");
    assert_eq!(local.trace_id(), remote.trace_id(), "{name}");
    let n = local.layout().n_traces;
    for t in 0..n {
        assert_eq!(
            local.header_bytes(t).unwrap(),
            remote.header_bytes(t).unwrap(),
            "{name} {t}"
        );
        assert_eq!(
            local.sample_bytes(t).unwrap(),
            remote.sample_bytes(t).unwrap(),
            "{name} {t}"
        );
    }
    let a = local.par_traces().unwrap();
    let b = remote.par_traces().unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"), "{name}");
}

#[test]
fn test_s37_every_fixture_matches_mmap() {
    let mut opened = 0;
    for path in fixtures() {
        let name = path.strip_prefix(data_dir()).unwrap().display().to_string();
        let local = File::open(&path);
        let bytes = std::fs::read(&path).unwrap();
        let opts = OpenOptions::default();

        let (_, mem) = in_memory(&bytes, tight());
        let mem_file = mem.open_segy(&opts);
        let fs = ObjectReader::open(path.to_str().unwrap(), [("block_size", "2048")]).unwrap();
        let fs_file = fs.open_segy(&opts);

        match local {
            Ok(local) => {
                opened += 1;
                assert_same(&name, &local, &mem_file.unwrap());
                assert_same(&name, &local, &fs_file.unwrap());
            }
            Err(e) => {
                // A file the mmap path refuses is refused the same way.
                let want = std::mem::discriminant(&e);
                let m = mem_file
                    .err()
                    .unwrap_or_else(|| panic!("{name} opened remotely"));
                let f = fs_file
                    .err()
                    .unwrap_or_else(|| panic!("{name} opened via file://"));
                assert_eq!(want, std::mem::discriminant(&m), "{name}: {e} vs {m}");
                assert_eq!(want, std::mem::discriminant(&f), "{name}: {e} vs {f}");
                assert_eq!(e.to_string(), m.to_string(), "{name}");
            }
        }
    }
    assert!(opened > 40, "only {opened} fixtures opened");
}

#[test]
fn test_s37_url_forms_and_options() {
    let path = data_dir().join("small.sgy");
    let local = File::open(&path).unwrap();
    let url = url::Url::from_file_path(path.canonicalize().unwrap()).unwrap();
    for location in [path.to_str().unwrap().to_string(), url.to_string()] {
        let r = ObjectReader::open(&location, std::iter::empty::<(&str, &str)>()).unwrap();
        assert_eq!(r.len(), std::fs::metadata(&path).unwrap().len());
        assert_same(
            &location,
            &local,
            &r.open_segy(&OpenOptions::default()).unwrap(),
        );
    }

    // A typo in a credential key must not be dropped silently.
    let err = ObjectReader::open("s3://bucket/key.sgy", [("aws_secret_acess_key", "x")])
        .err()
        .unwrap();
    assert!(err.to_string().contains("aws_secret_acess_key"), "{err}");
    let err = ObjectReader::open(path.to_str().unwrap(), [("region", "x")])
        .err()
        .unwrap();
    assert!(err.to_string().contains("region"), "{err}");
    let err = ObjectReader::open(path.to_str().unwrap(), [("block_size", "0")])
        .err()
        .unwrap();
    assert!(err.to_string().contains("block_size"), "{err}");

    let mut o = ReaderOptions::default();
    assert!(o.set("request_timeout", "2.5").unwrap());
    assert_eq!(o.request_timeout, Duration::from_millis(2500));
    assert!(!o.set("aws_region", "eu-west-1").unwrap());
    assert!(o.set("request_timeout", "-1").is_err());

    // A missing object is NotFound, not a generic failure.
    let err = ObjectReader::open(
        data_dir().join("no-such.sgy").to_str().unwrap(),
        std::iter::empty::<(&str, &str)>(),
    )
    .err()
    .unwrap();
    match err {
        segyrs::Error::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        other => panic!("{other}"),
    }
}

#[test]
fn test_s37_cache_coalesces_and_counts_requests() {
    let bytes = std::fs::read(data_dir().join("f3.sgy")).unwrap();
    let len = bytes.len() as u64;
    let block = 16 * 1024;
    let opts = ReaderOptions {
        block_size: block,
        readahead: 0,
        coalesce_gap: 0,
        cache_bytes: Some(1 << 30),
        ..ReaderOptions::default()
    };

    // Open scans every trace header from rayon workers. Each block is fetched once.
    let (store, reader) = in_memory(&bytes, opts.clone());
    let file = reader.open_segy(&OpenOptions::default()).unwrap();
    let blocks = len.div_ceil(block);
    let open_requests = store.ranges().len() as u64;
    assert!(
        open_requests <= blocks,
        "{open_requests} requests for {blocks} blocks"
    );
    assert!(store.ranges().iter().all(|r| r.end - r.start <= block));
    let mut seen = store.ranges();
    seen.sort_by_key(|r| r.start);
    seen.dedup();
    assert_eq!(
        seen.len() as u64,
        open_requests,
        "a block was fetched twice"
    );
    assert_eq!(reader.stats().requests, open_requests);

    // Everything is cached now: decoding the whole file sends nothing.
    file.par_traces().unwrap();
    assert_eq!(store.ranges().len() as u64, open_requests);
    assert!(reader.stats().hits > 0);

    // Prefetch: 10 traces a crossline apart are 10 requests with no coalescing,
    // and one request when the gap allows merging. Reads then wait for those
    // requests and send nothing new.
    let ncross = file.geometry().crosslines.len() as u64;
    let picks: Vec<u64> = (0..10).map(|i| i * ncross + 3).collect();
    let ranges = trace_ranges(&file, picks.iter().copied());
    for (gap, want) in [(0, 10), (1 << 20, 1)] {
        let (s, r) = in_memory(
            &bytes,
            ReaderOptions {
                coalesce_gap: gap,
                block_size: 1 << 20,
                ..opts.clone()
            },
        );
        prefetch_traces(&file, &r, picks.iter().copied()).unwrap();
        for range in &ranges {
            let mut buf = vec![0u8; (range.end - range.start) as usize];
            r.read_at(range.start, &mut buf).unwrap();
            assert_eq!(&buf[..], &bytes[range.start as usize..range.end as usize]);
        }
        assert_eq!(s.ranges().len(), want, "coalesce_gap={gap}");
        if want == 1 {
            assert_eq!(s.ranges()[0], ranges[0].start..ranges[9].end);
        }
        assert_eq!(r.stats().misses, 0);
    }
}

#[test]
fn test_s37_single_flight_long_reads_eviction_readahead() {
    let bytes: Vec<u8> = (0..64 * 1024u32).map(|i| (i * 7 % 251) as u8).collect();
    let block = 4096u64;

    // 16 threads miss the same block at once: one request.
    let (store, reader) = in_memory(
        &bytes,
        ReaderOptions {
            block_size: block,
            readahead: 0,
            ..ReaderOptions::default()
        },
    );
    store.delay_ms.store(100, Ordering::Relaxed);
    std::thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(|| {
                let mut buf = [0u8; 100];
                reader.read_at(5000, &mut buf).unwrap();
                assert_eq!(&buf[..], &bytes[5000..5100]);
            });
        }
    });
    assert_eq!(store.ranges(), vec![4096..8192]);
    store.delay_ms.store(0, Ordering::Relaxed);

    // A read of 3 blocks goes straight to the caller as 3 parallel requests.
    let mut big = vec![0u8; 3 * block as usize];
    reader.read_at(20_000, &mut big).unwrap();
    assert_eq!(&big[..], &bytes[20_000..20_000 + 3 * block as usize]);
    assert_eq!(store.ranges().len(), 4);
    assert_eq!(reader.cached_bytes(), block, "a long read was cached");

    // The cache stays within its budget.
    let (_, small) = in_memory(
        &bytes,
        ReaderOptions {
            block_size: block,
            readahead: 0,
            cache_bytes: Some(2 * block),
            ..ReaderOptions::default()
        },
    );
    for b in 0..10u64 {
        let mut buf = [0u8; 10];
        small.read_at(b * block + 1, &mut buf).unwrap();
        assert_eq!(
            &buf[..],
            &bytes[(b * block + 1) as usize..(b * block + 11) as usize]
        );
        assert!(small.cached_bytes() <= 2 * block);
    }

    // A sequential stream keeps read-ahead in flight: 16 blocks, few misses.
    let (store, seq) = in_memory(
        &bytes,
        ReaderOptions {
            block_size: block,
            readahead: 4,
            ..ReaderOptions::default()
        },
    );
    let mut out = vec![0u8; bytes.len()];
    for (i, chunk) in out.chunks_mut(240).enumerate() {
        seq.read_at(i as u64 * 240, chunk).unwrap();
    }
    assert_eq!(out, bytes);
    let mut r = store.ranges();
    r.sort_by_key(|r| r.start);
    r.dedup();
    assert_eq!(r.len(), 16, "each block once");
    assert_eq!(store.ranges().len(), 16);
    assert!(seq.stats().misses <= 3, "{:?}", seq.stats());
}

#[test]
fn test_s37_timeouts_retry_then_fail() {
    let bytes = vec![1u8; 10_000];
    let opts = |retries| ReaderOptions {
        block_size: 4096,
        readahead: 0,
        request_timeout: Duration::from_millis(100),
        retries,
        ..ReaderOptions::default()
    };
    // The first range request stalls past the deadline. The retry succeeds.
    let (store, reader) = in_memory(&bytes, opts(2));
    store.stall_first.store(1, Ordering::Relaxed);
    let mut buf = [0u8; 10];
    reader.read_at(0, &mut buf).unwrap();
    assert_eq!(buf, [1u8; 10]);
    assert_eq!(reader.stats().requests, 2);

    // With no retries the stall is a TimedOut error, and a later read recovers.
    let (store, reader) = in_memory(&bytes, opts(0));
    store.stall_first.store(1, Ordering::Relaxed);
    match reader.read_at(0, &mut buf) {
        Err(segyrs::Error::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::TimedOut),
        other => panic!("{other:?}"),
    }
    reader.read_at(0, &mut buf).unwrap();

    // Past the end is the same error the mmap path gives.
    match reader.read_at(9_995, &mut buf) {
        Err(segyrs::Error::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof),
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_s11_workers_reopen_by_path() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<File<ObjectReader>>();
    send_sync::<ObjectReader>();

    let path = data_dir().join("small.sgy");
    let location = path.to_str().unwrap().to_string();
    let reference = File::open(&path).unwrap().decode(0, 1).unwrap();
    let firsts: Vec<String> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let location = location.clone();
                s.spawn(move || {
                    let r =
                        ObjectReader::open(&location, std::iter::empty::<(&str, &str)>()).unwrap();
                    let f = r.open_segy(&OpenOptions::default()).unwrap();
                    format!("{:?}", f.decode(0, 1).unwrap())
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    assert_eq!(firsts[0], format!("{reference:?}"));
    assert_eq!(firsts[0], firsts[1]);

    // Opening never starts a child process.
    #[cfg(target_os = "linux")]
    {
        let mut children = String::new();
        for task in std::fs::read_dir("/proc/self/task").unwrap() {
            let p = task.unwrap().path().join("children");
            children.push_str(&std::fs::read_to_string(p).unwrap_or_default());
        }
        assert!(children.trim().is_empty(), "child processes: {children}");
    }
}

#[test]
fn coalesce_merges_within_gap() {
    let r = coalesce(&[0..10, 12..20, 100..110, 105..120], 2);
    assert_eq!(r, vec![0..20, 100..120]);
    assert_eq!(coalesce(&[0..10, 13..20], 2), vec![0..10, 13..20]);
}

/// In-memory store that records every range request, and can delay or stall.
#[derive(Debug)]
struct Counting {
    inner: InMemory,
    ranges: Mutex<Vec<Range<u64>>>,
    delay_ms: AtomicU64,
    stall_first: AtomicU64,
}

impl Counting {
    fn new(bytes: &[u8]) -> Self {
        let inner = InMemory::new();
        let payload = PutPayload::from(bytes.to_vec());
        futures::executor::block_on(inner.put(&StorePath::from("f.sgy"), payload)).unwrap();
        Self {
            inner,
            ranges: Mutex::new(Vec::new()),
            delay_ms: AtomicU64::new(0),
            stall_first: AtomicU64::new(0),
        }
    }

    fn ranges(&self) -> Vec<Range<u64>> {
        self.ranges.lock().unwrap().clone()
    }
}

impl fmt::Display for Counting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Counting({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for Counting {
    async fn put_opts(
        &self,
        location: &StorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &StorePath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &StorePath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if let Some(GetRange::Bounded(r)) = &options.range {
            self.ranges.lock().unwrap().push(r.clone());
            if self.stall_first.swap(0, Ordering::Relaxed) > 0 {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            let ms = self.delay_ms.load(Ordering::Relaxed);
            if ms > 0 {
                tokio::time::sleep(Duration::from_millis(ms)).await;
            }
        }
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<StorePath>>,
    ) -> BoxStream<'static, object_store::Result<StorePath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&StorePath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&StorePath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &StorePath,
        to: &StorePath,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
