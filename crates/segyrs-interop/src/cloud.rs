//! `ReadAt` over Rust `object_store`: s3, gs, az, http, file and in-memory stores.
//!
//! The core reads headers and traces with many small positional reads from rayon
//! workers. Sent one by one to an object store, each would be a round trip. This
//! reader serves them from a byte-range cache instead:
//!
//! - A miss fetches the aligned block around it (`block_size`). Other threads that
//!   need the same bytes wait for that one request (single flight).
//! - A sequential stream (a miss right after cached bytes, or the first touch of a
//!   read-ahead block) keeps `readahead` blocks in flight in the background.
//! - `prefetch` takes the ranges a selection needs, merges neighbours closer than
//!   `coalesce_gap`, splits long spans at `block_size`, and fetches them in parallel.
//! - A read of `block_size` bytes or more bypasses the cache and is fetched as
//!   parallel block-sized requests straight into the caller's buffer.
//! - Ready bytes are evicted least recently used once the cache passes
//!   `cache_bytes`, which defaults to a quarter of the RAM available at open.
//!
//! All requests run on one shared tokio runtime and at most `concurrency` are in
//! flight per reader. Callers block on a channel, never inside the runtime, so a
//! Python binding can release the GIL around any call. Each request has a deadline
//! (`request_timeout`) and a stalled request is retried up to `retries` times with
//! doubling backoff. HTTP status retries stay with object_store's own `RetryConfig`.

use std::collections::BTreeMap;
use std::future::Future;
use std::io::{self, ErrorKind};
use std::ops::Range;
use std::path::{Component, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt, ObjectStoreScheme, RetryConfig};
use segyrs::{Error, ReadAt, Result};
use tokio::sync::Semaphore;
use url::Url;

/// Tuning for [`ObjectReader`]. Every field can also be set by name through
/// [`ReaderOptions::set`], which is how a `storage_options` dict reaches it.
#[derive(Clone, Debug)]
pub struct ReaderOptions {
    /// Alignment and size of a cache fill, and the split size of long requests.
    pub block_size: u64,
    /// Blocks kept in flight ahead of a sequential stream.
    pub readahead: usize,
    /// `prefetch` merges two ranges whose gap is at most this many bytes.
    pub coalesce_gap: u64,
    /// Requests in flight at once for one reader.
    pub concurrency: usize,
    /// Cache budget in bytes. `None` takes a quarter of the RAM available at open.
    pub cache_bytes: Option<u64>,
    /// Deadline for one request attempt.
    pub request_timeout: Duration,
    /// Extra attempts after a request misses its deadline.
    pub retries: u32,
}

impl Default for ReaderOptions {
    // Defaults are starting points, not measured optima.
    fn default() -> Self {
        Self {
            block_size: 4 << 20,
            readahead: 8,
            coalesce_gap: 1 << 20,
            concurrency: 32,
            cache_bytes: None,
            request_timeout: Duration::from_secs(60),
            retries: 3,
        }
    }
}

impl ReaderOptions {
    /// Keys this reader consumes from a `storage_options` map.
    pub const KEYS: [&'static str; 7] = [
        "block_size",
        "readahead",
        "coalesce_gap",
        "concurrency",
        "cache_bytes",
        "request_timeout",
        "retries",
    ];

    /// Set one option from its string form. Returns `Ok(false)` for a key that is
    /// not a reader option, so the caller can hand it to the store builder.
    pub fn set(&mut self, key: &str, value: &str) -> Result<bool> {
        let bad = |what: &str| invalid(format!("storage option {key}={value:?}: {what}"));
        let int = || {
            value
                .trim()
                .parse::<u64>()
                .map_err(|_| bad("not an integer"))
        };
        match key {
            "block_size" => self.block_size = positive(int()?, &bad)?,
            "readahead" => self.readahead = int()? as usize,
            "coalesce_gap" => self.coalesce_gap = int()?,
            "concurrency" => self.concurrency = positive(int()?, &bad)? as usize,
            "cache_bytes" => self.cache_bytes = Some(int()?),
            "request_timeout" => {
                let secs = value
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|s| s.is_finite() && *s > 0.0)
                    .ok_or_else(|| bad("seconds must be a positive number"))?;
                self.request_timeout = Duration::from_secs_f64(secs);
            }
            "retries" => {
                self.retries = u32::try_from(int()?).map_err(|_| bad("too large"))?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

fn positive(v: u64, bad: &dyn Fn(&str) -> Error) -> Result<u64> {
    if v == 0 {
        Err(bad("must be greater than zero"))
    } else {
        Ok(v)
    }
}

/// Request and cache counters, for tests and for tuning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Range requests sent, retries included.
    pub requests: u64,
    /// Bytes received.
    pub bytes: u64,
    /// `read_at` calls served without a new request.
    pub hits: u64,
    /// `read_at` calls that had to start a request.
    pub misses: u64,
}

/// A remote or local object read through `object_store`, usable as the `R` in
/// `segyrs::File<R>`. Cloning is cheap and shares the cache.
#[derive(Clone)]
pub struct ObjectReader {
    inner: Arc<Inner>,
}

struct Inner {
    store: Arc<dyn ObjectStore>,
    path: StorePath,
    len: u64,
    opts: ReaderOptions,
    cap: u64,
    permits: Arc<Semaphore>,
    cache: Mutex<Cache>,
    ready: Condvar,
    requests: AtomicU64,
    bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ObjectReader {
    /// Open `path` in `store`. One `HEAD` request reads the object size.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        path: StorePath,
        options: ReaderOptions,
    ) -> Result<Self> {
        let head_store = Arc::clone(&store);
        let head_path = path.clone();
        let meta =
            block_on(async move { head_store.head(&head_path).await })?.map_err(store_error)?;
        let cap = options.cache_bytes.unwrap_or_else(default_cache_bytes);
        Ok(Self {
            inner: Arc::new(Inner {
                store,
                path,
                len: meta.size,
                permits: Arc::new(Semaphore::new(options.concurrency.max(1))),
                opts: options,
                cap,
                cache: Mutex::new(Cache::default()),
                ready: Condvar::new(),
                requests: AtomicU64::new(0),
                bytes: AtomicU64::new(0),
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
            }),
        })
    }

    /// Open a URL (`s3://`, `gs://`, `az://`, `abfs://`, `http(s)://`, `file://`)
    /// or a local path. Store credentials come from the environment first, then
    /// from `storage_options`, which also carries the [`ReaderOptions`] keys.
    /// An unknown key is an error, never silently dropped.
    pub fn open<I, K, V>(location: &str, storage_options: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut reader_opts = ReaderOptions::default();
        let mut store_opts = Vec::new();
        for (k, v) in storage_options {
            let key = k.as_ref().to_ascii_lowercase();
            if !reader_opts.set(&key, v.as_ref())? {
                store_opts.push((key, v.as_ref().to_string()));
            }
        }
        let (store, path) = build_store(location, &store_opts, &reader_opts)?;
        Self::new(store, path, reader_opts)
    }

    pub fn len(&self) -> u64 {
        self.inner.len
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Cache budget in bytes.
    pub fn cache_bytes(&self) -> u64 {
        self.inner.cap
    }

    pub fn stats(&self) -> Stats {
        let i = &self.inner;
        Stats {
            requests: i.requests.load(Ordering::Relaxed),
            bytes: i.bytes.load(Ordering::Relaxed),
            hits: i.hits.load(Ordering::Relaxed),
            misses: i.misses.load(Ordering::Relaxed),
        }
    }

    /// Bytes currently cached or in flight.
    pub fn cached_bytes(&self) -> u64 {
        self.inner.lock().bytes
    }

    /// Open a SEG-Y object. The same as `segyrs::File::from_reader` with this
    /// reader and its length.
    pub fn open_segy(&self, options: &segyrs::OpenOptions) -> Result<segyrs::File<ObjectReader>> {
        segyrs::File::from_reader(self.clone(), self.len(), options)
    }
}

impl ReadAt for ObjectReader {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.inner.read_at(offset, buf)
    }

    fn prefetch(&self, ranges: &[Range<u64>]) -> Result<()> {
        self.inner.prefetch(ranges);
        Ok(())
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn read_at(self: &Arc<Self>, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or_else(|| invalid("read range overflows".into()))?;
        if end > self.len {
            return Err(Error::Io(io::Error::new(
                ErrorKind::UnexpectedEof,
                "read past end of object",
            )));
        }
        let block = self.opts.block_size;
        let mut cur = offset;
        let mut missed = false;
        let mut cache = self.lock();
        while cur < end {
            match cache.find(cur) {
                Some((start, ext)) => match &ext.state {
                    State::Ready(data) => {
                        let stop = ext.end.min(end);
                        let src = &data[(cur - start) as usize..(stop - start) as usize];
                        buf[(cur - offset) as usize..(stop - offset) as usize].copy_from_slice(src);
                        let ahead = ext.ahead;
                        let ext_end = ext.end;
                        cache.touch(start);
                        if ahead {
                            cache.clear_ahead(start);
                            self.readahead(&mut cache, ext_end);
                        }
                        cur = stop;
                    }
                    State::Pending => {
                        cache = self.ready.wait(cache).unwrap_or_else(|e| e.into_inner());
                    }
                },
                None => {
                    missed = true;
                    let gap_end = cache.next_start(cur).unwrap_or(u64::MAX).min(end);
                    if gap_end - cur >= block {
                        // Long read: fetch straight into the caller, bypassing the cache.
                        drop(cache);
                        let dst = &mut buf[(cur - offset) as usize..(gap_end - offset) as usize];
                        self.fetch_into(cur, dst)?;
                        cur = gap_end;
                        cache = self.lock();
                        continue;
                    }
                    let a = cur / block * block;
                    let b = (a + block).min(self.len);
                    let sequential = a > 0 && cache.find(a - 1).is_some();
                    let mine = cache.reserve(a..b, block, false);
                    self.make_room(&mut cache);
                    if sequential {
                        self.readahead(&mut cache, b);
                    }
                    drop(cache);
                    let got = self.fetch_reserved(&mine);
                    cache = self.lock();
                    // Copy from what this thread fetched. A small cache may have
                    // evicted it already, and refetching would loop.
                    for (r, data) in got? {
                        if r.start <= cur && cur < r.end {
                            let stop = r.end.min(end);
                            buf[(cur - offset) as usize..(stop - offset) as usize].copy_from_slice(
                                &data[(cur - r.start) as usize..(stop - r.start) as usize],
                            );
                            cur = stop;
                        }
                    }
                }
            }
        }
        drop(cache);
        let counter = if missed { &self.misses } else { &self.hits };
        counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Keep `readahead` blocks in flight after `from`, within half the budget.
    fn readahead(self: &Arc<Self>, cache: &mut Cache, from: u64) {
        let n = self.opts.readahead as u64;
        if n == 0 || from >= self.len {
            return;
        }
        let block = self.opts.block_size;
        let to = from.saturating_add(n * block).min(self.len);
        let room = (self.cap / 2).saturating_sub(cache.pending_bytes);
        let to = to.min(from.saturating_add(room));
        if to <= from {
            return;
        }
        let spans = cache.reserve(from..to, block, true);
        self.make_room(cache);
        self.spawn_fetches(spans);
    }

    fn prefetch(self: &Arc<Self>, ranges: &[Range<u64>]) {
        let block = self.opts.block_size;
        let mut sorted: Vec<Range<u64>> = ranges
            .iter()
            .map(|r| r.start.min(self.len)..r.end.min(self.len))
            .filter(|r| r.start < r.end)
            .collect();
        sorted.sort_by_key(|r| r.start);
        let merged = coalesce(&sorted, self.opts.coalesce_gap);
        let mut cache = self.lock();
        let mut budget = (self.cap / 2).saturating_sub(cache.pending_bytes);
        let mut spans = Vec::new();
        for span in merged {
            if budget == 0 {
                break;
            }
            let end = span.end.min(span.start.saturating_add(budget));
            let got = cache.reserve(span.start..end, block, false);
            budget = budget.saturating_sub(got.iter().map(|r| r.end - r.start).sum());
            spans.extend(got);
        }
        self.make_room(&mut cache);
        drop(cache);
        self.spawn_fetches(spans);
    }

    fn make_room(&self, cache: &mut Cache) {
        cache.evict_to(self.cap);
    }

    /// Fetch spans already reserved as pending by this thread, and wait for them.
    fn fetch_reserved(self: &Arc<Self>, spans: &[Range<u64>]) -> Result<Vec<(Range<u64>, Bytes)>> {
        let this = Arc::clone(self);
        let owned = spans.to_vec();
        let results = block_on(async move {
            let jobs = owned.into_iter().map(|r| {
                let this = Arc::clone(&this);
                async move {
                    let got = this.get(r.clone()).await;
                    (r, got)
                }
            });
            futures::future::join_all(jobs).await
        })?;
        let mut first_err = None;
        let mut fetched = Vec::with_capacity(results.len());
        let mut cache = self.lock();
        for (range, got) in results {
            match got {
                Ok(data) => {
                    cache.complete(range.start, data.clone());
                    fetched.push((range, data));
                }
                Err(e) => {
                    cache.remove(range.start);
                    first_err.get_or_insert(e);
                }
            }
        }
        drop(cache);
        self.ready.notify_all();
        first_err.map_or(Ok(fetched), Err)
    }

    fn spawn_fetches(self: &Arc<Self>, spans: Vec<Range<u64>>) {
        for range in spans {
            let this = Arc::clone(self);
            runtime().spawn(async move {
                let got = this.get(range.clone()).await;
                let mut cache = this.lock();
                match got {
                    Ok(data) => cache.complete(range.start, data),
                    Err(_) => cache.remove(range.start),
                }
                drop(cache);
                this.ready.notify_all();
            });
        }
    }

    /// Long read: block-sized requests in parallel, copied into `dst`.
    fn fetch_into(self: &Arc<Self>, start: u64, dst: &mut [u8]) -> Result<()> {
        let block = self.opts.block_size;
        let end = start + dst.len() as u64;
        let parts: Vec<Range<u64>> = (start..end)
            .step_by(block as usize)
            .map(|a| a..(a + block).min(end))
            .collect();
        let this = Arc::clone(self);
        let got = block_on(async move {
            let jobs = parts.into_iter().map(|r| {
                let this = Arc::clone(&this);
                async move { this.get(r.clone()).await.map(|d| (r, d)) }
            });
            futures::future::try_join_all(jobs).await
        })??;
        for (r, data) in got {
            dst[(r.start - start) as usize..(r.end - start) as usize].copy_from_slice(&data);
        }
        Ok(())
    }

    /// One range request with a deadline, a concurrency permit, and retries on stall.
    async fn get(&self, range: Range<u64>) -> Result<Bytes> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| other("request pool closed"))?;
        let want = range.end - range.start;
        let mut backoff = Duration::from_millis(100);
        let mut attempt = 0;
        loop {
            self.requests.fetch_add(1, Ordering::Relaxed);
            let call = self.store.get_range(&self.path, range.clone());
            match tokio::time::timeout(self.opts.request_timeout, call).await {
                Ok(Ok(data)) => {
                    if data.len() as u64 != want {
                        return Err(Error::Io(io::Error::new(
                            ErrorKind::UnexpectedEof,
                            format!(
                                "range {}..{} returned {} bytes",
                                range.start,
                                range.end,
                                data.len()
                            ),
                        )));
                    }
                    self.bytes.fetch_add(want, Ordering::Relaxed);
                    return Ok(data);
                }
                Ok(Err(e)) => return Err(store_error(e)),
                Err(_) if attempt < self.opts.retries => {
                    attempt += 1;
                    tokio::time::sleep(backoff).await;
                    backoff = backoff.saturating_mul(2);
                }
                Err(_) => {
                    return Err(Error::Io(io::Error::new(
                        ErrorKind::TimedOut,
                        format!(
                            "range {}..{} missed its {:?} deadline {} times",
                            range.start,
                            range.end,
                            self.opts.request_timeout,
                            attempt + 1
                        ),
                    )))
                }
            }
        }
    }
}

/// Merge sorted ranges whose gap is at most `gap` bytes.
pub fn coalesce(sorted: &[Range<u64>], gap: u64) -> Vec<Range<u64>> {
    let mut out: Vec<Range<u64>> = Vec::new();
    for r in sorted {
        match out.last_mut() {
            Some(last) if r.start <= last.end.saturating_add(gap) => {
                last.end = last.end.max(r.end);
            }
            _ => out.push(r.clone()),
        }
    }
    out
}

#[derive(Default)]
struct Cache {
    /// Non-overlapping extents keyed by start offset.
    map: BTreeMap<u64, Extent>,
    /// Ready extents by last use: tick to start.
    lru: BTreeMap<u64, u64>,
    bytes: u64,
    pending_bytes: u64,
    tick: u64,
}

struct Extent {
    end: u64,
    state: State,
    tick: u64,
    /// Fetched by read-ahead and not read yet.
    ahead: bool,
}

enum State {
    Pending,
    Ready(Bytes),
}

impl Cache {
    fn find(&self, pos: u64) -> Option<(u64, &Extent)> {
        self.map
            .range(..=pos)
            .next_back()
            .filter(|(_, e)| e.end > pos)
            .map(|(s, e)| (*s, e))
    }

    fn next_start(&self, pos: u64) -> Option<u64> {
        self.map.range(pos..).next().map(|(s, _)| *s)
    }

    /// Mark the uncovered parts of `want` as pending, split at `piece` bytes.
    fn reserve(&mut self, want: Range<u64>, piece: u64, ahead: bool) -> Vec<Range<u64>> {
        let mut gaps = Vec::new();
        let mut cur = want.start;
        if let Some((_, e)) = self.find(cur) {
            cur = e.end;
        }
        if cur >= want.end {
            return Vec::new();
        }
        for (s, e) in self.map.range(cur..want.end) {
            if *s > cur {
                gaps.push(cur..*s);
            }
            cur = cur.max(e.end);
        }
        if cur < want.end {
            gaps.push(cur..want.end);
        }
        let mut out = Vec::new();
        for g in gaps {
            let mut a = g.start;
            while a < g.end {
                let b = a.saturating_add(piece).min(g.end);
                self.map.insert(
                    a,
                    Extent {
                        end: b,
                        state: State::Pending,
                        tick: 0,
                        ahead,
                    },
                );
                self.bytes += b - a;
                self.pending_bytes += b - a;
                out.push(a..b);
                a = b;
            }
        }
        out
    }

    fn complete(&mut self, start: u64, data: Bytes) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(ext) = self.map.get_mut(&start) {
            if matches!(ext.state, State::Pending) {
                self.pending_bytes -= ext.end - start;
            }
            ext.state = State::Ready(data);
            ext.tick = tick;
            self.lru.insert(tick, start);
        }
    }

    fn remove(&mut self, start: u64) {
        if let Some(ext) = self.map.remove(&start) {
            self.bytes -= ext.end - start;
            match ext.state {
                State::Pending => self.pending_bytes -= ext.end - start,
                State::Ready(_) => {
                    self.lru.remove(&ext.tick);
                }
            }
        }
    }

    fn touch(&mut self, start: u64) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(ext) = self.map.get_mut(&start) {
            self.lru.remove(&ext.tick);
            ext.tick = tick;
            self.lru.insert(tick, start);
        }
    }

    fn clear_ahead(&mut self, start: u64) {
        if let Some(ext) = self.map.get_mut(&start) {
            ext.ahead = false;
        }
    }

    /// Drop least recently used ready extents until `bytes <= cap`. Pending
    /// extents stay: someone is waiting for them.
    fn evict_to(&mut self, cap: u64) {
        while self.bytes > cap {
            let Some((_, start)) = self.lru.pop_first() else {
                break;
            };
            if let Some(ext) = self.map.remove(&start) {
                self.bytes -= ext.end - start;
            }
        }
    }
}

fn default_cache_bytes() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    // A quarter of what is free now leaves the decode buffers and the caller room.
    (sys.available_memory() / 4).max(64 << 20)
}

fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("segyrs-io")
            .enable_all()
            .build()
            .expect("segyrs-io runtime starts")
    })
}

/// Run `fut` on the shared runtime and wait for it from a plain thread. Works
/// from inside another tokio runtime too, because the caller never enters ours.
pub(crate) fn block_on<F>(fut: F) -> Result<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    runtime().spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv()
        .map_err(|_| other("I/O task stopped before it finished"))
}

fn build_store(
    location: &str,
    options: &[(String, String)],
    reader: &ReaderOptions,
) -> Result<(Arc<dyn ObjectStore>, StorePath)> {
    let url = to_url(location)?;
    let (scheme, path) =
        ObjectStoreScheme::parse(&url).map_err(|e| invalid(format!("{location:?}: {e}")))?;
    let retry = RetryConfig {
        retry_timeout: reader.request_timeout,
        ..RetryConfig::default()
    };
    macro_rules! configure {
        ($builder:expr, $key:ty, $name:literal) => {{
            let mut b = $builder.with_url(url.as_str()).with_retry(retry);
            for (k, v) in options {
                let key: $key = k.parse().map_err(|_| {
                    invalid(format!("unknown storage option {k:?} for {} URLs", $name))
                })?;
                b = b.with_config(key, v.clone());
            }
            Arc::new(b.build().map_err(store_error)?) as Arc<dyn ObjectStore>
        }};
    }
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            no_options(options, "file")?;
            Arc::new(object_store::local::LocalFileSystem::new())
        }
        ObjectStoreScheme::Memory => {
            return Err(invalid(
                "memory:// has no shared store; build ObjectReader::new with an InMemory store"
                    .into(),
            ))
        }
        ObjectStoreScheme::AmazonS3 => configure!(
            object_store::aws::AmazonS3Builder::from_env(),
            object_store::aws::AmazonS3ConfigKey,
            "s3"
        ),
        ObjectStoreScheme::GoogleCloudStorage => configure!(
            object_store::gcp::GoogleCloudStorageBuilder::from_env(),
            object_store::gcp::GoogleConfigKey,
            "gs"
        ),
        ObjectStoreScheme::MicrosoftAzure => configure!(
            object_store::azure::MicrosoftAzureBuilder::from_env(),
            object_store::azure::AzureConfigKey,
            "az"
        ),
        ObjectStoreScheme::Http => {
            let base = &url[..url::Position::BeforePath];
            let mut b = object_store::http::HttpBuilder::new()
                .with_url(base)
                .with_retry(retry);
            for (k, v) in options {
                let key: object_store::ClientConfigKey = k
                    .parse()
                    .map_err(|_| invalid(format!("unknown storage option {k:?} for http URLs")))?;
                b = b.with_config(key, v.clone());
            }
            Arc::new(b.build().map_err(store_error)?)
        }
        other => return Err(invalid(format!("unsupported URL scheme {other:?}"))),
    };
    Ok((store, path))
}

/// The store and object path a URL (or local path) names, configured from the
/// environment and then `storage_options`, as [`ObjectReader::open`] does. For
/// writers such as `zarr::ObjectSink`, so reader cache keys are an error here.
pub fn store_for<I, K, V>(
    location: &str,
    storage_options: I,
) -> Result<(Arc<dyn ObjectStore>, StorePath)>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut probe = ReaderOptions::default();
    let mut store_opts = Vec::new();
    for (k, v) in storage_options {
        let key = k.as_ref().to_ascii_lowercase();
        if probe.set(&key, v.as_ref())? {
            return Err(invalid(format!(
                "{key:?} tunes the read cache and has no meaning for a write"
            )));
        }
        store_opts.push((key, v.as_ref().to_string()));
    }
    build_store(location, &store_opts, &ReaderOptions::default())
}

fn no_options(options: &[(String, String)], scheme: &str) -> Result<()> {
    match options.first() {
        Some((k, _)) => Err(invalid(format!(
            "unknown storage option {k:?} for {scheme} URLs"
        ))),
        None => Ok(()),
    }
}

/// A URL, or a local path turned into a `file://` URL. A one-letter scheme is a
/// Windows drive, not a URL.
fn to_url(location: &str) -> Result<Url> {
    match Url::parse(location) {
        Ok(url) if url.scheme().len() > 1 => Ok(url),
        _ => {
            // object_store refuses `.` and `..` segments, so fold them here.
            let mut abs = PathBuf::new();
            for part in std::path::absolute(location)?.components() {
                match part {
                    Component::CurDir => {}
                    Component::ParentDir => {
                        abs.pop();
                    }
                    other => abs.push(other),
                }
            }
            Url::from_file_path(&abs)
                .map_err(|_| invalid(format!("{location:?} is neither a URL nor a local path")))
        }
    }
}

fn store_error(e: object_store::Error) -> Error {
    let kind = match &e {
        object_store::Error::NotFound { .. } => ErrorKind::NotFound,
        object_store::Error::PermissionDenied { .. }
        | object_store::Error::Unauthenticated { .. } => ErrorKind::PermissionDenied,
        object_store::Error::InvalidPath { .. }
        | object_store::Error::UnknownConfigurationKey { .. } => ErrorKind::InvalidInput,
        _ => ErrorKind::Other,
    };
    Error::Io(io::Error::new(kind, e))
}

fn invalid(msg: String) -> Error {
    Error::Io(io::Error::new(ErrorKind::InvalidInput, msg))
}

fn other(msg: &str) -> Error {
    Error::Io(io::Error::other(msg.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_inside_an_existing_extent_is_empty() {
        // A budget-clipped read-ahead window can sit inside one extent. That
        // once panicked in `BTreeMap::range` and left other readers waiting.
        let mut c = Cache::default();
        assert_eq!(c.reserve(100..1100, 1000, true), vec![100..1100]);
        assert!(c.reserve(200..300, 1000, true).is_empty());
        assert_eq!(c.reserve(0..1200, 1000, false), vec![0..100, 1100..1200]);
        assert_eq!(c.pending_bytes, 1200);
    }
}
