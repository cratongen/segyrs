use crate::pio::FileExt;
use std::fs::{self, File as FsFile, OpenOptions as FsOpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use crate::error::Result;
use crate::file::{File, ReadAt};
use crate::header::HeaderColumns;
use crate::ops::budget::over_budget;
use crate::ops::filter::spec_for;
use crate::ops::{invalid, stream, trace_offset, Budget, Col, OutputSpec, TraceSink};

/// One sort key: a header field name and its direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortKey {
    pub field: String,
    pub descending: bool,
}

impl SortKey {
    pub fn asc(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            descending: false,
        }
    }

    pub fn desc(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            descending: true,
        }
    }
}

/// How [`resort`] moves the traces once their order is known.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SortStrategy {
    /// `Gather` when the input is already in order. `InMemory` when twice the
    /// trace data fits the budget. Otherwise `Gather` when the file is on a
    /// non-rotational Linux block device, and `External` when it is on a
    /// rotating disk or the medium is unknown (network file systems included).
    #[default]
    Auto,
    /// One parallel sequential read of all trace data, then a reorder in RAM.
    /// Over budget is an error.
    InMemory,
    /// Read each trace from the input in output order, in parallel chunks. No
    /// temporary file, one random read per trace.
    Gather,
    /// Sorted runs on disk, then a k-way merge. Sequential reads of the input,
    /// one temporary file the size of the trace data.
    External,
}

#[derive(Clone, Debug)]
pub struct ResortOptions {
    pub budget: Budget,
    /// Directory for the run file. `None` is the system temporary directory.
    pub tmp_dir: Option<PathBuf>,
    pub strategy: SortStrategy,
    /// New binary-header trace-sorting code (bytes 3229-3230). `None` keeps it.
    pub sorting_code: Option<i16>,
}

impl ResortOptions {
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            tmp_dir: None,
            strategy: SortStrategy::Auto,
            sorting_code: None,
        }
    }
}

/// What [`resort`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SortReport {
    pub already_sorted: bool,
    /// The strategy that ran, with `Auto` resolved.
    pub strategy: SortStrategy,
    /// Sorted runs written to disk. Zero when nothing spilled.
    pub runs: usize,
    /// Trace bytes written to the run file.
    pub spilled_bytes: u64,
}

/// Input trace numbers in output order: by `keys`, ties by input position.
pub fn sort_order<R: ReadAt>(file: &File<R>, keys: &[SortKey]) -> Result<Vec<u64>> {
    if keys.is_empty() {
        return Err(invalid("resort needs at least one key"));
    }
    let n = file.layout().n_traces as usize;
    let k = keys.len();
    let spec = spec_for(file, keys.iter().map(|key| key.field.as_str()))?;
    let cols: HeaderColumns = file.par_header_columns(&spec)?;
    let mut table = vec![0u64; n * k];
    for (j, key) in keys.iter().enumerate() {
        let col = Col::find(&cols, &key.field)?;
        let flip = if key.descending { u64::MAX } else { 0 };
        table
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, row)| row[j] = col.key_at(i) ^ flip);
    }
    let mut order: Vec<u64> = (0..n as u64).collect();
    order.par_sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        table[a * k..a * k + k]
            .cmp(&table[b * k..b * k + k])
            .then(a.cmp(&b))
    });
    Ok(order)
}

/// Write every trace of `file` into `sink`, ordered by `keys` (external sort).
///
/// The keys are scanned from the headers and sorted in RAM, which costs
/// `n_traces * (8 * keys + 8)` bytes and is charged to the budget. The trace
/// bytes then move by `opts.strategy` (see [`SortStrategy`]). With `External`, worker threads each
/// read a contiguous block of input traces, order it, and write it as a sorted
/// run into one temporary file. The merge then fills each output chunk from
/// every run at once: each run holds a contiguous slice of the chunk's ranks,
/// so all k slices are read in parallel and scattered into place. Live trace
/// buffers stay within the budget in both phases. Trace bytes are copied, never
/// decoded, and the temporary file is removed on success and on error.
pub fn resort<R: ReadAt, S: TraceSink + Send>(
    file: &File<R>,
    keys: &[SortKey],
    opts: &ResortOptions,
    sink: &mut S,
) -> Result<SortReport> {
    let n = file.layout().n_traces as usize;
    let index_bytes = n as u128 * (8 * keys.len() as u128 + 8);
    opts.budget.check(index_bytes, "the sort key table")?;
    let data_budget = opts.budget.get() - index_bytes as u64;
    let _ra = file.readahead();
    let order = sort_order(file, keys)?;
    let mut spec = OutputSpec::from_file(file)?;
    spec.n_traces = Some(n as u64);
    if let Some(code) = opts.sorting_code {
        spec.set_sorting_code(code);
    }
    let stride = spec.trace_len();
    let hl = spec.trace_header_len;
    let already_sorted = order.par_iter().enumerate().all(|(i, &t)| i as u64 == t);
    let data_bytes = n as u128 * stride as u128;
    let fits = 2 * data_bytes <= u128::from(data_budget);
    let strategy = match opts.strategy {
        SortStrategy::Auto if already_sorted => SortStrategy::Gather,
        SortStrategy::Auto if fits => SortStrategy::InMemory,
        SortStrategy::Auto if rotational(file.path()) == Some(false) => SortStrategy::Gather,
        SortStrategy::Auto => SortStrategy::External,
        s => s,
    };
    if strategy == SortStrategy::InMemory && !fits {
        return Err(over_budget(
            2 * data_bytes,
            data_budget,
            "an in-memory resort",
        ));
    }
    sink.begin(&spec)?;
    let mut report = SortReport {
        already_sorted,
        strategy,
        ..SortReport::default()
    };
    match strategy {
        SortStrategy::External if !already_sorted => {
            let dir = opts.tmp_dir.clone().unwrap_or_else(std::env::temp_dir);
            let (runs, spilled) = external(file, order, stride, hl, data_budget, &dir, sink)?;
            report.runs = runs;
            report.spilled_bytes = spilled;
        }
        SortStrategy::InMemory => {
            let mut all = vec![0u8; data_bytes as usize];
            let per = n.div_ceil(rayon::current_num_threads()).max(1);
            all.par_chunks_mut(per * stride)
                .enumerate()
                .try_for_each(|(c, buf)| {
                    file.reader()
                        .read_at(trace_offset(file, (c * per) as u64), buf)
                })?;
            let budget = Budget::bytes(data_budget - data_bytes as u64);
            stream(n, stride, hl, budget, sink, |j, out| {
                let t = order[j] as usize;
                out.copy_from_slice(&all[t * stride..(t + 1) * stride]);
                Ok(())
            })?;
        }
        _ => {
            let budget = Budget::bytes(data_budget);
            stream(n, stride, hl, budget, sink, |j, out| {
                file.reader().read_at(trace_offset(file, order[j]), out)
            })?;
        }
    }
    sink.finish()?;
    Ok(report)
}

/// Whether `path` is on a rotating disk, from Linux `/proc/self/mountinfo`
/// and sysfs. `None` when unknown: not Linux, no path, or a source that is not
/// a block device.
fn rotational(path: &Path) -> Option<bool> {
    let path = fs::canonicalize(path).ok()?;
    let info = fs::read_to_string("/proc/self/mountinfo").ok()?;
    let mut best: Option<(usize, &str)> = None;
    for line in info.lines() {
        let (Some(mount), Some((_, tail))) = (line.split(' ').nth(4), line.split_once(" - "))
        else {
            continue;
        };
        let Some(source) = tail.split(' ').nth(1) else {
            continue;
        };
        if path.starts_with(mount) && best.is_none_or(|(len, _)| mount.len() >= len) {
            best = Some((mount.len(), source));
        }
    }
    let device = fs::canonicalize(best?.1).ok()?;
    let block = Path::new("/sys/class/block").join(device.file_name()?);
    [
        block.join("queue/rotational"),
        block.join("../queue/rotational"),
    ]
    .iter()
    .find_map(|q| fs::read_to_string(q).ok())
    .map(|flag| flag.trim() == "1")
}

/// A temporary file removed when dropped.
struct RunFile {
    path: PathBuf,
    file: FsFile,
}

impl RunFile {
    fn create(dir: &Path) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = format!(
            "segyrs-resort-{}-{nanos}-{}.runs",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        let path = dir.join(name);
        let file = FsOpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self { path, file })
    }
}

impl Drop for RunFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn external<R: ReadAt, S: TraceSink + Send>(
    file: &File<R>,
    order: Vec<u64>,
    stride: usize,
    hl: usize,
    budget: u64,
    dir: &Path,
    sink: &mut S,
) -> Result<(usize, u64)> {
    let n = order.len();
    // rank[t] is the output position of input trace t. The order is dropped
    // once the ranks exist, so the index stays within what the budget was charged.
    let mut rank = vec![0u64; n];
    for (j, &t) in order.iter().enumerate() {
        rank[t as usize] = j as u64;
    }
    drop(order);

    // Run generation. Each concurrent task holds its input block, the sorted
    // copy, and 16 bytes of ordering per trace.
    let per_trace = 2 * stride as u64 + 16;
    if budget < per_trace {
        return Err(over_budget(
            per_trace.into(),
            budget,
            "one trace of the sort",
        ));
    }
    let threads = rayon::current_num_threads() as u64;
    let tasks = (budget / per_trace).clamp(1, threads.max(1));
    let block = ((budget / tasks) / per_trace).max(1) as usize;
    let runs = n.div_ceil(block);
    let tmp = RunFile::create(dir)?;
    tmp.file.set_len(n as u64 * stride as u64)?;
    let mut blocks: Vec<(usize, &mut [u64])> = rank.chunks_mut(block).enumerate().collect();
    for wave in blocks.chunks_mut(tasks as usize) {
        wave.par_iter_mut()
            .try_for_each(|(r, ranks)| -> Result<()> {
                let a = *r * block;
                let len = ranks.len();
                let mut input = vec![0u8; len * stride];
                file.reader()
                    .read_at(trace_offset(file, a as u64), &mut input)?;
                let mut local: Vec<(u64, u32)> = ranks
                    .iter()
                    .enumerate()
                    .map(|(i, &rk)| (rk, i as u32))
                    .collect();
                local.sort_unstable();
                let mut sorted = vec![0u8; len * stride];
                for (j, &(rk, i)) in local.iter().enumerate() {
                    let i = i as usize;
                    sorted[j * stride..(j + 1) * stride]
                        .copy_from_slice(&input[i * stride..(i + 1) * stride]);
                    ranks[j] = rk;
                }
                drop(input);
                tmp.file.write_all_at(&sorted, (a * stride) as u64)?;
                Ok(())
            })?;
    }
    drop(blocks);

    // Merge. Three chunk buffers are live: the chunk the sink is writing, the
    // staging buffer the runs are read into, and the chunk being assembled.
    let chunk = ((budget / (3 * stride as u64 + 16)) as usize).clamp(1, n);
    let run_ranks: Vec<&[u64]> = rank.chunks(block).collect();
    let mut cursor = vec![0usize; runs];
    let mut staging = vec![0u8; chunk * stride];
    let mut cur = vec![0u8; chunk * stride];
    let mut next = vec![0u8; chunk * stride];
    let mut pos = vec![0usize; chunk];
    let mut assemble = |r0: usize, out: &mut [u8], cursor: &mut [usize]| -> Result<()> {
        let r1 = (r0 + chunk).min(n);
        let count = r1 - r0;
        // Slice of each run that holds ranks r0..r1.
        let mut segs = Vec::with_capacity(runs);
        let mut at = 0usize;
        for (r, ranks) in run_ranks.iter().enumerate() {
            let lo = cursor[r];
            let hi = lo + ranks[lo..].partition_point(|&x| (x as usize) < r1);
            for (i, &rk) in ranks[lo..hi].iter().enumerate() {
                pos[rk as usize - r0] = at + i;
            }
            if hi > lo {
                segs.push((r * block + lo, hi - lo));
            }
            at += hi - lo;
            cursor[r] = hi;
        }
        let mut parts: Vec<(usize, &mut [u8])> = Vec::with_capacity(segs.len());
        let mut rest = &mut staging[..count * stride];
        for &(first, len) in &segs {
            let (head, tail) = rest.split_at_mut(len * stride);
            parts.push((first, head));
            rest = tail;
        }
        parts
            .into_par_iter()
            .try_for_each(|(first, buf)| tmp.file.read_exact_at(buf, (first * stride) as u64))?;
        let staged = &staging[..count * stride];
        let pos = &pos[..count];
        out[..count * stride]
            .par_chunks_mut(stride)
            .zip(pos.par_iter())
            .for_each(|(dst, &p)| dst.copy_from_slice(&staged[p * stride..(p + 1) * stride]));
        Ok(())
    };
    let mut r0 = 0usize;
    let mut len = chunk.min(n);
    assemble(0, &mut cur, &mut cursor)?;
    loop {
        let next_r0 = r0 + len;
        if next_r0 >= n {
            sink.write_block(&cur[..len * stride], stride, hl)?;
            break;
        }
        let next_len = chunk.min(n - next_r0);
        let (wrote, built) = rayon::join(
            || sink.write_block(&cur[..len * stride], stride, hl),
            || assemble(next_r0, &mut next, &mut cursor),
        );
        wrote?;
        built?;
        std::mem::swap(&mut cur, &mut next);
        r0 = next_r0;
        len = next_len;
    }
    Ok((runs, n as u64 * stride as u64))
}
