/// One survey axis inferred from the trace headers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Inline,
    Crossline,
    Offset,
}

impl Axis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Crossline => "crossline",
            Self::Offset => "offset",
        }
    }
}

/// Sorting is a report. Open does not fail because a file is unsorted or sparse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryKind {
    InlineMajor,
    CrosslineMajor,
    OffsetMajor,
    Unsorted,
    Sparse,
    Unstructured,
}

impl GeometryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InlineMajor => "inline_major",
            Self::CrosslineMajor => "crossline_major",
            Self::OffsetMajor => "offset_major",
            Self::Unsorted => "unsorted",
            Self::Sparse => "sparse",
            Self::Unstructured => "unstructured",
        }
    }

    pub fn is_grid(self) -> bool {
        matches!(
            self,
            Self::InlineMajor | Self::CrosslineMajor | Self::OffsetMajor
        )
    }
}

#[derive(Clone, Debug)]
pub struct Geometry {
    pub kind: GeometryKind,
    pub inline_byte: u16,
    pub crossline_byte: u16,
    pub offset_byte: u16,
    /// Coordinate values along each axis, in on-disk axis order for a sorted grid.
    pub inlines: Vec<i32>,
    pub crosslines: Vec<i32>,
    pub offsets: Vec<i32>,
    /// Slowest axis first. Empty when the file is not a sorted grid.
    pub axes: Vec<Axis>,
    pub duplicate: Option<(i32, i32, i32)>,
    /// Absent `(inline, crossline)` slots when `kind` is [`GeometryKind::Sparse`].
    pub missing_pairs: Vec<(i32, i32)>,
}

pub fn infer(
    il: &[i32],
    xl: &[i32],
    off: &[i32],
    inline_byte: u16,
    crossline_byte: u16,
    offset_byte: u16,
) -> Geometry {
    let base = Geometry {
        kind: GeometryKind::Unstructured,
        inline_byte,
        crossline_byte,
        offset_byte,
        inlines: Vec::new(),
        crosslines: Vec::new(),
        offsets: Vec::new(),
        axes: Vec::new(),
        duplicate: None,
        missing_pairs: Vec::new(),
    };
    let n = il.len();
    if n == 0 || (il.iter().all(|v| *v == 0) && xl.iter().all(|v| *v == 0)) {
        return base;
    }
    if let Some(key) = duplicate_key(il, xl, off) {
        let mut g = base;
        g.kind = GeometryKind::Unsorted;
        g.duplicate = Some(key);
        g.inlines = unique_in_order(il);
        g.crosslines = unique_in_order(xl);
        g.offsets = unique_in_order(off);
        return g;
    }
    let uil = unique_in_order(il);
    let uxl = unique_in_order(xl);
    let uoff = unique_in_order(off);
    let nil = uil.len();
    let nxl = uxl.len();
    let noff = uoff.len();
    let n64 = n as u64;

    if (nil as u64) * (nxl as u64) == n64 {
        if let Some(order) = match_2d(il, xl, nil, nxl) {
            return Geometry {
                kind: kind_of(order.slow),
                inline_byte,
                crossline_byte,
                offset_byte,
                inlines: order.inlines,
                crosslines: order.crosslines,
                offsets: uoff,
                axes: order.axes,
                duplicate: None,
                missing_pairs: Vec::new(),
            };
        }
        let mut g = base;
        g.kind = GeometryKind::Unsorted;
        g.inlines = uil;
        g.crosslines = uxl;
        g.offsets = uoff;
        return g;
    }

    let cells3 = (nil as u64)
        .saturating_mul(nxl as u64)
        .saturating_mul(noff as u64);
    if noff > 1 && cells3 == n64 {
        if let Some(order) = match_3d(il, xl, off) {
            return Geometry {
                kind: kind_of(order.axes[0]),
                inline_byte,
                crossline_byte,
                offset_byte,
                inlines: order.inlines,
                crosslines: order.crosslines,
                offsets: order.offsets,
                axes: order.axes,
                duplicate: None,
                missing_pairs: Vec::new(),
            };
        }
        let mut g = base;
        g.kind = GeometryKind::Unsorted;
        g.inlines = uil;
        g.crosslines = uxl;
        g.offsets = uoff;
        return g;
    }

    let cells2 = (nil as u64).saturating_mul(nxl as u64);
    if cells2 > n64 && cells2 <= n64.saturating_mul(2) {
        let mut g = base;
        g.kind = GeometryKind::Sparse;
        g.missing_pairs = absent_pairs(il, xl, &uil, &uxl);
        g.inlines = uil;
        g.crosslines = uxl;
        g.offsets = uoff;
        return g;
    }
    if noff > 1 && cells3 > n64 && cells3 <= n64.saturating_mul(2) {
        let mut g = base.clone();
        g.kind = GeometryKind::Sparse;
        g.missing_pairs = absent_pairs(il, xl, &uil, &uxl);
        g.inlines = uil;
        g.crosslines = uxl;
        g.offsets = uoff;
        return g;
    }
    base
}

/// `(min, max)` of a line axis whose dense box would be absurd for this trace count.
///
/// The check looks at the min/max rectangle only. It does not allocate that box.
pub fn absurd_box(il: &[i32], xl: &[i32]) -> Option<(i64, i64)> {
    if il.is_empty() || xl.is_empty() {
        return None;
    }
    let min_il = i64::from(*il.iter().min().unwrap());
    let max_il = i64::from(*il.iter().max().unwrap());
    let min_xl = i64::from(*xl.iter().min().unwrap());
    let max_xl = i64::from(*xl.iter().max().unwrap());
    let span_il = (max_il - min_il).saturating_add(1).max(0) as u128;
    let span_xl = (max_xl - min_xl).saturating_add(1).max(0) as u128;
    let cells = span_il.saturating_mul(span_xl);
    let n = il.len() as u128;
    let dwarfs = cells > n.saturating_mul(8) && cells > 1_048_576;
    if !dwarfs {
        return None;
    }
    if span_il >= span_xl {
        Some((min_il, max_il))
    } else {
        Some((min_xl, max_xl))
    }
}

fn absent_pairs(il: &[i32], xl: &[i32], uil: &[i32], uxl: &[i32]) -> Vec<(i32, i32)> {
    use std::collections::HashSet;
    let have: HashSet<(i32, i32)> = il.iter().copied().zip(xl.iter().copied()).collect();
    let mut missing = Vec::new();
    for &i in uil {
        for &x in uxl {
            if !have.contains(&(i, x)) {
                missing.push((i, x));
            }
        }
    }
    missing
}

/// Coordinate scalar from the SEG-Y `scalco` / `scalel` rule.
///
/// A positive scalar multiplies, a negative scalar divides, and zero means 1.
pub fn apply_scalar(raw: i32, scalar: i16) -> f64 {
    let raw = f64::from(raw);
    match scalar {
        0 => raw,
        s if s > 0 => raw * f64::from(s),
        // Divide rather than multiply by a rounded reciprocal: -100 must give
        // the same float64 as raw / 100.
        s => raw / -f64::from(s),
    }
}

fn kind_of(slow: Axis) -> GeometryKind {
    match slow {
        Axis::Inline => GeometryKind::InlineMajor,
        Axis::Crossline => GeometryKind::CrosslineMajor,
        Axis::Offset => GeometryKind::OffsetMajor,
    }
}

fn duplicate_key(il: &[i32], xl: &[i32], off: &[i32]) -> Option<(i32, i32, i32)> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for i in 0..il.len() {
        let key = (il[i], xl[i], off[i]);
        if !seen.insert(key) {
            return Some(key);
        }
    }
    None
}

fn unique_in_order(values: &[i32]) -> Vec<i32> {
    let mut seen = std::collections::HashSet::new();
    values.iter().copied().filter(|v| seen.insert(*v)).collect()
}

struct Order2 {
    slow: Axis,
    axes: Vec<Axis>,
    inlines: Vec<i32>,
    crosslines: Vec<i32>,
}

fn match_2d(il: &[i32], xl: &[i32], nil: usize, nxl: usize) -> Option<Order2> {
    if try_2d(il, xl, nil, nxl) {
        return Some(Order2 {
            slow: Axis::Inline,
            axes: vec![Axis::Inline, Axis::Crossline],
            inlines: take_every(il, nxl),
            crosslines: xl[..nxl].to_vec(),
        });
    }
    if try_2d(xl, il, nxl, nil) {
        return Some(Order2 {
            slow: Axis::Crossline,
            axes: vec![Axis::Crossline, Axis::Inline],
            inlines: il[..nil].to_vec(),
            crosslines: take_every(xl, nil),
        });
    }
    None
}

/// `slow` is constant for `nfast` traces, then steps. `fast` repeats the first run.
fn try_2d(slow: &[i32], fast: &[i32], nslow: usize, nfast: usize) -> bool {
    if nslow == 0 || nfast == 0 || slow.len() != nslow * nfast {
        return false;
    }
    let fast_pat = &fast[..nfast];
    if !all_unique(fast_pat) {
        return false;
    }
    let mut slow_pat = Vec::with_capacity(nslow);
    for s in 0..nslow {
        let base = s * nfast;
        let s0 = slow[base];
        if slow[base..base + nfast].iter().any(|v| *v != s0) {
            return false;
        }
        if &fast[base..base + nfast] != fast_pat {
            return false;
        }
        slow_pat.push(s0);
    }
    all_unique(&slow_pat)
}

fn take_every(values: &[i32], step: usize) -> Vec<i32> {
    values.iter().step_by(step).copied().collect()
}

struct Order3 {
    axes: Vec<Axis>,
    inlines: Vec<i32>,
    crosslines: Vec<i32>,
    offsets: Vec<i32>,
}

fn match_3d(il: &[i32], xl: &[i32], off: &[i32]) -> Option<Order3> {
    const PERMS: [[Axis; 3]; 6] = [
        [Axis::Inline, Axis::Crossline, Axis::Offset],
        [Axis::Inline, Axis::Offset, Axis::Crossline],
        [Axis::Crossline, Axis::Inline, Axis::Offset],
        [Axis::Crossline, Axis::Offset, Axis::Inline],
        [Axis::Offset, Axis::Inline, Axis::Crossline],
        [Axis::Offset, Axis::Crossline, Axis::Inline],
    ];
    let cols = [il, xl, off];
    let n = il.len();
    for perm in PERMS {
        let counts = perm.map(|axis| unique_in_order(col(cols, axis)).len());
        if counts[0] * counts[1] * counts[2] != n {
            continue;
        }
        if let Some(pats) = try_3d(cols, perm, counts) {
            return Some(Order3 {
                axes: perm.to_vec(),
                inlines: pats[0].clone(),
                crosslines: pats[1].clone(),
                offsets: pats[2].clone(),
            });
        }
    }
    None
}

fn col(cols: [&[i32]; 3], axis: Axis) -> &[i32] {
    match axis {
        Axis::Inline => cols[0],
        Axis::Crossline => cols[1],
        Axis::Offset => cols[2],
    }
}

fn try_3d(cols: [&[i32]; 3], perm: [Axis; 3], counts: [usize; 3]) -> Option<[Vec<i32>; 3]> {
    let (ns, nm, nf) = (counts[0], counts[1], counts[2]);
    if ns == 0 || nm == 0 || nf == 0 {
        return None;
    }
    let slow = col(cols, perm[0]);
    let mid = col(cols, perm[1]);
    let fast = col(cols, perm[2]);
    let fast_pat = fast[..nf].to_vec();
    let mut mid_pat = Vec::with_capacity(nm);
    for m in 0..nm {
        mid_pat.push(mid[m * nf]);
    }
    let mut slow_pat = Vec::with_capacity(ns);
    for s in 0..ns {
        slow_pat.push(slow[s * nm * nf]);
    }
    if !all_unique(&slow_pat) || !all_unique(&mid_pat) || !all_unique(&fast_pat) {
        return None;
    }
    for (s, &slow_v) in slow_pat.iter().enumerate() {
        for (m, &mid_v) in mid_pat.iter().enumerate() {
            for (f, &fast_v) in fast_pat.iter().enumerate() {
                let i = s * nm * nf + m * nf + f;
                if slow[i] != slow_v || mid[i] != mid_v || fast[i] != fast_v {
                    return None;
                }
            }
        }
    }
    let mut inline = Vec::new();
    let mut cross = Vec::new();
    let mut offset = Vec::new();
    for (axis, pat) in perm.iter().zip([&slow_pat, &mid_pat, &fast_pat]) {
        match axis {
            Axis::Inline => inline.clone_from(pat),
            Axis::Crossline => cross.clone_from(pat),
            Axis::Offset => offset.clone_from(pat),
        }
    }
    Some([inline, cross, offset])
}

fn all_unique(values: &[i32]) -> bool {
    unique_in_order(values).len() == values.len()
}

/// `(inline, crossline)` to the file trace numbers that carry that pair.
///
/// Works on every layout, including unsorted files, so a label lookup does not
/// assume the trace sits at the dense grid slot. The index is one permutation
/// of the trace numbers, stably sorted by key in parallel, plus the sorted keys:
/// twelve bytes per trace and no allocation per key. Traces for one pair stay in
/// file order, which is how a constant-fold gather keeps its offsets.
#[derive(Clone, Debug)]
pub struct TraceIndex {
    keys: Vec<(i32, i32)>,
    order: Vec<u32>,
}

impl TraceIndex {
    pub fn build(inline: &[i32], crossline: &[i32]) -> Self {
        use rayon::prelude::*;
        let n = inline.len().min(crossline.len());
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.par_sort_by_key(|&i| (inline[i as usize], crossline[i as usize]));
        let keys = order
            .par_iter()
            .map(|&i| (inline[i as usize], crossline[i as usize]))
            .collect();
        Self { keys, order }
    }

    /// File trace numbers that carry `(inline, crossline)`, in file order.
    pub fn get(&self, inline: i32, crossline: i32) -> Option<&[u32]> {
        let key = (inline, crossline);
        let lo = self.keys.partition_point(|k| *k < key);
        let hi = self.keys.partition_point(|k| *k <= key);
        (lo < hi).then(|| &self.order[lo..hi])
    }
}

/// One axis of a [`Regular`] grid: `origin + i * step` for `i` in `0..count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub axis: Axis,
    pub origin: i32,
    pub step: i32,
    pub count: usize,
}

impl Run {
    fn at(&self, i: usize) -> i32 {
        (i64::from(self.origin) + i as i64 * i64::from(self.step)) as i32
    }

    fn values(&self) -> Vec<i32> {
        (0..self.count).map(|i| self.at(i)).collect()
    }

    /// `(min, max)` of the run, in `i64` so a span cannot overflow.
    fn bounds(&self) -> (i64, i64) {
        let a = i64::from(self.origin);
        let b = a + (self.count as i64 - 1) * i64::from(self.step);
        (a.min(b), a.max(b))
    }
}

/// A sorted grid whose keys step by a constant on every axis, so the key of
/// trace `t` is arithmetic on `t`. `runs` is slowest axis first. A post-stack
/// grid has its constant offset as the slowest run, with `count == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Regular {
    pub runs: [Run; 3],
}

/// Key words of one trace in `[inline, crossline, offset]` order.
pub(crate) type Key = [i32; 3];

const AXES: [Axis; 3] = [Axis::Inline, Axis::Crossline, Axis::Offset];

fn slot(axis: Axis) -> usize {
    match axis {
        Axis::Inline => 0,
        Axis::Crossline => 1,
        Axis::Offset => 2,
    }
}

impl Regular {
    /// `[inline, crossline, offset]` the grid puts at trace `t`.
    pub fn predict(&self, t: usize) -> Key {
        let [slow, mid, fast] = self.runs;
        let mut key = [0; 3];
        key[slot(fast.axis)] = fast.at(t % fast.count);
        key[slot(mid.axis)] = mid.at((t / fast.count) % mid.count);
        key[slot(slow.axis)] = slow.at(t / (fast.count * mid.count));
        key
    }

    /// Find the grid from a few dozen keys, or `None` when the first traces do
    /// not describe one. `key(t)` reads trace `t`. The search reads traces 0
    /// and 1, then finds where each run ends by exponential and binary search.
    /// It does not prove that the rest of the file follows: the caller checks.
    pub fn detect<E>(
        n: usize,
        mut key: impl FnMut(usize) -> Result<Key, E>,
    ) -> Result<Option<Self>, E> {
        if n < 4 {
            return Ok(None);
        }
        let k0 = key(0)?;
        let Some((fast, df)) = only_change(&k0, &key(1)?) else {
            return Ok(None);
        };
        let nf = first_break(n, |t| Ok(steps(&k0, &key(t)?, fast, df, t)))?;
        if nf >= n || n % nf != 0 {
            return Ok(None);
        }
        let Some((mid, dm)) = only_change(&k0, &key(nf)?).filter(|m| m.0 != fast) else {
            return Ok(None);
        };
        let groups = n / nf;
        let nm = first_break(groups, |g| Ok(steps(&k0, &key(g * nf)?, mid, dm, g)))?;
        let slow = 3 - fast - mid;
        let run = |s: usize, step: i64, count: usize| -> Option<Run> {
            let r = Run {
                axis: AXES[s],
                origin: k0[s],
                step: i32::try_from(step).ok()?,
                count,
            };
            let (lo, hi) = r.bounds();
            (lo >= i64::from(i32::MIN) && hi <= i64::from(i32::MAX)).then_some(r)
        };
        let (Some(f), Some(m)) = (run(fast, df, nf), run(mid, dm, nm)) else {
            return Ok(None);
        };
        let s = if nm == groups {
            // Two axes: only inline by crossline with one offset is reported
            // that way by `infer`. Anything else goes to the full scan.
            if AXES[slow] != Axis::Offset {
                return Ok(None);
            }
            run(slow, 0, 1)
        } else {
            let block = nf * nm;
            if n % block != 0 {
                return Ok(None);
            }
            match only_change(&k0, &key(block)?) {
                Some((s, ds)) if s == slow => run(slow, ds, n / block),
                _ => None,
            }
        };
        let Some(s) = s else {
            return Ok(None);
        };
        let grid = Self { runs: [s, m, f] };
        // `infer` reports these axes only with two or more lines each way.
        if grid.run(Axis::Inline).count < 2 || grid.run(Axis::Crossline).count < 2 {
            return Ok(None);
        }
        Ok(Some(grid))
    }

    fn run(&self, axis: Axis) -> Run {
        *self
            .runs
            .iter()
            .find(|r| r.axis == axis)
            .expect("a grid has one run per axis")
    }

    /// True when [`absurd_box`] refuses this grid's inline/crossline rectangle.
    pub fn absurd(&self, n: usize) -> bool {
        let span = |a: Axis| {
            let (lo, hi) = self.run(a).bounds();
            (hi - lo + 1) as u128
        };
        let cells = span(Axis::Inline).saturating_mul(span(Axis::Crossline));
        cells > (n as u128).saturating_mul(8) && cells > 1_048_576
    }

    /// The [`Geometry`] that [`infer`] reports for a file with exactly these keys.
    pub fn geometry(&self, inline_byte: u16, crossline_byte: u16, offset_byte: u16) -> Geometry {
        let [slow, mid, fast] = self.runs;
        let axes = if slow.count == 1 {
            vec![mid.axis, fast.axis]
        } else {
            vec![slow.axis, mid.axis, fast.axis]
        };
        Geometry {
            kind: kind_of(axes[0]),
            inline_byte,
            crossline_byte,
            offset_byte,
            inlines: self.run(Axis::Inline).values(),
            crosslines: self.run(Axis::Crossline).values(),
            offsets: self.run(Axis::Offset).values(),
            axes,
            duplicate: None,
            missing_pairs: Vec::new(),
        }
    }

    /// First trace whose key is not the predicted one, checked in parallel.
    pub fn first_mismatch(&self, il: &[i32], xl: &[i32], off: &[i32]) -> Option<usize> {
        use rayon::prelude::*;
        (0..il.len())
            .into_par_iter()
            .with_min_len(4096)
            .find_first(|&t| self.predict(t) != [il[t], xl[t], off[t]])
    }
}

/// The one slot where `b` differs from `a`, with the difference.
fn only_change(a: &Key, b: &Key) -> Option<(usize, i64)> {
    let mut found = None;
    for s in 0..3 {
        if a[s] != b[s] {
            if found.is_some() {
                return None;
            }
            found = Some((s, i64::from(b[s]) - i64::from(a[s])));
        }
    }
    found
}

/// `k` equals `k0` except that slot `s` is `k0[s] + i * step`.
fn steps(k0: &Key, k: &Key, s: usize, step: i64, i: usize) -> bool {
    (0..3).all(|j| {
        let want = i64::from(k0[j]) + if j == s { i as i64 * step } else { 0 };
        i64::from(k[j]) == want
    })
}

/// Smallest `i` in `2..=n` where `holds(i)` is false, `n` when none is (index
/// 1 already holds). Assumes `holds` is true on a prefix and false after it,
/// which is what a regular grid gives; the caller verifies the result.
fn first_break<E>(n: usize, mut holds: impl FnMut(usize) -> Result<bool, E>) -> Result<usize, E> {
    let (mut lo, mut hi) = (1usize, 2usize);
    while hi < n && holds(hi)? {
        lo = hi;
        hi = hi.saturating_mul(2);
    }
    let mut hi = hi.min(n);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if holds(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(hi)
}

#[cfg(test)]
mod regular_tests {
    use super::*;

    type Layout = [(Axis, i32, i32, usize); 3];

    fn keys(runs: Layout) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let grid = Regular {
            runs: runs.map(|(axis, origin, step, count)| Run {
                axis,
                origin,
                step,
                count,
            }),
        };
        let n: usize = runs.iter().map(|r| r.3).product();
        let mut cols = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..n {
            let k = grid.predict(t);
            cols.0.push(k[0]);
            cols.1.push(k[1]);
            cols.2.push(k[2]);
        }
        cols
    }

    fn detect(il: &[i32], xl: &[i32], off: &[i32]) -> Option<Regular> {
        Regular::detect(il.len(), |t| Ok::<_, ()>([il[t], xl[t], off[t]])).unwrap()
    }

    #[test]
    fn a_detected_grid_reports_what_infer_reports() {
        use Axis::*;
        let layouts: [Layout; 6] = [
            [
                (Offset, 0, 0, 1),
                (Inline, 10, 1, 7),
                (Crossline, 100, 2, 9),
            ],
            [
                (Offset, 5, 0, 1),
                (Crossline, 300, -3, 8),
                (Inline, 1, 1, 6),
            ],
            [
                (Inline, 1, 1, 4),
                (Crossline, 20, 1, 5),
                (Offset, 100, 50, 3),
            ],
            [(Inline, 1, 1, 4), (Offset, 0, 25, 3), (Crossline, 7, 1, 5)],
            [(Offset, 0, 10, 2), (Crossline, 1, 1, 3), (Inline, 9, -1, 4)],
            [(Crossline, 2, 2, 3), (Inline, 1, 1, 5), (Offset, 0, 1, 2)],
        ];
        for runs in layouts {
            let (il, xl, off) = keys(runs);
            let grid = detect(&il, &xl, &off).unwrap_or_else(|| panic!("no grid for {runs:?}"));
            assert_eq!(grid.first_mismatch(&il, &xl, &off), None);
            assert_eq!(
                format!("{:?}", grid.geometry(189, 193, 37)),
                format!("{:?}", infer(&il, &xl, &off, 189, 193, 37)),
                "{runs:?}"
            );
        }
    }

    #[test]
    fn a_broken_or_irregular_layout_is_caught() {
        use Axis::*;
        let (mut il, xl, off) =
            keys([(Offset, 0, 0, 1), (Inline, 1, 1, 10), (Crossline, 1, 1, 10)]);
        il[57] = 99;
        // The search may miss trace 57. Verification must not.
        if let Some(grid) = detect(&il, &xl, &off) {
            assert_eq!(grid.first_mismatch(&il, &xl, &off), Some(57));
        }
        let n = 100;
        let xl: Vec<i32> = (0..n).map(|t| 1 << (t % 10)).collect();
        let il: Vec<i32> = (0..n).map(|t| (t / 10) as i32).collect();
        let off = vec![0; n];
        if let Some(grid) = detect(&il, &xl, &off) {
            assert!(grid.first_mismatch(&il, &xl, &off).is_some());
        }
        // One inline only: left to the full scan.
        let (il, xl, off) = keys([(Offset, 0, 0, 1), (Inline, 1, 1, 1), (Crossline, 1, 1, 50)]);
        assert!(detect(&il, &xl, &off).is_none());
    }
}
