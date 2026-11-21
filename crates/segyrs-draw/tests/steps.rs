//! A Source over a core File someone else opened, and stepped (decimated)
//! windows on sections and slices.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use segyrs::{File, OpenOptions, ReadAt, Samples};
use segyrs_draw::{Dim, PlaneSpec, Source};

fn small() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-data/small.sgy")
}

/// In-memory bytes that count every byte read.
#[derive(Clone)]
struct Counted {
    data: Arc<Vec<u8>>,
    bytes: Arc<Mutex<usize>>,
}

impl Counted {
    fn take(&self) -> usize {
        std::mem::take(&mut *self.bytes.lock().unwrap())
    }
}

impl ReadAt for Counted {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> segyrs::Result<()> {
        *self.bytes.lock().unwrap() += buf.len();
        let s = offset as usize;
        buf.copy_from_slice(&self.data[s..s + buf.len()]);
        Ok(())
    }
}

/// Sample `s` of the trace at (inline position `i`, crossline position `x`), from the core.
fn core_value(file: &File, i: usize, x: usize, s: usize) -> f32 {
    let g = file.geometry();
    let t = file
        .trace_index()
        .get(g.inlines[i], g.crosslines[x])
        .unwrap()[0] as usize;
    match file.decode(t, 1).unwrap() {
        Samples::F32(v) => v[s],
        other => panic!("small.sgy is IBM float, got {other:?}"),
    }
}

#[test]
fn from_file_reuses_the_open_scan_and_index() {
    let r = Counted {
        data: Arc::new(std::fs::read(small()).unwrap()),
        bytes: Arc::default(),
    };
    let len = r.data.len() as u64;
    let file = Arc::new(File::from_reader(r.clone(), len, &OpenOptions::default()).unwrap());
    let index = file.trace_index() as *const _;
    r.take();
    let src = Source::from_file(file.clone()).unwrap();
    // Only trace 0's header (the recording delay) is read; no header scan.
    assert!(r.take() <= 240, "from_file must not scan the headers again");
    assert!(std::ptr::eq(src.file().trace_index(), index));
    assert_eq!(src.dims(), [Dim::Inline, Dim::Crossline, Dim::Time]);
}

#[test]
fn stepped_section_reads_every_step_th_position_and_sample() {
    let src = Source::open(small(), &OpenOptions::default()).unwrap();
    let file = File::open(small()).unwrap();
    let mut spec = PlaneSpec::new(vec![(Dim::Inline, 2)], 800, 480);
    spec.x_range = Some(0..5);
    spec.x_step = 2;
    spec.y_range = Some(3..40);
    spec.y_step = 5;
    let s = src.section(&spec).unwrap();
    let xs = [0, 2, 4];
    let ys: Vec<usize> = (3..40).step_by(5).collect();
    assert_eq!((s.width, s.height), (xs.len(), ys.len()));
    assert_eq!(s.traces_read, xs.len());
    let crosslines = src.coords(Dim::Crossline).unwrap();
    assert_eq!(s.x.labels, xs.map(|x| crosslines[x]));
    let times = src.coords(Dim::Time).unwrap();
    assert_eq!(s.y.labels, ys.iter().map(|y| times[*y]).collect::<Vec<_>>());
    for (c, x) in xs.iter().enumerate() {
        for (r, y) in ys.iter().enumerate() {
            assert_eq!(s.value(c, r), core_value(&file, 2, *x, *y), "x {x} y {y}");
        }
    }
}

#[test]
fn stepped_slice_reads_every_step_th_trace() {
    let src = Source::open(small(), &OpenOptions::default()).unwrap();
    let file = File::open(small()).unwrap();
    let mut spec = PlaneSpec::new(vec![(Dim::Time, 7)], 800, 480);
    spec.x_range = Some(1..5);
    spec.x_step = 2;
    spec.y_range = Some(0..5);
    spec.y_step = 3;
    let s = src.section(&spec).unwrap();
    let (xs, ys) = ([1, 3], [0, 3]);
    assert_eq!((s.width, s.height), (2, 2));
    assert_eq!(s.traces_read, 4);
    for (c, i) in xs.iter().enumerate() {
        for (r, x) in ys.iter().enumerate() {
            assert_eq!(s.value(c, r), core_value(&file, *i, *x, 7));
        }
    }
}

#[test]
fn zero_step_is_refused() {
    let src = Source::open(small(), &OpenOptions::default()).unwrap();
    let mut spec = PlaneSpec::new(vec![(Dim::Inline, 0)], 80, 48);
    spec.y_step = 0;
    assert!(src.section(&spec).is_err());
}
