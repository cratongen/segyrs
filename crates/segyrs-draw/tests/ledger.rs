//! Ledger tests for the M4 picture path. Fixtures under `test-data/` are read
//! and never written; mutated copies live in memory only.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use segyrs::{File, GainConvention, OpenOptions, ReadAt, Samples};
use segyrs_draw::{
    lut_index, map_image, notebook, picture, render, Colormap, Coord, Dim, Error, PlaneSpec,
    Polarity, Source, Style, Wiggle, MARGINS,
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name)
}

fn bytes_of(name: &str) -> Vec<u8> {
    std::fs::read(data(name)).unwrap()
}

/// In-memory bytes that log every read.
#[derive(Clone)]
struct Counted {
    data: Arc<Vec<u8>>,
    hits: Arc<Mutex<Vec<(u64, usize)>>>,
}

impl Counted {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data: Arc::new(data),
            hits: Arc::default(),
        }
    }

    fn take(&self) -> Vec<(u64, usize)> {
        std::mem::take(&mut *self.hits.lock().unwrap())
    }
}

impl ReadAt for Counted {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> segyrs::Result<()> {
        self.hits.lock().unwrap().push((offset, buf.len()));
        let s = offset as usize;
        let e = s + buf.len();
        if e > self.data.len() {
            return Err(segyrs::Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short",
            )));
        }
        buf.copy_from_slice(&self.data[s..e]);
        Ok(())
    }
}

/// Source over a counting reader, with the open's header reads discarded.
fn counted(data: Vec<u8>) -> (Source<Counted>, Counted) {
    let r = Counted::new(data);
    let len = r.data.len() as u64;
    let src = Source::from_reader(r.clone(), len, &OpenOptions::default()).unwrap();
    r.take();
    (src, r)
}

fn to_f32(s: Samples) -> Vec<f32> {
    match s {
        Samples::I8(v) => v.into_iter().map(f32::from).collect(),
        Samples::U8(v) => v.into_iter().map(f32::from).collect(),
        Samples::I16(v) => v.into_iter().map(f32::from).collect(),
        Samples::U16(v) => v.into_iter().map(f32::from).collect(),
        Samples::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        Samples::U32(v) => v.into_iter().map(|x| x as f32).collect(),
        Samples::I64(v) => v.into_iter().map(|x| x as f32).collect(),
        Samples::U64(v) => v.into_iter().map(|x| x as f32).collect(),
        Samples::F32(v) => v,
        Samples::F64(v) => v.into_iter().map(|x| x as f32).collect(),
    }
}

/// Every trace of the file decoded by the core, as f32.
fn core_traces(file: &File) -> Vec<Vec<f32>> {
    let ns = file.layout().n_samples as usize;
    let all = to_f32(file.decode(0, file.layout().n_traces as usize).unwrap());
    all.chunks(ns.max(1)).map(|c| c.to_vec()).collect()
}

/// Sample byte range of trace `t` in a file with one 240-byte header per trace.
fn sample_bytes(file: &File, t: usize) -> Range<u64> {
    let l = file.layout();
    let w = l.sample_format.width() as u64;
    let start = l.data_start
        + t as u64 * file.stride()
        + 240 * (1 + u64::from(l.extra_trace_header_blocks));
    start..start + u64::from(l.n_samples) * w
}

fn png_size(png: &[u8]) -> (u32, u32) {
    let dec = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = dec.read_info().expect("png header decodes");
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("png frame decodes");
    (info.width, info.height)
}

fn inline_traces(file: &File, il_pos: usize) -> Vec<usize> {
    let il = file.geometry().inlines[il_pos];
    (0..file.layout().n_traces as usize)
        .filter(|t| file.inline()[*t] == il)
        .collect()
}

/// Symmetric p98 reference over `values`, finite only, by sorting.
fn reference_p98(values: impl IntoIterator<Item = f32>) -> f32 {
    let mut m: Vec<f32> = values
        .into_iter()
        .filter(|v| v.is_finite())
        .map(f32::abs)
        .collect();
    m.sort_by(f32::total_cmp);
    let k = ((0.98 * m.len() as f64).ceil() as usize).max(1) - 1;
    m[k]
}

fn ibm_word(v: f32) -> [u8; 4] {
    // Encode a finite f32 as IBM float (big-endian), truncating the fraction.
    if v == 0.0 {
        return [0; 4];
    }
    let sign = if v < 0.0 { 0x80u32 << 24 } else { 0 };
    let mut m = f64::from(v.abs());
    let mut exp = 64i32;
    while m >= 1.0 {
        m /= 16.0;
        exp += 1;
    }
    while m < 0.0625 {
        m *= 16.0;
        exp -= 1;
    }
    let frac = (m * f64::from(1u32 << 24)) as u32;
    (sign | ((exp as u32) << 24) | frac).to_be_bytes()
}

/// The proof from viewer.md: one plane from a fixture, read through a
/// counter. The PNG decodes, the clip is symmetric, and no sample outside the
/// plane was read.
#[test]
fn test_s06_one_plane_png_reads_only_that_plane() {
    let raw = bytes_of("small.sgy");
    let (src, reader) = counted(raw);
    let core = File::open(data("small.sgy")).unwrap();
    assert_eq!(src.dims(), &[Dim::Inline, Dim::Crossline, Dim::Time]);
    let spec = PlaneSpec::new(vec![(Dim::Inline, 2)], 400, 300);
    let pic = picture(&src, &spec, &Style::default()).unwrap();
    assert_eq!(png_size(&pic.png), (400 + 64 + 16, 300 + 28 + 40));

    let plane = inline_traces(&core, 2);
    let allowed: Vec<Range<u64>> = plane.iter().map(|t| sample_bytes(&core, *t)).collect();
    let hits = reader.take();
    assert!(!hits.is_empty());
    for (off, len) in &hits {
        let end = off + *len as u64;
        assert!(
            allowed.iter().any(|r| r.start <= *off && end <= r.end),
            "read {off}+{len} is outside inline 3's samples"
        );
    }
    let total: usize = hits.iter().map(|h| h.1).sum();
    let per_trace = (core.layout().n_samples as usize) * core.format().width();
    assert_eq!(
        total,
        plane.len() * per_trace,
        "each sample of the plane is read once"
    );

    // Symmetric clip: one magnitude, -clip and +clip land on the LUT ends.
    let traces = core_traces(&core);
    let want = reference_p98(plane.iter().flat_map(|t| traces[*t].clone()));
    let clip = pic.section.p98.unwrap();
    assert_eq!(clip, want);
    assert_eq!(lut_index(-clip, clip, Polarity::Normal), Some(0));
    assert_eq!(lut_index(clip, clip, Polarity::Normal), Some(255));
    assert_eq!(lut_index(0.0, clip, Polarity::Normal), Some(128));
    assert!(pic.text.contains("inline 3"), "{}", pic.text);
}

/// Zoomed out, the plane is binned to the pixel grid with max-abs, and the
/// float grid is the pixel-grid shape. Each sample is read once. An explicit
/// clip reads exactly the same bytes.
#[test]
fn test_x09_bins_to_pixel_grid_once() {
    let (src, reader) = counted(bytes_of("small.sgy"));
    let core = File::open(data("small.sgy")).unwrap();
    let traces = core_traces(&core);
    let plane = inline_traces(&core, 0);
    let spec = PlaneSpec::new(vec![(Dim::Inline, 0)], 2, 7);
    let s = src.section(&spec).unwrap();
    assert_eq!((s.width, s.height), (2, 7));
    let auto_hits = reader.take();
    let mut seen = auto_hits.clone();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), auto_hits.len(), "no byte range read twice");
    for col in 0..2 {
        let cols = s.x.bin(col);
        for row in 0..7 {
            let rows = s.y.bin(row);
            let mut best = f32::NAN;
            for c in cols.clone() {
                for r in rows.clone() {
                    let v = traces[plane[c]][r];
                    if best.is_nan() || v.abs() > best.abs() {
                        best = v;
                    }
                }
            }
            assert_eq!(s.value(col, row), best, "bin ({col}, {row})");
        }
    }
    let style = Style {
        clip: Some(2.0),
        ..Style::default()
    };
    picture(&src, &spec, &style).unwrap();
    let mut set_hits = reader.take();
    set_hits.sort();
    let mut auto_sorted = auto_hits;
    auto_sorted.sort();
    assert_eq!(set_hits, auto_sorted);
}

/// A 3D selection does not draw itself, names `sel`, and reads nothing. The
/// notebook default picture is one plane (decision 5).
#[test]
fn test_x10_cube_refuses_and_names_sel() {
    let (src, reader) = counted(bytes_of("small.sgy"));
    let err = src.section(&PlaneSpec::new(vec![], 800, 480)).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, Error::NotTwoD { .. }));
    for hint in ["sel(inline=...)", "sel(crossline=...)", "sel(time=...)"] {
        assert!(msg.contains(hint), "{msg}");
    }
    assert!(reader.take().is_empty(), "a refused picture reads nothing");

    let fixed = src.default_fixed();
    assert_eq!(fixed, vec![(Dim::Inline, 0)]);
    let pic = notebook(&src, &fixed, None, None).unwrap();
    assert_eq!(pic.section.x.dim, Dim::Crossline);
    let core = File::open(data("small.sgy")).unwrap();
    let total: usize = reader.take().iter().map(|h| h.1).sum();
    assert_eq!(total, 5 * core.layout().n_samples as usize * 4);
    // The grid is bounded by the pixel cap, not by the cube.
    assert!(pic.section.width <= 800 && pic.section.height <= 480);
}

/// A horizontal axis that is not monotonic raises UnsortedCoordinate, unless
/// the view is sorted with its samples.
#[test]
fn test_x11_unsorted_axis_sorts_or_raises() {
    let mut raw = bytes_of("small.sgy");
    let core = File::open(data("small.sgy")).unwrap();
    let plane = inline_traces(&core, 1);
    let order = [30i32, 10, 50, 20, 40];
    for (t, x) in plane.iter().zip(order) {
        let h = (core.layout().data_start + *t as u64 * core.stride()) as usize;
        raw[h + 180..h + 184].copy_from_slice(&x.to_be_bytes());
        raw[h + 70..h + 72].copy_from_slice(&1i16.to_be_bytes());
    }
    let (src, reader) = counted(raw);
    let mut spec = PlaneSpec::new(vec![(Dim::Inline, 1)], 100, 100);
    spec.x_coord = Some(Coord::CdpX);
    let err = src.section(&spec).unwrap_err();
    assert!(
        matches!(err, Error::UnsortedCoordinate { ref coord } if coord == "cdp_x"),
        "{err}"
    );
    assert!(reader.take().is_empty());

    spec.sort = true;
    let s = src.section(&spec).unwrap();
    assert_eq!(s.x.labels, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let cols: Vec<u32> = s
        .column_traces
        .clone()
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let want: Vec<u32> = [1usize, 3, 0, 4, 2]
        .iter()
        .map(|i| plane[*i] as u32)
        .collect();
    assert_eq!(cols, want, "columns move with their coordinate");
    let traces = core_traces(&core);
    assert_eq!(s.value(0, 0), traces[plane[1]][0]);
}

/// The clip is the symmetric p98 of live, finite samples of the drawn view.
/// A dead trace (masked through trace id 2) and non-finite samples do not
/// move it, and the dead trace is not read.
#[test]
fn test_x12_clip_ignores_dead_and_non_finite() {
    let core = File::open(data("small.sgy")).unwrap();
    let traces = core_traces(&core);
    let plane = inline_traces(&core, 2);
    let ns = core.layout().n_samples as usize;
    let header = |t: usize| (core.layout().data_start + t as u64 * core.stride()) as usize;
    let sample = |t: usize, i: usize| header(t) + 240 + 4 * i;

    // IBM copy: one dead trace full of huge values, one overflow spike (IBM
    // 0x7fffffff is beyond f32, so it decodes to +inf).
    let mut raw = bytes_of("small.sgy");
    let dead = plane[1];
    raw[header(dead) + 28..header(dead) + 30].copy_from_slice(&2i16.to_be_bytes());
    for i in 0..ns {
        raw[sample(dead, i)..sample(dead, i) + 4].copy_from_slice(&ibm_word(1.0e6));
    }
    let spiked = plane[3];
    raw[sample(spiked, 7)..sample(spiked, 7) + 4].copy_from_slice(&[0x7f, 0xff, 0xff, 0xff]);

    let (src, reader) = counted(raw.clone());
    let mut spec = PlaneSpec::new(vec![(Dim::Inline, 2)], 400, 300);
    spec.live = Some(src.trace_id_mask());
    let s = src.section(&spec).unwrap();
    let live: Vec<f32> = plane
        .iter()
        .filter(|t| **t != dead)
        .flat_map(|t| {
            let mut v = traces[*t].clone();
            if *t == spiked {
                v[7] = f32::INFINITY;
            }
            v
        })
        .collect();
    let want = reference_p98(live);
    assert!(want.is_finite() && want < 1.0e3);
    assert_eq!(s.p98, Some(want));
    let dead_range = sample_bytes(&core, dead);
    assert!(
        reader.take().iter().all(|(o, _)| !dead_range.contains(o)),
        "the masked trace is not read"
    );
    let dead_col = plane.iter().position(|t| *t == dead).unwrap();
    assert!(
        s.value(dead_col, 0).is_nan(),
        "a masked trace draws as absent"
    );

    // Without a mask every stored trace is live (decision 7): the dead trace
    // counts, so the clip moves. The spike still does not.
    let open = src
        .section(&PlaneSpec::new(vec![(Dim::Inline, 2)], 400, 300))
        .unwrap();
    assert!(open.p98.unwrap() > want);

    // IEEE copy of the same file with a NaN and an infinity in the plane.
    let mut ieee = bytes_of("small.sgy");
    ieee[3224..3226].copy_from_slice(&5u16.to_be_bytes());
    for (t, tr) in traces.iter().enumerate() {
        for (i, v) in tr.iter().enumerate() {
            ieee[sample(t, i)..sample(t, i) + 4].copy_from_slice(&v.to_be_bytes());
        }
    }
    ieee[sample(plane[0], 3)..sample(plane[0], 3) + 4].copy_from_slice(&f32::NAN.to_be_bytes());
    ieee[sample(plane[4], 9)..sample(plane[4], 9) + 4]
        .copy_from_slice(&f32::NEG_INFINITY.to_be_bytes());
    let (src, _) = counted(ieee);
    let s = src
        .section(&PlaneSpec::new(vec![(Dim::Inline, 2)], 400, 300))
        .unwrap();
    let mut finite: Vec<f32> = plane.iter().flat_map(|t| traces[*t].clone()).collect();
    finite[3] = f32::NAN;
    finite[4 * ns + 9] = f32::NEG_INFINITY;
    assert_eq!(s.p98, Some(reference_p98(finite)));
    assert!(s.p98.unwrap().is_finite());
}

/// Default image axes are the 1D dims even where cdp_x is a 2D coordinate.
/// The map is a separate picture and reads headers only.
#[test]
fn test_x13_default_axes_are_dims_map_reads_no_samples() {
    let (src, reader) = counted(bytes_of("acute-small.sgy"));
    let core = File::open(data("acute-small.sgy")).unwrap();
    let plane = inline_traces(&core, 0);
    let cdp_x: Vec<i32> = plane.iter().map(|t| core.cdp_x()[*t]).collect();
    assert!(
        cdp_x.windows(2).any(|w| w[0] != w[1]),
        "cdp_x varies along the inline"
    );
    let other = inline_traces(&core, 1);
    assert_ne!(
        core.cdp_x()[plane[0]],
        core.cdp_x()[other[0]],
        "and across inlines"
    );

    let s = src
        .section(&PlaneSpec::new(src.default_fixed(), 300, 200))
        .unwrap();
    assert_eq!((s.x.dim, s.y.dim), (Dim::Crossline, Dim::Time));
    assert_eq!(s.x.name, "crossline");
    assert_eq!(s.y.name, "time (ms)");
    reader.take();

    let img = map_image(&src, 300, 300).unwrap();
    assert_eq!(png_size(&img.to_png().unwrap()), (300 + 80, 300 + 68));
    assert!(
        reader.take().is_empty(),
        "the map reads no sample and no header"
    );
}

/// A time slice reads one sample per trace, at that sample.
#[test]
fn time_slice_reads_one_sample_per_trace() {
    let (src, reader) = counted(bytes_of("small.sgy"));
    let core = File::open(data("small.sgy")).unwrap();
    let s = src
        .section(&PlaneSpec::new(vec![(Dim::Time, 10)], 64, 64))
        .unwrap();
    assert_eq!((s.x.dim, s.y.dim), (Dim::Inline, Dim::Crossline));
    let hits = reader.take();
    assert_eq!(hits.len(), core.layout().n_traces as usize);
    let traces = core_traces(&core);
    for t in 0..core.layout().n_traces as usize {
        let at = sample_bytes(&core, t).start + 10 * 4;
        assert!(hits.contains(&(at, 4)), "trace {t}");
    }
    let il =
        s.x.labels
            .iter()
            .position(|v| *v == f64::from(core.inline()[7]))
            .unwrap();
    let xl =
        s.y.labels
            .iter()
            .position(|v| *v == f64::from(core.crossline()[7]))
            .unwrap();
    assert_eq!(s.value(il, xl), traces[7][10]);
    assert_eq!(s.fixed[0].coord, src.coords(Dim::Time).unwrap()[10]);
}

/// A pre-stack grid gets an offset dim; a gather is offset by time.
#[test]
fn prestack_gather_and_default() {
    let src = Source::open(data("small-ps.sgy"), &OpenOptions::default()).unwrap();
    assert_eq!(
        src.dims(),
        &[Dim::Inline, Dim::Crossline, Dim::Offset, Dim::Time]
    );
    assert_eq!(
        src.default_fixed(),
        vec![(Dim::Inline, 0), (Dim::Offset, 0)]
    );
    let g = src
        .section(&PlaneSpec::new(
            vec![(Dim::Inline, 1), (Dim::Crossline, 2)],
            50,
            50,
        ))
        .unwrap();
    assert_eq!(g.x.dim, Dim::Offset);
    let core = File::open(data("small-ps.sgy")).unwrap();
    let traces = core_traces(&core);
    let (il, xl) = (core.geometry().inlines[1], core.geometry().crosslines[2]);
    for (c, off) in g.x.labels.iter().enumerate() {
        let t = (0..traces.len())
            .find(|t| {
                core.inline()[*t] == il
                    && core.crossline()[*t] == xl
                    && f64::from(core.offset()[*t]) == *off
            })
            .unwrap();
        assert_eq!(g.value(c, 0), traces[t][0]);
    }
    let err = src
        .section(&PlaneSpec::new(vec![(Dim::Inline, 0)], 50, 50))
        .unwrap_err();
    assert!(err.to_string().contains("fix 1 of its dims"), "{err}");
}

/// The window decoder agrees with the core decoder on every sample format,
/// both byte orders, for whole traces and for a window.
#[test]
fn window_decode_matches_core_for_every_format() {
    let dir = data("multiformats");
    let mut n = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let opts = OpenOptions {
            gain: Some(GainConvention::Rev0),
            ..OpenOptions::default()
        };
        let core = File::open_with(&path, &opts).unwrap();
        let src = Source::open(&path, &opts).unwrap();
        let traces = core_traces(&core);
        let ns = core.layout().n_samples as usize;
        let last = src.dims().len() - 1;
        let fixed: Vec<(Dim, usize)> = if last == 1 {
            vec![]
        } else {
            src.default_fixed()
        };
        let mut spec = PlaneSpec::new(fixed, 10_000, 10_000);
        spec.y_range = Some(1..ns);
        let s = src.section(&spec).unwrap();
        let cols = s.column_traces.clone().unwrap();
        for (c, t) in cols.iter().enumerate() {
            let t = t.unwrap() as usize;
            for r in 0..s.height {
                let (a, b) = (s.value(c, r), traces[t][r + 1]);
                assert!(
                    if b.is_finite() { a == b } else { a.is_nan() },
                    "{} trace {t} sample {}: {a} vs {b}",
                    path.display(),
                    r + 1
                );
            }
        }
        n += 1;
    }
    assert_eq!(n, 28);
}

/// Wiggle and variable area draw ink inside the plot when traces fit.
#[test]
fn wiggle_area_draws_when_traces_fit() {
    let src = Source::open(data("small.sgy"), &OpenOptions::default()).unwrap();
    let s = src
        .section(&PlaneSpec::new(vec![(Dim::Inline, 0)], 200, 200))
        .unwrap();
    let ink = |style: &Style| {
        let img = render(&s, style);
        img.rgba.chunks(4).filter(|p| p[0] < 128).count()
    };
    let none = Style {
        density: false,
        ..Style::default()
    };
    let line = Style {
        wiggle: Wiggle::Line,
        ..none
    };
    let area = Style {
        wiggle: Wiggle::Area,
        ..none
    };
    assert!(ink(&line) > ink(&none));
    assert!(ink(&area) > ink(&line));
    let seismic = Style {
        colormap: Colormap::Seismic,
        ..Style::default()
    };
    assert_ne!(
        render(&s, &seismic).rgba,
        render(&s, &Style::default()).rgba
    );
}

/// #83: a wiggle that does not fit says so in the text and on the image, and
/// `wiggle_every` reads one trace in N so the wiggles fit.
#[test]
fn wiggle_fallback_is_explicit_and_wiggle_every_decimates() {
    let src = Source::open(data("small.sgy"), &OpenOptions::default()).unwrap();
    // 5 traces on 10 px: 2 px per trace, under the 3 px rule.
    let spec = PlaneSpec::new(vec![(Dim::Inline, 0)], 10, 50);
    let dense = Style {
        wiggle: Wiggle::Line,
        ..Style::default()
    };
    let pic = picture(&src, &spec, &dense).unwrap();
    let note = pic.note.clone().expect("fallback note");
    assert!(
        note.starts_with("wiggle needs >= 3 px per trace (5 traces on 10 px); showing density"),
        "{note}"
    );
    assert!(pic.text.contains(&note), "{}", pic.text);

    let every = Style {
        wiggle_every: 2,
        density: false,
        ..dense
    };
    let pic = picture(&src, &spec, &every).unwrap();
    assert_eq!(pic.note.as_deref(), Some("wiggle_every=2: one trace in 2"));
    let all = src.coords(Dim::Crossline).unwrap();
    assert_eq!(pic.section.x.labels, vec![all[0], all[2], all[4]]);
    assert_eq!(pic.section.traces_read, 3);
    // Ink inside the plot area: the wiggles are drawn.
    let plot_ink = |style: &Style| {
        let img = render(&pic.section, style);
        let (ml, _, mt, _) = MARGINS;
        let mut n = 0;
        for y in mt..mt + 50 {
            for x in ml..ml + 10 {
                n += usize::from(img.rgba[(y * img.width + x) * 4] < 128);
            }
        }
        n
    };
    let off = Style {
        wiggle: Wiggle::Off,
        ..every
    };
    assert!(plot_ink(&every) > plot_ink(&off));

    // A time slice cannot decimate for wiggles: refused, nothing silently dropped.
    let slice = PlaneSpec::new(vec![(Dim::Time, 0)], 10, 50);
    assert!(picture(&src, &slice, &every).is_err());
}
