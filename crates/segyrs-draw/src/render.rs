//! Float grid to RGBA: LUT, absent-cell hatch, wiggles, frame, ticks, PNG.

use std::sync::Arc;

use rayon::prelude::*;
use segyrs::{apply_scalar, ReadAt};
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

use crate::clip::usable;
use crate::color::{lut_index, Style, Wiggle};
use crate::font;
use crate::plane::{fmt_num, PlaneSpec, Section};
use crate::source::{Dim, Source};
use crate::{Error, Result};

/// Margins around the plot area in pixels: left, right, top, bottom.
pub const MARGINS: (usize, usize, usize, usize) = (64, 16, 28, 40);

/// Plot-area cap for a notebook image. A product limit so a notebook does not
/// embed a full-resolution section; it is not a throughput figure.
pub const NOTEBOOK_PIXELS: (usize, usize) = (800, 480);

/// Narrowest trace spacing, in pixels, at which wiggles are drawn. Below it
/// wiggles merge into a black box, so only density is painted, and
/// [`wiggle_note`] says so in the caption and the text.
pub const WIGGLE_MIN_SPACING: f32 = 3.0;

const INK: [u8; 4] = [0, 0, 0, 255];
const HATCH: [[u8; 4]; 2] = [[205, 205, 205, 255], [242, 242, 242, 255]];

/// An RGBA8 image, row-major, not premultiplied.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Image {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgba: vec![255; width * height * 4],
        }
    }

    fn put(&mut self, x: i64, y: i64, c: [u8; 4]) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            let i = (y as usize * self.width + x as usize) * 4;
            self.rgba[i..i + 4].copy_from_slice(&c);
        }
    }

    fn text(&mut self, s: &str, x: i64, y: i64) {
        font::draw(s, x, y, |px, py| self.put(px, py, INK));
    }

    /// Encode as an 8-bit RGBA PNG.
    pub fn to_png(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, self.width as u32, self.height as u32);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().map_err(|e| Error::Png(e.to_string()))?;
            w.write_image_data(&self.rgba)
                .map_err(|e| Error::Png(e.to_string()))?;
        }
        Ok(out)
    }
}

/// The clip the LUT uses: the explicit clip, else the section's p98, divided
/// by the gain. Always finite and positive.
pub fn effective_clip(section: &Section, style: &Style) -> f32 {
    let base = usable(style.clip.or(section.p98), &section.values);
    let gain = if style.gain.is_finite() && style.gain > 0.0 {
        style.gain
    } else {
        1.0
    };
    base / gain
}

/// Draw a section with its axes. The plot area is `section.pixels`.
pub fn render(section: &Section, style: &Style) -> Image {
    let (pw, ph) = section.pixels;
    let (ml, mr, mt, mb) = MARGINS;
    let mut img = Image::new(pw + ml + mr, ph + mt + mb);
    let clip = effective_clip(section, style);
    let stride = img.width * 4;
    let cols: Vec<usize> = (0..pw).map(|px| px * section.width / pw).collect();
    let lut = style.colormap.lut();
    img.rgba[mt * stride..(mt + ph) * stride]
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(py, line)| {
            let row = py * section.height / ph;
            let src = &section.values[row * section.width..(row + 1) * section.width];
            for (px, col) in cols.iter().enumerate() {
                let c = match lut_index(src[*col], clip, style.polarity) {
                    _ if !style.density && !src[*col].is_nan() => [255, 255, 255, 255],
                    Some(i) => lut[i as usize],
                    None => HATCH[((px + py) / 4) % 2],
                };
                let o = (ml + px) * 4;
                line[o..o + 4].copy_from_slice(&c);
            }
        });
    if style.wiggle != Wiggle::Off {
        draw_wiggles(&mut img, section, style, clip);
    }
    frame(&mut img, pw, ph);
    let x_to_px = |v: f64| label_pixel(&section.x.labels, v, pw);
    let y_to_px = |v: f64| label_pixel(&section.y.labels, v, ph);
    x_ticks(&mut img, &section.x.labels, pw, ph, &x_to_px);
    y_ticks(&mut img, &section.y.labels, ph, &y_to_px);
    let source = if style.clip.is_some() {
        "set"
    } else {
        "p98 of view"
    };
    let title = format!(
        "{}   clip ±{} ({source})   {}",
        section.title(),
        fmt_num(f64::from(clip)),
        style.colormap.as_str()
    );
    img.text(&title, ml as i64, 4);
    if let Some(note) = wiggle_note(section, style) {
        img.text(&note, ml as i64, 16);
    }
    img.text(&section.y.name, 4, 16);
    let xn = &section.x.name;
    img.text(
        xn,
        (ml + pw / 2) as i64 - font::width(xn) as i64 / 2,
        (mt + ph + 26) as i64,
    );
    img
}

/// What happened to a requested wiggle: why it fell back to density (#83),
/// or how it was decimated. `None` when no wiggle was asked for.
pub fn wiggle_note(section: &Section, style: &Style) -> Option<String> {
    if style.wiggle == Wiggle::Off {
        return None;
    }
    let shown = if style.density {
        "showing density"
    } else {
        "showing nothing"
    };
    if !section.y.dim.is_vertical() {
        return Some(format!("wiggle needs time or depth free; {shown}"));
    }
    let n = section.x.labels.len();
    let pw = section.pixels.0;
    let every = style.wiggle_every.max(1);
    if section.column_traces.is_none() || (pw as f32 / section.width as f32) < WIGGLE_MIN_SPACING {
        let more = (n as f32 * WIGGLE_MIN_SPACING / pw as f32).ceil() as usize;
        return Some(format!(
            "wiggle needs >= {WIGGLE_MIN_SPACING} px per trace ({n} traces on {pw} px); {shown}; try wiggle_every={}",
            every * more.max(2)
        ));
    }
    (every > 1).then(|| format!("wiggle_every={every}: one trace in {every}"))
}

/// Wiggle (and variable area) per trace, only when each trace owns at least
/// [`WIGGLE_MIN_SPACING`] pixels and the grid is one column per trace.
fn draw_wiggles(img: &mut Image, s: &Section, style: &Style, clip: f32) {
    let Some(traces) = &s.column_traces else {
        return;
    };
    let (pw, ph) = s.pixels;
    let spacing = pw as f32 / s.width as f32;
    if spacing < WIGGLE_MIN_SPACING {
        return;
    }
    let Some(mut pm) = Pixmap::new(pw as u32, ph as u32) else {
        return;
    };
    let exc = style.xcur * spacing;
    let sign = style.polarity.sign();
    let mut paint = Paint::default();
    paint.set_color_rgba8(0, 0, 0, 255);
    paint.anti_alias = true;
    let stroke = Stroke {
        width: 1.0,
        ..Stroke::default()
    };
    let row_y = |r: usize| (r as f32 + 0.5) * ph as f32 / s.height as f32;
    for (c, t) in traces.iter().enumerate() {
        if t.is_none() {
            continue;
        }
        let xc = (c as f32 + 0.5) * spacing;
        let dx: Vec<f32> = (0..s.height)
            .map(|r| {
                let v = s.value(c, r);
                if v.is_nan() {
                    0.0
                } else {
                    (v * sign / clip).clamp(-1.0, 1.0) * exc
                }
            })
            .collect();
        if style.wiggle == Wiggle::Area {
            // Positive lobes, with the zero crossings interpolated.
            let mut pb = PathBuilder::new();
            pb.move_to(xc, row_y(0));
            for r in 0..s.height {
                if r > 0 && (dx[r - 1] > 0.0) != (dx[r] > 0.0) {
                    let f = dx[r - 1] / (dx[r - 1] - dx[r]);
                    let y = row_y(r - 1) + f * (row_y(r) - row_y(r - 1));
                    pb.line_to(xc, y);
                }
                pb.line_to(xc + dx[r].max(0.0), row_y(r));
            }
            pb.line_to(xc, row_y(s.height - 1));
            pb.close();
            if let Some(path) = pb.finish() {
                pm.fill_path(
                    &path,
                    &paint,
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
        }
        let mut pb = PathBuilder::new();
        pb.move_to(xc + dx[0], row_y(0));
        for (r, d) in dx.iter().enumerate().skip(1) {
            pb.line_to(xc + d, row_y(r));
        }
        if s.height == 1 {
            pb.line_to(xc + dx[0], row_y(0) + 1.0);
        }
        if let Some(path) = pb.finish() {
            pm.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }
    // Composite the premultiplied pixmap over the plot area.
    let (ml, _, mt, _) = MARGINS;
    let src = pm.data();
    for y in 0..ph {
        for x in 0..pw {
            let si = (y * pw + x) * 4;
            let a = u32::from(src[si + 3]);
            if a == 0 {
                continue;
            }
            let di = ((mt + y) * img.width + ml + x) * 4;
            for ch in 0..3 {
                let d = u32::from(img.rgba[di + ch]);
                img.rgba[di + ch] = (u32::from(src[si + ch]) + d * (255 - a) / 255).min(255) as u8;
            }
        }
    }
}

fn frame(img: &mut Image, pw: usize, ph: usize) {
    let (ml, _, mt, _) = MARGINS;
    let (l, t) = (ml as i64 - 1, mt as i64 - 1);
    let (r, b) = ((ml + pw) as i64, (mt + ph) as i64);
    for x in l..=r {
        img.put(x, t, INK);
        img.put(x, b, INK);
    }
    for y in t..=b {
        img.put(l, y, INK);
        img.put(r, y, INK);
    }
}

/// Fractional pixel of coordinate `v` along an axis whose data positions
/// carry `labels` (monotonic), drawn over `pixels`. `None` outside the axis.
fn label_pixel(labels: &[f64], v: f64, pixels: usize) -> Option<f64> {
    let n = labels.len();
    if n == 0 {
        return None;
    }
    let scale = pixels as f64 / n as f64;
    if n == 1 {
        return (v == labels[0]).then_some(0.5 * scale);
    }
    let inc = labels[n - 1] >= labels[0];
    let (lo, hi) = if inc {
        (labels[0], labels[n - 1])
    } else {
        (labels[n - 1], labels[0])
    };
    if v < lo || v > hi {
        return None;
    }
    // First position whose label is at or past v in drawing order.
    let k = labels.partition_point(|l| if inc { *l < v } else { *l > v });
    let p = if k == 0 {
        0.0
    } else {
        let (a, b) = (labels[k - 1], labels[k.min(n - 1)]);
        let f = if b != a { (v - a) / (b - a) } else { 0.0 };
        (k - 1) as f64 + f
    };
    Some((p + 0.5) * scale)
}

/// Round tick values over `[lo, hi]`, at most about `max` of them. An axis of
/// whole numbers (line numbers, trace numbers) never gets a fractional step.
fn tick_values(lo: f64, hi: f64, max: usize, whole: bool) -> (Vec<f64>, f64) {
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    if hi == lo || max == 0 {
        return (vec![lo], 1.0);
    }
    let raw = (hi - lo) / max as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10.0 * mag);
    let step = if whole { step.max(1.0) } else { step };
    let mut v = (lo / step).ceil() * step;
    let mut out = Vec::new();
    while v <= hi + step * 1e-9 {
        out.push(if v.abs() < step * 1e-9 { 0.0 } else { v });
        v += step;
    }
    (out, step)
}

fn tick_label(v: f64, step: f64) -> String {
    let d = if step >= 1.0 && step.fract() == 0.0 {
        0
    } else {
        (-step.log10().floor()).max(0.0) as usize + 1
    };
    let s = format!("{v:.d$}");
    if d > 0 {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

fn whole(labels: &[f64]) -> bool {
    labels.len() > 2 && labels.iter().all(|v| v.fract() == 0.0)
}

fn axis_ends(labels: &[f64]) -> (f64, f64) {
    (
        labels.first().copied().unwrap_or(0.0),
        labels.last().copied().unwrap_or(0.0),
    )
}

fn x_ticks(
    img: &mut Image,
    labels: &[f64],
    pw: usize,
    ph: usize,
    to_px: &dyn Fn(f64) -> Option<f64>,
) {
    let (ml, _, mt, _) = MARGINS;
    let (a, b) = axis_ends(labels);
    let (ticks, step) = tick_values(a, b, (pw / 70).max(2), whole(labels));
    let base = (mt + ph) as i64;
    for v in ticks {
        let Some(px) = to_px(v) else { continue };
        let x = ml as i64 + px as i64;
        for d in 1..=4 {
            img.put(x, base + d, INK);
        }
        let s = tick_label(v, step);
        img.text(&s, x - font::width(&s) as i64 / 2, base + 7);
    }
}

fn y_ticks(img: &mut Image, labels: &[f64], ph: usize, to_px: &dyn Fn(f64) -> Option<f64>) {
    let (ml, _, mt, _) = MARGINS;
    let (a, b) = axis_ends(labels);
    let (ticks, step) = tick_values(a, b, (ph / 40).max(2), whole(labels));
    let left = ml as i64 - 1;
    for v in ticks {
        let Some(py) = to_px(v) else { continue };
        let y = mt as i64 + py as i64;
        for d in 1..=4 {
            img.put(left - d, y, INK);
        }
        let s = tick_label(v, step);
        img.text(&s, left - 7 - font::width(&s) as i64, y - 3);
    }
}

/// A rendered section: PNG bytes, a one-line description for `text/plain`,
/// and the float grid it came from.
#[derive(Clone, Debug)]
pub struct Picture {
    pub png: Vec<u8>,
    pub text: String,
    pub section: Section,
    /// [`wiggle_note`]: why a requested wiggle fell back to density, or its
    /// decimation. Also in `text` and on the image.
    pub note: Option<String>,
}

/// Read one plane, colour it and encode it.
pub fn picture<R: ReadAt + 'static>(
    source: &Source<R>,
    spec: &PlaneSpec,
    style: &Style,
) -> Result<Picture> {
    let section = if style.wiggle != Wiggle::Off && style.wiggle_every > 1 {
        if spec.fixed.iter().any(|(d, _)| d.is_vertical()) {
            return Err(Error::Empty(
                "wiggle_every needs a section with time or depth free, not a slice",
            ));
        }
        let mut spec = spec.clone();
        spec.x_step = spec.x_step.saturating_mul(style.wiggle_every);
        source.section(&spec)?
    } else {
        source.section(spec)?
    };
    let png = render(&section, style).to_png()?;
    let clip = effective_clip(&section, style);
    let note = wiggle_note(&section, style);
    let text = format!(
        "segyrs section: {} | x {} ({} positions), y {} ({} positions) | grid {}x{} | clip ±{} ({}) | {}{}{}",
        if section.fixed.is_empty() {
            "whole volume".to_string()
        } else {
            section.title()
        },
        section.x.name,
        section.x.labels.len(),
        section.y.name,
        section.y.labels.len(),
        section.width,
        section.height,
        fmt_num(f64::from(clip)),
        if style.clip.is_some() {
            "set"
        } else {
            "p98 of drawn view"
        },
        style.colormap.as_str(),
        if style.polarity == crate::Polarity::Reversed {
            ", reversed"
        } else {
            ""
        },
        match (style.wiggle, &note) {
            (_, Some(n)) => format!(" | {n}"),
            (Wiggle::Off, None) => String::new(),
            (w, None) => format!(" | {}", if w == Wiggle::Area { "area" } else { "wiggle" }),
        },
    );
    Ok(Picture {
        png,
        text,
        section,
        note,
    })
}

/// The notebook image of one plane: plot area capped at [`NOTEBOOK_PIXELS`],
/// grey, normal polarity, p98 unless `clip` is set. A selection that leaves
/// more than two dims free is refused with [`Error::NotTwoD`], which names
/// the `sel` calls that make a plane.
pub fn notebook<R: ReadAt + 'static>(
    source: &Source<R>,
    fixed: &[(Dim, usize)],
    clip: Option<f32>,
    live: Option<Arc<[bool]>>,
) -> Result<Picture> {
    let mut spec = PlaneSpec::new(fixed.to_vec(), NOTEBOOK_PIXELS.0, NOTEBOOK_PIXELS.1);
    spec.live = live;
    let style = Style {
        clip,
        ..Style::default()
    };
    picture(source, &spec, &style)
}

/// A map of `cdp_x` against `cdp_y` (scalars applied), one dot per trace,
/// equal scale on both axes. It reads trace headers only, which the open
/// already scanned: no sample byte is read.
pub fn map_image<R: ReadAt + 'static>(source: &Source<R>, pw: usize, ph: usize) -> Result<Image> {
    if pw == 0 || ph == 0 {
        return Err(Error::Empty("pixel grid has no pixels"));
    }
    let f = source.file();
    let xs: Vec<f64> = f
        .cdp_x()
        .iter()
        .zip(f.scalco())
        .map(|(v, s)| apply_scalar(*v, *s))
        .collect();
    let ys: Vec<f64> = f
        .cdp_y()
        .iter()
        .zip(f.scalco())
        .map(|(v, s)| apply_scalar(*v, *s))
        .collect();
    if xs.is_empty() {
        return Err(Error::Empty("no traces to map"));
    }
    let span = |v: &[f64]| {
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    };
    let (x0, x1) = span(&xs);
    let (y0, y1) = span(&ys);
    let pad = 0.05 * (x1 - x0).max(y1 - y0).max(1.0);
    let scale = (pw as f64 / (x1 - x0 + 2.0 * pad)).min(ph as f64 / (y1 - y0 + 2.0 * pad));
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let to_x = move |v: f64| (v - cx) * scale + pw as f64 / 2.0;
    let to_y = move |v: f64| ph as f64 / 2.0 - (v - cy) * scale;
    let (ml, mr, mt, mb) = MARGINS;
    let mut img = Image::new(pw + ml + mr, ph + mt + mb);
    for (x, y) in xs.iter().zip(&ys) {
        let (px, py) = (to_x(*x) as i64, to_y(*y) as i64);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (qx, qy) = (px + dx, py + dy);
                if qx >= 0 && qy >= 0 && (qx as usize) < pw && (qy as usize) < ph {
                    img.put(ml as i64 + qx, mt as i64 + qy, INK);
                }
            }
        }
    }
    frame(&mut img, pw, ph);
    let xl = [cx - pw as f64 / 2.0 / scale, cx + pw as f64 / 2.0 / scale];
    let yl = [cy + ph as f64 / 2.0 / scale, cy - ph as f64 / 2.0 / scale];
    let x_px = |v: f64| (v >= xl[0] && v <= xl[1]).then(|| to_x(v));
    let y_px = |v: f64| (v <= yl[0] && v >= yl[1]).then(|| to_y(v));
    x_ticks(&mut img, &xl, pw, ph, &x_px);
    y_ticks(&mut img, &yl, ph, &y_px);
    img.text(
        &format!("map of {} traces (headers only)", xs.len()),
        ml as i64,
        4,
    );
    img.text("cdp_y", 4, 16);
    img.text("cdp_x", (ml + pw / 2) as i64 - 15, (mt + ph + 26) as i64);
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_round_and_inside() {
        let (t, step) = tick_values(0.0, 196.0, 5, false);
        assert_eq!(step, 50.0);
        assert_eq!(t, vec![0.0, 50.0, 100.0, 150.0]);
        let (t, _) = tick_values(20.0, 24.0, 10, true);
        assert_eq!(t, vec![20.0, 21.0, 22.0, 23.0, 24.0]);
        assert_eq!(tick_label(0.25, 0.05), "0.25");
        assert_eq!(tick_label(150.0, 50.0), "150");
    }

    #[test]
    fn label_pixels_follow_either_direction() {
        let inc = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(label_pixel(&inc, 1.0, 400), Some(50.0));
        assert_eq!(label_pixel(&inc, 2.5, 400), Some(200.0));
        let dec = [4.0, 3.0, 2.0, 1.0];
        assert_eq!(label_pixel(&dec, 4.0, 400), Some(50.0));
        assert_eq!(label_pixel(&dec, 0.0, 400), None);
    }
}
