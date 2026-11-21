//! Drawing arguments as the Python bindings pass them: option names as text,
//! a plane as fixed positions plus free windows. The notebook picture and the
//! viewer window are built in separate extension modules and parse here.

use std::ops::Range;

use segyrs::ReadAt;

use crate::{Colormap, Dim, Polarity, Source, Style, Wiggle};

/// A [`Style`] from option names. The error names the bad value.
pub fn style_from_names(
    clip: Option<f32>,
    colormap: &str,
    polarity: &str,
    gain: f32,
    wiggle: &str,
    wiggle_every: usize,
    density: bool,
) -> Result<Style, String> {
    let colormap = Colormap::parse(colormap).ok_or_else(|| {
        format!("colormap {colormap:?} is not \"grey\", \"seismic\", or \"amber\"")
    })?;
    let polarity = match polarity {
        "normal" => Polarity::Normal,
        "reversed" => Polarity::Reversed,
        p => return Err(format!("polarity {p:?} is not \"normal\" or \"reversed\"")),
    };
    let wiggle = match wiggle {
        "off" => Wiggle::Off,
        "line" => Wiggle::Line,
        "area" => Wiggle::Area,
        w => {
            return Err(format!(
                "wiggle {w:?} is not \"off\", \"line\", or \"area\""
            ))
        }
    };
    if wiggle_every == 0 {
        return Err("wiggle_every must be at least 1".into());
    }
    if let Some(c) = clip {
        if !(c.is_finite() && c > 0.0) {
            return Err(format!("clip {c} is not a positive finite number"));
        }
    }
    Ok(Style {
        clip,
        gain,
        colormap,
        polarity,
        density,
        wiggle,
        wiggle_every,
        ..Style::default()
    })
}

/// One free dim of a plane: its window of positions and the step inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub range: Range<usize>,
    pub step: usize,
}

/// Fixed `(dim, position)` pairs and the free windows, in volume order.
pub type Plane = (Vec<(Dim, usize)>, Vec<Window>);

fn dim(name: &str) -> Result<Dim, String> {
    Dim::parse(name).ok_or_else(|| format!("{name:?} is not a dim"))
}

/// The plane to draw: the caller's fixed dims, plus the notebook default
/// (`Source::default_fixed`) at the start of each window while more than two
/// dims are free; then the free windows in the source's dim order. A window
/// is `(dim, start, stop, step)`.
pub fn resolve_plane<R: ReadAt + 'static>(
    src: &Source<R>,
    fixed: &[(String, usize)],
    ranges: &[(String, usize, usize, usize)],
) -> Result<Plane, String> {
    let mut fixed: Vec<(Dim, usize)> = fixed
        .iter()
        .map(|(d, i)| Ok((dim(d)?, *i)))
        .collect::<Result<_, String>>()?;
    let mut free: Vec<(Dim, Window)> = ranges
        .iter()
        .map(|(d, a, b, s)| {
            if *s == 0 {
                return Err(format!("{d} step must be at least 1"));
            }
            Ok((
                dim(d)?,
                Window {
                    range: *a..*b,
                    step: *s,
                },
            ))
        })
        .collect::<Result<_, String>>()?;
    for (d, _) in src.default_fixed() {
        if free.len() <= 2 {
            break;
        }
        if let Some(k) = free.iter().position(|(f, _)| *f == d) {
            let (_, r) = free.remove(k);
            fixed.push((d, r.range.start));
        }
    }
    let order = |d: Dim| src.dims().iter().position(|x| *x == d);
    free.sort_by_key(|(d, _)| order(*d));
    Ok((fixed, free.into_iter().map(|(_, r)| r).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_names_are_checked() {
        assert!(style_from_names(None, "grey", "normal", 1.0, "off", 1, true).is_ok());
        let bad = [
            style_from_names(None, "jet", "normal", 1.0, "off", 1, true),
            style_from_names(None, "grey", "up", 1.0, "off", 1, true),
            style_from_names(None, "grey", "normal", 1.0, "dots", 1, true),
            style_from_names(None, "grey", "normal", 1.0, "off", 0, true),
            style_from_names(Some(-1.0), "grey", "normal", 1.0, "off", 1, true),
        ];
        for (err, word) in bad
            .into_iter()
            .zip(["jet", "up", "dots", "wiggle_every", "clip"])
        {
            assert!(err.unwrap_err().contains(word));
        }
    }
}
