//! Colormaps (256-entry LUTs shared by the CPU compositor and the GPU shader),
//! polarity, and the mapping from a display value to a LUT index.

use std::sync::OnceLock;

/// In-tree colormaps. None of them is claimed to be perceptually uniform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Colormap {
    /// Black (negative clip) to white (positive clip). The default.
    #[default]
    Grey,
    /// Diverging: dark blue, blue, white at zero, red, dark red.
    Seismic,
    /// Sequential with rising lightness: black, purple, orange, pale yellow.
    /// For one-sided attributes.
    Amber,
}

impl Colormap {
    pub const ALL: [Colormap; 3] = [Colormap::Grey, Colormap::Seismic, Colormap::Amber];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Grey => "grey",
            Self::Seismic => "seismic",
            Self::Amber => "amber",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "grey" | "gray" => Self::Grey,
            "seismic" | "bwr" => Self::Seismic,
            "amber" => Self::Amber,
            _ => return None,
        })
    }

    /// Row of this map in [`lut_rows`].
    pub fn row(self) -> usize {
        self as usize
    }

    fn anchors(self) -> &'static [(f32, [f32; 3])] {
        match self {
            Self::Grey => &[(0.0, [0.0, 0.0, 0.0]), (1.0, [1.0, 1.0, 1.0])],
            Self::Seismic => &[
                (0.0, [0.0, 0.0, 0.3]),
                (0.25, [0.0, 0.0, 1.0]),
                (0.5, [1.0, 1.0, 1.0]),
                (0.75, [1.0, 0.0, 0.0]),
                (1.0, [0.5, 0.0, 0.0]),
            ],
            Self::Amber => &[
                (0.0, [0.0, 0.0, 0.0]),
                (0.3, [0.33, 0.06, 0.38]),
                (0.65, [0.86, 0.36, 0.12]),
                (1.0, [1.0, 0.95, 0.7]),
            ],
        }
    }

    /// The 256 RGBA entries of this map.
    pub fn lut(self) -> &'static [[u8; 4]; 256] {
        &lut_rows()[self.row()]
    }
}

/// Every map's LUT, one row per [`Colormap::row`]. The GPU uploads exactly
/// these bytes, so the window and the PNG agree.
pub fn lut_rows() -> &'static [[[u8; 4]; 256]; 3] {
    static ROWS: OnceLock<[[[u8; 4]; 256]; 3]> = OnceLock::new();
    ROWS.get_or_init(|| {
        let mut rows = [[[0u8; 4]; 256]; 3];
        for map in Colormap::ALL {
            let a = map.anchors();
            for (i, px) in rows[map.row()].iter_mut().enumerate() {
                let t = i as f32 / 255.0;
                let k = a
                    .windows(2)
                    .position(|w| t <= w[1].0)
                    .unwrap_or(a.len() - 2);
                let (t0, c0) = a[k];
                let (t1, c1) = a[k + 1];
                let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
                for ch in 0..3 {
                    let v = c0[ch] + (c1[ch] - c0[ch]) * f;
                    px[ch] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                px[3] = 255;
            }
        }
        rows
    })
}

/// Normal draws positive amplitudes at the top of the LUT. Reversed flips
/// the sign on the way into the LUT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Polarity {
    #[default]
    Normal,
    Reversed,
}

impl Polarity {
    pub fn sign(self) -> f32 {
        match self {
            Self::Normal => 1.0,
            Self::Reversed => -1.0,
        }
    }
}

/// Wiggle trace display, drawn only where every trace has its own columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wiggle {
    #[default]
    Off,
    /// Wiggle line.
    Line,
    /// Wiggle line plus variable area (positive lobes filled).
    Area,
}

/// How a section is coloured. Changing any field rereads nothing.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    /// Symmetric clip. `None` uses the section's p98.
    pub clip: Option<f32>,
    /// Amplitude gain: the effective clip is `clip / gain`.
    pub gain: f32,
    pub colormap: Colormap,
    pub polarity: Polarity,
    /// Paint the variable-density image.
    pub density: bool,
    pub wiggle: Wiggle,
    /// Wiggle excursion at the clip, in trace spacings.
    pub xcur: f32,
    /// Draw one trace in `wiggle_every` (`picture` reads only those along the
    /// section), so a dense section still shows wiggles. 1 draws every trace.
    pub wiggle_every: usize,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            clip: None,
            gain: 1.0,
            colormap: Colormap::Grey,
            polarity: Polarity::Normal,
            density: true,
            wiggle: Wiggle::Off,
            xcur: 1.0,
            wiggle_every: 1,
        }
    }
}

/// LUT index of display value `v` for effective clip `clip` (> 0).
/// NaN has no index.
#[inline]
pub fn lut_index(v: f32, clip: f32, polarity: Polarity) -> Option<u8> {
    if v.is_nan() {
        return None;
    }
    let t = ((v * polarity.sign() / clip) + 1.0) * 0.5;
    Some((t.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diverging_white_at_zero_and_ends_fixed() {
        let s = Colormap::Seismic.lut();
        assert_eq!(s[0], [0, 0, 77, 255]);
        assert_eq!(s[255], [128, 0, 0, 255]);
        let mid = lut_index(0.0, 1.0, Polarity::Normal).unwrap() as usize;
        assert!(s[mid].iter().take(3).all(|c| *c >= 250));
        let g = Colormap::Grey.lut();
        assert_eq!(g[0], [0, 0, 0, 255]);
        assert_eq!(g[255], [255, 255, 255, 255]);
    }

    #[test]
    fn polarity_flips_and_clip_saturates() {
        assert_eq!(lut_index(5.0, 1.0, Polarity::Normal), Some(255));
        assert_eq!(lut_index(5.0, 1.0, Polarity::Reversed), Some(0));
        assert_eq!(lut_index(f32::NAN, 1.0, Polarity::Normal), None);
        assert_eq!(lut_index(f32::INFINITY, 1.0, Polarity::Normal), Some(255));
    }

    #[test]
    fn sequential_lightness_rises() {
        let a = Colormap::Amber.lut();
        let l = |p: [u8; 4]| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
        assert!(a.windows(2).all(|w| l(w[1]) >= l(w[0]) - 0.5));
    }
}
