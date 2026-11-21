//! Symmetric clip of the drawn view.

/// The 98th percentile (nearest rank) of `|v|` over the finite values, or
/// `None` when there are none. NaN and infinities never reach the ranking,
/// so a dead trace stored as NaN or a single overflowed spike cannot make
/// the clip non-finite.
pub fn p98(values: &[f32]) -> Option<f32> {
    percentile_abs(values, 98.0)
}

/// Nearest-rank percentile `q` (0 to 100) of `|v|` over the finite values.
pub fn percentile_abs(values: &[f32], q: f64) -> Option<f32> {
    let mut mags: Vec<f32> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .collect();
    if mags.is_empty() {
        return None;
    }
    let rank = ((q / 100.0) * mags.len() as f64).ceil() as usize;
    let k = rank.clamp(1, mags.len()) - 1;
    let (_, v, _) = mags.select_nth_unstable_by(k, f32::total_cmp);
    Some(*v)
}

/// A usable clip: positive and finite. Falls back to the largest magnitude,
/// then to 1, so an all-zero view still draws.
pub(crate) fn usable(clip: Option<f32>, values: &[f32]) -> f32 {
    if let Some(c) = clip.filter(|c| c.is_finite() && *c > 0.0) {
        return c;
    }
    let max = values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0f32, |m, v| m.max(v.abs()));
    if max > 0.0 {
        max
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_over_finite_magnitudes() {
        let mut v: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        v[0] = -1.0;
        v.push(f32::NAN);
        v.push(f32::INFINITY);
        v.push(f32::NEG_INFINITY);
        assert_eq!(p98(&v), Some(98.0));
        assert_eq!(p98(&[f32::NAN]), None);
        assert_eq!(p98(&[-3.0]), Some(3.0));
    }

    #[test]
    fn usable_never_zero() {
        assert_eq!(usable(Some(0.0), &[0.0, 0.0]), 1.0);
        assert_eq!(usable(None, &[0.0, -4.0]), 4.0);
        assert_eq!(usable(Some(2.5), &[9.0]), 2.5);
    }
}
