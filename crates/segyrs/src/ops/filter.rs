use rayon::prelude::*;

use crate::error::Result;
use crate::file::{File, ReadAt};
use crate::header::{HeaderColumns, HeaderDtype, HeaderField, HeaderSpec};
use crate::ops::{invalid, Col};

/// A test on one named trace-header column (X-29).
#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    /// `min <= value <= max`. A `None` end is open.
    Range {
        field: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// The value equals one of `values`.
    In { field: String, values: Vec<f64> },
}

impl Predicate {
    pub fn range(field: impl Into<String>, min: Option<f64>, max: Option<f64>) -> Self {
        Self::Range {
            field: field.into(),
            min,
            max,
        }
    }

    pub fn is_in(field: impl Into<String>, values: impl IntoIterator<Item = f64>) -> Self {
        Self::In {
            field: field.into(),
            values: values.into_iter().collect(),
        }
    }

    pub fn field(&self) -> &str {
        match self {
            Self::Range { field, .. } | Self::In { field, .. } => field,
        }
    }
}

/// Trace numbers, ascending, whose headers pass every predicate (X-29).
///
/// The test runs in parallel over the columns. No per-trace object is built.
pub fn filter_columns(cols: &HeaderColumns, preds: &[Predicate]) -> Result<Vec<u64>> {
    let n = cols.len();
    let mut keep = vec![true; n];
    for pred in preds {
        let col = Col::find(cols, pred.field())?;
        if col.len() != n {
            return Err(invalid(format!("column {} is not {n} long", pred.field())));
        }
        match pred {
            Predicate::Range { min, max, .. } => {
                let lo = min.unwrap_or(f64::NEG_INFINITY);
                let hi = max.unwrap_or(f64::INFINITY);
                keep.par_iter_mut().enumerate().for_each(|(i, k)| {
                    if *k {
                        let v = col.f64_at(i);
                        *k = v >= lo && v <= hi;
                    }
                });
            }
            Predicate::In { values, .. } => {
                let mut set: Vec<f64> = values.iter().map(|v| v + 0.0).collect();
                set.sort_by(f64::total_cmp);
                keep.par_iter_mut().enumerate().for_each(|(i, k)| {
                    if *k {
                        let v = col.f64_at(i) + 0.0;
                        *k = set.binary_search_by(|p| p.total_cmp(&v)).is_ok();
                    }
                });
            }
        }
    }
    Ok((0..n)
        .into_par_iter()
        .filter(|&i| keep[i])
        .map(|i| i as u64)
        .collect())
}

/// Scan only the header fields the predicates name, then [`filter_columns`].
///
/// `inline`, `crossline`, and `offset` follow the bytes the file was opened
/// with. Other names come from [`HeaderSpec::standard`].
pub fn filter<R: ReadAt>(file: &File<R>, preds: &[Predicate]) -> Result<Vec<u64>> {
    if preds.is_empty() {
        return Ok((0..file.layout().n_traces).collect());
    }
    let spec = spec_for(file, preds.iter().map(Predicate::field))?;
    let cols = file.par_header_columns(&spec)?;
    filter_columns(&cols, preds)
}

/// A header spec holding just the named fields.
pub(crate) fn spec_for<'a, R: ReadAt>(
    file: &File<R>,
    names: impl Iterator<Item = &'a str>,
) -> Result<HeaderSpec> {
    let standard = HeaderSpec::standard();
    let geometry = file.geometry();
    let mut fields: Vec<HeaderField> = Vec::new();
    for name in names {
        if fields.iter().any(|f| f.name.as_deref() == Some(name)) {
            continue;
        }
        let field = match name {
            "inline" => HeaderField::new(geometry.inline_byte, HeaderDtype::I32, name),
            "crossline" => HeaderField::new(geometry.crossline_byte, HeaderDtype::I32, name),
            "offset" => HeaderField::new(geometry.offset_byte, HeaderDtype::I32, name),
            _ => standard
                .fields
                .iter()
                .find(|f| f.name.as_deref() == Some(name))
                .cloned()
                .ok_or_else(|| invalid(format!("no standard header field named {name}")))?,
        };
        fields.push(field);
    }
    Ok(HeaderSpec { fields })
}
