use std::io;

use crate::error::{Error, Result};

/// Width and signedness of one trace-header field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderDtype {
    I8,
    I16,
    I32,
    I64,
    U16,
    U32,
    F64,
}

impl HeaderDtype {
    pub fn width(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

/// One trace-header field. `byte` is the SEG-Y number, starting at 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderField {
    pub byte: u16,
    pub dtype: HeaderDtype,
    pub name: Option<String>,
}

impl HeaderField {
    pub fn new(byte: u16, dtype: HeaderDtype, name: impl Into<String>) -> Self {
        Self {
            byte,
            dtype,
            name: Some(name.into()),
        }
    }
}

/// The fields a columnar scan reads.
#[derive(Clone, Debug)]
pub struct HeaderSpec {
    pub fields: Vec<HeaderField>,
}

impl HeaderSpec {
    /// Rev 0 / rev 1 trace header, plus the rev 2.1 names at the same bytes.
    ///
    /// Inline is byte 189 and crossline is byte 193, both `I32`. A caller can
    /// replace any field, including one wider than the standard word.
    pub fn standard() -> Self {
        const ROWS: &[(u16, HeaderDtype, &str)] = &[
            (1, HeaderDtype::I32, "trace_seq_line"),
            (5, HeaderDtype::I32, "trace_seq_file"),
            (9, HeaderDtype::I32, "field_record"),
            (13, HeaderDtype::I32, "trace_in_record"),
            (17, HeaderDtype::I32, "source_point"),
            (21, HeaderDtype::I32, "cdp"),
            (25, HeaderDtype::I32, "trace_in_cdp"),
            (29, HeaderDtype::I16, "trace_id"),
            (31, HeaderDtype::I16, "n_summed_v"),
            (33, HeaderDtype::I16, "n_summed_h"),
            (35, HeaderDtype::I16, "data_use"),
            (37, HeaderDtype::I32, "offset"),
            (41, HeaderDtype::I32, "group_elev"),
            (45, HeaderDtype::I32, "source_elev"),
            (49, HeaderDtype::I32, "source_depth"),
            (53, HeaderDtype::I32, "group_datum"),
            (57, HeaderDtype::I32, "source_datum"),
            (61, HeaderDtype::I32, "water_depth_source"),
            (65, HeaderDtype::I32, "water_depth_group"),
            (69, HeaderDtype::I16, "scalel"),
            (71, HeaderDtype::I16, "scalco"),
            (73, HeaderDtype::I32, "source_x"),
            (77, HeaderDtype::I32, "source_y"),
            (81, HeaderDtype::I32, "group_x"),
            (85, HeaderDtype::I32, "group_y"),
            (89, HeaderDtype::I16, "coord_units"),
            (91, HeaderDtype::I16, "weathering_vel"),
            (93, HeaderDtype::I16, "subweathering_vel"),
            (95, HeaderDtype::I16, "uphole_source"),
            (97, HeaderDtype::I16, "uphole_group"),
            (99, HeaderDtype::I16, "source_static"),
            (101, HeaderDtype::I16, "group_static"),
            (103, HeaderDtype::I16, "total_static"),
            (105, HeaderDtype::I16, "lag_a"),
            (107, HeaderDtype::I16, "lag_b"),
            (109, HeaderDtype::I16, "delay"),
            (111, HeaderDtype::I16, "mute_start"),
            (113, HeaderDtype::I16, "mute_end"),
            (115, HeaderDtype::U16, "ns"),
            (117, HeaderDtype::U16, "sample_interval"),
            (119, HeaderDtype::I16, "gain_type"),
            (121, HeaderDtype::I16, "gain_constant"),
            (123, HeaderDtype::I16, "gain_initial"),
            (125, HeaderDtype::I16, "correlated"),
            (127, HeaderDtype::I16, "sweep_start"),
            (129, HeaderDtype::I16, "sweep_end"),
            (131, HeaderDtype::I16, "sweep_length"),
            (133, HeaderDtype::I16, "sweep_type"),
            (135, HeaderDtype::I16, "sweep_taper_start"),
            (137, HeaderDtype::I16, "sweep_taper_end"),
            (139, HeaderDtype::I16, "taper_type"),
            (141, HeaderDtype::I16, "alias_freq"),
            (143, HeaderDtype::I16, "alias_slope"),
            (145, HeaderDtype::I16, "notch_freq"),
            (147, HeaderDtype::I16, "notch_slope"),
            (149, HeaderDtype::I16, "low_cut"),
            (151, HeaderDtype::I16, "high_cut"),
            (153, HeaderDtype::I16, "low_cut_slope"),
            (155, HeaderDtype::I16, "high_cut_slope"),
            (157, HeaderDtype::I16, "year"),
            (159, HeaderDtype::I16, "day"),
            (161, HeaderDtype::I16, "hour"),
            (163, HeaderDtype::I16, "minute"),
            (165, HeaderDtype::I16, "second"),
            (167, HeaderDtype::I16, "time_basis"),
            (169, HeaderDtype::I16, "trace_weight"),
            (171, HeaderDtype::I16, "geophone_roll"),
            (173, HeaderDtype::I16, "geophone_first"),
            (175, HeaderDtype::I16, "geophone_last"),
            (177, HeaderDtype::I16, "gap"),
            (179, HeaderDtype::I16, "over_travel"),
            (181, HeaderDtype::I32, "cdp_x"),
            (185, HeaderDtype::I32, "cdp_y"),
            (189, HeaderDtype::I32, "inline"),
            (193, HeaderDtype::I32, "crossline"),
            (197, HeaderDtype::I32, "shotpoint"),
            (201, HeaderDtype::I16, "shotpoint_scalar"),
            (203, HeaderDtype::I16, "trace_unit"),
            (205, HeaderDtype::I32, "transduction_mant"),
            (209, HeaderDtype::I16, "transduction_exp"),
            (211, HeaderDtype::I16, "transduction_unit"),
            (213, HeaderDtype::I16, "device_id"),
            (215, HeaderDtype::I16, "time_scalar"),
            (217, HeaderDtype::I16, "source_orient"),
            (225, HeaderDtype::I32, "source_meas_mant"),
            (229, HeaderDtype::I16, "source_meas_exp"),
            (231, HeaderDtype::I16, "source_meas_unit"),
        ];
        Self {
            fields: ROWS
                .iter()
                .map(|(byte, dtype, name)| HeaderField::new(*byte, *dtype, *name))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
enum Column {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    F64(Vec<f64>),
}

/// One column after [`HeaderColumns::into_columns`] moves it out.
#[derive(Clone, Debug)]
pub struct NamedColumn {
    pub name: String,
    pub byte: u16,
    pub dtype: HeaderDtype,
    pub values: ColumnValues,
}

#[derive(Clone, Debug)]
pub enum ColumnValues {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    F64(Vec<f64>),
}

/// One decoded value, before it is stored in a column.
#[derive(Clone, Copy, Debug)]
pub enum FieldValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U16(u16),
    U32(u32),
    F64(f64),
}

/// Columnar trace headers. Each column has one entry per trace.
#[derive(Clone, Debug)]
pub struct HeaderColumns {
    fields: Vec<HeaderField>,
    cols: Vec<Column>,
}

impl HeaderColumns {
    pub fn from_values(fields: Vec<HeaderField>, rows: Vec<Vec<FieldValue>>) -> Result<Self> {
        let n = rows.len();
        let mut cols = Vec::with_capacity(fields.len());
        for (c, field) in fields.iter().enumerate() {
            cols.push(empty_column(field.dtype, n));
            for (row_i, row) in rows.iter().enumerate() {
                push_value(&mut cols[c], row[c], row_i, field)?;
            }
        }
        Ok(Self { fields, cols })
    }

    /// Columns already filled by a scan, one per field, in field order.
    pub fn from_columns(fields: Vec<HeaderField>, values: Vec<ColumnValues>) -> Result<Self> {
        if fields.len() != values.len() {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} fields but {} columns", fields.len(), values.len()),
            )));
        }
        let cols = values
            .into_iter()
            .map(|v| match v {
                ColumnValues::I8(v) => Column::I8(v),
                ColumnValues::I16(v) => Column::I16(v),
                ColumnValues::I32(v) => Column::I32(v),
                ColumnValues::I64(v) => Column::I64(v),
                ColumnValues::U16(v) => Column::U16(v),
                ColumnValues::U32(v) => Column::U32(v),
                ColumnValues::F64(v) => Column::F64(v),
            })
            .collect();
        Ok(Self { fields, cols })
    }

    pub fn len(&self) -> usize {
        self.cols.first().map(column_len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn i32(&self, name: &str) -> Result<&[i32]> {
        match self.column(name)? {
            Column::I32(v) => Ok(v),
            _ => Err(type_error(name, "i32")),
        }
    }

    pub fn i16(&self, name: &str) -> Result<&[i16]> {
        match self.column(name)? {
            Column::I16(v) => Ok(v),
            _ => Err(type_error(name, "i16")),
        }
    }

    pub fn u16(&self, name: &str) -> Result<&[u16]> {
        match self.column(name)? {
            Column::U16(v) => Ok(v),
            _ => Err(type_error(name, "u16")),
        }
    }

    pub fn i64(&self, name: &str) -> Result<&[i64]> {
        match self.column(name)? {
            Column::I64(v) => Ok(v),
            _ => Err(type_error(name, "i64")),
        }
    }

    pub fn u32(&self, name: &str) -> Result<&[u32]> {
        match self.column(name)? {
            Column::U32(v) => Ok(v),
            _ => Err(type_error(name, "u32")),
        }
    }

    pub fn f64(&self, name: &str) -> Result<&[f64]> {
        match self.column(name)? {
            Column::F64(v) => Ok(v),
            _ => Err(type_error(name, "f64")),
        }
    }

    pub fn fields(&self) -> &[HeaderField] {
        &self.fields
    }

    /// Hands each column's buffer to the caller. The allocation is the one the scan filled.
    pub fn into_columns(self) -> Vec<NamedColumn> {
        self.fields
            .into_iter()
            .zip(self.cols)
            .map(|(field, col)| NamedColumn {
                name: field.name.unwrap_or_default(),
                byte: field.byte,
                dtype: field.dtype,
                values: match col {
                    Column::I8(v) => ColumnValues::I8(v),
                    Column::I16(v) => ColumnValues::I16(v),
                    Column::I32(v) => ColumnValues::I32(v),
                    Column::I64(v) => ColumnValues::I64(v),
                    Column::U16(v) => ColumnValues::U16(v),
                    Column::U32(v) => ColumnValues::U32(v),
                    Column::F64(v) => ColumnValues::F64(v),
                },
            })
            .collect()
    }

    fn column(&self, name: &str) -> Result<&Column> {
        let idx = self
            .fields
            .iter()
            .position(|field| field.name.as_deref() == Some(name))
            .ok_or_else(|| {
                Error::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("no header field named {name}"),
                ))
            })?;
        Ok(&self.cols[idx])
    }
}

fn empty_column(dtype: HeaderDtype, n: usize) -> Column {
    match dtype {
        HeaderDtype::I8 => Column::I8(Vec::with_capacity(n)),
        HeaderDtype::I16 => Column::I16(Vec::with_capacity(n)),
        HeaderDtype::I32 => Column::I32(Vec::with_capacity(n)),
        HeaderDtype::I64 => Column::I64(Vec::with_capacity(n)),
        HeaderDtype::U16 => Column::U16(Vec::with_capacity(n)),
        HeaderDtype::U32 => Column::U32(Vec::with_capacity(n)),
        HeaderDtype::F64 => Column::F64(Vec::with_capacity(n)),
    }
}

fn column_len(col: &Column) -> usize {
    match col {
        Column::I8(v) => v.len(),
        Column::I16(v) => v.len(),
        Column::I32(v) => v.len(),
        Column::I64(v) => v.len(),
        Column::U16(v) => v.len(),
        Column::U32(v) => v.len(),
        Column::F64(v) => v.len(),
    }
}

fn push_value(col: &mut Column, value: FieldValue, row: usize, field: &HeaderField) -> Result<()> {
    let ok = match (col, value) {
        (Column::I8(v), FieldValue::I8(x)) => {
            v.push(x);
            true
        }
        (Column::I16(v), FieldValue::I16(x)) => {
            v.push(x);
            true
        }
        (Column::I32(v), FieldValue::I32(x)) => {
            v.push(x);
            true
        }
        (Column::I64(v), FieldValue::I64(x)) => {
            v.push(x);
            true
        }
        (Column::U16(v), FieldValue::U16(x)) => {
            v.push(x);
            true
        }
        (Column::U32(v), FieldValue::U32(x)) => {
            v.push(x);
            true
        }
        (Column::F64(v), FieldValue::F64(x)) => {
            v.push(x);
            true
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(Error::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "header field {} row {row} does not match its dtype",
                field.name.as_deref().unwrap_or("(unnamed)")
            ),
        )))
    }
}

fn type_error(name: &str, want: &str) -> Error {
    Error::Io(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("header field {name} is not {want}"),
    ))
}
