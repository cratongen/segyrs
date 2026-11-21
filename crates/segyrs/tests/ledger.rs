use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use segyrs::{
    apply_scalar, ibm_to_f32, Endian, Error, File, GainConvention, GeometryKind, HeaderDtype,
    HeaderField, HeaderSpec, OpenOptions, ReadAt, SampleFormat,
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name)
}

fn open_data(name: &str) -> File {
    File::open(data(name)).unwrap_or_else(|err| panic!("open {name}: {err}"))
}

struct Mem {
    data: Vec<u8>,
}

impl ReadAt for Mem {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> segyrs::Result<()> {
        let start = offset as usize;
        let end = start.checked_add(buf.len()).ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "range",
            ))
        })?;
        if end > self.data.len() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read",
            )));
        }
        buf.copy_from_slice(&self.data[start..end]);
        Ok(())
    }
}

struct Count {
    data: Vec<u8>,
    hits: Arc<Mutex<Vec<(u64, usize)>>>,
}

impl ReadAt for Count {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> segyrs::Result<()> {
        self.hits.lock().unwrap().push((offset, buf.len()));
        Mem {
            data: self.data.clone(),
        }
        .read_at(offset, buf)
    }
}

struct Built {
    format: u16,
    ns: u32,
    dt: u16,
    major: u8,
    minor: u8,
    little: bool,
    sig: [u8; 4],
    fixed: u16,
    extra: u32,
    binary_ns: u32,
    ext_binary_ns: u32,
    text_byte0: u8,
    traces: Vec<OneTrace>,
}

#[derive(Clone)]
struct OneTrace {
    il: i32,
    xl: i32,
    off: i32,
    ns_header: u16,
    ext_ns: u32,
    scalco: i16,
    cdp_x: i32,
    trace_id: i16,
    samples: Vec<u8>,
}

impl Built {
    fn ieee(samples: Vec<f32>) -> Self {
        let ns = samples.len() as u32;
        let mut raw = Vec::new();
        for value in samples {
            raw.extend(value.to_be_bytes());
        }
        Self::one(5, ns, raw, 1, 1, 0)
    }

    fn one(format: u16, ns: u32, samples: Vec<u8>, il: i32, xl: i32, off: i32) -> Self {
        Self {
            format,
            ns,
            dt: 4000,
            major: 1,
            minor: 0,
            little: false,
            sig: [0, 0, 0, 0],
            fixed: 1,
            extra: 0,
            binary_ns: ns,
            ext_binary_ns: 0,
            text_byte0: b'C',
            traces: vec![OneTrace {
                il,
                xl,
                off,
                ns_header: u16::try_from(ns).unwrap_or(u16::MAX),
                ext_ns: 0,
                scalco: 0,
                cdp_x: 0,
                trace_id: 1,
                samples,
            }],
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let width = SampleFormat::from_code(self.format).unwrap().width();
        let blocks = 1 + self.extra;
        let stride = 240 * blocks as usize + self.ns as usize * width;
        let mut buf = vec![0u8; 3600 + self.traces.len() * stride];
        buf[0] = self.text_byte0;
        if self.text_byte0 == 0x43 {
            buf[1] = b' ';
            buf[2] = b'1';
        }
        if self.text_byte0 == 0xc3 {
            buf[1] = 0x40;
            buf[2] = 0xf1;
        }
        put_u16(&mut buf, 3216, self.dt, self.little);
        put_u16(
            &mut buf,
            3220,
            u16::try_from(self.binary_ns).unwrap_or(u16::MAX),
            self.little,
        );
        put_u16(&mut buf, 3224, self.format, self.little);
        put_u32(&mut buf, 3268, self.ext_binary_ns, self.little);
        buf[3296..3300].copy_from_slice(&self.sig);
        buf[3500] = self.major;
        buf[3501] = self.minor;
        put_u16(&mut buf, 3502, self.fixed, self.little);
        put_u32(&mut buf, 3506, self.extra, self.little);
        for (i, trace) in self.traces.iter().enumerate() {
            let at = 3600 + i * stride;
            put_i32(&mut buf, at + 36, trace.off, self.little);
            put_i16(&mut buf, at + 28, trace.trace_id, self.little);
            put_i16(&mut buf, at + 70, trace.scalco, self.little);
            put_u16(&mut buf, at + 114, trace.ns_header, self.little);
            put_i32(&mut buf, at + 180, trace.cdp_x, self.little);
            put_i32(&mut buf, at + 188, trace.il, self.little);
            put_i32(&mut buf, at + 192, trace.xl, self.little);
            if self.extra > 0 {
                put_u32(&mut buf, at + 240 + 136, trace.ext_ns, self.little);
                put_u16(&mut buf, at + 240 + 156, self.extra as u16, self.little);
            }
            let sample_at = at + 240 * blocks as usize;
            buf[sample_at..sample_at + trace.samples.len()].copy_from_slice(&trace.samples);
        }
        buf
    }
}

fn put_u16(buf: &mut [u8], at: usize, v: u16, little: bool) {
    let b = if little {
        v.to_le_bytes()
    } else {
        v.to_be_bytes()
    };
    buf[at..at + 2].copy_from_slice(&b);
}

fn put_i16(buf: &mut [u8], at: usize, v: i16, little: bool) {
    put_u16(buf, at, v as u16, little);
}

fn put_u32(buf: &mut [u8], at: usize, v: u32, little: bool) {
    let b = if little {
        v.to_le_bytes()
    } else {
        v.to_be_bytes()
    };
    buf[at..at + 4].copy_from_slice(&b);
}

fn put_i32(buf: &mut [u8], at: usize, v: i32, little: bool) {
    put_u32(buf, at, v as u32, little);
}

fn open_mem(bytes: Vec<u8>, options: &OpenOptions) -> segyrs::Result<File<Mem>> {
    let len = bytes.len() as u64;
    File::from_reader(Mem { data: bytes }, len, options)
}

fn must(bytes: Vec<u8>) -> File<Mem> {
    open_mem(bytes, &OpenOptions::default()).expect("open synthetic file")
}

fn err_of<T>(result: segyrs::Result<T>) -> Error {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(err) => err,
    }
}

#[test]
fn s01_ibm_overflow_is_inf_and_raw_bytes_round_trip() {
    for word in [0x6110_0000, 0x6110_0001, 0x611f_ffff, 0x6120_0000] {
        let value = ibm_to_f32(word);
        assert!(value.is_infinite() && value.is_sign_positive(), "{word:#x}");
        assert!(!value.is_nan());
    }
    assert_eq!(ibm_to_f32(0), 0.0);
    assert_eq!(ibm_to_f32(0x4110_0000), 1.0);
    assert!(!ibm_to_f32(0x4110_0000).is_nan());

    let mut samples = Vec::new();
    for word in [0x4110_0000u32, 0x6110_0000] {
        samples.extend(word.to_be_bytes());
    }
    let mut built = Built::one(1, 2, samples, 1, 1, 0);
    built.dt = 1000;
    let file = must(built.bytes());
    assert_eq!(file.format(), SampleFormat::IbmFloat32);
    assert_eq!(file.format().name(), "ibm_float32");
    let decoded = file.decode(0, 1).unwrap();
    let values = decoded.as_f32().unwrap();
    assert_eq!(values[0], 1.0);
    assert!(values[1].is_infinite());
    let raw = file.sample_bytes(0).unwrap();
    assert_eq!(&raw[..4], &0x4110_0000u32.to_be_bytes());
    assert_eq!(
        ibm_to_f32(u32::from_be_bytes(raw[..4].try_into().unwrap())),
        values[0]
    );
}

#[test]
fn s02_format4_needs_gain_and_both_conventions() {
    let err = err_of(File::open(data("multiformats/Format4msb.sgy")));
    assert!(matches!(err, Error::Format4NeedsGain), "{err}");

    let word = [0x00u8, 0x03, 0x00, 0x05];
    let built = Built::one(4, 1, word.to_vec(), 1, 1, 0);
    let bytes = built.bytes();
    let closed = err_of(open_mem(bytes.clone(), &OpenOptions::default()));
    assert!(matches!(closed, Error::Format4NeedsGain), "{closed}");

    let rev0 = open_mem(
        bytes.clone(),
        &OpenOptions {
            gain: Some(GainConvention::Rev0),
            ..OpenOptions::default()
        },
    )
    .unwrap();
    let rev2 = open_mem(
        bytes,
        &OpenOptions {
            gain: Some(GainConvention::Rev2),
            ..OpenOptions::default()
        },
    )
    .unwrap();
    let a = rev0.decode(0, 1).unwrap().as_f64().unwrap()[0];
    let b = rev2.decode(0, 1).unwrap().as_f64().unwrap()[0];
    assert!((a - 40.0).abs() < 1e-9, "{a}");
    assert!((b - 0.625).abs() < 1e-9, "{b}");
}

#[test]
fn s03_sample_format_override() {
    let ieee_one = 1.0f32.to_be_bytes().to_vec();
    let mut built = Built::one(1, 1, ieee_one, 1, 1, 0);
    built.format = 1;
    let bytes = built.bytes();
    let trusted = must(bytes.clone());
    let wrong = trusted.decode(0, 1).unwrap().as_f32().unwrap()[0];
    assert_ne!(wrong, 1.0);

    let forced = open_mem(
        bytes,
        &OpenOptions {
            sample_format: Some(SampleFormat::IeeeFloat32),
            ..OpenOptions::default()
        },
    )
    .unwrap();
    assert_eq!(forced.layout().sample_format, SampleFormat::IeeeFloat32);
    assert_eq!(forced.binary_header_bytes()[24..26], 1u16.to_be_bytes());
    assert_eq!(forced.decode(0, 1).unwrap().as_f32().unwrap()[0], 1.0);
}

#[test]
fn s04_every_sample_format_and_int24_wider_than_i16() {
    let root = data("multiformats");
    for code in [1u16, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16] {
        for (suffix, endian) in [("msb", Endian::Big), ("lsb", Endian::Little)] {
            let name = format!("Format{code}{suffix}.sgy");
            let file = File::open(root.join(&name)).unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_eq!(file.format().code(), code, "{name}");
            assert_eq!(file.endian(), endian, "{name}");
            assert!(file.layout().n_traces > 0, "{name}");
            let decoded = file.decode(0, 1).unwrap();
            assert_eq!(decoded.len(), file.layout().n_samples as usize, "{name}");
        }
    }
    let mut raw = vec![0u8; 3];
    raw.copy_from_slice(&0x01_00_00u32.to_be_bytes()[1..]);
    let file = must(Built::one(7, 1, raw, 1, 1, 0).bytes());
    let value = file.decode(0, 1).unwrap().as_i32().unwrap()[0];
    assert_eq!(value, 65536);
    assert!(i16::try_from(value).is_err());
}

#[test]
fn s05_sample_interval_is_unsigned() {
    let mut built = Built::one(5, 1, 1.0f32.to_be_bytes().to_vec(), 1, 1, 0);
    built.dt = 0x9c40;
    let file = must(built.bytes());
    assert_eq!(file.layout().sample_interval_us, 40_000);
}

#[test]
fn s07_columnar_header_scan_matches_single_trace() {
    let file = open_data("small.sgy");
    let n = file.layout().n_traces as usize;
    let cols = file.par_header_columns(&HeaderSpec::standard()).unwrap();
    assert_eq!(cols.len(), n);
    let inline = cols.i32("inline").unwrap();
    let crossline = cols.i32("crossline").unwrap();
    let offset = cols.i32("offset").unwrap();
    for index in [0, n / 2, n - 1] {
        let raw = file.header_bytes(index as u64).unwrap();
        let il = i32::from_be_bytes(raw[188..192].try_into().unwrap());
        let xl = i32::from_be_bytes(raw[192..196].try_into().unwrap());
        let off = i32::from_be_bytes(raw[36..40].try_into().unwrap());
        assert_eq!(inline[index], il);
        assert_eq!(crossline[index], xl);
        assert_eq!(offset[index], off);
    }
    let spec = HeaderSpec {
        fields: vec![
            HeaderField::new(189, HeaderDtype::I32, "inline"),
            HeaderField::new(193, HeaderDtype::I32, "crossline"),
            HeaderField::new(37, HeaderDtype::I32, "offset"),
            HeaderField::new(71, HeaderDtype::I16, "scalco"),
            HeaderField::new(181, HeaderDtype::I32, "cdp_x"),
            HeaderField::new(185, HeaderDtype::I32, "cdp_y"),
            HeaderField::new(73, HeaderDtype::I32, "source_x"),
            HeaderField::new(77, HeaderDtype::I32, "source_y"),
            HeaderField::new(81, HeaderDtype::I32, "group_x"),
            HeaderField::new(85, HeaderDtype::I32, "group_y"),
        ],
    };
    let wide = file.par_header_columns(&spec).unwrap();
    assert_eq!(wide.len(), n);
    assert_eq!(wide.i32("inline").unwrap(), inline);
}

#[test]
fn s09_header_byte_zero_matches_file_offset() {
    let path = data("small.sgy");
    let file = File::open(&path).unwrap();
    let disk = std::fs::read(&path).unwrap();
    let stride = file.stride() as usize;
    let start = file.layout().data_start as usize;
    for i in 0..file.layout().n_traces {
        let raw = file.header_bytes(i).unwrap();
        let at = start + i as usize * stride;
        assert_eq!(raw[0], disk[at], "trace {i}");
        assert_eq!(raw.len(), 240);
    }
}

#[test]
fn s12_open_reads_headers_only() {
    let mut built = Built::ieee(vec![1.0, 2.0, 3.0, 4.0]);
    built.traces[0].il = 1;
    built.traces[0].xl = 1;
    let second = OneTrace {
        il: 1,
        xl: 2,
        off: 0,
        ns_header: 4,
        ext_ns: 0,
        scalco: 0,
        cdp_x: 0,
        trace_id: 1,
        samples: built.traces[0].samples.clone(),
    };
    let third = OneTrace {
        il: 2,
        xl: 1,
        ..second.clone()
    };
    let fourth = OneTrace {
        il: 2,
        xl: 2,
        ..second.clone()
    };
    built.traces.push(second);
    built.traces.push(third);
    built.traces.push(fourth);
    let bytes = built.bytes();
    let hits = Arc::new(Mutex::new(Vec::new()));
    let len = bytes.len() as u64;
    let file = File::from_reader(
        Count {
            data: bytes,
            hits: Arc::clone(&hits),
        },
        len,
        &OpenOptions::default(),
    )
    .unwrap();
    let sample_at =
        file.layout().data_start + u64::from(file.layout().extra_trace_header_blocks + 1) * 240;
    for (off, n) in hits.lock().unwrap().iter().copied() {
        let sample_end = sample_at + u64::from(file.layout().n_samples) * 4;
        let overlaps_samples = (0..file.layout().n_traces).any(|i| {
            let a = sample_at + i * file.stride();
            let b = a + (sample_end - sample_at);
            off < b && a < off + n as u64
        });
        assert!(!overlaps_samples, "open read {off}+{n} touched samples");
    }
    hits.lock().unwrap().clear();
    let nxl = 2;
    let one = file.decode(0, nxl).unwrap();
    assert_eq!(one.len(), nxl * file.layout().n_samples as usize);
    assert!(one.len() < (file.layout().n_traces as usize) * file.layout().n_samples as usize);
    let reads = hits.lock().unwrap().clone();
    assert!(!reads.is_empty());
    for (off, _) in reads {
        let later = file.layout().data_start + 2 * file.stride();
        assert!(
            off < later,
            "decode of the first inline read {off}, past {later}"
        );
    }
}

#[test]
fn s14_unsorted_opens_and_sorted_cube_reports_kind() {
    let mut built = Built::one(8, 1, vec![0], 1, 1, 0);
    built.traces = vec![
        trace_at(&built.traces[0], 1, 1),
        trace_at(&built.traces[0], 2, 2),
        trace_at(&built.traces[0], 1, 2),
        trace_at(&built.traces[0], 2, 1),
    ];
    let file = must(built.bytes());
    assert_eq!(file.geometry().kind, GeometryKind::Unsorted);
    assert_eq!(file.layout().n_traces, 4);

    let cube = open_data("small.sgy");
    assert!(
        cube.geometry().kind.is_grid(),
        "small.sgy kind {}",
        cube.geometry().kind.as_str()
    );
    assert!(!cube.geometry().inlines.is_empty());
    assert!(!cube.geometry().crosslines.is_empty());
}

#[test]
fn s15_unsorted_index_returns_the_trace_with_that_pair() {
    let mut built = Built::one(8, 1, vec![1], 1, 1, 0);
    built.traces = vec![
        trace_at(&built.traces[0], 1, 1),
        trace_at(&built.traces[0], 2, 2),
        trace_at(&built.traces[0], 1, 2),
        trace_at(&built.traces[0], 2, 1),
    ];
    for (i, trace) in built.traces.iter_mut().enumerate() {
        trace.samples = vec![10 + i as u8];
    }
    let file = must(built.bytes());
    assert_eq!(file.geometry().kind, GeometryKind::Unsorted);
    let hit = file.trace_index().get(2, 1).unwrap();
    assert_eq!(hit, &[3]);
    assert_eq!(file.decode_indices(hit).unwrap().as_i8().unwrap(), &[13]);
    assert!(file.trace_index().get(9, 9).is_none());
    assert_eq!(file.samples_read(), 1);
}

#[test]
fn s16_sparse_lists_missing_pair_and_duplicate_is_an_error() {
    let seed = {
        let mut seed_file = Built::one(8, 1, vec![1], 0, 0, 0);
        seed_file.traces.remove(0)
    };
    let mut pairs = Vec::new();
    for il in 1..=3 {
        for xl in 1..=3 {
            if (il, xl) != (3, 3) {
                pairs.push(trace_at(&seed, il, xl));
            }
        }
    }
    let mut built = Built::one(8, 1, vec![1], 1, 1, 0);
    built.traces = pairs;
    let file = must(built.bytes());
    assert_eq!(file.geometry().kind, GeometryKind::Sparse);
    assert_eq!(file.geometry().missing_pairs, vec![(3, 3)]);

    let mut dup = Built::one(8, 1, vec![1], 5, 7, 0);
    dup.traces.push(trace_at(&dup.traces[0], 5, 7));
    let err = err_of(open_mem(dup.bytes(), &OpenOptions::default()));
    match err {
        Error::DuplicateKey { inline, crossline } => {
            assert_eq!((inline, crossline), (5, 7));
        }
        other => panic!("{other}"),
    }
}

#[test]
fn s20_ascii_and_ebcdic_textual_headers() {
    let ascii = must(Built::one(8, 1, vec![0], 1, 1, 0).bytes());
    assert_eq!(ascii.textual_header()[0].encoding.as_str(), "ascii");
    assert_eq!(ascii.textual_header()[0].cards.len(), 40);
    assert!(ascii.textual_header()[0].cards[0].starts_with("C 1"));

    let mut ebcdic = Built::one(8, 1, vec![0], 1, 1, 0);
    ebcdic.text_byte0 = 0xc3;
    let file = must(ebcdic.bytes());
    assert_eq!(file.textual_header()[0].encoding.as_str(), "ebcdic");
    assert!(
        file.textual_header()[0].cards[0].starts_with("C 1"),
        "{}",
        file.textual_header()[0].cards[0]
    );

    let fixture = open_data("small.sgy");
    assert_eq!(fixture.textual_header()[0].cards.len(), 40);
    assert!(!fixture.textual_header()[0].cards[0].is_empty());
}

#[test]
fn s22_endian_signature_and_zero_defaults_big() {
    let mut classic = Built::one(5, 1, 1.0f32.to_be_bytes().to_vec(), 1, 1, 0);
    classic.sig = [0, 0, 0, 0];
    let file = must(classic.bytes());
    assert_eq!(file.endian(), Endian::Big);

    let mut little = Built::one(5, 1, 1.0f32.to_le_bytes().to_vec(), 7, 8, 0);
    little.little = true;
    little.major = 2;
    little.minor = 0;
    little.sig = [4, 3, 2, 1];
    let file = must(little.bytes());
    assert_eq!(file.endian(), Endian::Little);
    assert_eq!(file.layout().revision, segyrs::Revision::Rev20);
    assert_eq!(file.decode(0, 1).unwrap().as_f32().unwrap()[0], 1.0);
    assert_eq!(file.inline()[0], 7);

    let small = open_data("small.sgy");
    assert_eq!(small.endian(), Endian::Big);
    let lsb = open_data("small-lsb.sgy");
    assert_eq!(lsb.endian(), Endian::Little);
}

#[test]
fn s23_decoded_headers_are_native_and_raw_bytes_stay_file_order() {
    let mut built = Built::one(5, 1, 1.0f32.to_le_bytes().to_vec(), 1000, 3, 0);
    built.little = true;
    built.sig = [4, 3, 2, 1];
    built.major = 2;
    let file = must(built.bytes());
    assert_eq!(file.inline()[0], 1000);
    let raw = file.header_bytes(0).unwrap();
    assert_eq!(&raw[188..192], &1000i32.to_le_bytes());
    let native = i32::from_le_bytes(raw[188..192].try_into().unwrap());
    assert_eq!(native, file.inline()[0]);
    assert_eq!(file.layout().source_endian, Endian::Little);
}

#[test]
fn s24_extended_trace_header_sample_count() {
    let ns = 40_000u32;
    let mut samples = vec![0u8; ns as usize];
    samples[ns as usize - 1] = 42;
    let mut built = Built::one(8, ns, samples, 1, 1, 0);
    built.major = 2;
    built.minor = 0;
    built.sig = [1, 2, 3, 4];
    built.binary_ns = 32767;
    built.extra = 1;
    built.fixed = 1;
    built.traces[0].ns_header = 32767;
    built.traces[0].ext_ns = ns;
    let file = must(built.bytes());
    assert_eq!(file.layout().n_samples, ns);
    let decoded = file.decode(0, 1).unwrap();
    assert_eq!(decoded.as_i8().unwrap()[ns as usize - 1], 42);
}

#[test]
fn s25_sample_count_endian_and_trace_length_are_distinct() {
    let bytes = Built::one(8, 4, vec![1, 2, 3, 4], 1, 1, 0).bytes();
    let mut short = bytes.clone();
    short.truncate(short.len() - 4);
    let err = err_of(open_mem(short, &OpenOptions::default()));
    match err {
        Error::SampleCountMismatch {
            remainder,
            file_len,
            expected,
            ..
        } => {
            assert_ne!(remainder, 0, "file_len={file_len} expected={expected}");
            assert_eq!(file_len, expected + remainder);
        }
        other => panic!("{other}"),
    }

    let mut mixed = Built::one(8, 4, vec![1, 2, 3, 4], 1, 1, 0);
    mixed.fixed = 0;
    mixed.traces[0].ns_header = 4;
    mixed.traces.push(OneTrace {
        ns_header: 5,
        il: 1,
        xl: 2,
        ..mixed.traces[0].clone()
    });
    let err = err_of(open_mem(mixed.bytes(), &OpenOptions::default()));
    match err {
        Error::NonUniformTraceLength {
            trace,
            binary_ns,
            trace_ns,
        } => {
            assert_eq!(trace, 1);
            assert_eq!(binary_ns, 4);
            assert_eq!(trace_ns, 5);
        }
        other => panic!("{other}"),
    }

    let mut little = Built::one(5, 1, 1.0f32.to_le_bytes().to_vec(), 1, 1, 0);
    little.little = true;
    little.major = 2;
    little.sig = [4, 3, 2, 1];
    let err = err_of(open_mem(
        little.bytes(),
        &OpenOptions {
            endian: Some(Endian::Big),
            ..OpenOptions::default()
        },
    ));
    match err {
        Error::EndianMismatch { signature } => assert_eq!(signature, 0x0403_0201),
        other => panic!("{other}"),
    }
}

#[test]
fn x01_int16_stays_int16_with_no_nan_substitute() {
    let file = open_data("f3.sgy");
    assert_eq!(file.format(), SampleFormat::Int16);
    let samples = file.decode(0, 1).unwrap();
    let values = samples.as_i16().expect("int16 samples");
    assert!(!values.is_empty());
    assert!(samples.as_f32().is_none());
}

#[test]
fn x02_dead_trace_id_does_not_promote_integers() {
    let mut built = Built::one(
        3,
        2,
        7i16.to_be_bytes()
            .into_iter()
            .chain(7i16.to_be_bytes())
            .collect(),
        1,
        1,
        0,
    );
    built.traces[0].trace_id = 2;
    let file = must(built.bytes());
    assert_eq!(file.trace_id()[0], 2);
    let values = file.decode(0, 1).unwrap().as_i16().unwrap().to_vec();
    assert_eq!(values, vec![7, 7]);
}

#[test]
fn x03_samples_are_not_run_through_mask_and_scale() {
    let mut built = Built::one(3, 1, 5i16.to_be_bytes().to_vec(), 1, 1, 0);
    built.traces[0].scalco = -10;
    let file = must(built.bytes());
    assert_eq!(file.decode(0, 1).unwrap().as_i16().unwrap()[0], 5);
    assert_eq!(file.scalco()[0], -10);
}

#[test]
fn x04_both_endians_decode_to_native() {
    let msb = File::open(data("multiformats/Format5msb.sgy")).unwrap();
    let lsb = File::open(data("multiformats/Format5lsb.sgy")).unwrap();
    assert_eq!(msb.endian(), Endian::Big);
    assert_eq!(lsb.endian(), Endian::Little);
    let a = msb.decode(0, 1).unwrap();
    let b = lsb.decode(0, 1).unwrap();
    assert_eq!(a.as_f32().unwrap(), b.as_f32().unwrap());
    assert_ne!(msb.layout().source_endian, lsb.layout().source_endian);
}

#[test]
fn x05_scalco_positive_negative_and_zero() {
    assert_eq!(apply_scalar(1000, 10), 10_000.0);
    assert_eq!(apply_scalar(1000, -10), 100.0);
    assert_eq!(apply_scalar(1000, 0), 1000.0);
    let mut built = Built::one(8, 1, vec![0], 1, 1, 0);
    built.traces[0].scalco = -100;
    built.traces[0].cdp_x = 500_000;
    let file = must(built.bytes());
    assert_eq!(file.cdp_x()[0], 500_000);
    assert_eq!(apply_scalar(file.cdp_x()[0], file.scalco()[0]), 5_000.0);
}

#[test]
fn x06_integer_samples_ignore_coordinate_scalar() {
    let mut built = Built::one(2, 1, 25i32.to_be_bytes().to_vec(), 1, 1, 0);
    built.traces[0].scalco = 100;
    let file = must(built.bytes());
    assert_eq!(file.decode(0, 1).unwrap().as_i32().unwrap()[0], 25);
}

#[test]
fn x07_all_zero_trace_stays_present() {
    let file = must(Built::one(3, 4, vec![0; 8], 1, 1, 0).bytes());
    assert_eq!(file.layout().n_traces, 1);
    assert_eq!(file.decode(0, 1).unwrap().as_i16().unwrap(), &[0, 0, 0, 0]);
}

#[test]
fn x08_ibm_format_name_with_f32_samples() {
    let file = File::open(data("multiformats/Format1msb.sgy")).unwrap();
    assert_eq!(file.format().name(), "ibm_float32");
    assert!(file.decode(0, 1).unwrap().as_f32().is_some());
}

#[test]
fn x09_one_inline_is_smaller_than_the_volume() {
    let file = open_data("small.sgy");
    let nil = file.geometry().inlines.len();
    let nxl = file.geometry().crosslines.len();
    assert!(nil > 1 && nxl > 1, "nil={nil} nxl={nxl}");
    let inline = file.decode(0, nxl).unwrap();
    assert_eq!(inline.len(), nxl * file.layout().n_samples as usize);
    let volume = (file.layout().n_traces as usize) * file.layout().n_samples as usize;
    assert!(inline.len() < volume);
}

#[test]
fn x10_volume_reports_two_line_axes() {
    let volume = open_data("small.sgy");
    assert!(volume.geometry().inlines.len() > 1);
    assert!(volume.geometry().crosslines.len() > 1);
    let line = open_data("1xN.sgy");
    let single = line
        .geometry()
        .inlines
        .len()
        .min(line.geometry().crosslines.len());
    assert_eq!(single, 1, "{}", line.geometry().kind.as_str());
}

#[test]
fn x16_absurd_inline_span_is_geometry_error() {
    let mut built = Built::one(8, 1, vec![0], 1, 1, 0);
    built
        .traces
        .push(trace_at(&built.traces[0], 1_000_000_000, 1));
    let err = err_of(open_mem(built.bytes(), &OpenOptions::default()));
    match err {
        Error::Geometry { min, max, n_traces } => {
            assert_eq!(min, 1);
            assert_eq!(max, 1_000_000_000);
            assert_eq!(n_traces, 2);
        }
        other => panic!("{other}"),
    }
}

fn trace_at(seed: &OneTrace, il: i32, xl: i32) -> OneTrace {
    let mut out = seed.clone();
    out.il = il;
    out.xl = xl;
    out
}

#[test]
fn s13_read_window_matches_on_the_map_and_on_positional_reads() {
    let mapped = open_data("small.sgy");
    let ns = mapped.layout().n_samples as usize;
    let full = mapped.decode(0, 25).unwrap();
    let full = full.as_f32().unwrap();
    let before = mapped.samples_read();
    let picked = [7u64, 3, 24];
    let window = mapped.read_window(&picked, 5, 12).unwrap();
    assert_eq!(mapped.samples_read() - before, 3);
    let window = window.as_f32().unwrap();
    assert_eq!(window.len(), 3 * 7);
    for (row, &t) in picked.iter().enumerate() {
        let t = t as usize;
        assert_eq!(
            &window[row * 7..row * 7 + 7],
            &full[t * ns + 5..t * ns + 12]
        );
    }

    let bytes = std::fs::read(data("small-lsb.sgy")).unwrap();
    let positional = open_mem(bytes, &OpenOptions::default()).unwrap();
    let lsb = positional.read_window(&picked, 5, 12).unwrap();
    assert_eq!(lsb.as_f32().unwrap(), window);
    assert!(mapped.read_window(&[25], 0, 1).is_err());
    assert!(mapped.read_window(&[0], 3, ns + 1).is_err());
    assert!(mapped.read_window(&[], 0, ns).unwrap().is_empty());
}

#[test]
fn x18_one_short_trace_is_trace_length_error_with_its_index() {
    let mut built = Built::one(8, 4, vec![1, 2, 3, 4], 1, 1, 0);
    built.fixed = 0;
    built.traces.push(OneTrace {
        xl: 2,
        ..built.traces[0].clone()
    });
    built.traces.push(OneTrace {
        xl: 3,
        ..built.traces[0].clone()
    });
    let mut bytes = built.bytes();
    // Trace 1 really is one sample short: its header says 3 and the file lost a byte.
    let stride = 240 + 4;
    let t1 = 3600 + stride;
    bytes[t1 + 114..t1 + 116].copy_from_slice(&3u16.to_be_bytes());
    bytes.remove(t1 + 240 + 3);
    match err_of(open_mem(bytes, &OpenOptions::default())) {
        Error::TraceLength {
            trace,
            expected,
            got,
        } => {
            assert_eq!(trace, 1);
            assert_eq!(expected, 244);
            assert_eq!(got, 243);
        }
        other => panic!("{other}"),
    }
}
