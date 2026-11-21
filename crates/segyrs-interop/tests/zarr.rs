//! Zarr v3 export: round trip through the crate's own reader, the chunking
//! rules (X-14), missing chunks (X-34), and a check of every `zarr.json`
//! against the v3 core spec.
#![cfg(feature = "zarr")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use segyrs::{File, GeometryKind, Samples};
use segyrs_interop::zarr::{export, DirSink, ZarrArray, ZarrOptions};
use serde_json::Value;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-data")
}

fn out_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("zarr-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn le(s: &Samples) -> Vec<u8> {
    match s {
        Samples::I8(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::U8(v) => v.clone(),
        Samples::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::U64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Samples::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn i64s(root: &Path, name: &str) -> Vec<i64> {
    let a = ZarrArray::open(root, name).unwrap();
    assert_eq!(a.data_type, "int64");
    a.read_all()
        .unwrap()
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Check the export against the file without trusting the writer's grid: each
/// trace is located through the coordinate arrays the store itself holds.
fn assert_round_trip(name: &str, file: &File, root: &Path) {
    let data = ZarrArray::open(root, "data").unwrap();
    let mask = ZarrArray::open(root, "trace_mask")
        .unwrap()
        .read_all()
        .unwrap();
    let all = data.read_all().unwrap();
    let w = data.itemsize();
    let ns = file.layout().n_samples as usize;
    let trace_bytes = ns * w;
    let n = file.layout().n_traces as usize;
    let dims: Vec<&str> = data.dimension_names.iter().map(String::as_str).collect();
    let cells: Vec<usize> = if dims[0] == "trace" {
        (0..n).collect()
    } else {
        let pos = |name: &str| -> HashMap<i64, usize> {
            i64s(root, name)
                .into_iter()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect()
        };
        let (il, xl) = (pos("inline"), pos("crossline"));
        let off = (dims.len() == 4).then(|| pos("offset"));
        let nxl = data.shape[1] as usize;
        let noff = if dims.len() == 4 {
            data.shape[2] as usize
        } else {
            1
        };
        (0..n)
            .map(|t| {
                let i = il[&i64::from(file.inline()[t])];
                let x = xl[&i64::from(file.crossline()[t])];
                let o = off.as_ref().map_or(0, |p| p[&i64::from(file.offset()[t])]);
                (i * nxl + x) * noff + o
            })
            .collect()
    };
    let mut seen = vec![false; mask.len()];
    for (t, &cell) in cells.iter().enumerate() {
        let want = le(&file.decode(t, 1).unwrap());
        assert_eq!(
            &all[cell * trace_bytes..(cell + 1) * trace_bytes],
            &want[..],
            "{name} trace {t}"
        );
        assert_eq!(mask[cell], 1, "{name} trace {t}");
        seen[cell] = true;
    }
    for (cell, s) in seen.iter().enumerate() {
        if !s {
            assert_eq!(mask[cell], 0, "{name} cell {cell}");
            assert!(all[cell * trace_bytes..(cell + 1) * trace_bytes]
                .iter()
                .all(|b| *b == 0));
        }
    }
}

/// Rules from the Zarr v3 core specification (metadata, chunk grid, chunk key
/// encoding, data types, fill value, codecs) applied to every node in a store,
/// plus every chunk file: a valid key and the exact encoded size.
fn assert_v3_schema(root: &Path) {
    let mut nodes = Vec::new();
    collect_meta(root, &mut nodes);
    assert!(!nodes.is_empty());
    let types = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ];
    let allowed_array_keys = [
        "zarr_format",
        "node_type",
        "shape",
        "data_type",
        "chunk_grid",
        "chunk_key_encoding",
        "fill_value",
        "codecs",
        "attributes",
        "dimension_names",
        "storage_transformers",
    ];
    for path in nodes {
        let m: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let at = path.display();
        assert_eq!(m["zarr_format"], 3, "{at}");
        let attrs = m
            .get("attributes")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        assert!(attrs.is_object(), "{at}");
        for bad in ["_FillValue", "scale_factor", "add_offset", "missing_value"] {
            assert!(attrs.get(bad).is_none(), "{at} has {bad}");
        }
        match m["node_type"].as_str() {
            Some("group") => {
                for k in m.as_object().unwrap().keys() {
                    assert!(
                        ["zarr_format", "node_type", "attributes"].contains(&k.as_str()),
                        "{at}: {k}"
                    );
                }
            }
            Some("array") => {
                for k in m.as_object().unwrap().keys() {
                    assert!(allowed_array_keys.contains(&k.as_str()), "{at}: {k}");
                }
                let shape: Vec<u64> = m["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect();
                let rank = shape.len();
                let dtype = m["data_type"].as_str().unwrap();
                assert!(types.contains(&dtype), "{at}: {dtype}");
                assert_eq!(m["chunk_grid"]["name"], "regular", "{at}");
                let chunks: Vec<u64> = m["chunk_grid"]["configuration"]["chunk_shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect();
                assert_eq!(chunks.len(), rank, "{at}");
                assert!(chunks.iter().all(|c| *c > 0), "{at}");
                assert_eq!(m["chunk_key_encoding"]["name"], "default", "{at}");
                let sep = &m["chunk_key_encoding"]["configuration"]["separator"];
                assert!(sep == "/" || sep == ".", "{at}");
                let fill = &m["fill_value"];
                match dtype {
                    "bool" => assert!(fill.is_boolean(), "{at}"),
                    d if d.starts_with("float") => {
                        assert!(fill.is_number() || fill.is_string(), "{at}")
                    }
                    _ => assert!(fill.is_i64() || fill.is_u64(), "{at}"),
                }
                let codecs = m["codecs"].as_array().unwrap();
                let array_to_bytes: Vec<_> =
                    codecs.iter().filter(|c| c["name"] == "bytes").collect();
                assert_eq!(
                    array_to_bytes.len(),
                    1,
                    "{at}: exactly one array-to-bytes codec"
                );
                let endian = &array_to_bytes[0]["configuration"]["endian"];
                assert!(endian == "little" || endian == "big", "{at}");
                let names = m["dimension_names"].as_array().unwrap();
                assert_eq!(names.len(), rank, "{at}");
                // Chunk files: keys inside the grid, size = chunk cells * item size.
                let dir = path.parent().unwrap();
                let counts: Vec<u64> = shape
                    .iter()
                    .zip(&chunks)
                    .map(|(s, c)| s.div_ceil(*c))
                    .collect();
                let item = match dtype {
                    "bool" | "int8" | "uint8" => 1,
                    "int16" | "uint16" => 2,
                    "int32" | "uint32" | "float32" => 4,
                    _ => 8,
                };
                let size = chunks.iter().product::<u64>() * item;
                for f in walk(&dir.join("c")) {
                    let rel = f.strip_prefix(dir.join("c")).unwrap();
                    let idx: Vec<u64> = rel
                        .components()
                        .map(|c| c.as_os_str().to_str().unwrap().parse().unwrap())
                        .collect();
                    assert_eq!(idx.len(), rank, "{}", f.display());
                    assert!(
                        idx.iter().zip(&counts).all(|(i, n)| i < n),
                        "{}",
                        f.display()
                    );
                    assert_eq!(
                        std::fs::metadata(&f).unwrap().len(),
                        size,
                        "{}",
                        f.display()
                    );
                }
            }
            other => panic!("{at}: node_type {other:?}"),
        }
    }
}

fn collect_meta(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            collect_meta(&p, out);
        } else if p.file_name().unwrap() == "zarr.json" {
            out.push(p);
        }
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd {
            let p = e.unwrap().path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}

fn fixtures() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = walk(&data_dir())
        .into_iter()
        .filter(|p| p.file_name().unwrap() != "README.md")
        .collect();
    v.sort();
    v
}

#[test]
fn zarr_round_trip_and_v3_schema_every_fixture() {
    let mut exported = 0;
    for path in fixtures() {
        let Ok(file) = File::open(&path) else {
            continue;
        };
        let name = path
            .strip_prefix(data_dir())
            .unwrap()
            .display()
            .to_string()
            .replace('/', "_");
        let root = out_dir(&name);
        let summary = export(&file, &DirSink::new(&root), &ZarrOptions::default()).unwrap();
        assert_v3_schema(&root);
        assert_round_trip(&name, &file, &root);
        assert_eq!(
            *summary.chunks.last().unwrap(),
            u64::from(file.layout().n_samples.max(1))
        );
        exported += 1;
    }
    assert!(exported > 40, "{exported}");
}

#[test]
fn test_x14_default_chunks_follow_disk_order() {
    let file = File::open(data_dir().join("small.sgy")).unwrap();
    assert_eq!(file.geometry().kind, GeometryKind::InlineMajor);
    let ns = u64::from(file.layout().n_samples);
    let trace = ns * 4;
    let two = ZarrOptions {
        target_chunk_bytes: 2 * trace,
        ..ZarrOptions::default()
    };
    let root = out_dir("x14-small");
    let s = export(&file, &DirSink::new(&root), &two).unwrap();
    // Whole traces, grouped along crossline, the fast on-disk axis.
    assert_eq!(s.dims, ["inline", "crossline", "time"]);
    assert_eq!(s.chunks, vec![1, 2, ns]);
    assert_round_trip("x14-small", &file, &root);

    // Default budget: the fast axis fills first, then the slow one.
    let root = out_dir("x14-small-default");
    let s = export(&file, &DirSink::new(&root), &ZarrOptions::default()).unwrap();
    assert_eq!(s.chunks, vec![5, 5, ns]);

    // The same cube stored crossline-major groups along inline instead.
    let src = std::fs::read(data_dir().join("small.sgy")).unwrap();
    let (start, stride) = (file.layout().data_start as usize, file.stride() as usize);
    let mut xmajor = src[..start].to_vec();
    for x in 0..5 {
        for i in 0..5 {
            let t = i * 5 + x;
            xmajor.extend_from_slice(&src[start + t * stride..start + (t + 1) * stride]);
        }
    }
    let xpath = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("x14-xmajor.sgy");
    std::fs::write(&xpath, &xmajor).unwrap();
    let xm = File::open(&xpath).unwrap();
    assert_eq!(xm.geometry().kind, GeometryKind::CrosslineMajor);
    let root = out_dir("x14-xmajor");
    let s = export(&xm, &DirSink::new(&root), &two).unwrap();
    assert_eq!(s.dims, ["inline", "crossline", "time"]);
    assert_eq!(s.chunks, vec![2, 1, ns]);
    assert_round_trip("x14-xmajor", &xm, &root);

    // Pre-stack, inline-major with offset fastest: offset fills, then crossline.
    let ps = File::open(data_dir().join("small-ps-dec-xl-inc-il-off.sgy")).unwrap();
    let ns = u64::from(ps.layout().n_samples);
    let four = ZarrOptions {
        target_chunk_bytes: 4 * ns * 4,
        ..ZarrOptions::default()
    };
    let root = out_dir("x14-ps");
    let s = export(&ps, &DirSink::new(&root), &four).unwrap();
    assert_eq!(s.dims, ["inline", "crossline", "offset", "time"]);
    assert_eq!(ps.geometry().offsets.len(), 2);
    assert_eq!(s.chunks, vec![1, 2, 2, ns]);
    assert_round_trip("x14-ps", &ps, &root);

    // Rechunk only when asked, including along samples.
    let root = out_dir("x14-rechunk");
    let ask = ZarrOptions {
        chunks: Some(vec![3, 2, 1, 4]),
        ..ZarrOptions::default()
    };
    let s = export(&ps, &DirSink::new(&root), &ask).unwrap();
    assert_eq!(s.chunks, vec![3, 2, 1, 4]);
    assert_v3_schema(&root);
    assert_round_trip("x14-rechunk", &ps, &root);
    let bad = ZarrOptions {
        chunks: Some(vec![1, 1]),
        ..ZarrOptions::default()
    };
    assert!(export(&ps, &DirSink::new(out_dir("x14-bad")), &bad).is_err());

    // A crossline read from the store equals the matching traces in the file.
    let data = ZarrArray::open(&root, "data").unwrap();
    let all = data.read_all().unwrap();
    let sh: Vec<usize> = data.shape.iter().map(|v| *v as usize).collect();
    let (ils, xls, offs) = (
        i64s(&root, "inline"),
        i64s(&root, "crossline"),
        i64s(&root, "offset"),
    );
    let tb = sh[3] * data.itemsize();
    for (x, xl) in xls.iter().enumerate() {
        for (i, il) in ils.iter().enumerate() {
            for (o, off) in offs.iter().enumerate() {
                let cell = (i * sh[1] + x) * sh[2] + o;
                let t = (0..ps.layout().n_traces as usize)
                    .find(|&t| {
                        i64::from(ps.crossline()[t]) == *xl
                            && i64::from(ps.inline()[t]) == *il
                            && i64::from(ps.offset()[t]) == *off
                    })
                    .unwrap();
                assert_eq!(
                    &all[cell * tb..(cell + 1) * tb],
                    &le(&ps.decode(t, 1).unwrap())[..]
                );
            }
        }
    }
}

#[test]
fn test_x34_missing_chunk_reads_fill() {
    // small.sgy without trace 7: a 5 x 5 grid with one hole.
    let src = std::fs::read(data_dir().join("small.sgy")).unwrap();
    let probe = File::open(data_dir().join("small.sgy")).unwrap();
    let start = probe.layout().data_start as usize;
    let stride = probe.stride() as usize;
    let mut bytes = src[..start + 7 * stride].to_vec();
    bytes.extend_from_slice(&src[start + 8 * stride..]);
    let sparse_path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("x34-sparse.sgy");
    std::fs::write(&sparse_path, &bytes).unwrap();
    let file = File::open(&sparse_path).unwrap();
    assert_eq!(file.geometry().kind, GeometryKind::Sparse);
    let (hole_il, hole_xl) = (probe.inline()[7], probe.crossline()[7]);

    let root = out_dir("x34");
    let one = ZarrOptions {
        target_chunk_bytes: 1,
        ..ZarrOptions::default()
    };
    let s = export(&file, &DirSink::new(&root), &one).unwrap();
    assert_eq!(s.chunks_missing, 1);
    assert_eq!(s.chunks_written, 24);
    assert_v3_schema(&root);
    assert_round_trip("x34", &file, &root);

    let il = i64s(&root, "inline")
        .iter()
        .position(|v| *v == i64::from(hole_il))
        .unwrap() as u64;
    let xl = i64s(&root, "crossline")
        .iter()
        .position(|v| *v == i64::from(hole_xl))
        .unwrap() as u64;
    assert!(!root.join(format!("data/c/{il}/{xl}/0")).exists());
    let data = ZarrArray::open(&root, "data").unwrap();
    let chunk = data.read_chunk(&[il, xl, 0]).unwrap();
    assert!(chunk.iter().all(|b| *b == 0));
    let mask = ZarrArray::open(&root, "trace_mask").unwrap();
    assert_eq!(mask.read_chunk(&[il, xl]).unwrap(), vec![0]);
    assert_eq!(mask.read_chunk(&[0, 0]).unwrap(), vec![1]);
}

#[cfg(feature = "cloud")]
#[test]
fn zarr_export_from_remote_reader_and_to_object_store() {
    use object_store::memory::InMemory;
    use object_store::path::Path as StorePath;
    use object_store::{ObjectStore, ObjectStoreExt};
    use segyrs_interop::cloud::ObjectReader;
    use segyrs_interop::zarr::ObjectSink;
    use std::sync::Arc;

    let path = data_dir().join("small-ps.sgy");
    let local = File::open(&path).unwrap();
    let remote = ObjectReader::open(path.to_str().unwrap(), [("block_size", "4096")])
        .unwrap()
        .open_segy(&segyrs::OpenOptions::default())
        .unwrap();
    let (a, b) = (out_dir("remote-a"), out_dir("remote-b"));
    export(&local, &DirSink::new(&a), &ZarrOptions::default()).unwrap();
    export(&remote, &DirSink::new(&b), &ZarrOptions::default()).unwrap();
    let files = |root: &Path| -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
        walk(root)
            .into_iter()
            .map(|p| {
                (
                    p.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(&p).unwrap(),
                )
            })
            .collect()
    };
    let want = files(&a);
    assert_eq!(want, files(&b));

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let sink = ObjectSink::new(Arc::clone(&store), StorePath::from("out.zarr"));
    export(&remote, &sink, &ZarrOptions::default()).unwrap();
    for (rel, bytes) in &want {
        let key = StorePath::from(format!("out.zarr/{}", rel.display()));
        let got = futures::executor::block_on(async {
            store.get(&key).await.unwrap().bytes().await.unwrap()
        });
        assert_eq!(&got[..], &bytes[..], "{}", rel.display());
    }
}
