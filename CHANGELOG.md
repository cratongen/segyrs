# Changelog

segyrs follows [semantic versioning](https://semver.org/). Each release lists
what changed for someone who uses the Python package.

## 0.1.0 (unreleased)

First public release.

- Licensed under Apache-2.0 only, with one `LICENSE` file. The unreleased
  tree was MIT OR Apache-2.0 until 2026-11-21.

- Read and write SEG-Y rev 0, 1, 2.0 and 2.1 with a labelled, lazy `Volume`
  (`segyrs.open`, `sel`, `isel`, header columns, geometry report).
- Write and edit in place (`segyrs.create`, `Volume.to_segy`, `segyrs.edit`,
  `segyrs.validate`), keeping the source sample format.
- Streaming whole-file work: `resort`, `filter`, `slice_horizon`,
  `to_regular`, `concat`, `stats`, `clip`, `rms`.
- Pictures drawn in Rust: `Volume.plot` and the notebook image.
- One abi3 wheel per platform for CPython 3.11 and newer.
- Experimental object-store reads (`segyrs.open` on `s3://`, `gs://`, `az://`
  and `http(s)://` URLs) and Zarr v3 export (`Volume.to_zarr`), built into
  the segyrs wheel. segyrs is the only package of this release and has no
  extras.
- The native viewer window (`segyrs.view`) is not in this release on PyPI. It
  builds from the repository (`mise run build`) and will be published later;
  without it, `segyrs.view` raises `ImportError` that says so. The segyrs
  wheel links no window or GPU crate.
