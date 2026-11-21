# Viewer and notebook

segyrs draws its own pictures. There is no matplotlib, no xarray and no dask. Two crates do it:

| Crate | What it holds | Window dependencies |
| --- | --- | --- |
| `segyrs-draw` | Plane read, binning to the pixel grid, symmetric p98 clip, colormaps, ticks, wiggle and variable area, PNG, the header map | none |
| `segyrs-view` | The native window: eframe on wgpu, with a softbuffer CPU fallback | behind the `window` feature (on by default) |

`crates/segyrs` (the reader) depends on neither.

## One picture of one plane

```rust
use segyrs::OpenOptions;
use segyrs_draw::{picture, Dim, PlaneSpec, Source, Style};

let src = Source::open("test-data/small.sgy", &OpenOptions::default())?;
let spec = PlaneSpec::new(vec![(Dim::Inline, 2)], 800, 480); // inline position 2, plot area 800x480
let pic = picture(&src, &spec, &Style::default())?;
std::fs::write("inline.png", &pic.png)?;
println!("{}", pic.text); // what was drawn, the clip and where it came from
```

`cargo run -p segyrs-draw --example section -- FILE OUT.png [DIM INDEX | map] [grey|seismic|amber|wiggle|area]` does the same from the command line (an example, not a CLI).

Dims are `inline`, `crossline`, `offset` (only on a pre-stack grid), then `time` (or `depth` with `Source::depth(true)`). A file that is not a complete sorted grid has dims `trace` and `time`. A plane fixes every dim except two, by position (`isel`). The first free dim is the image x axis, the second is y.

| Fixed | Picture |
| --- | --- |
| `inline` | crossline by time section |
| `crossline` | inline by time section |
| `time` | inline by crossline slice, one sample from every trace |
| `inline`, `crossline` (pre-stack) | offset by time gather |
| nothing, on a `(trace, time)` file | the whole line |

## What is read

`Source::open` maps the file and lets the core scan the trace headers once. No sample byte is read then. `Source::from_file` draws from a core `File` that is already open, through any owner (`Box`, `Arc`, or the Python `File`): its header columns and trace index are reused, and only trace 0's header is read again, for the recording delay (`from_file_reuses_the_open_scan_and_index`). Sample windows are read through that File's reader, so an object-store file draws the same way as a mapped one. A grid cell finds its trace through the core's `(inline, crossline)` index (`File::trace_index`), the same index `sel` uses, so the picture and the Volume agree on which trace sits where.

`Source::section` reads only the traces of the plane, and only the samples inside the vertical window. A time or depth slice reads one sample per trace. Traces are decoded on rayon straight to f32. The test `test_s06_one_plane_png_reads_only_that_plane` puts a read counter under the source and checks that every byte read lies inside that plane's samples, each read once.

The float grid is never larger than the plot area. When more traces or samples than pixels fall in one cell, the cell keeps the sample of largest magnitude, so a thin peak survives zooming out (`test_x09_bins_to_pixel_grid_once`). When fewer, the image stretches each cell into a block (nearest). The window can also interpolate (linear); the PNG does not.

The displayed value is f32. int32, int64, uint32, uint64 and float64 samples that f32 cannot hold are rounded for display only. The window's cursor readout reads the exact file value of one sample when the cell is one sample of one trace.

## Clip

With no clip set, the clip is the symmetric 98th percentile (nearest rank) of `|v|` over the finite values of the drawn grid. Only the drawn view is used; the rest of the file is not read. NaN and infinities never enter the ranking. The colour range is `-clip` to `+clip`, divided by the gain.

The p98 is taken over the binned grid, so on a zoomed-out view it follows max-abs bins, not every raw sample. The title says `p98 of view` to make that visible. A clip passed by the caller (`Style::clip`, Python `vmin`/`vmax`) reads exactly the same bytes.

Dead traces follow roadmap decision 7: every stored trace is live unless the caller passes a live mask (`PlaneSpec::live`, the Volume's `trace_mask`). `Source::trace_id_mask()` builds one from trace identification code 2. A masked trace is not read, stays out of the clip, and draws as a grey hatch (`test_x12_clip_ignores_dead_and_non_finite`). Non-finite samples also draw as the hatch.

## Axes

Default axes are the dims: crossline and time for an inline, never a 2D coordinate such as `cdp_x` (`test_x13_default_axes_are_dims_map_reads_no_samples`). Time is the recording delay of trace 0 (with the time scalar of bytes 215-216) plus the index times the header sample interval, in milliseconds.

`PlaneSpec::x_coord` labels a section's x axis with a header coordinate instead (`cdp_x`, `cdp_y`, `inline`, `crossline`, `offset`). If that axis is not monotonic, the call fails with `UnsortedCoordinate` and reads nothing. With `sort: true` the columns are sorted together with their coordinate (`test_x11_unsorted_axis_sorts_or_raises`).

`map_image` draws `cdp_x` against `cdp_y` (scalars applied), one dot per trace, equal scale. It uses the header columns from the open and reads nothing more.

## Colour

Three LUTs are written in this repository, 256 entries each: `grey` (default), `seismic` (dark blue, blue, white at zero, red, dark red) and `amber` (sequential, rising lightness, for one-sided attributes). None is claimed to be perceptually uniform. Polarity reversed flips the sign before the LUT. The window uploads the same LUT bytes, and `gpu_matches_cpu_lut_when_an_adapter_exists` checks that the GPU colours match the CPU within one level.

Wiggles (`Wiggle::Line`) and variable area (`Wiggle::Area`, positive lobes filled) are drawn with tiny-skia only when every trace owns at least 3 pixels. Below that they would merge into a black box, so only density is painted, and the fallback is never silent: `Picture::note` (also in `Picture::text` and on the image under the title) says, for example, `wiggle needs >= 3 px per trace (1000 traces on 800 px); showing density; try wiggle_every=4`. A wiggle asked for on a time slice says `wiggle needs time or depth free`. `xcur` is the excursion at the clip in trace spacings.

`Style::wiggle_every = N` (Python `wiggle_every=N`) draws one trace in N so a dense section still shows wiggles. `picture` reads only those traces (`PlaneSpec::x_step`), so the density under the wiggles is decimated too, and the note says `wiggle_every=N: one trace in N`. `picture` refuses it on a time or depth slice. The window applies it to a section while a wiggle is drawn (`ViewState::spec`) and ignores it on a slice.

## Stepped windows

`PlaneSpec::x_step` and `y_step` keep one position in N along each image axis, counted from the start of `x_range` and `y_range`, on sections and on slices. A section reads the vertical window of each kept trace in one piece and keeps every `y_step`-th sample; a slice reads only the kept traces (`stepped_section_reads_every_step_th_position_and_sample`, `stepped_slice_reads_every_step_th_trace`). `wiggle_every` multiplies `x_step`.

Tick labels use a 5x7 bitmap font drawn for segyrs, so the draw crate needs no font file.

## Notebook image

`segyrs_draw::notebook(&source, &fixed, clip, live)` returns PNG bytes and a `text/plain` line for a 2D selection. The plot area is capped at 800 by 480 pixels (`NOTEBOOK_PIXELS`), a product limit so a notebook does not embed a full-resolution section. A selection with more than two free dims fails with `NotTwoD`, whose message names the `sel` calls that make a plane, and reads nothing (`test_x10_cube_refuses_and_names_sel`). `Source::default_fixed()` gives the plane roadmap decision 5 describes: the first index of the slowest grid axis (and of offset on a pre-stack grid).

## Window

`segyrs_view::run(Arc<Source>, ViewOptions)` opens one window on the calling thread and blocks until it closes. `cargo run -p segyrs-view --example view -- FILE [gpu|cpu]` opens it on a file.

- With a wgpu adapter: eframe with the wgpu renderer. The float grid is an R32Float texture read with `textureLoad`, so no float-filtering feature is required. The LUT, clip, gain, polarity and colormap are a uniform. Changing them rereads nothing; a new plane, index or zoom reads the plane again, on a worker thread.
- With no adapter, or `SEGYRS_VIEW_RENDERER=cpu`: a winit window that presents the draw crate's image with softbuffer. Keys: left and right step, page up and down step by 10, tab changes the plane, `p` polarity, `c` colormap, `w` wiggle, `+` and `-` gain, `a` back to the p98 clip, `r` reset zoom, escape closes. The title shows the cursor readout.
- The toolbar holds the plane (inline section, crossline section, time slice, gather), an index stepper per fixed dim, the clip (p98 or a value), gain, polarity, colormap, wiggle, nearest or linear, and reset zoom. The wheel zooms around the cursor, dragging pans, and a double click resets. The status line is the cursor readout.

The event loop is created once per process and reused, so `run` can be called again after a window closes. It has to be the main thread. On a machine with no display `run` returns `ViewError::NoDisplay`; the notebook image is the path there. `ViewOptions::interrupt` is polled about five times a second and closes the window when it returns true.

## Python

The notebook picture is in the segyrs wheel: `crates/segyrs-py/src/view.rs` (class `_segyrs.Drawer`, one per opened file) and `python/segyrs/_view.py`. A `Drawer` wraps the Volume's own `_segyrs.File` (`Source::from_file`), so drawing never scans the headers a second time, and a Volume opened from a URL draws through its object-store reader (`test_remote_volume_draws`).

The window is experimental and is not in the segyrs wheel. It is the separate `segyrs-view` package (`crates/segyrs-view-py`, module `_segyrs_view`), built with the `window` feature. It is not part of the 0.1.0 release on PyPI and will be published later. From a checkout, `mise run build` installs segyrs and the viewer into `.venv`:

```bash
mise install
mise run build
```

Without it, `segyrs.view` raises `ImportError` that says the viewer is not in the 0.1.0 release on PyPI and names `mise run build`. `Volume.plot` and the notebook image need no viewer. The extension opens its own `segyrs::File` on the Volume's reader, which `_segyrs.File.reader()` hands over as a capsule (`segyrs::bridge`), so it reads the same map or the same remote cache. It scans the trace headers once more when the window opens.

- `Volume.plot(**opts)` returns a `segyrs.Picture` (`png`, `text`, `note`, `traces_read`). The Volume's selection becomes positions in the opened file: dims selected away are `PlaneSpec::fixed`, kept dims are `x_range` and `y_range`, and a kept dim selected with a step (`isel(crossline=slice(0, 1000, 10))`, or a list with a constant step) becomes `x_step` or `y_step`, so only those traces and samples are read. A Volume with more than two free dims is fixed with `Source::default_fixed()` at the start of its own selection (roadmap decision 5). With no `vmin`/`vmax` the clip is the p98 of the drawn view, otherwise `max(|vmin|, |vmax|)`. The Volume's `trace_mask` is `live` when the file was opened with `dead_trace`.
- `Volume._repr_mimebundle_` returns `image/png` and `text/plain` from the same call at `NOTEBOOK_PIXELS`. A stepped selection draws decimated. A selection that is not a plane (one dim, or a kept dim that is not ascending with a constant step, such as `[0, 1, 3]` or a reversed slice) returns `text/plain` only, with the reason.
- `segyrs.view(volume_or_path, ...)` calls `segyrs_view::run` with the plane's `fixed`, the clip, `live` and the style, with the interpreter detached. It starts on the selection: `ViewOptions::region` holds the window and step of the two free dims, so the window opens zoomed to the selected window and reads one position in each step until the plane changes. `wiggle_every=N` (which implies `wiggle="line"`) thins a wiggled section. A scattered selection raises `ValueError` before any window. `ViewOptions::interrupt` re-attaches to run `check_signals`, so Ctrl-C closes the window and raises `KeyboardInterrupt`. `ViewError::NoDisplay` and every other `ViewError` raise `SegyrsError`.

`test_s06_quick_start_inline_reads_only_that_inline` (python/tests/test_view.py) runs the README quick start on `test-data/small.sgy`: the picture reads only that inline's traces (`Picture.traces_read`) and the Volume's own read counter stays at zero.

## Performance

Not measured. No speed figure in this page comes from a benchmark, and none should be quoted until one is run on this machine.
