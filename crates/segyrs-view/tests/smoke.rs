//! Headless smoke tests. No test opens a window. The GPU comparison runs only
//! when wgpu finds an adapter and says so when it skips.

use std::path::PathBuf;
use std::sync::Arc;

use segyrs::OpenOptions;
use segyrs_draw::{lut_index, Colormap, Dim, Polarity, Source, Style, Wiggle};
use segyrs_view::state::{planes, ViewState, Zoom};
use segyrs_view::{check, Region, ViewError, ViewOptions};

fn small() -> Arc<Source> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-data/small.sgy");
    Arc::new(Source::open(p, &OpenOptions::default()).unwrap())
}

#[test]
fn planes_and_stepping() {
    let src = small();
    let p = planes(src.dims());
    let names: Vec<String> = p.iter().map(|k| k.name()).collect();
    assert_eq!(names, ["inline section", "crossline section", "time slice"]);
    let mut st = ViewState::new(src.clone(), &[], Style::default());
    assert_eq!(st.fixed(), vec![(Dim::Inline, 0)]);
    st.step(-3);
    assert_eq!(st.index_of(Dim::Inline), 0);
    st.step(99);
    assert_eq!(st.index_of(Dim::Inline), 4);
    st.set_plane(2);
    assert_eq!(st.fixed(), vec![(Dim::Time, 0)]);
    st.zoom.zoom(0.5, 0.0, 0.0);
    let spec = st.spec((100, 100));
    assert_eq!(spec.x_range, Some(0..3));
    st.load((100, 100));
    let s = st.section.as_ref().unwrap();
    assert_eq!(
        s.traces_read, 9,
        "a zoomed slice reads only the traces in view"
    );
    assert!(st
        .readout(0, 0)
        .unwrap()
        .starts_with("inline 1, crossline 20"));
}

#[test]
fn bad_selection_fails_before_any_window() {
    let src = small();
    let o = ViewOptions {
        fixed: vec![(Dim::Inline, 9)],
        ..Default::default()
    };
    assert!(matches!(
        segyrs_view::run(src.clone(), o),
        Err(ViewError::Draw(segyrs_draw::Error::Index { .. }))
    ));
    let o = ViewOptions {
        fixed: vec![(Dim::Inline, 0), (Dim::Crossline, 0)],
        ..Default::default()
    };
    let err = check(&src, &o).unwrap_err();
    assert!(err.to_string().contains("cannot be drawn"), "{err}");
}

/// `segyrs.view(vol)` starts on the selection: zoomed to its window, read
/// one position in its step, with `wiggle_every` thinning a wiggled section.
#[test]
fn starts_on_the_selection_region_and_wiggle_every() {
    let src = small();
    let style = Style {
        wiggle: Wiggle::Line,
        wiggle_every: 2,
        ..Style::default()
    };
    let region = Region {
        x: 1..5,
        y: 5..41,
        x_step: 1,
        y_step: 3,
    };
    let o = ViewOptions {
        fixed: vec![(Dim::Inline, 2)],
        region: Some(region.clone()),
        style,
        ..Default::default()
    };
    check(&src, &o).unwrap();
    let mut st = ViewState::new(src.clone(), &o.fixed, style);
    st.set_region(&region);
    let spec = st.spec((800, 480));
    assert_eq!(spec.fixed, vec![(Dim::Inline, 2)]);
    assert_eq!((spec.x_range, spec.y_range), (Some(1..5), Some(5..41)));
    assert_eq!((spec.x_step, spec.y_step), (2, 3));
    st.load((800, 480));
    let s = st.section.as_ref().unwrap();
    assert_eq!((s.width, s.height), (2, 12));
    assert_eq!(s.traces_read, 2);
    // Another plane drops the steps; a slice ignores wiggle_every.
    st.set_plane(2);
    let spec = st.spec((800, 480));
    assert_eq!((spec.x_step, spec.y_step), (1, 1));
}

#[test]
fn a_region_outside_the_plane_fails_before_any_window() {
    let src = small();
    let o = ViewOptions {
        fixed: vec![(Dim::Inline, 0)],
        region: Some(Region {
            x: 0..6,
            y: 0..10,
            x_step: 1,
            y_step: 1,
        }),
        ..Default::default()
    };
    assert!(matches!(
        segyrs_view::run(src, o),
        Err(ViewError::Draw(segyrs_draw::Error::Window { .. }))
    ));
}

#[test]
fn zoom_fractions_round_trip_to_the_same_positions() {
    for n in 1..1200usize {
        for (a, b) in [(0, n), (n / 3, n - n / 5), (n - 1, n), (n / 7, n / 7 + 1)] {
            if a < b {
                assert_eq!(Zoom::range(Zoom::fractions(&(a..b), n), n), a..b, "n {n}");
            }
        }
    }
}

#[cfg(feature = "window")]
mod window {
    use super::*;
    use eframe::egui;
    use eframe::wgpu;
    use segyrs_view::app::SectionApp;
    use segyrs_view::gpu::{uniforms, Gpu, Grid, ABSENT, SHADER};

    fn input(events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1000.0, 700.0),
            )),
            events,
            ..Default::default()
        }
    }

    fn frame<R: segyrs::ReadAt + 'static>(
        ctx: &egui::Context,
        app: &mut SectionApp<R>,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        let mut out = ctx.run_ui(input(events), |ui| app.frame_ui(ui));
        out.textures_delta.clear();
        out
    }

    /// The whole window UI runs for real on a bare egui context: toolbar,
    /// paint callback, axes, readout. Arrow keys step the plane.
    #[test]
    fn headless_frames_step_and_paint() {
        let src = small();
        let state = ViewState::new(src.clone(), &[], Style::default());
        let mut app = SectionApp::new(state, None, 8192);
        let ctx = egui::Context::default();
        let out = frame(&ctx, &mut app, vec![]);
        let s = app
            .state
            .section
            .clone()
            .expect("first frame reads the plane");
        assert_eq!(s.fixed[0].index, 0);
        assert_eq!((s.x.dim, s.y.dim), (Dim::Crossline, Dim::Time));
        let callbacks = out
            .shapes
            .iter()
            .filter(|c| matches!(c.shape, egui::Shape::Callback(_)))
            .count();
        assert_eq!(callbacks, 1, "the section is one wgpu paint callback");

        let key = egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Default::default(),
        };
        frame(&ctx, &mut app, vec![key]);
        frame(&ctx, &mut app, vec![]);
        let s = app.state.section.clone().unwrap();
        assert_eq!(s.fixed[0].index, 1, "right arrow steps the inline");
        assert!(app.state.error.is_none());

        // An interrupt (Ctrl-C from Python) closes the window.
        app.interrupt = Some(segyrs_view::Interrupt(Arc::new(|| true)));
        let out = frame(&ctx, &mut app, vec![]);
        let closes = out.viewport_output.values().any(|v| {
            v.commands
                .iter()
                .any(|c| matches!(c, egui::ViewportCommand::Close))
        });
        assert!(closes);
    }

    /// The shader parses and validates with naga, with or without a GPU.
    #[test]
    fn shader_validates_without_a_gpu() {
        use wgpu::naga;
        let module = naga::front::wgsl::parse_str(SHADER).expect("wgsl parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("wgsl validates");
    }

    /// With an adapter, the GPU colours four cells exactly like the CPU LUT.
    #[test]
    fn gpu_matches_cpu_lut_when_an_adapter_exists() {
        if !segyrs_view::probe_adapter() {
            eprintln!("no wgpu adapter on this machine: GPU comparison skipped");
            return;
        }
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut gpu = Gpu::new(&device, &queue, format, 1);
        let grid = Grid {
            width: 4,
            height: 1,
            values: vec![-1.0, 0.0, 0.5, ABSENT],
            version: 1,
        };
        for (cmap, pol) in [
            (Colormap::Grey, Polarity::Normal),
            (Colormap::Seismic, Polarity::Reversed),
        ] {
            let style = Style {
                colormap: cmap,
                polarity: pol,
                ..Style::default()
            };
            gpu.upload(&device, &queue, &grid);
            gpu.write_uniforms(&queue, uniforms(1.0, &style, false));
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 4,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&Default::default());
            let out = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                gpu.draw(&mut pass);
            }
            enc.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &out,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                target.size(),
            );
            queue.submit([enc.finish()]);
            out.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let px = out.slice(..).get_mapped_range().unwrap().to_vec();
            for (i, v) in [-1.0f32, 0.0, 0.5].iter().enumerate() {
                let want = cmap.lut()[lut_index(*v, 1.0, pol).unwrap() as usize];
                let got = &px[i * 4..i * 4 + 4];
                for ch in 0..3 {
                    assert!(
                        (i32::from(got[ch]) - i32::from(want[ch])).abs() <= 1,
                        "{cmap:?} cell {i}: gpu {got:?} cpu {want:?}"
                    );
                }
            }
            let hatch = &px[12..15];
            assert!(
                hatch == [205, 205, 205] || hatch == [242, 242, 242],
                "absent cell {hatch:?}"
            );
            out.unmap();
        }
    }
}

#[test]
fn window_errors_drop_the_build_path_of_the_dependency() {
    let raw = "os error at /home/ci/.cargo/registry/src/index.crates.io-0/winit-0.30.13/src/platform_impl/linux/mod.rs:765: neither WAYLAND_DISPLAY nor WAYLAND_SOCKET nor DISPLAY is set.";
    assert_eq!(
        segyrs_view::without_source_location(raw),
        "neither WAYLAND_DISPLAY nor WAYLAND_SOCKET nor DISPLAY is set"
    );
    assert_eq!(
        segyrs_view::without_source_location("EventLoop can't be recreated"),
        "EventLoop can't be recreated"
    );
}
