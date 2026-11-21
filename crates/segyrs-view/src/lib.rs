//! Native window for segyrs sections.
//!
//! [`run`] opens one window on the calling (main) thread and blocks until it
//! closes. It uses eframe with the wgpu renderer when an adapter exists: the
//! float grid is an R32Float texture and the LUT, clip and polarity live in
//! the shader. With no adapter it presents the draw crate's RGBA image with
//! softbuffer. Both paths share [`state::ViewState`], so they show the same
//! planes. Every window reuses one event loop, so `run` can be called again
//! after a window closes.
//!
//! Without the `window` feature only the state is built and [`run`] returns
//! [`ViewError::Disabled`].

pub mod state;

#[cfg(feature = "window")]
pub mod app;
#[cfg(feature = "window")]
pub mod gpu;
#[cfg(feature = "window")]
pub mod soft;

use std::ops::Range;
use std::sync::Arc;

use segyrs::ReadAt;
use segyrs_draw::{Dim, Source, Style};

/// Which presenter to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Renderer {
    /// wgpu when an adapter exists, otherwise the CPU window. The environment
    /// variable `SEGYRS_VIEW_RENDERER=cpu` forces the CPU window.
    #[default]
    Auto,
    Gpu,
    Cpu,
}

/// What `segyrs.view(volume, ...)` passes down.
#[derive(Clone, Debug, Default)]
pub struct ViewOptions {
    /// Starting plane as positional indices of the fixed dims. Empty: the
    /// notebook default (first index of the slowest grid axis).
    pub fixed: Vec<(Dim, usize)>,
    /// Starting window of the two free dims. `None`: the whole plane.
    pub region: Option<Region>,
    /// Starting style. `clip: Some(..)` is the caller's `vmin`/`vmax` magnitude.
    /// `wiggle_every` decimates the traces of a section while a wiggle is on.
    pub style: Style,
    /// Per-trace live mask (the Volume's `trace_mask`). `None`: all live.
    pub live: Option<Arc<[bool]>>,
    /// Window title; the file name is a good one.
    pub title: String,
    pub renderer: Renderer,
    /// Polled about five times a second while the window is open; `true`
    /// closes it. The Python glue re-attaches and runs `check_signals` here,
    /// so Ctrl-C closes the window.
    pub interrupt: Option<Interrupt>,
}

/// The part of the starting plane to show first: positions along the image
/// x and y dims (the free dims in volume order) and the step along each. The
/// window opens zoomed to it; the steps hold until the plane changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub x: Range<usize>,
    pub y: Range<usize>,
    pub x_step: usize,
    pub y_step: usize,
}

/// A cheap check the window polls; see [`ViewOptions::interrupt`].
#[derive(Clone)]
pub struct Interrupt(pub Arc<dyn Fn() -> bool + Send + Sync>);

impl std::fmt::Debug for Interrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Interrupt")
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ViewError {
    #[error("segyrs-view was built without the `window` feature")]
    Disabled,

    /// No display server, or the event loop could not start. A remote kernel
    /// has no display: use the notebook image there.
    #[error("cannot open a window ({0}); on a remote or headless machine use the notebook image instead")]
    NoDisplay(String),

    /// The event loop must live on the process main thread.
    #[error("the viewer must run on the main thread: {0}")]
    Thread(String),

    #[error(transparent)]
    Draw(#[from] segyrs_draw::Error),

    #[error("window: {0}")]
    Window(String),
}

/// A window-system error without the build path and line that winit puts in
/// front of an OS error (`os error at /path/to/src/file.rs:765: message`).
pub fn without_source_location(message: &str) -> String {
    let Some(rest) = message.strip_prefix("os error at ") else {
        return message.trim_end_matches('.').to_string();
    };
    let tail = rest.find(".rs:").and_then(|i| {
        let after = &rest[i + 4..];
        let digits = after.len() - after.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        after[digits..].strip_prefix(": ")
    });
    tail.unwrap_or(message).trim_end_matches('.').to_string()
}

/// Check the starting selection without reading samples or opening a window.
pub fn check<R: ReadAt + 'static>(
    source: &Source<R>,
    options: &ViewOptions,
) -> Result<(), ViewError> {
    let dims = source.dims();
    for (d, i) in &options.fixed {
        let len = source.len_of(*d).ok_or(segyrs_draw::Error::NoDim {
            dim: *d,
            dims: dims.to_vec(),
        })?;
        if *i >= len {
            return Err(segyrs_draw::Error::Index {
                dim: *d,
                index: *i,
                len,
            }
            .into());
        }
    }
    if !options.fixed.is_empty() && dims.len() - options.fixed.len() != 2 {
        let free = dims
            .iter()
            .copied()
            .filter(|d| !options.fixed.iter().any(|(f, _)| f == d))
            .collect();
        return Err(segyrs_draw::Error::NotTwoD {
            free,
            dims: dims.to_vec(),
        }
        .into());
    }
    if let Some(w) = &options.region {
        let fixed = if options.fixed.is_empty() {
            source.default_fixed()
        } else {
            options.fixed.clone()
        };
        let free: Vec<Dim> = dims
            .iter()
            .copied()
            .filter(|d| !fixed.iter().any(|(f, _)| f == d))
            .collect();
        for (dim, range, step) in [(free[0], &w.x, w.x_step), (free[1], &w.y, w.y_step)] {
            let len = source.len_of(dim).unwrap_or(0);
            if range.start >= range.end || range.end > len || step == 0 {
                return Err(segyrs_draw::Error::Window {
                    dim,
                    len,
                    range: range.clone(),
                }
                .into());
            }
        }
    }
    if let Some(live) = &options.live {
        let want = source.file().layout().n_traces as usize;
        if live.len() != want {
            return Err(segyrs_draw::Error::Mask {
                got: live.len(),
                want,
            }
            .into());
        }
    }
    Ok(())
}

/// Open a window on `source` and block until it is closed. Returns the
/// presenter that ran.
#[cfg(not(feature = "window"))]
pub fn run<R: ReadAt + 'static>(
    source: Arc<Source<R>>,
    options: ViewOptions,
) -> Result<Renderer, ViewError> {
    check(&source, &options)?;
    Err(ViewError::Disabled)
}

/// Open a window on `source` and block until it is closed. Returns the
/// presenter that ran.
#[cfg(feature = "window")]
pub fn run<R: ReadAt + 'static>(
    source: Arc<Source<R>>,
    options: ViewOptions,
) -> Result<Renderer, ViewError> {
    window::run(source, options)
}

/// True when wgpu finds an adapter. Creates no window and no surface.
#[cfg(feature = "window")]
pub fn probe_adapter() -> bool {
    use eframe::wgpu;
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).is_ok()
}

#[cfg(feature = "window")]
mod window {
    use std::cell::{Cell, RefCell};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;
    use std::sync::Arc;

    use eframe::egui;
    use eframe::UserEvent;
    use segyrs::ReadAt;
    use segyrs_draw::Source;
    use winit::event_loop::EventLoop;
    use winit::platform::run_on_demand::EventLoopExtRunOnDemand;

    use super::{check, probe_adapter, Renderer, ViewError, ViewOptions};
    use crate::app::SectionApp;
    use crate::soft::SoftApp;
    use crate::state::ViewState;

    thread_local! {
        // winit allows one event loop per process; keep it for the next window.
        static LOOP: RefCell<Option<EventLoop<UserEvent>>> = const { RefCell::new(None) };
        // A failed first build cannot be retried, so later calls repeat its reason.
        static FAILED: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
        p.downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic".into())
    }

    fn with_loop<T>(f: impl FnOnce(&mut EventLoop<UserEvent>) -> T) -> Result<T, ViewError> {
        LOOP.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                if let Some(reason) = FAILED.with(|f| f.borrow().clone()) {
                    return Err(ViewError::NoDisplay(format!(
                        "{reason}; the first attempt in this process failed and the window \
                         system cannot be retried, restart Python"
                    )));
                }
                let built = catch_unwind(|| EventLoop::<UserEvent>::with_user_event().build())
                    .map_err(|p| ViewError::Thread(panic_text(p)))?
                    .map_err(|e| {
                        let reason = super::without_source_location(&e.to_string());
                        FAILED.with(|f| *f.borrow_mut() = Some(reason.clone()));
                        ViewError::NoDisplay(reason)
                    })?;
                *slot = Some(built);
            }
            Ok(f(slot.as_mut().expect("event loop just built")))
        })
    }

    fn state<R: ReadAt + 'static>(source: &Arc<Source<R>>, o: &ViewOptions) -> ViewState<R> {
        let mut s = ViewState::new(source.clone(), &o.fixed, o.style);
        s.live = o.live.clone();
        if let Some(w) = &o.region {
            s.set_region(w);
        }
        s
    }

    pub fn run<R: ReadAt + 'static>(
        source: Arc<Source<R>>,
        options: ViewOptions,
    ) -> Result<Renderer, ViewError> {
        check(&source, &options)?;
        let title = if options.title.is_empty() {
            "segyrs".to_string()
        } else {
            options.title.clone()
        };
        let cpu_env = std::env::var("SEGYRS_VIEW_RENDERER").is_ok_and(|v| v == "cpu");
        let choice = match options.renderer {
            Renderer::Auto if cpu_env || !probe_adapter() => Renderer::Cpu,
            Renderer::Auto => Renderer::Gpu,
            r => r,
        };
        if choice == Renderer::Gpu {
            let created = Rc::new(Cell::new(false));
            let flag = created.clone();
            let st = state(&source, &options);
            let interrupt = options.interrupt.clone();
            let ran = with_loop(|el| {
                let native = eframe::NativeOptions {
                    viewport: egui::ViewportBuilder::default()
                        .with_title(title.clone())
                        .with_inner_size([1100.0, 720.0]),
                    renderer: eframe::Renderer::Wgpu,
                    ..Default::default()
                };
                let mut app = eframe::create_native(
                    &title,
                    native,
                    Box::new(move |cc| {
                        let rs = cc
                            .wgpu_render_state
                            .as_ref()
                            .ok_or("eframe started without a wgpu render state")?;
                        crate::gpu::install(rs, 1);
                        let max = rs.device.limits().max_texture_dimension_2d as usize;
                        flag.set(true);
                        let mut app = SectionApp::new(st, Some(&cc.egui_ctx), max);
                        app.interrupt = interrupt;
                        Ok(Box::new(app))
                    }),
                    el,
                );
                catch_unwind(AssertUnwindSafe(|| el.run_app_on_demand(&mut app)))
            })?;
            match ran {
                Err(p) => return Err(ViewError::Window(panic_text(p))),
                Ok(Err(e)) if created.get() => return Err(ViewError::Window(e.to_string())),
                Ok(_) if created.get() => return Ok(Renderer::Gpu),
                // The GPU window never came up (surface or device failed): fall back.
                Ok(_) if options.renderer == Renderer::Gpu => {
                    return Err(ViewError::Window("the wgpu window could not start".into()))
                }
                Ok(_) => {}
            }
        }
        let mut app = SoftApp::new(state(&source, &options), title);
        app.interrupt = options.interrupt.clone();
        let ran =
            with_loop(|el| catch_unwind(AssertUnwindSafe(|| el.run_app_on_demand(&mut app))))?;
        match ran {
            Err(p) => Err(ViewError::Window(panic_text(p))),
            Ok(Err(e)) => Err(ViewError::Window(e.to_string())),
            Ok(Ok(())) => match app.error {
                Some(e) => Err(ViewError::NoDisplay(e)),
                None => Ok(Renderer::Cpu),
            },
        }
    }
}
