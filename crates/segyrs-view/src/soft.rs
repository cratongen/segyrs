//! The CPU fallback window: when no GPU adapter exists, the draw crate's RGBA
//! image is presented with softbuffer. Keys replace the toolbar; the title
//! bar carries the state and the cursor readout.

use std::num::NonZeroU32;
use std::rc::Rc;

use eframe::UserEvent;
use segyrs::ReadAt;
use segyrs_draw::{render, MARGINS};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::state::ViewState;

/// Keys of the CPU window, shown in its help line.
pub const KEYS: &str = "left/right step, tab plane, p polarity, c colormap, w wiggle, +/- gain, a auto clip, r reset zoom, esc close";

pub struct SoftApp<R: ReadAt + 'static> {
    pub state: ViewState<R>,
    title: String,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    cursor: Option<(f64, f64)>,
    last: Option<segyrs_draw::PlaneSpec>,
    pub error: Option<String>,
    pub interrupt: Option<crate::Interrupt>,
}

impl<R: ReadAt + 'static> SoftApp<R> {
    pub fn new(state: ViewState<R>, title: String) -> Self {
        Self {
            state,
            title,
            window: None,
            surface: None,
            cursor: None,
            last: None,
            error: None,
            interrupt: None,
        }
    }

    /// Plot-area size for a window of `w` x `h` pixels.
    fn plot_size(w: u32, h: u32) -> (usize, usize) {
        let (ml, mr, mt, mb) = MARGINS;
        (
            (w as usize).saturating_sub(ml + mr).max(1),
            (h as usize).saturating_sub(mt + mb).max(1),
        )
    }

    fn redraw(&mut self) {
        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else {
            return;
        };
        let size = window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return;
        };
        let plot = Self::plot_size(size.width, size.height);
        let spec = self.state.spec(plot);
        if self.last.as_ref() != Some(&spec) {
            self.state.load(plot);
            self.last = Some(spec);
        }
        if surface.resize(w, h).is_err() {
            return;
        }
        let Ok(mut buf) = surface.buffer_mut() else {
            return;
        };
        buf.fill(0x00ff_ffff);
        if let Some(s) = &self.state.section {
            let img = render(s, &self.state.style);
            let cols = img.width.min(size.width as usize);
            for y in 0..img.height.min(size.height as usize) {
                for x in 0..cols {
                    let i = (y * img.width + x) * 4;
                    let p = &img.rgba[i..i + 4];
                    buf[y * size.width as usize + x] =
                        (u32::from(p[0]) << 16) | (u32::from(p[1]) << 8) | u32::from(p[2]);
                }
            }
        }
        let _ = buf.present();
        let readout = self
            .cursor
            .and_then(|(x, y)| {
                let (ml, _, mt, _) = MARGINS;
                let (px, py) = (x as i64 - ml as i64, y as i64 - mt as i64);
                (px >= 0 && py >= 0 && (px as usize) < plot.0 && (py as usize) < plot.1)
                    .then(|| self.state.readout(px as usize, py as usize))
                    .flatten()
            })
            .or_else(|| self.state.error.clone())
            .unwrap_or_else(|| format!("{} [{}]", self.state.kind().name(), KEYS));
        window.set_title(&format!("{} (cpu): {readout}", self.title));
    }

    fn key(&mut self, event_loop: &ActiveEventLoop, key: &Key) {
        match key {
            Key::Named(NamedKey::ArrowLeft) => self.state.step(-1),
            Key::Named(NamedKey::ArrowRight) => self.state.step(1),
            Key::Named(NamedKey::PageDown) => self.state.step(-10),
            Key::Named(NamedKey::PageUp) => self.state.step(10),
            Key::Named(NamedKey::Tab) => {
                let n = self.state.planes.len();
                self.state.set_plane((self.state.plane + 1) % n);
            }
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Character(c) => match c.as_str() {
                "p" => self.state.flip_polarity(),
                "c" => self.state.cycle_colormap(),
                "w" => self.state.cycle_wiggle(),
                "+" | "=" => self.state.style.gain *= 1.25,
                "-" => self.state.style.gain /= 1.25,
                "a" => self.state.style.clip = None,
                "r" => self.state.zoom = Default::default(),
                _ => return,
            },
            _ => return,
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl<R: ReadAt + 'static> ApplicationHandler<UserEvent> for SoftApp<R> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!("{} (cpu)", self.title))
            .with_inner_size(winit::dpi::LogicalSize::new(1100.0, 700.0));
        let made = event_loop
            .create_window(attrs)
            .map_err(|e| e.to_string())
            .and_then(|w| {
                let w = Rc::new(w);
                let ctx = softbuffer::Context::new(w.clone()).map_err(|e| e.to_string())?;
                let s = softbuffer::Surface::new(&ctx, w.clone()).map_err(|e| e.to_string())?;
                Ok((w, s))
            });
        match made {
            Ok((w, s)) => {
                w.request_redraw();
                self.window = Some(w);
                self.surface = Some(s);
            }
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::Resized(_) => {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some((position.x, position.y));
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => self.key(event_loop, &logical_key),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(stop) = &self.interrupt {
            if (stop.0)() {
                event_loop.exit();
                return;
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(200),
            ));
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.surface = None;
        self.window = None;
    }
}
