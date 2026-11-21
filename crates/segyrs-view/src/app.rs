//! The eframe window: toolbar, the section as a wgpu paint callback, axes,
//! zoom and pan, and the cursor readout. Plane reads run on a worker thread
//! so the window keeps drawing while a large section decodes.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use eframe::egui_wgpu;
use segyrs::ReadAt;
use segyrs_draw::{effective_clip, Colormap, PlaneSpec, Section, Source, Wiggle};

use crate::gpu::{uniforms, Grid, SectionCallback};
use crate::state::ViewState;

const MARGIN_L: f32 = 64.0;
const MARGIN_B: f32 = 36.0;
const MARGIN_T: f32 = 8.0;
const MARGIN_R: f32 = 12.0;

type Reply = (PlaneSpec, Result<Section, String>);

/// Reads planes off the UI thread. Only the newest request is served.
struct Worker {
    tx: Sender<PlaneSpec>,
    rx: Receiver<Reply>,
}

impl Worker {
    fn spawn<R: ReadAt + 'static>(source: Arc<Source<R>>, ctx: egui::Context) -> Self {
        let (tx, req) = channel::<PlaneSpec>();
        let (out, rx) = channel::<Reply>();
        std::thread::Builder::new()
            .name("segyrs-view-read".into())
            .spawn(move || {
                while let Ok(mut spec) = req.recv() {
                    while let Ok(newer) = req.try_recv() {
                        spec = newer;
                    }
                    let r = source.section(&spec).map_err(|e| e.to_string());
                    if out.send((spec, r)).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            })
            .expect("spawn reader thread");
        Self { tx, rx }
    }
}

pub struct SectionApp<R: ReadAt + 'static> {
    pub state: ViewState<R>,
    worker: Option<Worker>,
    requested: Option<PlaneSpec>,
    grid: Option<Arc<Grid>>,
    version: u64,
    pub max_texture: usize,
    pub linear: bool,
    clip_text: f32,
    pub readout: String,
    pub interrupt: Option<crate::Interrupt>,
}

impl<R: ReadAt + 'static> SectionApp<R> {
    /// `threaded`: read planes on a worker thread (the window) or inline (tests).
    pub fn new(state: ViewState<R>, ctx: Option<&egui::Context>, max_texture: usize) -> Self {
        let worker = ctx.map(|c| Worker::spawn(state.source.clone(), c.clone()));
        Self {
            state,
            worker,
            requested: None,
            grid: None,
            version: 0,
            max_texture,
            linear: false,
            clip_text: 1.0,
            readout: String::new(),
            interrupt: None,
        }
    }

    fn accept(&mut self, s: Section) {
        self.version += 1;
        self.grid = Some(Arc::new(Grid::from_section(&s, self.version)));
        if self.state.style.clip.is_none() {
            self.clip_text = effective_clip(&s, &self.state.style);
        }
        self.state.section = Some(Arc::new(s));
        self.state.error = None;
    }

    /// Ask for the plane at this plot size if it is not the one already asked for.
    fn request(&mut self, pixels: (usize, usize)) {
        let cap = self.max_texture.max(1);
        let spec = self.state.spec((pixels.0.min(cap), pixels.1.min(cap)));
        if self.requested.as_ref() == Some(&spec) {
            return;
        }
        self.requested = Some(spec.clone());
        match &self.worker {
            Some(w) => {
                let _ = w.tx.send(spec);
            }
            None => match self.state.source.section(&spec) {
                Ok(s) => self.accept(s),
                Err(e) => self.state.error = Some(e.to_string()),
            },
        }
    }

    fn poll(&mut self) {
        let Some(w) = &self.worker else { return };
        let mut latest = None;
        while let Ok(r) = w.rx.try_recv() {
            latest = Some(r);
        }
        if let Some((spec, r)) = latest {
            if self.requested.as_ref() == Some(&spec) {
                match r {
                    Ok(s) => self.accept(s),
                    Err(e) => self.state.error = Some(e),
                }
            }
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let names: Vec<String> = self.state.planes.iter().map(|p| p.name()).collect();
            let mut plane = self.state.plane;
            egui::ComboBox::from_id_salt("plane")
                .selected_text(names[plane].clone())
                .show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut plane, i, n);
                    }
                });
            self.state.set_plane(plane);

            for (dim, _) in self.state.fixed() {
                let len = self.state.len_of(dim);
                let mut i = self.state.index_of(dim);
                ui.separator();
                ui.label(dim.as_str());
                if ui.small_button("<").clicked() {
                    i = i.saturating_sub(1);
                }
                ui.add(egui::DragValue::new(&mut i).range(0..=len.saturating_sub(1)));
                if ui.small_button(">").clicked() {
                    i += 1;
                }
                self.state.set_index(dim, i);
                let coord = self
                    .state
                    .source
                    .coords(dim)
                    .map(|c| c[self.state.index_of(dim)]);
                if let Some(c) = coord {
                    ui.weak(format!("= {c}"));
                }
            }

            ui.separator();
            let mut auto = self.state.style.clip.is_none();
            ui.checkbox(&mut auto, "clip p98");
            if !auto {
                let speed = self.clip_text.abs().max(1e-6) * 0.01;
                ui.add(
                    egui::DragValue::new(&mut self.clip_text)
                        .speed(speed)
                        .range(f32::MIN_POSITIVE..=f32::MAX),
                );
                self.state.style.clip = Some(self.clip_text);
            } else {
                self.state.style.clip = None;
            }
            ui.label("gain");
            ui.add(
                egui::DragValue::new(&mut self.state.style.gain)
                    .speed(0.02)
                    .range(0.01..=1000.0),
            );
            let mut rev = self.state.style.polarity == segyrs_draw::Polarity::Reversed;
            if ui.checkbox(&mut rev, "reverse").changed() {
                self.state.flip_polarity();
            }
            egui::ComboBox::from_id_salt("cmap")
                .selected_text(self.state.style.colormap.as_str())
                .show_ui(ui, |ui| {
                    for c in Colormap::ALL {
                        ui.selectable_value(&mut self.state.style.colormap, c, c.as_str());
                    }
                });
            egui::ComboBox::from_id_salt("wiggle")
                .selected_text(match self.state.style.wiggle {
                    Wiggle::Off => "no wiggle",
                    Wiggle::Line => "wiggle",
                    Wiggle::Area => "wiggle + area",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.state.style.wiggle, Wiggle::Off, "no wiggle");
                    ui.selectable_value(&mut self.state.style.wiggle, Wiggle::Line, "wiggle");
                    ui.selectable_value(
                        &mut self.state.style.wiggle,
                        Wiggle::Area,
                        "wiggle + area",
                    );
                });
            ui.checkbox(&mut self.linear, "linear");
            if ui.button("reset zoom").clicked() {
                self.state.zoom = Default::default();
            }
        });
    }

    fn keys(&mut self, ui: &egui::Ui) {
        let (left, right, page_up, page_down) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::PageUp),
                i.key_pressed(egui::Key::PageDown),
            )
        });
        let step = if page_up || page_down { 10 } else { 1 };
        if left || page_down {
            self.state.step(-step);
        }
        if right || page_up {
            self.state.step(step);
        }
    }

    /// The whole window for one frame. Testable with a bare `egui::Context`.
    pub fn frame_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(stop) = &self.interrupt {
            if (stop.0)() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }
        self.poll();
        self.keys(ui);
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(e) = &self.state.error {
                    ui.colored_label(Color32::RED, e);
                } else {
                    ui.label(&self.readout);
                }
            });
        });
        egui::CentralPanel::default().show(ui, |ui| self.plot(ui));
    }

    fn plot(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        let rect = Rect::from_min_max(
            full.min + Vec2::new(MARGIN_L, MARGIN_T),
            full.max - Vec2::new(MARGIN_R, MARGIN_B),
        );
        if rect.width() < 4.0 || rect.height() < 4.0 {
            return;
        }
        let ppp = ui.ctx().pixels_per_point();
        let pixels = (
            (rect.width() * ppp).round() as usize,
            (rect.height() * ppp).round() as usize,
        );
        self.request(pixels);
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());

        // Zoom with the wheel around the cursor, pan by dragging, reset on double click.
        if let Some(p) = resp.hover_pos() {
            let f = ((p - rect.min) / rect.size()).clamp(Vec2::ZERO, Vec2::splat(1.0));
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let factor = (-(scroll as f64) / 200.0).exp();
                self.state.zoom.zoom(factor, f.x as f64, f.y as f64);
            }
        }
        if resp.dragged() {
            let d = resp.drag_delta() / rect.size();
            self.state.zoom.pan(-d.x as f64, -d.y as f64);
        }
        if resp.double_clicked() {
            self.state.zoom = Default::default();
        }

        let Some(section) = self.state.section.clone() else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "reading...",
                egui::FontId::default(),
                ui.visuals().text_color(),
            );
            return;
        };
        let clip = effective_clip(&section, &self.state.style);
        if let Some(grid) = &self.grid {
            let u = uniforms(clip, &self.state.style, self.linear);
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                SectionCallback {
                    grid: grid.clone(),
                    uniforms: u,
                },
            ));
        }
        self.wiggles(ui, rect, &section, clip);
        self.axes(ui, rect, &section);

        self.readout = match resp.hover_pos() {
            Some(p) => {
                let px = ((p.x - rect.min.x) / rect.width() * section.pixels.0 as f32) as usize;
                let py = ((p.y - rect.min.y) / rect.height() * section.pixels.1 as f32) as usize;
                self.state.readout(px, py).unwrap_or_default()
            }
            None => format!(
                "{}  clip ±{clip} ({})  grid {}x{}",
                section.title(),
                if self.state.style.clip.is_some() {
                    "set"
                } else {
                    "p98 of view"
                },
                section.width,
                section.height
            ),
        };
    }

    /// Wiggle lines with egui's painter when every trace has 3 points of room.
    fn wiggles(&self, ui: &egui::Ui, rect: Rect, s: &Section, clip: f32) {
        if self.state.style.wiggle == Wiggle::Off || s.column_traces.is_none() {
            return;
        }
        let spacing = rect.width() / s.width as f32;
        if spacing < 3.0 {
            return;
        }
        let exc = self.state.style.xcur * spacing;
        let sign = self.state.style.polarity.sign();
        let ink = Stroke::new(1.0, Color32::BLACK);
        for c in 0..s.width {
            let xc = rect.min.x + (c as f32 + 0.5) * spacing;
            let pts: Vec<Pos2> = (0..s.height)
                .map(|r| {
                    let v = s.value(c, r);
                    let dx = if v.is_nan() {
                        0.0
                    } else {
                        (v * sign / clip).clamp(-1.0, 1.0) * exc
                    };
                    Pos2::new(
                        xc + dx,
                        rect.min.y + (r as f32 + 0.5) * rect.height() / s.height as f32,
                    )
                })
                .collect();
            if self.state.style.wiggle == Wiggle::Area {
                for p in &pts {
                    if p.x > xc {
                        let h = rect.height() / s.height as f32;
                        ui.painter().rect_filled(
                            Rect::from_min_max(
                                Pos2::new(xc, p.y - h / 2.0),
                                Pos2::new(p.x, p.y + h / 2.0),
                            ),
                            0.0,
                            Color32::BLACK,
                        );
                    }
                }
            }
            ui.painter().add(egui::Shape::line(pts, ink));
        }
    }

    fn axes(&self, ui: &egui::Ui, rect: Rect, s: &Section) {
        let painter = ui.painter();
        let ink = ui.visuals().text_color();
        painter.rect_stroke(rect, 0.0, Stroke::new(1.0, ink), egui::StrokeKind::Outside);
        let font = egui::FontId::monospace(11.0);
        let tick = |a: &segyrs_draw::Axis, n: usize| -> Vec<(f32, String)> {
            let len = a.labels.len();
            let k = n.clamp(2, len.max(2));
            (0..k)
                .map(|i| {
                    let p = if k == 1 { 0 } else { i * (len - 1) / (k - 1) };
                    ((p as f32 + 0.5) / len as f32, fmt(a.labels[p]))
                })
                .collect()
        };
        for (f, label) in tick(&s.x, (rect.width() / 90.0) as usize) {
            let x = rect.min.x + f * rect.width();
            painter.line_segment(
                [Pos2::new(x, rect.max.y), Pos2::new(x, rect.max.y + 4.0)],
                Stroke::new(1.0, ink),
            );
            painter.text(
                Pos2::new(x, rect.max.y + 5.0),
                egui::Align2::CENTER_TOP,
                label,
                font.clone(),
                ink,
            );
        }
        for (f, label) in tick(&s.y, (rect.height() / 50.0) as usize) {
            let y = rect.min.y + f * rect.height();
            painter.line_segment(
                [Pos2::new(rect.min.x - 4.0, y), Pos2::new(rect.min.x, y)],
                Stroke::new(1.0, ink),
            );
            painter.text(
                Pos2::new(rect.min.x - 6.0, y),
                egui::Align2::RIGHT_CENTER,
                label,
                font.clone(),
                ink,
            );
        }
        painter.text(
            Pos2::new(rect.center().x, rect.max.y + 20.0),
            egui::Align2::CENTER_TOP,
            &s.x.name,
            font.clone(),
            ink,
        );
        painter.text(
            Pos2::new(rect.min.x - MARGIN_L + 2.0, rect.min.y - MARGIN_T),
            egui::Align2::LEFT_TOP,
            &s.y.name,
            font,
            ink,
        );
    }
}

fn fmt(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v:.2}")
    }
}

impl<R: ReadAt + 'static> eframe::App for SectionApp<R> {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.frame_ui(ui);
    }
}
