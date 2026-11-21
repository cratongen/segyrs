//! The section on the GPU: an R32Float texture of the float grid, the shared
//! LUT as a 256 x N texture, and clip, polarity and colormap as uniforms.
//! Changing any of those rewrites a 32-byte uniform; only a new grid uploads.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu, CallbackResources, CallbackTrait, ScreenDescriptor};
use segyrs_draw::{lut_rows, Section};

pub const SHADER: &str = include_str!("section.wgsl");

/// Absent cells are uploaded as this value: WGSL may assume no NaN exists.
pub const ABSENT: f32 = f32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub clip: f32,
    pub sign: f32,
    pub cmap: f32,
    pub linear: f32,
    pub srgb: f32,
    pub density: f32,
    pub _pad: [f32; 2],
}

/// Pipeline and textures, created once per window and kept in egui-wgpu's
/// callback resources.
pub struct Gpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    lut: wgpu::TextureView,
    grid: Option<(wgpu::Texture, wgpu::BindGroup, u64)>,
    srgb: bool,
}

impl Gpu {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: wgpu::TextureFormat,
        samples: u32,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("segyrs section"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let tex = |filterable| wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("segyrs section"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, tex(false)),
                entry(2, tex(false)),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("segyrs section"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("segyrs section"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: samples.max(1),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segyrs section uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rows = lut_rows();
        let lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("segyrs lut"),
            size: wgpu::Extent3d {
                width: 256,
                height: rows.len() as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes: Vec<u8> = rows.iter().flatten().flatten().copied().collect();
        queue.write_texture(
            lut_tex.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 256,
                height: rows.len() as u32,
                depth_or_array_layers: 1,
            },
        );
        Self {
            pipeline,
            layout,
            uniforms,
            lut: lut_tex.create_view(&Default::default()),
            grid: None,
            srgb: target.is_srgb(),
        }
    }

    /// Upload a float grid unless `version` is already on the GPU.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid: &Grid) {
        if self.grid.as_ref().is_some_and(|g| g.2 == grid.version) {
            return;
        }
        let size = wgpu::Extent3d {
            width: grid.width as u32,
            height: grid.height as u32,
            depth_or_array_layers: 1,
        };
        let reuse = self
            .grid
            .take()
            .filter(|(t, _, _)| t.size() == size)
            .map(|(t, b, _)| (t, b));
        let (texture, bind) = reuse.unwrap_or_else(|| {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("segyrs grid"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = t.create_view(&Default::default());
            let b = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segyrs section"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.lut),
                    },
                ],
            });
            (t, b)
        });
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&grid.values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(grid.width as u32 * 4),
                rows_per_image: None,
            },
            size,
        );
        self.grid = Some((texture, bind, grid.version));
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, mut u: Uniforms) {
        u.srgb = if self.srgb { 1.0 } else { 0.0 };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some((_, bind, _)) = &self.grid {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// A float grid ready for upload: absent cells already replaced by [`ABSENT`].
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
    pub version: u64,
}

impl Grid {
    pub fn from_section(s: &Section, version: u64) -> Self {
        Self {
            width: s.width,
            height: s.height,
            values: s
                .values
                .iter()
                .map(|v| if v.is_nan() { ABSENT } else { *v })
                .collect(),
            version,
        }
    }
}

pub fn uniforms(clip: f32, style: &segyrs_draw::Style, linear: bool) -> Uniforms {
    Uniforms {
        clip,
        sign: style.polarity.sign(),
        cmap: style.colormap.row() as f32,
        linear: if linear { 1.0 } else { 0.0 },
        srgb: 0.0,
        density: if style.density { 1.0 } else { 0.0 },
        _pad: [0.0; 2],
    }
}

/// The paint callback: carries the grid to upload (when new) and the uniforms.
pub struct SectionCallback {
    pub grid: Arc<Grid>,
    pub uniforms: Uniforms,
}

impl CallbackTrait for SectionCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(gpu) = resources.get_mut::<Gpu>() {
            gpu.upload(device, queue, &self.grid);
            gpu.write_uniforms(queue, self.uniforms);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        if let Some(gpu) = resources.get::<Gpu>() {
            gpu.draw(pass);
        }
    }
}

/// Register the pipeline with egui-wgpu. Called once when the window opens.
pub fn install(rs: &egui_wgpu::RenderState, samples: u32) {
    let gpu = Gpu::new(&rs.device, &rs.queue, rs.target_format, samples);
    rs.renderer.write().callback_resources.insert(gpu);
}
