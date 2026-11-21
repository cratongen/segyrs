// Variable-density section: float grid in, LUT colour out.
// The grid is R32Float and read with textureLoad, so no float filtering
// feature is needed. Linear mode interpolates four taps here.

struct U {
    clip: f32,
    sign: f32,
    cmap: f32,
    linear: f32,
    srgb: f32,
    density: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var grid: texture_2d<f32>;
@group(0) @binding(2) var lut: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle that covers the viewport egui gives the callback.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    var o: VOut;
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

// Absent cells are uploaded as f32::MAX.
fn absent(v: f32) -> bool {
    return v >= 3.0e38;
}

fn at(c: vec2<i32>, size: vec2<i32>) -> f32 {
    return textureLoad(grid, clamp(c, vec2<i32>(0), size - vec2<i32>(1)), 0).r;
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(grid));
    let p = in.uv * vec2<f32>(size);
    var v = at(vec2<i32>(floor(p)), size);
    if (u.linear > 0.5 && !absent(v)) {
        let q = p - vec2<f32>(0.5);
        let c = vec2<i32>(floor(q));
        let f = fract(q);
        let a = at(c, size);
        let b = at(c + vec2<i32>(1, 0), size);
        let d = at(c + vec2<i32>(0, 1), size);
        let e = at(c + vec2<i32>(1, 1), size);
        if (!(absent(a) || absent(b) || absent(d) || absent(e))) {
            v = mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
        }
    }
    var rgb: vec3<f32>;
    if (absent(v)) {
        let k = (i32(in.pos.x) + i32(in.pos.y)) / 4;
        rgb = select(vec3<f32>(242.0 / 255.0), vec3<f32>(205.0 / 255.0), (k & 1) == 0);
    } else if (u.density < 0.5) {
        rgb = vec3<f32>(1.0);
    } else {
        let t = clamp((v * u.sign / u.clip + 1.0) * 0.5, 0.0, 1.0);
        let idx = i32(round(t * 255.0));
        rgb = textureLoad(lut, vec2<i32>(idx, i32(u.cmap)), 0).rgb;
    }
    if (u.srgb > 0.5) {
        rgb = to_linear(rgb);
    }
    return vec4<f32>(rgb, 1.0);
}
