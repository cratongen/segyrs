//! Open the viewer on a SEG-Y file.
//!
//! `cargo run -p segyrs-view --example view -- FILE [gpu|cpu]`

use std::sync::Arc;

use segyrs::OpenOptions;
use segyrs_draw::Source;
use segyrs_view::{run, Renderer, ViewOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: view FILE [gpu|cpu]")?;
    let renderer = match args.next().as_deref() {
        Some("gpu") => Renderer::Gpu,
        Some("cpu") => Renderer::Cpu,
        _ => Renderer::Auto,
    };
    let source = Arc::new(Source::open(&path, &OpenOptions::default())?);
    let options = ViewOptions {
        title: path.clone(),
        renderer,
        ..Default::default()
    };
    let used = run(source, options)?;
    println!("closed ({used:?})");
    Ok(())
}
