//! Draw one plane of a SEG-Y file to a PNG.
//!
//! `cargo run -p segyrs-draw --example section -- FILE OUT.png [DIM INDEX] [STYLE]`
//! where DIM is inline, crossline, offset or time, and STYLE is one of grey,
//! seismic, amber, wiggle, area. Without DIM it draws the notebook default.

use segyrs::OpenOptions;
use segyrs_draw::{
    map_image, picture, Colormap, Dim, PlaneSpec, Source, Style, Wiggle, NOTEBOOK_PIXELS,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: section FILE OUT.png [DIM INDEX | map] [STYLE]".into());
    }
    let src = Source::open(&args[0], &OpenOptions::default())?;
    if args.get(2).map(String::as_str) == Some("map") {
        std::fs::write(&args[1], map_image(&src, 500, 500)?.to_png()?)?;
        return Ok(());
    }
    let fixed = match (args.get(2), args.get(3)) {
        (Some(d), Some(i)) => vec![(Dim::parse(d).ok_or("unknown dim")?, i.parse()?)],
        _ => src.default_fixed(),
    };
    let mut style = Style::default();
    match args.get(4).map(String::as_str) {
        Some("wiggle") => style.wiggle = Wiggle::Line,
        Some("area") => {
            style.wiggle = Wiggle::Area;
            style.density = false;
        }
        Some(name) => style.colormap = Colormap::parse(name).ok_or("unknown style")?,
        None => {}
    }
    let spec = PlaneSpec::new(fixed, NOTEBOOK_PIXELS.0, NOTEBOOK_PIXELS.1);
    let pic = picture(&src, &spec, &style)?;
    std::fs::write(&args[1], &pic.png)?;
    println!("{}", pic.text);
    Ok(())
}
