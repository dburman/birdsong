//! PNG spectrograms for saved clips: STFT magnitude in dB, mapped through a dark-to-bright colour
//! ramp. Deliberately simple; it is not meant to match BirdNET-Pi's sox rendering.
//!
//! Images are written as 8-bit indexed PNGs (256-colour palette), about a third of the size of
//! RGB, because spectrogram files count towards the clip retention size cap.

use std::io::BufWriter;
use std::path::Path;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::AudioError;

/// Rendering parameters. `Default` gives an 800×300 image of 0–12 kHz.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrogramOptions {
    pub width: u32,
    pub height: u32,
    pub fft_size: usize,
    pub hop: usize,
    /// Highest frequency shown.
    pub max_hz: f32,
    /// dB below the loudest bin that map to black.
    pub dynamic_range_db: f32,
}

impl Default for SpectrogramOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 300,
            fft_size: 1024,
            hop: 256,
            max_hz: 12_000.0,
            dynamic_range_db: 80.0,
        }
    }
}

/// Colour ramp stops (position, RGB), dark purple through orange to pale yellow.
const RAMP: [(f32, [f32; 3]); 5] = [
    (0.00, [0.0, 0.0, 4.0]),
    (0.25, [87.0, 16.0, 110.0]),
    (0.50, [188.0, 55.0, 84.0]),
    (0.75, [249.0, 142.0, 9.0]),
    (1.00, [252.0, 255.0, 164.0]),
];

fn colour(v: f32) -> [u8; 3] {
    let v = v.clamp(0.0, 1.0);
    for pair in RAMP.windows(2) {
        let ((p0, c0), (p1, c1)) = (pair[0], pair[1]);
        if v <= p1 {
            let t = (v - p0) / (p1 - p0);
            return [0, 1, 2].map(|i| (c0[i] + t * (c1[i] - c0[i])).round() as u8);
        }
    }
    [252, 255, 164]
}

/// The 256-entry RGB palette (768 bytes); index `i` is the colour for level `i / 255`.
pub fn palette() -> Vec<u8> {
    (0..=255u32)
        .flat_map(|i| colour(i as f32 / 255.0))
        .collect()
}

/// Render to row-major palette indices (`width × height`), low frequencies at the bottom.
pub fn render_indexed(samples: &[f32], sample_rate: u32, opts: &SpectrogramOptions) -> Vec<u8> {
    let (w, h) = (opts.width.max(1) as usize, opts.height.max(1) as usize);
    let n = opts.fft_size.max(16);
    let hop = opts.hop.max(1);

    let mut padded;
    let signal = if samples.len() < n {
        padded = samples.to_vec();
        padded.resize(n, 0.0);
        &padded[..]
    } else {
        samples
    };
    let frames = 1 + (signal.len() - n) / hop;
    let bin_hz = sample_rate as f32 / n as f32;
    let bins = ((opts.max_hz / bin_hz).floor() as usize).clamp(1, n / 2);

    let window: Vec<f32> = (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / n as f32).cos())
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let mut buf = vec![Complex::new(0.0, 0.0); n];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let mut db = vec![0.0f32; frames * bins];
    let mut max_db = f32::NEG_INFINITY;
    for f in 0..frames {
        let start = f * hop;
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(signal[start + i] * window[i], 0.0);
        }
        fft.process_with_scratch(&mut buf, &mut scratch);
        for (b, c) in buf.iter().take(bins).enumerate() {
            let v = 10.0 * (c.norm_sqr() + 1e-20).log10();
            db[f * bins + b] = v;
            max_db = max_db.max(v);
        }
    }
    // Near-silence stays dark instead of being stretched to full brightness.
    let floor = (max_db - opts.dynamic_range_db).max(-120.0);
    let range = (max_db - floor).max(1e-3);

    let mut out = vec![0u8; w * h];
    for x in 0..w {
        let f0 = x * frames / w;
        let f1 = ((x + 1) * frames / w).max(f0 + 1).min(frames);
        for y in 0..h {
            let row = h - 1 - y;
            let b0 = row * bins / h;
            let b1 = ((row + 1) * bins / h).max(b0 + 1).min(bins);
            let mut peak = f32::NEG_INFINITY;
            for f in f0..f1 {
                for b in b0..b1 {
                    peak = peak.max(db[f * bins + b]);
                }
            }
            let level = ((peak - floor) / range).clamp(0.0, 1.0);
            out[y * w + x] = (level * 255.0).round() as u8;
        }
    }
    out
}

/// Render to row-major RGB8 bytes (`width × height × 3`).
pub fn render_rgb(samples: &[f32], sample_rate: u32, opts: &SpectrogramOptions) -> Vec<u8> {
    let pal = palette();
    render_indexed(samples, sample_rate, opts)
        .into_iter()
        .flat_map(|i| {
            let o = i as usize * 3;
            [pal[o], pal[o + 1], pal[o + 2]]
        })
        .collect()
}

/// Render and write an indexed PNG. Returns the file size in bytes.
pub fn write_png(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    opts: &SpectrogramOptions,
) -> Result<u64, AudioError> {
    let io = |source| AudioError::Io {
        path: path.to_path_buf(),
        source,
    };
    let encode = |e: png::EncodingError| AudioError::Encode {
        path: path.to_path_buf(),
        message: e.to_string(),
    };
    let indices = render_indexed(samples, sample_rate, opts);
    let file = std::fs::File::create(path).map_err(io)?;
    let mut encoder =
        png::Encoder::new(BufWriter::new(file), opts.width.max(1), opts.height.max(1));
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_palette(palette());
    let mut writer = encoder.write_header().map_err(encode)?;
    writer.write_image_data(&indices).map_err(encode)?;
    writer.finish().map_err(encode)?;
    std::fs::metadata(path).map(|m| m.len()).map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, seconds: f32) -> Vec<f32> {
        (0..(seconds * 48_000.0) as usize)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / 48_000.0).sin())
            .collect()
    }

    #[test]
    fn png_has_requested_size_and_palette() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.png");
        let bytes = write_png(
            &path,
            &tone(3_000.0, 6.0),
            48_000,
            &SpectrogramOptions::default(),
        )
        .unwrap();
        assert!(bytes > 100);
        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let reader = decoder.read_info().unwrap();
        let info = reader.info();
        assert_eq!((info.width, info.height), (800, 300));
        assert_eq!(info.color_type, png::ColorType::Indexed);
        assert_eq!(info.palette.as_ref().map(|p| p.len()), Some(768));
    }

    #[test]
    fn tone_is_brightest_at_its_frequency() {
        let opts = SpectrogramOptions::default();
        let rgb = render_rgb(&tone(4_000.0, 2.0), 48_000, &opts);
        let (w, h) = (opts.width as usize, opts.height as usize);
        let brightness =
            |y: usize| -> u64 { (0..w).map(|x| rgb[(y * w + x) * 3 + 1] as u64).sum() };
        let brightest = (0..h).max_by_key(|&y| brightness(y)).unwrap();
        let expected = ((1.0 - 4_000.0 / 12_000.0) * h as f32) as usize; // row 200
        assert!(
            brightest.abs_diff(expected) <= 3,
            "brightest row {brightest}, expected about {expected}"
        );
    }

    #[test]
    fn silence_and_short_input_do_not_panic() {
        let opts = SpectrogramOptions::default();
        let dark = render_rgb(&vec![0.0; 48_000], 48_000, &opts);
        assert!(
            dark.chunks(3).all(|p| p == [0, 0, 4]),
            "silence renders dark"
        );
        assert_eq!(render_rgb(&[0.1; 10], 48_000, &opts).len(), 800 * 300 * 3);
        let pal = palette();
        assert_eq!(
            (&pal[..3], &pal[765..]),
            (&[0u8, 0, 4][..], &[252u8, 255, 164][..])
        );
    }
}
