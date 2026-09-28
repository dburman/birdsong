//! Pure-Rust re-implementation of BirdNET V2.4's in-graph mel-spectrogram frontend
//! (`MelSpecLayerSimple`, two instances) so the rest of the network can run in tract.
//!
//! Reference (Keras, per layer):
//! ```text
//! x = (x - min(x)) ; x = x / (max(x) + 1e-6) ; x = (x - 0.5) * 2
//! S = stft(x, frame_length, frame_step, hann(periodic), pad_end=False)   # complex
//! S = real(S)                       # tf.cast(complex -> float32) keeps the real part
//! M = S @ mel_filterbank            # (frames, n_mels)
//! M = M^2 ; M = M^(1 / (1 + exp(magnitude_scaling)))
//! M = reverse(M, mel axis) ; M = transpose -> (n_mels, frames)
//! ```
//! The two layers are concatenated on a trailing channel axis: output (96, 511, 2) NHWC.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Hyper-parameters of one `MelSpecLayerSimple` instance.
#[derive(Clone, Debug, PartialEq)]
pub struct MelParams {
    pub sample_rate: u32,
    pub frame_length: usize,
    pub frame_step: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    /// Trained scalar weight `magnitude_scaling`; exponent is `1 / (1 + exp(w))`.
    pub magnitude_scaling: f32,
}

impl MelParams {
    /// BirdNET V2.4 `MEL_SPEC1` (low band).
    pub const fn v24_spec1() -> Self {
        Self {
            sample_rate: 48_000,
            frame_length: 2048,
            frame_step: 278,
            n_mels: 96,
            fmin: 0.0,
            fmax: 3000.0,
            magnitude_scaling: 1.211_000_4,
        }
    }
    /// BirdNET V2.4 `MEL_SPEC2` (high band).
    pub const fn v24_spec2() -> Self {
        Self {
            sample_rate: 48_000,
            frame_length: 1024,
            frame_step: 280,
            n_mels: 96,
            fmin: 500.0,
            fmax: 15000.0,
            magnitude_scaling: 1.446_587_4,
        }
    }
    pub fn n_bins(&self) -> usize {
        self.frame_length / 2 + 1
    }
    pub fn n_frames(&self, n_samples: usize) -> usize {
        if n_samples < self.frame_length {
            0
        } else {
            1 + (n_samples - self.frame_length) / self.frame_step
        }
    }
}

/// HTK mel scale as used by `tf.signal.linear_to_mel_weight_matrix`.
fn hertz_to_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Port of `tf.signal.linear_to_mel_weight_matrix`. Returns row-major `(n_bins, n_mels)`.
pub fn linear_to_mel_weight_matrix(p: &MelParams) -> Vec<f32> {
    let n_bins = p.n_bins();
    let nyquist = p.sample_rate as f32 / 2.0;
    // TF drops the DC bin ("bands_to_zero = 1") and zero-pads it back afterwards.
    let bins_hz: Vec<f32> = (1..n_bins)
        .map(|i| nyquist * i as f32 / (n_bins - 1) as f32)
        .collect();
    let bins_mel: Vec<f32> = bins_hz.iter().map(|&h| hertz_to_mel(h)).collect();
    let lo = hertz_to_mel(p.fmin);
    let hi = hertz_to_mel(p.fmax);
    let edges: Vec<f32> = (0..p.n_mels + 2)
        .map(|i| lo + (hi - lo) * i as f32 / (p.n_mels + 1) as f32)
        .collect();
    let mut w = vec![0.0f32; n_bins * p.n_mels];
    for m in 0..p.n_mels {
        let (l, c, u) = (edges[m], edges[m + 1], edges[m + 2]);
        for (bi, &bm) in bins_mel.iter().enumerate() {
            let lower = (bm - l) / (c - l);
            let upper = (u - bm) / (u - c);
            let v = lower.min(upper).max(0.0);
            w[(bi + 1) * p.n_mels + m] = v;
        }
    }
    w
}

/// One mel-spectrogram layer with precomputed window, FFT plan and filterbank.
pub struct MelFrontend {
    params: MelParams,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    /// Row-major `(n_bins, n_mels)` filterbank, kept for reference/inspection.
    melfb: Vec<f32>,
    /// Per mel band: first non-zero bin and that band's contiguous weights.
    /// Each triangular filter touches only a handful of bins, so this cuts the
    /// projection from `n_bins × n_mels` to roughly `2 × n_bins` MACs per frame.
    bands: Vec<(usize, Vec<f32>)>,
}

fn band_ranges(melfb: &[f32], n_bins: usize, n_mels: usize) -> Vec<(usize, Vec<f32>)> {
    (0..n_mels)
        .map(|m| {
            let col: Vec<f32> = (0..n_bins).map(|b| melfb[b * n_mels + m]).collect();
            let lo = col.iter().position(|&w| w != 0.0).unwrap_or(0);
            let hi = col.iter().rposition(|&w| w != 0.0).map_or(lo, |i| i + 1);
            (lo, col[lo..hi].to_vec())
        })
        .collect()
}

impl MelFrontend {
    pub fn new(params: MelParams) -> Self {
        let n = params.frame_length;
        // tf.signal.hann_window(periodic=True): 0.5 - 0.5 cos(2πk/N)
        let window = (0..n)
            .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / n as f32).cos())
            .collect();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
        let melfb = linear_to_mel_weight_matrix(&params);
        let bands = band_ranges(&melfb, params.n_bins(), params.n_mels);
        Self {
            params,
            window,
            fft,
            melfb,
            bands,
        }
    }

    /// Replace the filterbank with an externally supplied one (row-major `(n_bins, n_mels)`).
    pub fn with_filterbank(mut self, melfb: Vec<f32>) -> Self {
        assert_eq!(melfb.len(), self.params.n_bins() * self.params.n_mels);
        self.bands = band_ranges(&melfb, self.params.n_bins(), self.params.n_mels);
        self.melfb = melfb;
        self
    }

    /// The `(n_bins, n_mels)` filterbank in use.
    pub fn filterbank(&self) -> &[f32] {
        &self.melfb
    }

    pub fn params(&self) -> &MelParams {
        &self.params
    }

    /// Compute the layer output for an already-normalised signal.
    /// Returns row-major `(n_mels, n_frames)`, mel axis flipped as in the reference.
    pub fn compute(&self, x: &[f32]) -> Vec<f32> {
        let p = &self.params;
        let n_frames = p.n_frames(x.len());
        let n_bins = p.n_bins();
        let exponent = 1.0 / (1.0 + p.magnitude_scaling.exp());
        let mut out = vec![0.0f32; p.n_mels * n_frames];
        let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); p.frame_length];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut re = vec![0.0f32; n_bins];
        for t in 0..n_frames {
            let start = t * p.frame_step;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex::new(x[start + i] * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut buf, &mut scratch);
            // real part only (the reference casts complex -> float32), then mel projection
            for (r, c) in re.iter_mut().zip(&buf) {
                *r = c.re;
            }
            for (m, (lo, weights)) in self.bands.iter().enumerate() {
                let acc: f32 = weights.iter().zip(&re[*lo..]).map(|(w, r)| w * r).sum();
                let v = (acc * acc).powf(exponent);
                // reverse mel axis, transpose to (mel, time)
                out[(p.n_mels - 1 - m) * n_frames + t] = v;
            }
        }
        out
    }
}

/// `(x - min) / (max(x - min) + 1e-6)`, then rescaled to `[-1, 1]`.
pub fn normalize(samples: &[f32]) -> Vec<f32> {
    let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
    let shifted: Vec<f32> = samples.iter().map(|v| v - min).collect();
    let max = shifted.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    shifted
        .iter()
        .map(|v| (v / (max + 1e-6) - 0.5) * 2.0)
        .collect()
}

/// The full BirdNET V2.4 frontend: two layers concatenated on a trailing channel axis.
pub struct BirdnetV24Frontend {
    spec1: MelFrontend,
    spec2: MelFrontend,
}

impl Default for BirdnetV24Frontend {
    fn default() -> Self {
        Self::new()
    }
}

impl BirdnetV24Frontend {
    pub fn new() -> Self {
        Self {
            spec1: MelFrontend::new(MelParams::v24_spec1()),
            spec2: MelFrontend::new(MelParams::v24_spec2()),
        }
    }

    pub fn with_layers(spec1: MelFrontend, spec2: MelFrontend) -> Self {
        Self { spec1, spec2 }
    }

    /// Output dims `(n_mels, n_frames, 2)` for a signal of `n_samples`.
    pub fn output_shape(&self, n_samples: usize) -> [usize; 3] {
        [
            self.spec1.params.n_mels,
            self.spec1.params.n_frames(n_samples),
            2,
        ]
    }

    /// Raw audio in, NHWC-flattened `(96, 511, 2)` spectrogram out (for 144 000 samples).
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let x = normalize(samples);
        let a = self.spec1.compute(&x);
        let b = self.spec2.compute(&x);
        debug_assert_eq!(a.len(), b.len());
        let mut out = Vec::with_capacity(a.len() * 2);
        for (va, vb) in a.iter().zip(&b) {
            out.push(*va);
            out.push(*vb);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn read_f32_file(path: &std::path::Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        Some(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        )
    }

    #[test]
    fn frame_counts_match_reference() {
        assert_eq!(MelParams::v24_spec1().n_frames(144_000), 511);
        assert_eq!(MelParams::v24_spec2().n_frames(144_000), 511);
        assert_eq!(MelParams::v24_spec1().n_bins(), 1025);
        assert_eq!(MelParams::v24_spec2().n_bins(), 513);
    }

    #[test]
    fn normalize_maps_to_unit_range() {
        let x = normalize(&[-2.0, 0.0, 2.0]);
        assert!((x[0] + 1.0).abs() < 1e-5);
        assert!(x[1].abs() < 1e-5);
        assert!((x[2] - 1.0).abs() < 1e-5);
    }

    /// Requires `models/MEL_SPEC{1,2}_melfb.f32` dumped by tools/convert_model/export_headless_v24.py.
    #[test]
    fn filterbank_matches_tensorflow_dump() {
        for (name, p) in [
            ("MEL_SPEC1", MelParams::v24_spec1()),
            ("MEL_SPEC2", MelParams::v24_spec2()),
        ] {
            let path = repo_root().join("models").join(format!("{name}_melfb.f32"));
            let Some(reference) = read_f32_file(&path) else {
                eprintln!("skipping: {} not found", path.display());
                return;
            };
            let ours = linear_to_mel_weight_matrix(&p);
            assert_eq!(ours.len(), reference.len(), "{name} shape");
            let max_err = ours
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // f32 rounding in TF's linspace vs ours: observed 1.2e-5 on MEL_SPEC1.
            assert!(max_err < 1e-4, "{name}: max filterbank error {max_err}");
        }
    }

    /// Requires `tools/fixtures/soundscape_15s.wav` and the binary reference spectrogram of its
    /// first chunk (`golden/soundscape_15s_chunk0_spec.f32`) dumped by export_headless_v24.py.
    #[test]
    fn spectrogram_matches_tensorflow_reference() {
        let wav = repo_root().join("tools/fixtures/soundscape_15s.wav");
        let refp = repo_root().join("tools/fixtures/golden/soundscape_15s_chunk0_spec.f32");
        let (Ok(mut reader), Some(reference)) =
            (hound::WavReader::open(&wav), read_f32_file(&refp))
        else {
            eprintln!("skipping: fixtures not found");
            return;
        };
        assert_eq!(reader.spec().sample_rate, 48_000);
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let fe = BirdnetV24Frontend::new();
        assert_eq!(fe.output_shape(144_000), [96, 511, 2]);
        let ours = fe.compute(&samples[..144_000]);
        assert_eq!(ours.len(), reference.len());
        let max_ref = reference.iter().copied().fold(0.0f32, f32::max);
        let max_err = ours
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_err = ours
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / ours.len() as f32;
        eprintln!("chunk 0: max_ref={max_ref:.4} max_err={max_err:.5} mean_err={mean_err:.6}");
        assert!(
            max_err < 1e-2 * max_ref.max(1.0),
            "spectrogram mismatch: max_err={max_err}"
        );
    }
}
