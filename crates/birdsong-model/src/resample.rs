//! 48 kHz → 32 kHz resampling for classifiers that run at 32 kHz (Perch v2).
//!
//! The ratio is exactly 2:3, so output sample `n` sits at input position `1.5 n`: either on an
//! input sample or halfway between two. Each output is a windowed-sinc low-pass (Blackman window)
//! evaluated at that position, so only two kernels are needed and they are built once.

use std::f64::consts::PI;

/// Input samples used on each side of an output position.
const HALF_TAPS: usize = 64;
/// Low-pass cutoff. Below the 16 kHz output Nyquist frequency so the transition band finishes
/// before content can alias back into the band the classifier uses.
const CUTOFF_HZ: f64 = 14_500.0;
const INPUT_RATE_HZ: f64 = 48_000.0;

/// A reusable 48 kHz → 32 kHz resampler.
#[derive(Clone, Debug)]
pub struct Resampler48To32 {
    /// Kernel for an output on an input sample, and for one halfway between two.
    kernels: [Vec<f32>; 2],
}

impl Default for Resampler48To32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Resampler48To32 {
    pub fn new() -> Self {
        Self {
            kernels: [kernel(0.0), kernel(0.5)],
        }
    }

    /// Output length for `input_len` samples: `ceil(input_len × 2 / 3)`.
    pub fn output_len(input_len: usize) -> usize {
        (input_len * 2).div_ceil(3)
    }

    /// Resample one block. Samples outside the block count as silence, so the first and last
    /// ~1.3 ms are slightly attenuated; windows are analysed independently, as the model expects.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        let taps = 2 * HALF_TAPS;
        (0..Self::output_len(input.len()))
            .map(|n| {
                let half_samples = 3 * n; // position in half input samples
                let kernel = &self.kernels[half_samples % 2];
                let first = (half_samples / 2) as isize + 1 - HALF_TAPS as isize;
                match usize::try_from(first) {
                    // Interior: every tap is inside the block.
                    Ok(start) if start + taps <= input.len() => kernel
                        .iter()
                        .zip(&input[start..start + taps])
                        .map(|(w, x)| w * x)
                        .sum(),
                    _ => kernel
                        .iter()
                        .enumerate()
                        .filter_map(|(j, w)| {
                            let k = usize::try_from(first + j as isize).ok()?;
                            input.get(k).map(|x| w * x)
                        })
                        .sum(),
                }
            })
            .collect()
    }
}

/// Kernel taps for an output at `base + frac`, covering input samples
/// `base + 1 − HALF_TAPS ..= base + HALF_TAPS`, normalised to unit gain at DC.
fn kernel(frac: f64) -> Vec<f32> {
    let fc = CUTOFF_HZ / INPUT_RATE_HZ;
    let taps: Vec<f64> = (0..2 * HALF_TAPS)
        .map(|j| {
            let d = j as f64 + 1.0 - HALF_TAPS as f64 - frac;
            let x = d / HALF_TAPS as f64;
            if x.abs() >= 1.0 {
                return 0.0;
            }
            let window = 0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos();
            2.0 * fc * sinc(2.0 * fc * d) * window
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    taps.into_iter().map(|t| (t / sum) as f32).collect()
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: f64, n: usize, amplitude: f64) -> Vec<f32> {
        (0..n)
            .map(|i| (amplitude * (2.0 * PI * hz * i as f64 / rate).sin()) as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// Ignore the edges, where the block boundary attenuates the signal.
    fn interior(x: &[f32]) -> &[f32] {
        &x[200..x.len() - 200]
    }

    #[test]
    fn lengths() {
        assert_eq!(Resampler48To32::output_len(240_000), 160_000);
        assert_eq!(Resampler48To32::output_len(3), 2);
        assert_eq!(Resampler48To32::output_len(4), 3);
        assert_eq!(
            Resampler48To32::new().process(&[0.0; 240_000]).len(),
            160_000
        );
    }

    #[test]
    fn dc_is_preserved() {
        let out = Resampler48To32::new().process(&[0.3; 9_600]);
        assert!(interior(&out).iter().all(|v| (v - 0.3).abs() < 1e-4));
    }

    #[test]
    fn passband_tones_match_the_ideal_32k_signal() {
        let r = Resampler48To32::new();
        for hz in [440.0, 3_000.0, 9_000.0] {
            let out = r.process(&tone(hz, 48_000.0, 48_000, 0.5));
            let ideal = tone(hz, 32_000.0, out.len(), 0.5);
            let worst = interior(&out)
                .iter()
                .zip(interior(&ideal))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 5e-3, "{hz} Hz: worst error {worst}");
        }
    }

    #[test]
    fn content_above_the_output_band_does_not_alias() {
        let r = Resampler48To32::new();
        for hz in [17_000.0, 20_000.0, 23_000.0] {
            let input = tone(hz, 48_000.0, 48_000, 0.5);
            let ratio = rms(interior(&r.process(&input))) / rms(&input);
            assert!(ratio < 3e-3, "{hz} Hz leaked {ratio}");
        }
    }
}
