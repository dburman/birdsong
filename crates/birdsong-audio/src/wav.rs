//! WAV reading (any PCM/float format, downmixed to mono) and writing (16-bit PCM mono).

use std::path::Path;

use birdsong_core::SAMPLE_RATE_HZ;

use crate::AudioError;

/// Decoded audio, downmixed to mono `f32` in `[-1, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct WavAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Channel count in the file before downmixing.
    pub channels: u16,
}

fn wav_err(path: &Path, source: hound::Error) -> AudioError {
    AudioError::Wav {
        path: path.to_path_buf(),
        source,
    }
}

/// Read a WAV file (8/16/24/32-bit integer or 32-bit float), averaging channels to mono.
pub fn read_wav(path: &Path) -> Result<WavAudio, AudioError> {
    let mut reader = hound::WavReader::open(path).map_err(|e| wav_err(path, e))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1);
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| wav_err(path, e))?,
        hound::SampleFormat::Int => {
            if !(1..=32).contains(&spec.bits_per_sample) {
                return Err(AudioError::Unsupported {
                    path: path.to_path_buf(),
                    message: format!("{}-bit integer samples", spec.bits_per_sample),
                });
            }
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()
                .map_err(|e| wav_err(path, e))?
        }
    };
    let samples = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok(WavAudio {
        samples,
        sample_rate: spec.sample_rate,
        channels,
    })
}

/// Read a WAV file that must already be 48 kHz (any channel count; downmixed).
/// Use ffmpeg ([`crate::FfmpegSource`]) for files that need resampling.
pub fn read_wav_48k_mono(path: &Path) -> Result<Vec<f32>, AudioError> {
    let audio = read_wav(path)?;
    if audio.sample_rate != SAMPLE_RATE_HZ {
        return Err(AudioError::Unsupported {
            path: path.to_path_buf(),
            message: format!(
                "sample rate {} Hz, expected {SAMPLE_RATE_HZ} Hz",
                audio.sample_rate
            ),
        });
    }
    Ok(audio.samples)
}

/// Write mono 16-bit PCM. Samples are scaled by 32 768 and clamped, the inverse of [`read_wav`].
/// Returns the file size in bytes.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<u64, AudioError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| wav_err(path, e))?;
    for &s in samples {
        let v = (s * 32_768.0)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(v).map_err(|e| wav_err(path, e))?;
    }
    writer.finalize().map_err(|e| wav_err(path, e))?;
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| AudioError::Io {
            path: path.to_path_buf(),
            source: e,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_16_bit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.wav");
        let samples: Vec<f32> = (-5..5).map(|i| i as f32 / 8.0).chain([1.5, -1.5]).collect();
        let bytes = write_wav(&path, &samples, 48_000).unwrap();
        assert_eq!(bytes, 44 + 2 * samples.len() as u64);
        let back = read_wav(&path).unwrap();
        assert_eq!(back.sample_rate, 48_000);
        assert_eq!(back.channels, 1);
        assert_eq!(
            &back.samples[..10],
            &samples[..10],
            "k/8 is exactly representable"
        );
        assert!(
            (back.samples[10] - 32_767.0 / 32_768.0).abs() < 1e-6,
            "clamped high"
        );
        assert_eq!(back.samples[11], -1.0, "clamped low");
    }

    #[test]
    fn stereo_float_is_downmixed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for (l, r) in [(1.0f32, 0.0f32), (0.5, 0.5), (-1.0, 0.0)] {
            w.write_sample(l).unwrap();
            w.write_sample(r).unwrap();
        }
        w.finalize().unwrap();
        let a = read_wav(&path).unwrap();
        assert_eq!(a.channels, 2);
        assert_eq!(a.samples, [0.5, 0.5, -0.5]);
        assert_eq!(read_wav_48k_mono(&path).unwrap().len(), 3);
    }

    #[test]
    fn wrong_rate_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.wav");
        write_wav(&path, &[0.0; 100], 44_100).unwrap();
        let err = read_wav_48k_mono(&path).unwrap_err().to_string();
        assert!(err.contains("44100"), "{err}");
        assert!(read_wav(&dir.path().join("missing.wav")).is_err());
    }

    #[test]
    fn fixture_matches_known_length() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/fixtures/soundscape_15s.wav");
        let Ok(samples) = read_wav_48k_mono(&path) else {
            eprintln!("skipping: fixture not present");
            return;
        };
        // Cut with `ffmpeg -t 15 -c copy`, which stops on a packet boundary: 15.0187 s.
        assert_eq!(samples.len(), 720_896);
    }
}
