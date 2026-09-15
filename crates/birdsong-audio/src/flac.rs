//! FLAC encoding of mono clips with the pure-Rust `flacenc` crate. Used for saved clips (about
//! half the size of WAV) and for BirdWeather soundscape uploads, which must be FLAC.

use std::path::Path;

use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::Verify;

use crate::AudioError;

/// Scale `[-1, 1]` floats to 16-bit PCM exactly like [`crate::wav::write_wav`].
fn to_pcm16(samples: &[f32]) -> Vec<i32> {
    samples
        .iter()
        .map(|&s| {
            (s * 32_768.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i32
        })
        .collect()
}

/// Shortest frame strict decoders accept. The FLAC format allows a shorter final frame, but some
/// decoders reject one, so a short tail is padded with silence (at most 15 samples, 0.3 ms).
const MIN_FRAME_SAMPLES: usize = 16;

/// Encode mono samples as a 16-bit FLAC stream in memory.
pub fn encode_flac(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, AudioError> {
    if samples.is_empty() {
        return Err(AudioError::FlacEncode("no samples to encode".into()));
    }
    let encoder = flacenc::config::Encoder::default();
    let block_size = encoder.block_size;
    let mut pcm = to_pcm16(samples);
    let tail = pcm.len() % block_size;
    if tail != 0 && tail < MIN_FRAME_SAMPLES {
        pcm.resize(pcm.len() + MIN_FRAME_SAMPLES - tail, 0);
    }
    let config = encoder
        .into_verified()
        .map_err(|(_, e)| AudioError::FlacEncode(format!("encoder configuration: {e:?}")))?;
    let source = flacenc::source::MemSource::from_samples(&pcm, 1, 16, sample_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, block_size)
        .map_err(|e| AudioError::FlacEncode(format!("{e:?}")))?;
    let mut sink = ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| AudioError::FlacEncode(format!("{e:?}")))?;
    Ok(sink.into_inner())
}

/// Write mono 16-bit FLAC. Returns the file size in bytes.
pub fn write_flac(path: &Path, samples: &[f32], sample_rate: u32) -> Result<u64, AudioError> {
    let bytes = encode_flac(samples, sample_rate)?;
    std::fs::write(path, &bytes).map_err(|source| AudioError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(bytes.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(bytes: &[u8]) -> (claxon::metadata::StreamInfo, Vec<i32>) {
        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(bytes)).expect("valid FLAC");
        let info = reader.streaminfo();
        let samples = reader
            .samples()
            .collect::<Result<Vec<_>, _>>()
            .expect("decodable");
        (info, samples)
    }

    #[test]
    fn round_trips_losslessly() {
        let samples: Vec<f32> = (0..62_400)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                0.4 * (2.0 * std::f32::consts::PI * 3_000.0 * t).sin()
                    + 0.05 * ((i * 7919 % 97) as f32 / 97.0 - 0.5)
            })
            .collect();
        let bytes = encode_flac(&samples, 48_000).unwrap();
        assert_eq!(&bytes[..4], b"fLaC");
        let (info, pcm) = decode(&bytes);
        assert_eq!(
            (info.sample_rate, info.channels, info.bits_per_sample),
            (48_000, 1, 16)
        );
        assert_eq!(info.samples, Some(62_400));
        assert_eq!(pcm, to_pcm16(&samples));
    }

    #[test]
    fn clamps_like_wav_and_pads_tiny_input() {
        let (info, pcm) = decode(&encode_flac(&[1.5, -1.5, 0.5, 0.0], 48_000).unwrap());
        assert_eq!(&pcm[..4], [32_767, -32_768, 16_384, 0]);
        assert_eq!(pcm.len(), 16, "padded to the minimum frame size");
        assert!(pcm[4..].iter().all(|&v| v == 0));
        assert_eq!(info.samples, Some(16));
    }

    #[test]
    fn short_final_block_is_padded_but_normal_lengths_are_exact() {
        // 4096 + 3 samples would end with a 3-sample frame.
        let (_, pcm) = decode(&encode_flac(&vec![0.25; 4_099], 48_000).unwrap());
        assert_eq!(pcm.len(), 4_096 + 16);
        let (_, pcm) = decode(&encode_flac(&vec![0.25; 216_000], 48_000).unwrap());
        assert_eq!(
            pcm.len(),
            216_000,
            "a 4.5 s clip has a long final block and is not padded"
        );
    }

    #[test]
    fn real_audio_is_much_smaller_than_wav() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/fixtures/soundscape_15s.wav");
        let Ok(samples) = crate::wav::read_wav_48k_mono(&path) else {
            eprintln!("skipping: fixture not present");
            return;
        };
        let clip = &samples[..6 * 48_000];
        let dir = tempfile::tempdir().unwrap();
        let flac = write_flac(&dir.path().join("c.flac"), clip, 48_000).unwrap();
        let wav = crate::wav::write_wav(&dir.path().join("c.wav"), clip, 48_000).unwrap();
        eprintln!(
            "6 s clip: WAV {wav} bytes, FLAC {flac} bytes ({:.0}%)",
            100.0 * flac as f64 / wav as f64
        );
        assert!((flac as f64) < 0.8 * wav as f64, "FLAC {flac} vs WAV {wav}");
        let (_, pcm) = decode(&std::fs::read(dir.path().join("c.flac")).unwrap());
        assert_eq!(pcm, to_pcm16(clip));
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(encode_flac(&[], 48_000).is_err());
    }
}
