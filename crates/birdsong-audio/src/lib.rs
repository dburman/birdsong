#![forbid(unsafe_code)]
//! Audio for Birdsong: capture, buffering, chunking and WAV files.
//!
//! ```text
//! AudioSource (FfmpegSource | WavFileSource) --AudioFrame--> Chunker --Chunk (3 s)--> inference
//!                                                               |
//!                                                               +-- RingBuffer (shared, for clips)
//! ```
//!
//! Every source produces mono `f32` samples at 48 kHz. Frames carry the UTC time of their first
//! sample; timestamps advance by sample count so they stay contiguous (see `docs/DECISIONS.md` #10).

mod chunker;
mod error;
mod ffmpeg;
mod frame;
mod ring;
mod source;
pub mod wav;
mod wav_source;

pub use chunker::{Chunk, Chunker, ChunkerEvent, MIN_TAIL_SAMPLES};
pub use error::AudioError;
pub use ffmpeg::{ffmpeg_args, redact_text, redact_url, FfmpegOptions, FfmpegSource};
pub use frame::{apply_gain, db_to_gain, delta_to_samples, samples_to_delta, AudioFrame};
pub use ring::{lock_ring, Extracted, RingBuffer, SharedRingBuffer};
pub use source::{source_from_config, AudioSource};
pub use wav_source::{Pacing, WavFileSource};
