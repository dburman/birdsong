use std::path::PathBuf;

use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::frame::{apply_gain, db_to_gain, samples_to_delta};
use crate::{wav, AudioError, AudioFrame, AudioSource};

/// How a [`WavFileSource`] delivers its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pacing {
    /// As fast as the receiver accepts them; timestamps start at `start_at`.
    Fast { start_at: DateTime<Utc> },
    /// At the recording's own speed, stamped with wall-clock time.
    Realtime,
}

/// Pure-Rust source replaying a 48 kHz WAV file (no ffmpeg). Used by tests and offline analysis.
pub struct WavFileSource {
    id: String,
    path: PathBuf,
    pacing: Pacing,
    gain: f32,
    frame_samples: usize,
}

impl WavFileSource {
    pub fn new(id: impl Into<String>, path: impl Into<PathBuf>, pacing: Pacing) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            pacing,
            gain: 1.0,
            frame_samples: 4_800,
        }
    }

    pub fn with_gain_db(mut self, gain_db: f32) -> Self {
        self.gain = db_to_gain(gain_db);
        self
    }

    pub fn with_frame_samples(mut self, frame_samples: usize) -> Self {
        self.frame_samples = frame_samples.max(1);
        self
    }
}

#[async_trait::async_trait]
impl AudioSource for WavFileSource {
    fn id(&self) -> &str {
        &self.id
    }

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<AudioFrame>,
        cancel: CancellationToken,
    ) -> Result<(), AudioError> {
        let path = self.path.clone();
        let mut samples = tokio::task::spawn_blocking(move || wav::read_wav_48k_mono(&path))
            .await
            .map_err(|e| AudioError::Task(e.to_string()))??;
        apply_gain(&mut samples, self.gain);

        let started = tokio::time::Instant::now();
        let wall_start = Utc::now();
        let mut offset = 0u64;
        for piece in samples.chunks(self.frame_samples) {
            let captured_at = match self.pacing {
                Pacing::Fast { start_at } => start_at + samples_to_delta(offset),
                Pacing::Realtime => {
                    let end = samples_to_delta(offset + piece.len() as u64);
                    let due = started + end.to_std().unwrap_or_default();
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return Ok(()),
                        _ = tokio::time::sleep_until(due) => {}
                    }
                    wall_start + samples_to_delta(offset)
                }
            };
            let frame = AudioFrame {
                samples: piece.to_vec(),
                captured_at,
            };
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Ok(()),
                sent = tx.send(frame) => if sent.is_err() { return Ok(()) },
            }
            offset += piece.len() as u64;
        }
        Ok(())
    }
}
