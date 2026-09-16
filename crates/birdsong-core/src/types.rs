use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Capture sample rate: audio sources, the ring buffer, clips and uploads all use it. A classifier
/// that needs another rate resamples its own window.
pub const SAMPLE_RATE_HZ: u32 = 48_000;
/// BirdNET V2.4's analysis window in seconds. Other classifiers set their own through
/// `Classifier::window_seconds`; this is also the shortest window any model may use.
pub const CHUNK_SECONDS: f32 = 3.0;
/// Samples in BirdNET V2.4's window (`3 s × 48 kHz`).
pub const CHUNK_SAMPLES: usize = 144_000;

/// One species detected in one analysis window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Database id; `None` until stored.
    pub id: Option<i64>,
    /// Start of the analysis window, UTC.
    pub detected_at: DateTime<Utc>,
    pub scientific_name: String,
    pub common_name: String,
    /// `0.0..=1.0`, after the sigmoid.
    pub confidence: f32,
    /// Which configured audio source produced it (`"mic0"`, `"rtsp1"`, …).
    pub source_id: String,
    /// Identifier of the classifier that produced it (`"birdnet-v2.4"`).
    pub model_id: String,
    /// Saved clip, relative to the clips directory; `None` if never saved or purged.
    pub clip_path: Option<String>,
}
