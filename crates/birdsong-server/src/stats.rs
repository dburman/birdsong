//! Lock-free counters shared by the pipeline and (later) the HTTP API.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use birdsong_audio::ClockStats;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Weight of the newest sample in the inference-time moving average.
const EWMA_ALPHA: f64 = 0.1;

/// Upper bounds of the buckets counting how confident the classifier was about the person class
/// that put a chunk within the privacy rank cutoff.
pub const PERSON_CONFIDENCE_BUCKETS: [f32; 7] = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0];

/// Chunks kept in the privacy log (an hour of 5 s windows).
pub const PRIVACY_LOG_CHUNKS: usize = 720;

/// A person class in one chunk, for the privacy log.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PersonClass {
    pub class: String,
    /// 1 = the chunk's top class.
    pub rank: usize,
    pub confidence: f32,
}

/// What the privacy filter saw in one chunk. No audio is kept.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrivacySample {
    pub at: DateTime<Utc>,
    pub source_id: String,
    /// Best-ranked person class of any kind.
    pub person: Option<PersonClass>,
    /// Best-ranked voice class.
    pub voice: Option<PersonClass>,
    /// The chunk itself counted as having a person (its neighbours are then masked too).
    pub person_present: bool,
}

/// Pipeline counters. Updated from the inference thread and storage task; read from anywhere.
#[derive(Debug, Default)]
pub struct PipelineStats {
    chunks_processed: AtomicU64,
    chunks_dropped: AtomicU64,
    masked_chunks: AtomicU64,
    /// Chunks with a person class within the privacy rank cutoff, by that class's confidence
    /// (non-cumulative, one per [`PERSON_CONFIDENCE_BUCKETS`] entry) and by class name.
    person_by_confidence: [AtomicU64; PERSON_CONFIDENCE_BUCKETS.len()],
    person_by_class: Mutex<BTreeMap<String, u64>>,
    /// The newest [`PRIVACY_LOG_CHUNKS`] chunks, oldest first.
    privacy_log: Mutex<VecDeque<PrivacySample>>,
    gaps: AtomicU64,
    detections: AtomicU64,
    unconfirmed_detections: AtomicU64,
    inference_errors: AtomicU64,
    store_errors: AtomicU64,
    clips_written: AtomicU64,
    clip_errors: AtomicU64,
    birdweather_soundscapes: AtomicU64,
    birdweather_detections: AtomicU64,
    birdweather_errors: AtomicU64,
    birdweather_skipped: AtomicU64,
    /// Exponentially weighted mean inference time in microseconds; 0 = no sample yet.
    inference_micros_ewma: AtomicU64,
    /// Start time of the newest processed chunk, microseconds since the epoch; 0 = none yet.
    last_chunk_at_micros: AtomicI64,
    /// Wall-clock time the newest chunk finished inference, microseconds since the epoch; 0 = none.
    last_processed_at_micros: AtomicI64,
    /// Capture clock corrections, updated by the live audio sources.
    clock: Arc<ClockStats>,
}

/// A consistent-enough copy of the counters, for logs and the API.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StatsSnapshot {
    pub chunks_processed: u64,
    pub chunks_dropped: u64,
    pub masked_chunks: u64,
    /// Chunks with a person class within the privacy rank cutoff, whether or not they were
    /// masked; shows what `detection.privacy_min_confidence` would change.
    pub privacy_person: PersonStats,
    /// Discontinuities in the audio timeline, whatever the cause (includes `clock_reanchors`).
    pub gaps: u64,
    /// Times a live source's timestamps jumped to the wall clock (a stall, a restart, or drift the
    /// gradual correction could not absorb).
    pub clock_reanchors: u64,
    /// Samples repeated / skipped to keep a live capture clock in step with the wall clock.
    pub clock_samples_inserted: u64,
    pub clock_samples_dropped: u64,
    /// Net capture clock correction in parts per million (positive: the microphone runs slow).
    pub clock_correction_ppm: Option<f64>,
    pub detections: u64,
    /// Detections dropped because their species was never confirmed (`min_detections`).
    pub unconfirmed_detections: u64,
    pub inference_errors: u64,
    pub store_errors: u64,
    pub clips_written: u64,
    pub clip_errors: u64,
    pub birdweather_soundscapes: u64,
    pub birdweather_detections: u64,
    pub birdweather_errors: u64,
    pub birdweather_skipped: u64,
    pub mean_inference_ms: Option<f64>,
    /// Start of the newest processed chunk (audio time).
    pub last_chunk_at: Option<DateTime<Utc>>,
    /// When the newest chunk was processed (wall clock); use this for liveness checks.
    pub last_processed_at: Option<DateTime<Utc>>,
}

/// See [`StatsSnapshot::privacy_person`].
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PersonStats {
    /// Chunks per confidence bucket, lowest first.
    pub by_confidence: Vec<PersonBucket>,
    /// Chunks per person class.
    pub by_class: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PersonBucket {
    /// Confidence above the previous bucket's bound and at most this.
    pub up_to: f32,
    pub chunks: u64,
}

impl PipelineStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// The counters live audio sources update with their clock corrections.
    pub fn clock(&self) -> Arc<ClockStats> {
        Arc::clone(&self.clock)
    }

    pub(crate) fn chunk_processed(&self, start_at: DateTime<Utc>, inference: Duration) {
        self.chunks_processed.fetch_add(1, Ordering::Relaxed);
        self.last_chunk_at_micros
            .store(start_at.timestamp_micros(), Ordering::Relaxed);
        self.last_processed_at_micros
            .store(Utc::now().timestamp_micros(), Ordering::Relaxed);
        let sample = inference.as_micros().min(u64::MAX as u128) as u64;
        // Single writer (the inference thread), so load-then-store is race free.
        let old = self.inference_micros_ewma.load(Ordering::Relaxed);
        let new = if old == 0 {
            sample.max(1)
        } else {
            (old as f64 * (1.0 - EWMA_ALPHA) + sample as f64 * EWMA_ALPHA) as u64
        };
        self.inference_micros_ewma
            .store(new.max(1), Ordering::Relaxed);
    }

    pub(crate) fn chunk_dropped(&self) {
        self.chunks_dropped.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn chunk_masked(&self) {
        self.masked_chunks.fetch_add(1, Ordering::Relaxed);
    }

    /// A person class ranked within the privacy cutoff.
    pub(crate) fn person_ranked(&self, class: &str, confidence: f32) {
        let bucket = PERSON_CONFIDENCE_BUCKETS
            .iter()
            .position(|&up_to| confidence <= up_to)
            .unwrap_or(PERSON_CONFIDENCE_BUCKETS.len() - 1);
        self.person_by_confidence[bucket].fetch_add(1, Ordering::Relaxed);
        let mut by_class = self
            .person_by_class
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *by_class.entry(class.to_string()).or_default() += 1;
    }

    pub(crate) fn privacy_sample(&self, sample: PrivacySample) {
        let mut log = self.privacy_log.lock().unwrap_or_else(|e| e.into_inner());
        if log.len() == PRIVACY_LOG_CHUNKS {
            log.pop_front();
        }
        log.push_back(sample);
    }

    /// The privacy log, oldest first.
    pub fn privacy_log(&self) -> Vec<PrivacySample> {
        self.privacy_log
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    pub(crate) fn gap(&self) {
        self.gaps.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn detections_stored(&self, n: u64) {
        self.detections.fetch_add(n, Ordering::Relaxed);
    }

    pub(crate) fn detections_unconfirmed(&self, n: u64) {
        self.unconfirmed_detections.fetch_add(n, Ordering::Relaxed);
    }

    pub(crate) fn inference_error(&self) {
        self.inference_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn store_error(&self) {
        self.store_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn clip_written(&self) {
        self.clips_written.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn clip_error(&self) {
        self.clip_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn birdweather_uploaded(&self, detections: u64) {
        self.birdweather_soundscapes.fetch_add(1, Ordering::Relaxed);
        self.birdweather_detections
            .fetch_add(detections, Ordering::Relaxed);
    }

    pub(crate) fn birdweather_skipped(&self) {
        self.birdweather_skipped.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn birdweather_error(&self) {
        self.birdweather_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        let ewma = self.inference_micros_ewma.load(Ordering::Relaxed);
        let last = self.last_chunk_at_micros.load(Ordering::Relaxed);
        let processed = self.last_processed_at_micros.load(Ordering::Relaxed);
        StatsSnapshot {
            chunks_processed: self.chunks_processed.load(Ordering::Relaxed),
            chunks_dropped: self.chunks_dropped.load(Ordering::Relaxed),
            masked_chunks: self.masked_chunks.load(Ordering::Relaxed),
            privacy_person: PersonStats {
                by_confidence: PERSON_CONFIDENCE_BUCKETS
                    .iter()
                    .zip(&self.person_by_confidence)
                    .map(|(&up_to, n)| PersonBucket {
                        up_to,
                        chunks: n.load(Ordering::Relaxed),
                    })
                    .collect(),
                by_class: self
                    .person_by_class
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
            },
            gaps: self.gaps.load(Ordering::Relaxed),
            clock_reanchors: self.clock.reanchors(),
            clock_samples_inserted: self.clock.samples_inserted(),
            clock_samples_dropped: self.clock.samples_dropped(),
            clock_correction_ppm: self.clock.correction_ppm(),
            detections: self.detections.load(Ordering::Relaxed),
            unconfirmed_detections: self.unconfirmed_detections.load(Ordering::Relaxed),
            inference_errors: self.inference_errors.load(Ordering::Relaxed),
            store_errors: self.store_errors.load(Ordering::Relaxed),
            clips_written: self.clips_written.load(Ordering::Relaxed),
            clip_errors: self.clip_errors.load(Ordering::Relaxed),
            birdweather_soundscapes: self.birdweather_soundscapes.load(Ordering::Relaxed),
            birdweather_detections: self.birdweather_detections.load(Ordering::Relaxed),
            birdweather_errors: self.birdweather_errors.load(Ordering::Relaxed),
            birdweather_skipped: self.birdweather_skipped.load(Ordering::Relaxed),
            mean_inference_ms: (ewma > 0).then(|| ewma as f64 / 1000.0),
            last_chunk_at: (last != 0)
                .then(|| DateTime::from_timestamp_micros(last))
                .flatten(),
            last_processed_at: (processed != 0)
                .then(|| DateTime::from_timestamp_micros(processed))
                .flatten(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn privacy_log_keeps_the_newest_chunks() {
        let s = PipelineStats::new();
        let t = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap();
        for i in 0..PRIVACY_LOG_CHUNKS + 3 {
            s.privacy_sample(PrivacySample {
                at: t + chrono::TimeDelta::seconds(5 * i as i64),
                source_id: "mic0".into(),
                person: None,
                voice: None,
                person_present: false,
            });
        }
        let log = s.privacy_log();
        assert_eq!(log.len(), PRIVACY_LOG_CHUNKS);
        assert_eq!(log[0].at, t + chrono::TimeDelta::seconds(15));
        assert!(log.windows(2).all(|w| w[0].at < w[1].at));
    }

    #[test]
    fn person_confidence_buckets() {
        let s = PipelineStats::new();
        s.person_ranked("Speech", 0.0004);
        s.person_ranked("Speech", 0.001); // bounds are inclusive
        s.person_ranked("Breathing", 0.03);
        s.person_ranked("Speech", 0.97);
        let p = s.snapshot().privacy_person;
        let counts: Vec<u64> = p.by_confidence.iter().map(|b| b.chunks).collect();
        assert_eq!(counts, [2, 0, 1, 0, 0, 0, 1]);
        assert_eq!(p.by_confidence[2].up_to, 0.05);
        assert_eq!(p.by_class["Speech"], 3);
        assert_eq!(p.by_class["Breathing"], 1);
    }

    #[test]
    fn counters_and_moving_average() {
        let s = PipelineStats::new();
        let empty = s.snapshot();
        assert_eq!(
            StatsSnapshot {
                privacy_person: PersonStats::default(),
                ..empty.clone()
            },
            StatsSnapshot::default()
        );
        assert!(empty
            .privacy_person
            .by_confidence
            .iter()
            .all(|b| b.chunks == 0));
        let t = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap();
        s.chunk_processed(t, Duration::from_millis(100));
        assert_eq!(
            s.snapshot().mean_inference_ms,
            Some(100.0),
            "first sample seeds the average"
        );
        s.chunk_processed(t, Duration::from_millis(200));
        assert_eq!(s.snapshot().mean_inference_ms, Some(110.0));
        s.chunk_dropped();
        s.detections_stored(3);
        let snap = s.snapshot();
        assert_eq!(
            (snap.chunks_processed, snap.chunks_dropped, snap.detections),
            (2, 1, 3)
        );
        assert_eq!(snap.last_chunk_at, Some(t));
        assert!(snap
            .last_processed_at
            .is_some_and(|p| (Utc::now() - p).num_seconds() < 5));
    }
}
