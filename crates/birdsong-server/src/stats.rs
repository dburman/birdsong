//! Lock-free counters shared by the pipeline and (later) the HTTP API.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Weight of the newest sample in the inference-time moving average.
const EWMA_ALPHA: f64 = 0.1;

/// Pipeline counters. Updated from the inference thread and storage task; read from anywhere.
#[derive(Debug, Default)]
pub struct PipelineStats {
    chunks_processed: AtomicU64,
    chunks_dropped: AtomicU64,
    masked_chunks: AtomicU64,
    gaps: AtomicU64,
    detections: AtomicU64,
    inference_errors: AtomicU64,
    store_errors: AtomicU64,
    clips_written: AtomicU64,
    clip_errors: AtomicU64,
    /// Exponentially weighted mean inference time in microseconds; 0 = no sample yet.
    inference_micros_ewma: AtomicU64,
    /// Start time of the newest processed chunk, microseconds since the epoch; 0 = none yet.
    last_chunk_at_micros: AtomicI64,
}

/// A consistent-enough copy of the counters, for logs and the API.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StatsSnapshot {
    pub chunks_processed: u64,
    pub chunks_dropped: u64,
    pub masked_chunks: u64,
    pub gaps: u64,
    pub detections: u64,
    pub inference_errors: u64,
    pub store_errors: u64,
    pub clips_written: u64,
    pub clip_errors: u64,
    pub mean_inference_ms: Option<f64>,
    pub last_chunk_at: Option<DateTime<Utc>>,
}

impl PipelineStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn chunk_processed(&self, start_at: DateTime<Utc>, inference: Duration) {
        self.chunks_processed.fetch_add(1, Ordering::Relaxed);
        self.last_chunk_at_micros
            .store(start_at.timestamp_micros(), Ordering::Relaxed);
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

    pub(crate) fn gap(&self) {
        self.gaps.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn detections_stored(&self, n: u64) {
        self.detections.fetch_add(n, Ordering::Relaxed);
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

    pub fn snapshot(&self) -> StatsSnapshot {
        let ewma = self.inference_micros_ewma.load(Ordering::Relaxed);
        let last = self.last_chunk_at_micros.load(Ordering::Relaxed);
        StatsSnapshot {
            chunks_processed: self.chunks_processed.load(Ordering::Relaxed),
            chunks_dropped: self.chunks_dropped.load(Ordering::Relaxed),
            masked_chunks: self.masked_chunks.load(Ordering::Relaxed),
            gaps: self.gaps.load(Ordering::Relaxed),
            detections: self.detections.load(Ordering::Relaxed),
            inference_errors: self.inference_errors.load(Ordering::Relaxed),
            store_errors: self.store_errors.load(Ordering::Relaxed),
            clips_written: self.clips_written.load(Ordering::Relaxed),
            clip_errors: self.clip_errors.load(Ordering::Relaxed),
            mean_inference_ms: (ewma > 0).then(|| ewma as f64 / 1000.0),
            last_chunk_at: (last != 0)
                .then(|| DateTime::from_timestamp_micros(last))
                .flatten(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn counters_and_moving_average() {
        let s = PipelineStats::new();
        assert_eq!(s.snapshot(), StatsSnapshot::default());
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
    }
}
