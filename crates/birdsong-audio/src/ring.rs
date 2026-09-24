use std::sync::{Arc, Mutex, MutexGuard};

use birdsong_core::SAMPLE_RATE_HZ;
use chrono::{DateTime, Utc};

use crate::frame::{delta_to_samples, samples_to_delta};

/// A ring buffer shared between the chunker (writer) and the clip writer (reader).
pub type SharedRingBuffer = Arc<Mutex<RingBuffer>>;

/// Lock a shared ring buffer. A panic in another holder cannot leave the buffer in an unsafe
/// state (every write is a plain copy), so a poisoned lock is simply recovered.
pub fn lock_ring(ring: &SharedRingBuffer) -> MutexGuard<'_, RingBuffer> {
    ring.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Samples cut from the buffer, with the UTC time of the first one.
#[derive(Clone, Debug, PartialEq)]
pub struct Extracted {
    pub samples: Vec<f32>,
    pub start_at: DateTime<Utc>,
}

/// Fixed-capacity, time-indexed buffer of the most recent mono 48 kHz samples.
///
/// Samples are addressed by their absolute index since the last [`RingBuffer::reset`]; index
/// `i` was captured at `anchor + i / 48000 s`. Only the newest `capacity` samples are retained.
#[derive(Clone, Debug)]
pub struct RingBuffer {
    buf: Vec<f32>,
    total: u64,
    anchor: DateTime<Utc>,
}

impl RingBuffer {
    /// A buffer holding `capacity_samples` (at least one).
    pub fn new(capacity_samples: usize) -> Self {
        Self {
            buf: vec![0.0; capacity_samples.max(1)],
            total: 0,
            anchor: DateTime::UNIX_EPOCH,
        }
    }

    /// A buffer holding `seconds` of 48 kHz audio.
    pub fn with_seconds(seconds: f32) -> Self {
        Self::new((seconds.max(0.0) * SAMPLE_RATE_HZ as f32).ceil() as usize)
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Forget everything; the next sample pushed is index 0, captured at `start_at`.
    pub fn reset(&mut self, start_at: DateTime<Utc>) {
        self.total = 0;
        self.anchor = start_at;
    }

    /// Samples pushed since the last reset (including ones no longer retained).
    pub fn total_samples(&self) -> u64 {
        self.total
    }

    /// Index of the oldest retained sample.
    pub fn oldest_index(&self) -> u64 {
        self.total.saturating_sub(self.buf.len() as u64)
    }

    /// Capture time of sample `index`.
    pub fn time_of(&self, index: u64) -> DateTime<Utc> {
        self.anchor + samples_to_delta(index)
    }

    /// Sample index nearest to time `at` (negative if before the anchor).
    pub fn index_at(&self, at: DateTime<Utc>) -> i64 {
        delta_to_samples(at - self.anchor)
    }

    /// Time just after the newest sample.
    pub fn end_time(&self) -> DateTime<Utc> {
        self.time_of(self.total)
    }

    /// Append samples, overwriting the oldest when full.
    /// Append `n` samples of silence: fills a forward jump in the audio timeline so the samples
    /// before it keep their times and stay extractable.
    pub fn push_silence(&mut self, n: u64) {
        let cap = self.buf.len() as u64;
        let fill = n.min(cap);
        for index in self.total + n - fill..self.total + n {
            self.buf[(index % cap) as usize] = 0.0;
        }
        self.total += n;
    }

    pub fn push(&mut self, samples: &[f32]) {
        let cap = self.buf.len();
        let skip = samples.len().saturating_sub(cap);
        let mut rest = &samples[skip..];
        let mut pos = ((self.total + skip as u64) % cap as u64) as usize;
        while !rest.is_empty() {
            let n = (cap - pos).min(rest.len());
            self.buf[pos..pos + n].copy_from_slice(&rest[..n]);
            rest = &rest[n..];
            pos = (pos + n) % cap;
        }
        self.total += samples.len() as u64;
    }

    /// Copy samples `[start, end)`; `None` unless the whole range is retained.
    pub fn copy_range(&self, start: u64, end: u64) -> Option<Vec<f32>> {
        if start > end || start < self.oldest_index() || end > self.total {
            return None;
        }
        let cap = self.buf.len();
        let mut remaining = (end - start) as usize;
        let mut out = Vec::with_capacity(remaining);
        let mut pos = (start % cap as u64) as usize;
        while remaining > 0 {
            let n = (cap - pos).min(remaining);
            out.extend_from_slice(&self.buf[pos..pos + n]);
            remaining -= n;
            pos = (pos + n) % cap;
        }
        Some(out)
    }

    /// Cut `duration_seconds` starting at `start_at`, clamped to what is retained.
    /// `None` when nothing of the requested span is available.
    pub fn extract(&self, start_at: DateTime<Utc>, duration_seconds: f32) -> Option<Extracted> {
        let start = self.index_at(start_at);
        let len = (duration_seconds.max(0.0) as f64 * SAMPLE_RATE_HZ as f64).round() as i64;
        let lo = start.max(self.oldest_index() as i64);
        let hi = start.saturating_add(len).min(self.total as i64);
        if hi <= lo {
            return None;
        }
        let samples = self.copy_range(lo as u64, hi as u64)?;
        Some(Extracted {
            samples,
            start_at: self.time_of(lo as u64),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeDelta, TimeZone};

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap()
    }

    fn ramp(from: u64, n: usize) -> Vec<f32> {
        (0..n as u64).map(|i| (from + i) as f32).collect()
    }

    #[test]
    fn push_wraps_and_copy_range_crosses_boundary() {
        let mut r = RingBuffer::new(10);
        r.reset(t0());
        r.push(&ramp(0, 7));
        r.push(&ramp(7, 7)); // total 14, retained 4..14, wrapped
        assert_eq!(r.total_samples(), 14);
        assert_eq!(r.oldest_index(), 4);
        assert_eq!(r.copy_range(4, 14).unwrap(), ramp(4, 10));
        assert_eq!(r.copy_range(8, 12).unwrap(), ramp(8, 4));
        assert_eq!(r.copy_range(9, 9).unwrap(), Vec::<f32>::new());
        assert!(r.copy_range(3, 8).is_none(), "overwritten");
        assert!(r.copy_range(10, 15).is_none(), "not yet written");
    }

    #[test]
    fn oversized_push_keeps_newest() {
        let mut r = RingBuffer::new(5);
        r.reset(t0());
        r.push(&ramp(0, 12));
        assert_eq!(r.total_samples(), 12);
        assert_eq!(r.copy_range(7, 12).unwrap(), ramp(7, 5));
        r.push(&ramp(12, 3));
        assert_eq!(r.copy_range(10, 15).unwrap(), ramp(10, 5));
    }

    #[test]
    fn time_indexing_and_extract() {
        let mut r = RingBuffer::with_seconds(2.0); // 96 000 samples
        r.reset(t0());
        r.push(&ramp(0, 144_000)); // 3 s written, last 2 s retained (48 000..144 000)
        assert_eq!(r.time_of(48_000), t0() + TimeDelta::seconds(1));
        assert_eq!(r.index_at(t0() + TimeDelta::milliseconds(1500)), 72_000);
        assert_eq!(r.end_time(), t0() + TimeDelta::seconds(3));

        let e = r
            .extract(t0() + TimeDelta::milliseconds(1500), 1.0)
            .unwrap();
        assert_eq!(e.samples, ramp(72_000, 48_000));
        assert_eq!(e.start_at, t0() + TimeDelta::milliseconds(1500));

        // Starts before the oldest retained sample: clamped at the front.
        let e = r.extract(t0(), 1.5).unwrap();
        assert_eq!(e.start_at, t0() + TimeDelta::seconds(1));
        assert_eq!(e.samples, ramp(48_000, 24_000));

        // Runs past the newest sample: clamped at the end.
        let e = r
            .extract(t0() + TimeDelta::milliseconds(2500), 5.0)
            .unwrap();
        assert_eq!(e.samples.len(), 24_000);

        // Entirely overwritten, entirely in the future, or empty: None.
        assert!(r.extract(t0(), 0.5).is_none());
        assert!(r.extract(t0() + TimeDelta::seconds(4), 1.0).is_none());
        assert!(r.extract(t0() + TimeDelta::seconds(2), 0.0).is_none());

        // A reset forgets old times.
        r.reset(t0() + TimeDelta::seconds(60));
        assert!(r.extract(t0() + TimeDelta::seconds(2), 0.5).is_none());
    }

    #[test]
    fn silence_fills_a_jump() {
        let t0 = chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 5, 1, 6, 0, 0).unwrap();
        let mut r = RingBuffer::new(10);
        r.reset(t0);
        r.push(&[1.0, 2.0, 3.0]);
        r.push_silence(4);
        r.push(&[5.0]);
        assert_eq!(r.total_samples(), 8);
        assert_eq!(
            r.copy_range(0, 8).unwrap(),
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0]
        );
        r.push_silence(25); // more than the capacity: only the newest samples are kept
        assert_eq!(r.total_samples(), 33);
        assert_eq!(r.copy_range(23, 33).unwrap(), vec![0.0; 10]);
    }
}
