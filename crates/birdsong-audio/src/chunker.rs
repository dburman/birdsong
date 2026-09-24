use std::sync::{Arc, Mutex};

use birdsong_core::SAMPLE_RATE_HZ;
use chrono::{DateTime, TimeDelta, Utc};

use crate::frame::{delta_to_samples, AudioFrame};
use crate::ring::{lock_ring, RingBuffer, SharedRingBuffer};

/// Frames are fed to the ring buffer in pieces of this size, so chunks are cut before the
/// samples they need can be overwritten, whatever the frame size.
const PIECE_SAMPLES: usize = 4_800;

/// One analysis window.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    /// Exactly [`Chunker::window_samples`] mono 48 kHz samples.
    pub samples: Arc<[f32]>,
    /// UTC time of the first sample.
    pub start_at: DateTime<Utc>,
    pub source_id: Arc<str>,
    /// The end was zero-padded (only for trailing windows from [`Chunker::finish`]).
    pub padded: bool,
}

/// What the chunker produced for a frame.
#[derive(Clone, Debug, PartialEq)]
pub enum ChunkerEvent {
    Chunk(Chunk),
    /// The frame did not continue the previous one (source restart, dropped audio, clock jump).
    /// Alignment and the ring buffer were reset; the pipeline should flush per-stream state.
    Gap {
        expected: DateTime<Utc>,
        got: DateTime<Utc>,
    },
}

/// Cuts a stream of [`AudioFrame`]s into windows of the classifier's length, stepping by
/// `window − overlap`, and keeps the
/// recent audio in a [`SharedRingBuffer`] for clip extraction.
pub struct Chunker {
    source_id: Arc<str>,
    window: usize,
    step: usize,
    ring: SharedRingBuffer,
    next_start: u64,
    started: bool,
    gap_tolerance: TimeDelta,
    gaps: u64,
}

impl Chunker {
    /// `window_seconds` is the classifier's window (3 s for BirdNET V2.4). `buffer_seconds` sizes
    /// the ring buffer; it is raised to at least one window plus one piece.
    pub fn new(
        source_id: impl Into<Arc<str>>,
        window_seconds: f32,
        overlap_seconds: f32,
        buffer_seconds: f32,
    ) -> Self {
        let window = seconds_to_samples(window_seconds).max(1);
        let min_capacity = window + PIECE_SAMPLES;
        let capacity =
            ((buffer_seconds.max(0.0) * SAMPLE_RATE_HZ as f32).ceil() as usize).max(min_capacity);
        Self {
            source_id: source_id.into(),
            window,
            step: Self::step_samples(window_seconds, overlap_seconds),
            ring: Arc::new(Mutex::new(RingBuffer::new(capacity))),
            next_start: 0,
            started: false,
            gap_tolerance: TimeDelta::seconds(1),
            gaps: 0,
        }
    }

    /// How far a frame's timestamp may deviate from the expected one before it counts as a gap.
    pub fn with_gap_tolerance(mut self, tolerance: TimeDelta) -> Self {
        self.gap_tolerance = tolerance;
        self
    }

    /// Samples between consecutive chunk starts: `(window − overlap) × 48 kHz`, at least 1.
    pub fn step_samples(window_seconds: f32, overlap_seconds: f32) -> usize {
        seconds_to_samples(window_seconds - overlap_seconds).max(1)
    }

    /// Samples in one window.
    pub fn window_samples(&self) -> usize {
        self.window
    }

    /// A trailing partial window shorter than half a window is dropped (BirdNET `splitSignal`
    /// minlen: 1.5 s of a 3 s window).
    pub fn min_tail_samples(&self) -> usize {
        self.window / 2
    }

    /// Handle for the clip writer.
    pub fn ring(&self) -> SharedRingBuffer {
        Arc::clone(&self.ring)
    }

    pub fn source_id(&self) -> &Arc<str> {
        &self.source_id
    }

    /// Gaps detected so far.
    pub fn gaps(&self) -> u64 {
        self.gaps
    }

    /// Feed one frame; returns any gap notice followed by the chunks completed by it.
    pub fn push(&mut self, frame: AudioFrame) -> Vec<ChunkerEvent> {
        let mut events = Vec::new();
        let ring = Arc::clone(&self.ring);
        let mut ring = lock_ring(&ring);

        if !self.started {
            ring.reset(frame.captured_at);
            self.next_start = 0;
            self.started = true;
        } else {
            let expected = ring.end_time();
            let jump = frame.captured_at - expected;
            if jump.abs() > self.gap_tolerance {
                self.gaps += 1;
                let missing = delta_to_samples(jump);
                // A forward jump (lost audio, or the source re-anchoring its clock to the wall
                // clock) is filled with silence so the audio before it stays in the buffer: clips
                // of detections still being confirmed need it. A backward jump, or one longer
                // than the buffer, starts the buffer afresh.
                let keep = missing > 0 && (missing as u64) < ring.capacity() as u64;
                if keep {
                    ring.push_silence(missing as u64);
                } else {
                    ring.reset(frame.captured_at);
                }
                tracing::warn!(
                    source = %self.source_id,
                    %expected,
                    got = %frame.captured_at,
                    buffered_audio_kept = keep,
                    "audio gap detected; realigning chunks"
                );
                events.push(ChunkerEvent::Gap {
                    expected,
                    got: frame.captured_at,
                });
                // Resume with the new audio; the partial window before the jump is dropped.
                self.next_start = ring.total_samples();
            }
        }

        for piece in frame.samples.chunks(PIECE_SAMPLES) {
            ring.push(piece);
            while ring.total_samples() >= self.next_start + self.window as u64 {
                let Some(samples) =
                    ring.copy_range(self.next_start, self.next_start + self.window as u64)
                else {
                    // Unreachable with capacity >= window + PIECE_SAMPLES; never loop forever.
                    tracing::error!(source = %self.source_id, "chunk no longer in ring buffer; skipping");
                    self.next_start = ring.total_samples();
                    break;
                };
                events.push(ChunkerEvent::Chunk(Chunk {
                    samples: samples.into(),
                    start_at: ring.time_of(self.next_start),
                    source_id: Arc::clone(&self.source_id),
                    padded: false,
                }));
                self.next_start += self.step as u64;
            }
        }
        events
    }

    /// End of stream: emit trailing windows of at least half a window, zero-padded. The next
    /// [`Chunker::push`] starts a fresh stream.
    pub fn finish(&mut self) -> Vec<Chunk> {
        let ring = Arc::clone(&self.ring);
        let ring = lock_ring(&ring);
        let mut out = Vec::new();
        if self.started {
            while ring.total_samples() >= self.next_start + self.min_tail_samples() as u64 {
                let end = ring
                    .total_samples()
                    .min(self.next_start + self.window as u64);
                let Some(mut samples) = ring.copy_range(self.next_start, end) else {
                    break;
                };
                let padded = samples.len() < self.window;
                samples.resize(self.window, 0.0);
                out.push(Chunk {
                    samples: samples.into(),
                    start_at: ring.time_of(self.next_start),
                    source_id: Arc::clone(&self.source_id),
                    padded,
                });
                self.next_start += self.step as u64;
            }
        }
        self.started = false;
        out
    }
}

fn seconds_to_samples(seconds: f32) -> usize {
    (f64::from(seconds.max(0.0)) * f64::from(SAMPLE_RATE_HZ)).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use birdsong_core::{CHUNK_SAMPLES, CHUNK_SECONDS};
    use chrono::TimeZone;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap()
    }

    /// `seconds` of a ramp (value = sample index) delivered in frames of `frame` samples.
    fn frames(seconds: f32, frame: usize, start: DateTime<Utc>) -> Vec<AudioFrame> {
        let n = (seconds * 48_000.0) as usize;
        let all: Vec<f32> = (0..n).map(|i| i as f32).collect();
        all.chunks(frame)
            .enumerate()
            .map(|(k, s)| AudioFrame {
                samples: s.to_vec(),
                captured_at: start + crate::samples_to_delta((k * frame) as u64),
            })
            .collect()
    }

    fn run(chunker: &mut Chunker, frames: Vec<AudioFrame>) -> (Vec<Chunk>, usize) {
        let mut chunks = Vec::new();
        let mut gaps = 0;
        for f in frames {
            for e in chunker.push(f) {
                match e {
                    ChunkerEvent::Chunk(c) => chunks.push(c),
                    ChunkerEvent::Gap { .. } => gaps += 1,
                }
            }
        }
        (chunks, gaps)
    }

    #[test]
    fn no_overlap_ten_seconds_gives_three_chunks() {
        let mut c = Chunker::new("mic0", CHUNK_SECONDS, 0.0, 90.0);
        let (chunks, gaps) = run(&mut c, frames(10.0, 4_800, t0()));
        assert_eq!(gaps, 0);
        assert_eq!(chunks.len(), 3);
        for (k, ch) in chunks.iter().enumerate() {
            assert_eq!(ch.start_at, t0() + TimeDelta::seconds(3 * k as i64));
            assert_eq!(ch.samples.len(), CHUNK_SAMPLES);
            assert_eq!(ch.samples[0], (k * 144_000) as f32);
            assert_eq!(
                ch.samples[CHUNK_SAMPLES - 1],
                (k * 144_000 + CHUNK_SAMPLES - 1) as f32
            );
            assert!(!ch.padded);
            assert_eq!(&*ch.source_id, "mic0");
        }
        assert!(c.finish().is_empty(), "1 s tail is below the 1.5 s minimum");
    }

    #[test]
    fn overlap_one_and_a_half_seconds_gives_five_chunks_plus_padded_tail() {
        let mut c = Chunker::new("mic0", CHUNK_SECONDS, 1.5, 90.0);
        let (chunks, _) = run(&mut c, frames(10.0, 4_800, t0()));
        assert_eq!(chunks.len(), 5);
        let starts: Vec<_> = chunks.iter().map(|ch| ch.samples[0] as usize).collect();
        assert_eq!(starts, [0, 72_000, 144_000, 216_000, 288_000]);
        assert_eq!(chunks[1].start_at, t0() + TimeDelta::milliseconds(1500));

        let tail = c.finish();
        assert_eq!(tail.len(), 1, "window at 7.5 s has 2.5 s of audio");
        assert!(tail[0].padded);
        assert_eq!(tail[0].samples[0], 360_000.0);
        assert_eq!(tail[0].samples[119_999], 479_999.0);
        assert_eq!(tail[0].samples[120_000], 0.0);
        assert_eq!(tail[0].start_at, t0() + TimeDelta::milliseconds(7500));
    }

    #[test]
    fn frame_size_does_not_change_chunks() {
        let mut a = Chunker::new("s", CHUNK_SECONDS, 0.5, 0.0); // minimum-size ring buffer
        let mut b = Chunker::new("s", CHUNK_SECONDS, 0.5, 0.0);
        let (ca, _) = run(&mut a, frames(20.0, 1_000, t0()));
        let (cb, _) = run(&mut b, frames(20.0, 20 * 48_000, t0())); // one frame larger than the ring
        assert_eq!(ca, cb);
        assert_eq!(ca.len(), 7);
    }

    #[test]
    fn forward_gap_realigns_and_keeps_buffered_audio() {
        let mut c = Chunker::new("mic0", CHUNK_SECONDS, 0.0, 90.0);
        let ring = c.ring();
        let mut fs = frames(4.0, 4_800, t0());
        // Second stretch starts 10 s later than it should.
        fs.extend(frames(6.0, 4_800, t0() + TimeDelta::seconds(14)));
        let (chunks, gaps) = run(&mut c, fs);
        assert_eq!(gaps, 1);
        assert_eq!(c.gaps(), 1);
        let starts: Vec<_> = chunks.iter().map(|ch| ch.start_at).collect();
        assert_eq!(
            starts,
            [
                t0(),
                t0() + TimeDelta::seconds(14),
                t0() + TimeDelta::seconds(17)
            ],
            "the 1 s left before the gap is not analysed; chunks realign to the new stretch"
        );
        assert_eq!(
            chunks[1].samples[0], 0.0,
            "new stretch starts at its own sample 0"
        );

        // The audio before the gap is still there, at its own time, for clips.
        let r = lock_ring(&ring);
        let before = r.extract(t0() + TimeDelta::seconds(1), 2.0).unwrap();
        assert_eq!(before.start_at, t0() + TimeDelta::seconds(1));
        assert_eq!(before.samples[0], 48_000.0);
        assert_eq!(before.samples[95_999], 143_999.0);
        // A clip across the gap has the real audio, then silence where audio is missing.
        let across = r.extract(t0() + TimeDelta::seconds(3), 2.0).unwrap();
        assert_eq!(across.samples[0], 144_000.0);
        assert_eq!(across.samples[47_999], 191_999.0);
        assert!(across.samples[48_000..].iter().all(|&s| s == 0.0));
        assert_eq!(r.end_time(), t0() + TimeDelta::seconds(20));
    }

    #[test]
    fn clock_reanchor_jump_keeps_audio() {
        // A live source re-anchoring to the wall clock moves its timestamps 2 s ahead with no
        // audio missing; recent audio must stay extractable.
        let mut c = Chunker::new("mic0", 5.0, 0.0, 90.0);
        let ring = c.ring();
        let mut fs = frames(12.0, 4_800, t0());
        fs.extend(frames(10.0, 4_800, t0() + TimeDelta::seconds(14)));
        let (chunks, gaps) = run(&mut c, fs);
        assert_eq!(gaps, 1);
        assert_eq!(chunks.len(), 4, "0 s and 5 s before, 14 s and 19 s after");
        let r = lock_ring(&ring);
        let clip = r.extract(t0() + TimeDelta::seconds(9), 6.0).unwrap();
        assert_eq!(clip.samples.len(), 288_000);
        assert_eq!(clip.samples[0], 432_000.0);
    }

    #[test]
    fn backward_or_huge_gaps_reset_the_buffer() {
        // Timestamps going backwards cannot be filled: start afresh.
        let mut c = Chunker::new("mic0", CHUNK_SECONDS, 0.0, 90.0);
        let mut fs = frames(4.0, 4_800, t0());
        fs.extend(frames(6.0, 4_800, t0() + TimeDelta::seconds(1)));
        let (chunks, gaps) = run(&mut c, fs);
        assert_eq!(gaps, 1);
        let starts: Vec<_> = chunks.iter().map(|ch| ch.start_at).collect();
        assert_eq!(
            starts,
            [
                t0(),
                t0() + TimeDelta::seconds(1),
                t0() + TimeDelta::seconds(4)
            ]
        );

        // A gap longer than the whole buffer would push everything out anyway.
        let mut c = Chunker::new("mic0", CHUNK_SECONDS, 0.0, 90.0);
        let ring = c.ring();
        let mut fs = frames(4.0, 4_800, t0());
        fs.extend(frames(6.0, 4_800, t0() + TimeDelta::seconds(104)));
        let (chunks, _) = run(&mut c, fs);
        assert_eq!(chunks[1].start_at, t0() + TimeDelta::seconds(104));
        assert!(lock_ring(&ring)
            .extract(t0() + TimeDelta::seconds(1), 2.0)
            .is_none());
    }

    #[test]
    fn small_jitter_is_not_a_gap() {
        let mut c = Chunker::new("mic0", CHUNK_SECONDS, 0.0, 90.0);
        let mut fs = frames(6.0, 4_800, t0());
        fs[30].captured_at += TimeDelta::milliseconds(200);
        let (chunks, gaps) = run(&mut c, fs);
        assert_eq!(gaps, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].start_at, t0() + TimeDelta::seconds(3));
    }

    #[test]
    fn five_second_windows() {
        let mut c = Chunker::new("mic0", 5.0, 1.0, 90.0);
        assert_eq!(c.window_samples(), 240_000);
        let (chunks, _) = run(&mut c, frames(14.0, 4_800, t0()));
        let starts: Vec<_> = chunks.iter().map(|ch| ch.samples[0] as usize).collect();
        assert_eq!(starts, [0, 192_000, 384_000], "4 s steps");
        assert!(chunks.iter().all(|ch| ch.samples.len() == 240_000));
        assert!(
            c.finish().is_empty(),
            "the 2 s left at 12 s is below half a window"
        );
    }

    #[test]
    fn ring_buffer_is_shared() {
        let mut c = Chunker::new("mic0", CHUNK_SECONDS, 0.0, 60.0);
        let ring = c.ring();
        run(&mut c, frames(5.0, 4_800, t0()));
        let r = lock_ring(&ring);
        assert_eq!(r.total_samples(), 240_000);
        let clip = r.extract(t0() + TimeDelta::seconds(1), 2.0).unwrap();
        assert_eq!(clip.samples[0], 48_000.0);
        assert_eq!(clip.samples.len(), 96_000);
    }
}
