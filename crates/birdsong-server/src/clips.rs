//! Saving detection clips: cut from the source's ring buffer, written as WAV (plus an optional
//! spectrogram PNG) under `<data_dir>/clips/<local date>/<Species>/`, and attached to every
//! detection of the window.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use birdsong_audio::{
    lock_ring, samples_to_delta, spectrogram, wav, SharedRingBuffer, SpectrogramOptions,
};
use birdsong_core::{local_date_and_hour, sanitize_name, Config, CHUNK_SECONDS, SAMPLE_RATE_HZ};
use birdsong_store::{ClipInfo, DetectionStore};
use chrono::{DateTime, TimeDelta, Utc};
use chrono_tz::Tz;
use tokio::sync::mpsc::{self, error::TryRecvError};

use crate::stats::PipelineStats;

/// Where and how clips are written.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipSettings {
    pub clips_dir: PathBuf,
    pub clip_seconds: f32,
    pub spectrograms: bool,
    pub timezone: Tz,
}

impl ClipSettings {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            clips_dir: cfg.storage.clips_dir(),
            clip_seconds: cfg.storage.clip_seconds,
            spectrograms: cfg.storage.spectrograms,
            timezone: cfg.station.timezone,
        }
    }
}

/// Start and length of the clip for a chunk: `clip_seconds` (at least 3) centred on the 3 s window
/// (BUILD_PLAN §7.7). The default 6 s adds 1.5 s either side.
pub fn clip_window(chunk_start: DateTime<Utc>, clip_seconds: f32) -> (DateTime<Utc>, f32) {
    let duration = clip_seconds.max(CHUNK_SECONDS);
    let pad = (duration - CHUNK_SECONDS) / 2.0;
    (
        chunk_start - TimeDelta::microseconds((f64::from(pad) * 1e6).round() as i64),
        duration,
    )
}

/// `YYYY-MM-DD/Common_Name/YYYY-MM-DDTHH-MM-SS.mmmZ_source_0.87.wav` (date in the station time zone,
/// time in UTC, confidence of the best detection).
pub fn clip_relative_path(
    chunk_start: DateTime<Utc>,
    source_id: &str,
    common_name: &str,
    confidence: f32,
    tz: Tz,
) -> String {
    let (date, _) = local_date_and_hour(chunk_start, tz);
    let or = |s: String, fallback: &str| {
        if s.is_empty() {
            fallback.to_string()
        } else {
            s
        }
    };
    format!(
        "{}/{}/{}_{}_{:.2}.wav",
        date.format("%Y-%m-%d"),
        or(sanitize_name(common_name), "Unknown"),
        chunk_start.format("%Y-%m-%dT%H-%M-%S%.3fZ"),
        or(sanitize_name(source_id), "source"),
        confidence
    )
}

/// Write the WAV (and spectrogram) atomically: `.tmp` then rename. A failed spectrogram is logged
/// and skipped; it never costs the clip.
pub fn write_clip_files(
    settings: &ClipSettings,
    relative: &str,
    samples: &[f32],
) -> anyhow::Result<ClipInfo> {
    let path = settings.clips_dir.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension("wav.tmp");
    wav::write_wav(&tmp, samples, SAMPLE_RATE_HZ)?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))?;
    let clip_bytes = std::fs::metadata(&path)
        .with_context(|| format!("reading {}", path.display()))?
        .len();

    let spectrogram_path = if settings.spectrograms {
        let relative_png = format!("{}.png", relative.strip_suffix(".wav").unwrap_or(relative));
        let png = settings.clips_dir.join(&relative_png);
        let tmp = png.with_extension("png.tmp");
        let written = spectrogram::write_png(
            &tmp,
            samples,
            SAMPLE_RATE_HZ,
            &SpectrogramOptions::default(),
        )
        .map_err(anyhow::Error::from)
        .and_then(|_| std::fs::rename(&tmp, &png).context("renaming spectrogram"));
        match written {
            Ok(()) => Some(relative_png),
            Err(e) => {
                tracing::warn!(path = %png.display(), error = %e, "spectrogram not written");
                let _ = std::fs::remove_file(&tmp);
                None
            }
        }
    } else {
        None
    };
    Ok(ClipInfo {
        clip_path: relative.to_string(),
        clip_bytes,
        spectrogram_path,
    })
}

/// A window whose detections were stored and need a clip.
#[derive(Clone, Debug)]
pub(crate) struct ClipJob {
    pub ids: Vec<i64>,
    pub source_id: String,
    pub start_at: DateTime<Utc>,
    pub common_name: String,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub(crate) enum ClipMsg {
    Job(ClipJob),
    /// No more audio will arrive for this source; stop waiting for it.
    SourceEnded(Arc<str>),
}

/// Handle clip jobs in order. A job waits (up to the clip length plus 5 s) until its source has
/// captured the audio after the window, unless that source has ended.
pub(crate) async fn clip_task(
    settings: ClipSettings,
    rings: HashMap<Arc<str>, SharedRingBuffer>,
    store: Arc<dyn DetectionStore>,
    mut rx: mpsc::Receiver<ClipMsg>,
    stats: Arc<PipelineStats>,
) {
    let mut pending: VecDeque<ClipMsg> = VecDeque::new();
    let mut ended: HashSet<Arc<str>> = HashSet::new();
    let mut closed = false;

    loop {
        let msg = match pending.pop_front() {
            Some(m) => m,
            None if closed => break,
            None => match rx.recv().await {
                Some(m) => m,
                None => break,
            },
        };
        let job = match msg {
            ClipMsg::SourceEnded(source) => {
                ended.insert(source);
                continue;
            }
            ClipMsg::Job(job) => job,
        };
        let Some(ring) = rings.get(job.source_id.as_str()) else {
            tracing::warn!(source = %job.source_id, "no ring buffer for source; clip not saved");
            stats.clip_error();
            continue;
        };

        let (start, duration) = clip_window(job.start_at, settings.clip_seconds);
        let needed_end = start
            + samples_to_delta((f64::from(duration) * f64::from(SAMPLE_RATE_HZ)).round() as u64);
        let deadline = Instant::now() + Duration::from_secs_f32(duration + 5.0);
        loop {
            let ring_end = lock_ring(ring).end_time();
            if ring_end >= needed_end
                || closed
                || ended.contains(job.source_id.as_str())
                || Instant::now() >= deadline
            {
                break;
            }
            loop {
                match rx.try_recv() {
                    Ok(ClipMsg::SourceEnded(source)) => {
                        ended.insert(Arc::clone(&source));
                        pending.push_back(ClipMsg::SourceEnded(source));
                    }
                    Ok(other) => pending.push_back(other),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let extracted = lock_ring(ring).extract(start, duration);
        let Some(clip) = extracted else {
            tracing::warn!(source = %job.source_id, start_at = %job.start_at, "audio no longer buffered; clip not saved");
            stats.clip_error();
            continue;
        };
        let relative = clip_relative_path(
            job.start_at,
            &job.source_id,
            &job.common_name,
            job.confidence,
            settings.timezone,
        );
        let write_settings = settings.clone();
        let written = tokio::task::spawn_blocking(move || {
            write_clip_files(&write_settings, &relative, &clip.samples)
        })
        .await;
        match written {
            Ok(Ok(info)) => match store.set_clip(&job.ids, Some(&info)).await {
                Ok(()) => {
                    stats.clip_written();
                    tracing::debug!(clip = %info.clip_path, bytes = info.clip_bytes, "clip saved");
                }
                Err(e) => {
                    stats.clip_error();
                    tracing::error!(error = %e, clip = %info.clip_path, "clip written but not recorded");
                }
            },
            Ok(Err(e)) => {
                stats.clip_error();
                tracing::error!(error = %format!("{e:#}"), "failed to write clip");
            }
            Err(e) => {
                stats.clip_error();
                tracing::error!(error = %e, "clip writing task panicked");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use birdsong_audio::RingBuffer;
    use chrono::TimeZone;

    fn t(secs: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 15, 10, 0, 0).unwrap() + TimeDelta::seconds(secs)
    }

    #[test]
    fn window_is_centred() {
        assert_eq!(
            clip_window(t(30), 6.0),
            (t(30) - TimeDelta::milliseconds(1500), 6.0)
        );
        assert_eq!(clip_window(t(30), 3.0), (t(30), 3.0));
        assert_eq!(
            clip_window(t(30), 1.0),
            (t(30), 3.0),
            "never shorter than the chunk"
        );
        assert_eq!(
            clip_window(t(30), 10.0).0,
            t(30) - TimeDelta::milliseconds(3500)
        );
    }

    #[test]
    fn relative_path_layout() {
        let p = clip_relative_path(
            t(0),
            "mic0",
            "Black-capped Chickadee",
            0.752,
            chrono_tz::America::New_York,
        );
        assert_eq!(
            p,
            "2026-05-15/Black_capped_Chickadee/2026-05-15T10-00-00.000Z_mic0_0.75.wav"
        );
        // 02:00 UTC is still the previous day in New York.
        let late = Utc.with_ymd_and_hms(2026, 5, 16, 2, 0, 0).unwrap();
        assert!(
            clip_relative_path(late, "", "", 0.9, chrono_tz::America::New_York)
                .starts_with("2026-05-15/Unknown/")
        );
    }

    #[test]
    fn extraction_is_exact_when_buffered_and_clamped_otherwise() {
        let mut ring = RingBuffer::with_seconds(90.0);
        ring.reset(t(0));
        ring.push(&vec![0.25; 20 * 48_000]);
        let (start, duration) = clip_window(t(6), 6.0);
        assert_eq!(
            ring.extract(start, duration).unwrap().samples.len(),
            6 * 48_000
        );
        let (start, duration) = clip_window(t(0), 6.0);
        let clamped = ring.extract(start, duration).unwrap();
        assert_eq!(
            clamped.samples.len(),
            216_000,
            "1.5 s before the recording started is not available"
        );
        assert_eq!(clamped.start_at, t(0));
        let (start, duration) = clip_window(t(18), 6.0);
        assert_eq!(
            ring.extract(start, duration).unwrap().samples.len(),
            168_000,
            "16.5 s to the newest sample at 20 s"
        );
    }

    #[test]
    fn files_are_written_atomically_with_spectrogram() {
        let dir = tempfile::tempdir().unwrap();
        let settings = ClipSettings {
            clips_dir: dir.path().join("clips"),
            clip_seconds: 6.0,
            spectrograms: true,
            timezone: chrono_tz::UTC,
        };
        let rel = "2026-05-15/Wren/2026-05-15T10-00-00.000Z_mic0_0.80.wav";
        let info = write_clip_files(&settings, rel, &vec![0.1; 48_000]).unwrap();
        assert_eq!(info.clip_path, rel);
        assert_eq!(info.clip_bytes, 44 + 2 * 48_000);
        let png = info.spectrogram_path.as_deref().unwrap();
        assert_eq!(
            png,
            "2026-05-15/Wren/2026-05-15T10-00-00.000Z_mic0_0.80.png"
        );
        assert!(settings.clips_dir.join(png).exists());
        let leftovers: Vec<_> = std::fs::read_dir(settings.clips_dir.join("2026-05-15/Wren"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }
}
