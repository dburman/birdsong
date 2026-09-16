//! Capture sources end to end. ffmpeg tests skip when ffmpeg is not installed.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use birdsong_audio::{
    wav, AudioError, AudioFrame, AudioSource, Chunker, ChunkerEvent, FfmpegOptions, FfmpegSource,
    Pacing, WavFileSource,
};
use birdsong_core::config::{AudioSourceConfig, AudioSourceKind};
use chrono::{TimeDelta, TimeZone, Utc};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/soundscape_15s.wav")
}

fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn source_cfg(kind: AudioSourceKind, target: &Path) -> AudioSourceConfig {
    let t = target.display().to_string();
    AudioSourceConfig {
        id: "test".into(),
        kind,
        device: (kind == AudioSourceKind::Alsa).then(|| t.clone()),
        url: (kind == AudioSourceKind::Rtsp).then(|| t.clone()),
        path: (kind == AudioSourceKind::File).then(|| target.to_path_buf()),
        gain_db: 0.0,
    }
}

/// Run a source to completion (or until `cancel_after`), collecting every frame.
async fn collect(
    source: Box<dyn AudioSource>,
    cancel_after: Option<Duration>,
) -> (Result<(), AudioError>, Vec<AudioFrame>, Duration) {
    let (tx, mut rx) = mpsc::channel(4_096);
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let handle = tokio::spawn(source.run(tx, cancel.clone()));
    if let Some(after) = cancel_after {
        let c = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            c.cancel();
        });
    }
    let mut frames = Vec::new();
    while let Some(f) = rx.recv().await {
        frames.push(f);
    }
    let result = handle.await.expect("source task panicked");
    (result, frames, started.elapsed())
}

fn concat(frames: &[AudioFrame]) -> Vec<f32> {
    frames
        .iter()
        .flat_map(|f| f.samples.iter().copied())
        .collect()
}

fn fast_opts() -> FfmpegOptions {
    FfmpegOptions {
        realtime_files: false,
        fast_start_at: Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap(),
        ..FfmpegOptions::default()
    }
}

#[tokio::test]
async fn wav_source_fast_reproduces_file() {
    let reference = wav::read_wav_48k_mono(&fixture()).unwrap();
    let start = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap();
    let src = WavFileSource::new("f", fixture(), Pacing::Fast { start_at: start });
    let (res, frames, _) = collect(Box::new(src), None).await;
    res.unwrap();
    assert_eq!(
        frames.len(),
        151,
        "720 896 samples in frames of 4 800, last one partial"
    );
    assert_eq!(frames[150].samples.len(), 896);
    assert_eq!(concat(&frames), reference);
    assert_eq!(frames[10].captured_at, start + TimeDelta::seconds(1));
}

#[tokio::test]
async fn wav_source_realtime_is_paced_and_cancellable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("half.wav");
    wav::write_wav(&path, &vec![0.1; 24_000], 48_000).unwrap(); // 0.5 s
    let (res, frames, elapsed) = collect(
        Box::new(WavFileSource::new("r", &path, Pacing::Realtime)),
        None,
    )
    .await;
    res.unwrap();
    assert_eq!(frames.len(), 5);
    assert!(
        elapsed >= Duration::from_millis(450),
        "played too fast: {elapsed:?}"
    );

    let src = WavFileSource::new("r", fixture(), Pacing::Realtime);
    let (res, frames, elapsed) = collect(Box::new(src), Some(Duration::from_millis(300))).await;
    res.unwrap();
    assert!(elapsed < Duration::from_secs(2), "cancel took {elapsed:?}");
    assert!(frames.len() <= 5, "{} frames", frames.len());
}

#[tokio::test]
async fn chunker_on_fixture_matches_direct_slices() {
    let reference = wav::read_wav_48k_mono(&fixture()).unwrap();
    let start = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap();
    let src = WavFileSource::new("f", fixture(), Pacing::Fast { start_at: start })
        .with_frame_samples(1_234);
    let (res, frames, _) = collect(Box::new(src), None).await;
    res.unwrap();

    let mut chunker = Chunker::new("f", birdsong_core::CHUNK_SECONDS, 0.0, 90.0);
    let mut chunks = Vec::new();
    for f in frames {
        for e in chunker.push(f) {
            if let ChunkerEvent::Chunk(c) = e {
                chunks.push(c);
            }
        }
    }
    chunks.extend(chunker.finish());
    assert_eq!(chunks.len(), 5, "same count as the golden export");
    for (k, c) in chunks.iter().enumerate() {
        assert_eq!(&c.samples[..], &reference[k * 144_000..(k + 1) * 144_000]);
        assert_eq!(c.start_at, start + TimeDelta::seconds(3 * k as i64));
    }
}

#[tokio::test]
async fn ffmpeg_file_fast_matches_hound() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not installed");
        return;
    }
    let reference = wav::read_wav_48k_mono(&fixture()).unwrap();
    let src =
        FfmpegSource::new(source_cfg(AudioSourceKind::File, &fixture()), fast_opts()).unwrap();
    let (res, frames, _) = collect(Box::new(src), None).await;
    res.unwrap();
    let ours = concat(&frames);
    assert_eq!(ours.len(), reference.len());
    let max_err = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-6,
        "ffmpeg s16→f32 differs from hound: {max_err}"
    );
    for w in frames.windows(2) {
        assert_eq!(
            w[1].captured_at - w[0].captured_at,
            TimeDelta::milliseconds(100)
        );
    }
}

#[tokio::test]
async fn ffmpeg_applies_gain() {
    if !ffmpeg_available() {
        return;
    }
    let reference = wav::read_wav_48k_mono(&fixture()).unwrap();
    let mut cfg = source_cfg(AudioSourceKind::File, &fixture());
    cfg.gain_db = 6.020_6; // ×2
    let (res, frames, _) =
        collect(Box::new(FfmpegSource::new(cfg, fast_opts()).unwrap()), None).await;
    res.unwrap();
    let ours = concat(&frames);
    let max_err = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - 2.0 * b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "{max_err}");
}

#[tokio::test]
async fn ffmpeg_realtime_file_cancels_promptly() {
    if !ffmpeg_available() {
        return;
    }
    let opts = FfmpegOptions {
        realtime_files: true,
        ..FfmpegOptions::default()
    };
    let src = FfmpegSource::new(source_cfg(AudioSourceKind::File, &fixture()), opts).unwrap();
    let (res, frames, elapsed) = collect(Box::new(src), Some(Duration::from_millis(1_000))).await;
    res.unwrap();
    assert!(elapsed < Duration::from_secs(3), "cancel took {elapsed:?}");
    assert!(
        !frames.is_empty() && frames.len() <= 30,
        "{} frames in ~1 s",
        frames.len()
    );
    let age = Utc::now() - frames[0].captured_at;
    assert!(
        age < TimeDelta::seconds(5),
        "realtime frames carry wall-clock time"
    );
}

#[tokio::test]
async fn ffmpeg_live_source_restarts_after_exit() {
    if !ffmpeg_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short.wav");
    wav::write_wav(&path, &vec![0.2; 14_400], 48_000).unwrap(); // 0.3 s
                                                                // kind=rtsp with a plain path: ffmpeg reads it quickly and exits, like a dropped stream.
    let opts = FfmpegOptions {
        restart_backoff_min: Duration::from_millis(50),
        restart_backoff_max: Duration::from_millis(100),
        ..FfmpegOptions::default()
    };
    let src = FfmpegSource::new(source_cfg(AudioSourceKind::Rtsp, &path), opts).unwrap();
    let (res, frames, _) = collect(Box::new(src), Some(Duration::from_millis(2_500))).await;
    res.unwrap();
    let total: usize = frames.iter().map(|f| f.samples.len()).sum();
    assert!(
        total >= 2 * 14_400,
        "expected at least one restart, got {total} samples"
    );
}

#[tokio::test]
async fn missing_ffmpeg_is_fatal_even_for_live_sources() {
    let opts = FfmpegOptions {
        ffmpeg_path: "/nonexistent/ffmpeg".into(),
        ..FfmpegOptions::default()
    };
    let src =
        FfmpegSource::new(source_cfg(AudioSourceKind::Alsa, Path::new("hw:9,0")), opts).unwrap();
    let (res, frames, elapsed) = collect(Box::new(src), Some(Duration::from_secs(10))).await;
    assert!(matches!(res, Err(AudioError::FfmpegNotFound(_))), "{res:?}");
    assert!(frames.is_empty());
    assert!(elapsed < Duration::from_secs(2));
}

#[tokio::test]
async fn ffmpeg_bad_file_reports_stderr() {
    if !ffmpeg_available() {
        return;
    }
    let src = FfmpegSource::new(
        source_cfg(AudioSourceKind::File, Path::new("/nonexistent/x.wav")),
        fast_opts(),
    )
    .unwrap();
    let (res, _, _) = collect(Box::new(src), None).await;
    match res {
        Err(AudioError::FfmpegFailed { stderr, .. }) => {
            assert!(stderr.contains("nonexistent"), "{stderr}")
        }
        other => panic!("expected FfmpegFailed, got {other:?}"),
    }
}
