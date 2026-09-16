//! Pipeline end to end with real models. Skips when `models/` is absent.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use birdsong_audio::{FfmpegOptions, FfmpegSource, Pacing, WavFileSource};
use birdsong_core::config::{AudioSourceConfig, AudioSourceKind};
use birdsong_core::Config;
use birdsong_model::ModelBundle;
use birdsong_server::{Backpressure, Pipeline, PipelineOptions, SourceSpec};
use birdsong_store::{DetectionQuery, DetectionStore, SqliteStore, StoreOptions};
use chrono::{TimeZone, Utc};
use tokio_util::sync::CancellationToken;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn models_present() -> bool {
    repo_root()
        .join("models/birdnet-v2.4-headless.onnx")
        .exists()
}

fn fixture() -> PathBuf {
    repo_root().join("tools/fixtures/soundscape_15s.wav")
}

fn config(data_dir: &Path) -> Config {
    config_with(data_dir, "")
}

/// `extra` is spliced in before the audio sources, so it may open its own tables.
fn config_with(data_dir: &Path, extra: &str) -> Config {
    Config::from_toml(&format!(
        r#"
[station]
latitude = 42.36
longitude = -71.06
timezone = "America/New_York"
{extra}
[[audio.sources]]
id = "file0"
kind = "file"
path = {fixture:?}
[model]
dir = {models:?}
[storage]
data_dir = {data:?}
"#,
        extra = extra,
        fixture = fixture().display().to_string(),
        models = repo_root().join("models").display().to_string(),
        data = data_dir.display().to_string(),
    ))
    .unwrap()
}

async fn setup(dir: &Path) -> (Config, ModelBundle, SqliteStore) {
    let cfg = config(dir);
    let bundle = ModelBundle::load(&cfg).unwrap();
    let store = SqliteStore::open(
        &cfg.storage.database_path(),
        StoreOptions::from_config(&cfg),
    )
    .await
    .unwrap();
    (cfg, bundle, store)
}

#[tokio::test(flavor = "multi_thread")]
async fn fixture_detections_are_stored_and_broadcast() {
    if !models_present() {
        eprintln!("skipping: models not present");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (cfg, bundle, store) = setup(dir.path()).await;
    let start = Utc.with_ymd_and_hms(2026, 5, 15, 10, 0, 0).unwrap();
    let source = WavFileSource::new("file0", fixture(), Pacing::Fast { start_at: start });
    let pipeline = Pipeline::with_sources(
        cfg,
        bundle,
        Arc::new(store.clone()),
        vec![SourceSpec {
            source: Box::new(source),
            backpressure: Backpressure::Wait,
        }],
        PipelineOptions {
            exit_on_eof: true,
            fast_files: true,
        },
    );
    let mut rx = pipeline.subscribe();
    let summary = tokio::time::timeout(
        Duration::from_secs(60),
        pipeline.run(CancellationToken::new()),
    )
    .await
    .expect("pipeline finished")
    .expect("pipeline ok");

    assert_eq!(summary.chunks_processed, 5, "{summary:?}");
    assert_eq!(summary.chunks_dropped, 0, "Wait backpressure never drops");
    assert!(summary.mean_inference_ms.is_some());

    let rows = store
        .list(&DetectionQuery {
            species: Some("Poecile atricapillus".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(
        !rows.is_empty(),
        "chickadee expected in the fixture; summary {summary:?}"
    );
    assert_eq!(
        rows.last().unwrap().detected_at,
        start,
        "first chunk starts at the source's start time"
    );
    assert_eq!(
        summary.detections as usize,
        store.list(&DetectionQuery::default()).await.unwrap().len()
    );

    let first = rx.try_recv().expect("broadcast of stored detection");
    assert!(first.id.is_some());

    assert!(summary.clips_written >= 1, "{summary:?}");
    assert_eq!(summary.clip_errors, 0, "{summary:?}");
    let chunk0 = store.get(rows.last().unwrap().id).await.unwrap().unwrap();
    let clip = chunk0
        .clip_path
        .expect("clip saved for the chickadee detection");
    assert_eq!(
        clip,
        "2026-05-15/Black_capped_Chickadee/2026-05-15T10-00-00.000Z_file0_0.75.flac"
    );
    let clips_dir = dir.path().join("clips");
    let mut reader = claxon::FlacReader::open(clips_dir.join(&clip)).expect("clip is FLAC");
    assert_eq!(reader.streaminfo().sample_rate, 48_000);
    let decoded: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
    assert_eq!(
        decoded.len(),
        216_000,
        "6 s window clamped at the start of the recording"
    );
    let png_rel = chunk0.spectrogram_path.expect("spectrogram saved");
    let decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(clips_dir.join(&png_rel)).unwrap(),
    ));
    let reader = decoder.read_info().unwrap();
    assert_eq!((reader.info().width, reader.info().height), (800, 300));
}

/// Run the fixture once and report `(stored, unconfirmed, chunks)`.
async fn run_fixture(extra: &str) -> (u64, u64, u64) {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config_with(dir.path(), extra);
    let bundle = ModelBundle::load(&cfg).unwrap();
    let store = SqliteStore::open(
        &cfg.storage.database_path(),
        StoreOptions::from_config(&cfg),
    )
    .await
    .unwrap();
    let start = Utc.with_ymd_and_hms(2026, 5, 15, 10, 0, 0).unwrap();
    let source = WavFileSource::new("file0", fixture(), Pacing::Fast { start_at: start });
    let pipeline = Pipeline::with_sources(
        cfg,
        bundle,
        Arc::new(store),
        vec![SourceSpec {
            source: Box::new(source),
            backpressure: Backpressure::Wait,
        }],
        PipelineOptions {
            exit_on_eof: true,
            fast_files: true,
        },
    );
    let summary = tokio::time::timeout(
        Duration::from_secs(60),
        pipeline.run(CancellationToken::new()),
    )
    .await
    .expect("pipeline finished")
    .expect("pipeline ok");
    (
        summary.detections,
        summary.unconfirmed_detections,
        summary.chunks_processed,
    )
}

/// `min_detections` holds a species back until it repeats. The same audio is analysed either way,
/// so every detection the model made is either stored or counted as unconfirmed.
#[tokio::test(flavor = "multi_thread")]
async fn repeat_confirmation_drops_one_off_detections() {
    if !models_present() {
        eprintln!("skipping: models not present");
        return;
    }
    let (base_stored, base_unconfirmed, base_chunks) = run_fixture("").await;
    assert_eq!(base_unconfirmed, 0, "confirmation is off by default");

    let (stored, unconfirmed, chunks) = run_fixture(
        r#"
[audio]
ring_buffer_seconds = 60.0

[detection]
min_detections = 2
confirmation_window_seconds = 15.0
"#,
    )
    .await;

    assert_eq!(chunks, base_chunks, "the same audio is analysed either way");
    assert!(
        stored <= base_stored,
        "confirmation only removes detections: {stored} vs {base_stored}"
    );
    assert_eq!(
        stored + unconfirmed,
        base_stored,
        "every detection is stored or counted unconfirmed"
    );
    assert!(
        unconfirmed > 0,
        "the fixture has at least one species heard only once"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_stops_a_live_source_promptly() {
    if !models_present() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (cfg, bundle, store) = setup(dir.path()).await;
    let source = WavFileSource::new("live", fixture(), Pacing::Realtime);
    let pipeline = Pipeline::with_sources(
        cfg,
        bundle,
        Arc::new(store),
        vec![SourceSpec {
            source: Box::new(source),
            backpressure: Backpressure::DropOldest,
        }],
        PipelineOptions::default(),
    );
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let handle = tokio::spawn(pipeline.run(cancel.clone()));
    tokio::time::sleep(Duration::from_millis(3_500)).await; // at least one chunk captured
    cancel.cancel();
    let summary = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("stopped")
        .unwrap()
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(9));
    assert!(summary.chunks_processed >= 1, "{summary:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn fatal_source_error_fails_the_run() {
    if !models_present() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (cfg, bundle, store) = setup(dir.path()).await;
    let src = AudioSourceConfig {
        id: "mic0".into(),
        kind: AudioSourceKind::Alsa,
        device: Some("hw:9,0".into()),
        url: None,
        path: None,
        gain_db: 0.0,
    };
    let opts = FfmpegOptions {
        ffmpeg_path: "/nonexistent/ffmpeg".into(),
        ..FfmpegOptions::default()
    };
    let source = FfmpegSource::new(src, opts).unwrap();
    let pipeline = Pipeline::with_sources(
        cfg,
        bundle,
        Arc::new(store),
        vec![SourceSpec {
            source: Box::new(source),
            backpressure: Backpressure::DropOldest,
        }],
        PipelineOptions::default(),
    );
    let err = tokio::time::timeout(
        Duration::from_secs(10),
        pipeline.run(CancellationToken::new()),
    )
    .await
    .expect("returned")
    .unwrap_err();
    let text = format!("{err:#}");
    assert!(
        text.contains("ffmpeg not found") && text.contains("mic0"),
        "{text}"
    );
}
