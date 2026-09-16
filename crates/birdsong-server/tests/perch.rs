//! The pipeline with Perch v2: 5 s windows end to end. Skips when the regional model from
//! `scripts/fetch-perch.sh` or the BirdNET labels are absent.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use birdsong_audio::{Pacing, WavFileSource};
use birdsong_core::Config;
use birdsong_model::ModelBundle;
use birdsong_server::{Backpressure, Pipeline, PipelineOptions, SourceSpec};
use birdsong_store::{DetectionQuery, DetectionStore, SqliteStore, StoreOptions};
use chrono::{TimeZone, Utc};
use tokio_util::sync::CancellationToken;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[tokio::test(flavor = "multi_thread")]
async fn perch_pipeline_uses_five_second_windows() {
    let models = repo_root().join("models");
    if !models
        .join("perch/perch_v2_north-america-east_no_dft_fp32.onnx")
        .exists()
        || !models.join("labels/en_us.txt").exists()
    {
        eprintln!("skipping: Perch model not present");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let fixture = repo_root().join("tools/fixtures/soundscape_15s.wav");
    let cfg = Config::from_toml(&format!(
        r#"
[station]
latitude = 42.36
longitude = -71.06
timezone = "America/New_York"
[birdweather]
token = "not-used-with-perch"
[[audio.sources]]
id = "file0"
kind = "file"
path = {fixture:?}
[model]
dir = {models:?}
kind = "perch-v2"
classifier = "perch/perch_v2_north-america-east_no_dft_fp32.onnx"
labels = "perch/perch_v2_north-america-east_labels.txt"
common_names = "labels/en_us.txt"
[detection]
min_confidence = 0.3
[storage]
data_dir = {data:?}
"#,
        fixture = fixture.display().to_string(),
        models = models.display().to_string(),
        data = dir.path().display().to_string(),
    ))
    .unwrap();
    let bundle = ModelBundle::load(&cfg).unwrap();
    let store = SqliteStore::open(
        &cfg.storage.database_path(),
        StoreOptions::from_config(&cfg),
    )
    .await
    .unwrap();
    let start = Utc.with_ymd_and_hms(2026, 5, 15, 10, 0, 0).unwrap();
    let source = WavFileSource::new("file0", fixture, Pacing::Fast { start_at: start });
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
    let summary = tokio::time::timeout(
        Duration::from_secs(120),
        pipeline.run(CancellationToken::new()),
    )
    .await
    .expect("pipeline finished")
    .expect("pipeline ok");

    // 15.02 s in 5 s steps: three full windows; the 0.02 s left is below half a window.
    assert_eq!(summary.chunks_processed, 3, "{summary:?}");
    assert_eq!(
        summary.birdweather_soundscapes + summary.birdweather_errors,
        0,
        "no uploads with Perch"
    );

    let rows = store
        .list(&DetectionQuery {
            species: Some("Poecile atricapillus".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let chickadee = rows.last().expect("chickadee in the first window");
    assert_eq!(chickadee.detected_at, start);
    assert_eq!(chickadee.model_id, "perch-v2");
    assert_eq!(chickadee.common_name, "Black-capped Chickadee");

    assert!(summary.clips_written >= 1, "{summary:?}");
    let clip = store
        .get(chickadee.id)
        .await
        .unwrap()
        .unwrap()
        .clip_path
        .expect("clip saved");
    let mut reader =
        claxon::FlacReader::open(dir.path().join("clips").join(&clip)).expect("clip is FLAC");
    let samples = reader.samples().count();
    assert_eq!(
        samples, 264_000,
        "6 s clip centred on the 5 s window, clamped at the start: 5.5 s"
    );
}
