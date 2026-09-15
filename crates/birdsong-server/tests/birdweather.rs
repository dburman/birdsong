//! BirdWeather uploads against an in-process mock of the two BirdWeather endpoints.
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use birdsong_audio::{Pacing, WavFileSource};
use birdsong_core::Config;
use birdsong_model::ModelBundle;
use birdsong_server::birdweather::{upload_window, BirdWeatherClient, Station, UploadJob};
use birdsong_server::{Backpressure, Pipeline, PipelineOptions, SourceSpec};
use birdsong_store::{SqliteStore, StoreOptions};
use chrono::{TimeDelta, TimeZone, Utc};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
struct Soundscape {
    token: String,
    query: HashMap<String, String>,
    content_type: String,
    body: Vec<u8>,
}

#[derive(Default)]
struct Mock {
    soundscapes: Vec<Soundscape>,
    detections: Vec<Value>,
    /// Status codes to return, in order, before succeeding.
    soundscape_failures: Vec<u16>,
    /// Scientific names answered with 422.
    refuse_species: Vec<String>,
    soundscape_success: bool,
}

type Shared = Arc<Mutex<Mock>>;

async fn soundscape(
    State(mock): State<Shared>,
    Path(token): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    let mut m = mock.lock().unwrap();
    if !m.soundscape_failures.is_empty() {
        let status = m.soundscape_failures.remove(0);
        return (
            StatusCode::from_u16(status).unwrap(),
            Json(json!({"success": false})),
        );
    }
    let content_type = headers
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_string())
        .unwrap_or_default();
    m.soundscapes.push(Soundscape {
        token,
        query,
        content_type,
        body: body.to_vec(),
    });
    let id = 41 + m.soundscapes.len() as i64;
    if m.soundscape_success {
        (
            StatusCode::CREATED,
            Json(json!({"success": true, "soundscape": {"id": id, "duration": 6.0}})),
        )
    } else {
        (
            StatusCode::CREATED,
            Json(json!({"success": false, "message": "Invalid station token"})),
        )
    }
}

async fn detection(
    State(mock): State<Shared>,
    Path(_token): Path<String>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let mut m = mock.lock().unwrap();
    let refused = m
        .refuse_species
        .iter()
        .any(|s| body["scientificName"] == s.as_str());
    m.detections.push(body);
    if refused {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"success": false, "message": "Species not found"})),
        )
    } else {
        (StatusCode::CREATED, Json(json!({"success": true})))
    }
}

async fn mock_server(mock: Mock) -> (String, Shared) {
    let shared: Shared = Arc::new(Mutex::new(mock));
    let app = Router::new()
        .route("/api/v1/stations/{token}/soundscapes", post(soundscape))
        .route("/api/v1/stations/{token}/detections", post(detection))
        .with_state(Arc::clone(&shared));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}/api/v1"), shared)
}

fn decode_flac(bytes: &[u8]) -> Vec<i32> {
    let mut reader =
        claxon::FlacReader::new(std::io::Cursor::new(bytes)).expect("soundscape is FLAC");
    assert_eq!(reader.streaminfo().sample_rate, 48_000);
    reader.samples().collect::<Result<_, _>>().unwrap()
}

fn job() -> UploadJob {
    let clip_start = Utc.with_ymd_and_hms(2026, 5, 15, 10, 0, 0).unwrap();
    UploadJob {
        clip_samples: (0..288_000)
            .map(|i| (i as f32 * 0.01).sin() * 0.2)
            .collect(),
        clip_start_at: clip_start,
        chunk_start_at: clip_start + TimeDelta::milliseconds(1500),
        detections: vec![
            (
                "Poecile atricapillus".into(),
                "Black-capped Chickadee".into(),
                0.81,
            ),
            ("Canis familiaris".into(), "Dog".into(), 0.72),
        ],
    }
}

const BOSTON: Station = Station {
    latitude: 42.36,
    longitude: -71.06,
    timezone: chrono_tz::America::New_York,
};

#[tokio::test(flavor = "multi_thread")]
async fn uploads_soundscape_then_detections_like_birdnet_pi() {
    let (api, mock) = mock_server(Mock {
        soundscape_success: true,
        refuse_species: vec!["Canis familiaris".into()],
        ..Default::default()
    })
    .await;
    let result = tokio::task::spawn_blocking(move || {
        let client = BirdWeatherClient::new(&api, "tok_123");
        upload_window(&client, BOSTON, &job())
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        (result.detections_posted, result.detections_refused),
        (1, 1),
        "422 for a non-bird is not an error"
    );

    let m = mock.lock().unwrap();
    assert_eq!(m.soundscapes.len(), 1);
    let s = &m.soundscapes[0];
    assert_eq!(s.token, "tok_123");
    assert_eq!(s.query["timestamp"], "2026-05-15T06:00:00.000-04:00");
    assert_eq!(s.query["type"], "flac");
    assert_eq!(s.content_type, "audio/flac");
    assert_eq!(decode_flac(&s.body).len(), 288_000);

    assert_eq!(m.detections.len(), 2);
    let d = &m.detections[0];
    assert_eq!(d["timestamp"], "2026-05-15T06:00:01.500-04:00");
    assert_eq!(d["soundscapeId"], 42);
    assert_eq!(d["soundscapeStartTime"], 1.5);
    assert_eq!(d["soundscapeEndTime"], 4.5);
    assert_eq!(
        (d["lat"].as_f64(), d["lon"].as_f64()),
        (Some(42.36), Some(-71.06))
    );
    assert_eq!(d["commonName"], "Black-capped Chickadee");
    assert_eq!(d["scientificName"], "Poecile atricapillus");
    assert_eq!(d["algorithm"], "2p4");
    assert!((d["confidence"].as_f64().unwrap() - 0.81).abs() < 1e-6);
}

#[tokio::test(flavor = "multi_thread")]
async fn transient_failures_are_retried_and_rejections_are_not() {
    let (api, mock) = mock_server(Mock {
        soundscape_success: true,
        soundscape_failures: vec![503, 429],
        ..Default::default()
    })
    .await;
    let api2 = api.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut client = BirdWeatherClient::new(&api2, "tok");
        client.retry_delays = vec![Duration::from_millis(10), Duration::from_millis(10)];
        upload_window(&client, BOSTON, &job())
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.detections_posted, 2);
    assert_eq!(
        mock.lock().unwrap().soundscapes.len(),
        1,
        "succeeded on the third attempt"
    );

    // success:false (for example a bad token) is not retried and stops before any detection.
    let (api, mock) = mock_server(Mock {
        soundscape_success: false,
        ..Default::default()
    })
    .await;
    let err = tokio::task::spawn_blocking(move || {
        let client = BirdWeatherClient::new(&api, "bad");
        upload_window(&client, BOSTON, &job())
    })
    .await
    .unwrap()
    .unwrap_err();
    assert!(err.to_string().contains("Invalid station token"), "{err}");
    let m = mock.lock().unwrap();
    assert_eq!(m.soundscapes.len(), 1);
    assert!(m.detections.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn unreachable_server_fails_after_retries() {
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let err = tokio::task::spawn_blocking(move || {
        let mut client = BirdWeatherClient::new(&format!("http://127.0.0.1:{port}/api/v1"), "tok");
        client.retry_delays = vec![Duration::from_millis(5)];
        upload_window(&client, BOSTON, &job())
    })
    .await
    .unwrap()
    .unwrap_err();
    assert!(err.is_transient(), "{err}");
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[tokio::test(flavor = "multi_thread")]
async fn pipeline_uploads_saved_clips() {
    if !repo_root()
        .join("models/birdnet-v2.4-headless.onnx")
        .exists()
    {
        eprintln!("skipping: models not present");
        return;
    }
    let (api, mock) = mock_server(Mock {
        soundscape_success: true,
        ..Default::default()
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let cfg = Config::from_toml(&format!(
        r#"
[station]
latitude = 42.36
longitude = -71.06
timezone = "America/New_York"
[[audio.sources]]
id = "file0"
kind = "file"
path = {fixture:?}
[model]
dir = {models:?}
[storage]
data_dir = {data:?}
[birdweather]
token = "station_token_1"
api_url = {api:?}
"#,
        fixture = repo_root()
            .join("tools/fixtures/soundscape_15s.wav")
            .display()
            .to_string(),
        models = repo_root().join("models").display().to_string(),
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
    let source = WavFileSource::new(
        "file0",
        repo_root().join("tools/fixtures/soundscape_15s.wav"),
        Pacing::Fast { start_at: start },
    );
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
    .unwrap()
    .unwrap();
    assert_eq!(summary.birdweather_errors, 0, "{summary:?}");
    assert_eq!(
        summary.birdweather_detections, summary.detections,
        "{summary:?}"
    );

    let m = mock.lock().unwrap();
    assert_eq!(m.soundscapes.len() as u64, summary.clips_written);
    let first = &m.soundscapes[0];
    assert_eq!(first.token, "station_token_1");
    assert_eq!(first.query["timestamp"], "2026-05-15T06:00:00.000-04:00");
    assert_eq!(
        decode_flac(&first.body).len(),
        216_000,
        "the saved 4.5 s clip is the soundscape"
    );
    let chickadee = m
        .detections
        .iter()
        .find(|d| d["scientificName"] == "Poecile atricapillus")
        .expect("chickadee uploaded");
    assert_eq!(
        chickadee["soundscapeStartTime"], 0.0,
        "the clip starts with the detection window"
    );
    assert_eq!(chickadee["soundscapeEndTime"], 3.0);
    assert_eq!(chickadee["timestamp"], "2026-05-15T06:00:00.000-04:00");
}
