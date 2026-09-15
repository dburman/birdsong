//! HTTP API against a temp database seeded with 50 detections. No models needed.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::Router;
use birdsong_audio::{spectrogram, wav, SpectrogramOptions};
use birdsong_core::{Config, Detection};
use birdsong_server::api::{self, AppState};
use birdsong_server::PipelineStats;
use birdsong_store::{ClipInfo, DetectionQuery, DetectionStore, Order, SqliteStore, StoreOptions};
use chrono::{DateTime, TimeDelta, Utc};
use http_body_util::BodyExt;
use serde_json::Value;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

const SPECIES: [(&str, &str); 3] = [
    ("Cardinalis cardinalis", "Northern Cardinal"),
    ("Poecile atricapillus", "Black-capped Chickadee"),
    ("Turdus migratorius", "American Robin"),
];

struct Env {
    _dir: tempfile::TempDir,
    state: AppState,
    router: Router,
    seeded_at: DateTime<Utc>,
}

fn detection(i: usize, at: DateTime<Utc>) -> Detection {
    let (sci, common) = SPECIES[i % 3];
    Detection {
        id: None,
        detected_at: at,
        scientific_name: sci.into(),
        common_name: common.into(),
        confidence: 0.70 + (i % 30) as f32 * 0.01,
        source_id: "mic0".into(),
        model_id: "birdnet-v2.4".into(),
        clip_path: None,
    }
}

/// 50 detections two minutes apart ending just now; detection id 10 has a real clip.
async fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let cfg = Config::from_toml(&format!(
        r#"
[station]
name = "Test station"
[[audio.sources]]
id = "cam"
kind = "rtsp"
url = "rtsp://user:secret@cam.local/stream"
[storage]
data_dir = {:?}
"#,
        dir.path().display().to_string()
    ))
    .unwrap();
    let store = SqliteStore::open(
        &cfg.storage.database_path(),
        StoreOptions::from_config(&cfg),
    )
    .await
    .unwrap();
    let seeded_at = Utc::now();
    for i in 0..50 {
        let at = seeded_at - TimeDelta::minutes(100) + TimeDelta::minutes(2 * i as i64);
        store.insert(&detection(i, at)).await.unwrap();
    }

    let clips_dir = cfg.storage.clips_dir();
    let rel = "2026-05-15/Black_capped_Chickadee/clip.wav";
    let png = "2026-05-15/Black_capped_Chickadee/clip.png";
    std::fs::create_dir_all(clips_dir.join("2026-05-15/Black_capped_Chickadee")).unwrap();
    let samples: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
    let bytes = wav::write_wav(&clips_dir.join(rel), &samples, 48_000).unwrap();
    spectrogram::write_png(
        &clips_dir.join(png),
        &samples,
        48_000,
        &SpectrogramOptions::default(),
    )
    .unwrap();
    store
        .set_clip(
            &[10],
            Some(&ClipInfo {
                clip_path: rel.into(),
                clip_bytes: bytes,
                spectrogram_path: Some(png.into()),
            }),
        )
        .await
        .unwrap();

    let state = AppState {
        store: Arc::new(store),
        clips_dir,
        config: Arc::new(cfg),
        stats: Arc::new(PipelineStats::new()),
        detections: broadcast::channel(16).0,
        model_id: "birdnet-v2.4".into(),
        started: Instant::now(),
        shutdown: CancellationToken::new(),
    };
    Env {
        router: api::router(state.clone()),
        state,
        seeded_at,
        _dir: dir,
    }
}

async fn send(
    router: &Router,
    uri: &str,
    headers: &[(&str, &str)],
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let mut req = Request::builder().uri(uri);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = router
        .clone()
        .oneshot(req.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = resp
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    (status, headers, body)
}

async fn json(router: &Router, uri: &str) -> (StatusCode, Value) {
    let (status, headers, body) = send(router, uri, &[]).await;
    assert!(
        headers
            .get(header::CONTENT_TYPE)
            .is_some_and(|v| v.to_str().unwrap().starts_with("application/json")),
        "{uri}: expected JSON, got {headers:?}"
    );
    (status, serde_json::from_slice(&body).unwrap())
}

fn ids(v: &Value) -> Vec<i64> {
    v["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_i64().unwrap())
        .collect()
}

async fn assert_error(router: &Router, uri: &str, status: StatusCode, contains: &str) {
    let (got, body) = json(router, uri).await;
    assert_eq!(got, status, "{uri}: {body}");
    let msg = body["error"].as_str().unwrap_or_default();
    assert!(
        msg.contains(contains),
        "{uri}: error {msg:?} should mention {contains:?}"
    );
}

#[tokio::test]
async fn health_reports_status_and_stats() {
    let env = env().await;
    let (status, body) = json(&env.router, "/api/v1/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model_id"], "birdnet-v2.4");
    assert_eq!(body["station"], "Test station");
    assert!(body["uptime_s"].is_u64());
    assert_eq!(body["stats"]["chunks_processed"], 0);
    assert!(body["last_chunk_at"].is_null());
}

#[tokio::test]
async fn list_shape_and_defaults() {
    let env = env().await;
    let (status, page) = json(&env.router, "/api/v1/detections").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        ids(&page),
        (1..=50).rev().collect::<Vec<_>>(),
        "newest first by default"
    );
    assert_eq!(page["next_after_id"], 50);
    assert!(page["next_before_id"].is_null(), "page was not full");
    let item = &page["items"][0];
    for field in [
        "id",
        "detected_at",
        "local_date",
        "local_hour",
        "scientific_name",
        "common_name",
        "confidence",
        "source_id",
        "model_id",
    ] {
        assert!(!item[field].is_null(), "missing {field}: {item}");
    }
    let (_, latest) = json(&env.router, "/api/v1/detections/latest?limit=5").await;
    assert_eq!(ids(&latest), [50, 49, 48, 47, 46]);
    assert_eq!(latest["next_before_id"], 46);
}

#[tokio::test]
async fn cursor_walk_visits_every_row_once() {
    let env = env().await;
    let mut cursor = 0;
    let mut seen: Vec<i64> = Vec::new();
    loop {
        let (status, page) = json(
            &env.router,
            &format!("/api/v1/detections?after_id={cursor}&order=asc&limit=7"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let page_ids = ids(&page);
        if page_ids.is_empty() {
            assert!(page["next_after_id"].is_null());
            break;
        }
        seen.extend(&page_ids);
        cursor = page["next_after_id"].as_i64().unwrap();
        assert_eq!(cursor, *page_ids.last().unwrap());
    }
    assert_eq!(seen, (1..=50).collect::<Vec<_>>());

    // Backwards with before_id.
    let (_, page) = json(&env.router, "/api/v1/detections?limit=20").await;
    let before = page["next_before_id"].as_i64().unwrap();
    let (_, older) = json(
        &env.router,
        &format!("/api/v1/detections?limit=20&before_id={before}"),
    )
    .await;
    assert_eq!(ids(&older), (11..=30).rev().collect::<Vec<_>>());
}

#[tokio::test]
async fn filters_and_bad_parameters() {
    let env = env().await;
    let all = env
        .state
        .store
        .list(&DetectionQuery {
            limit: Some(1000),
            order: Order::Asc,
            ..Default::default()
        })
        .await
        .unwrap();

    let (_, page) = json(
        &env.router,
        "/api/v1/detections?min_confidence=0.9&limit=1000",
    )
    .await;
    let expected = all.iter().filter(|r| r.confidence >= 0.9).count();
    assert_eq!(ids(&page).len(), expected);
    assert!(page["items"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["confidence"].as_f64().unwrap() >= 0.9 - 1e-6));

    let (_, page) = json(
        &env.router,
        "/api/v1/detections?species=Cardinalis%20cardinalis&limit=1000",
    )
    .await;
    assert_eq!(ids(&page).len(), 17);

    let since = all[10].detected_at.to_rfc3339();
    let until = all[20].detected_at.to_rfc3339();
    let uri = format!(
        "/api/v1/detections?order=asc&since={}&until={}",
        urlencode(&since),
        urlencode(&until)
    );
    let (_, page) = json(&env.router, &uri).await;
    assert_eq!(ids(&page), (11..=20).collect::<Vec<_>>());

    let (_, page) = json(&env.router, "/api/v1/detections?limit=50000").await;
    assert_eq!(ids(&page).len(), 50, "limit is clamped, not rejected");

    assert_error(
        &env.router,
        "/api/v1/detections?limit=abc",
        StatusCode::BAD_REQUEST,
        "limit",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/detections?order=sideways",
        StatusCode::BAD_REQUEST,
        "order",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/detections?since=yesterday",
        StatusCode::BAD_REQUEST,
        "RFC 3339",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/detections?min_confidence=2",
        StatusCode::BAD_REQUEST,
        "min_confidence",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/detections?after_id=x",
        StatusCode::BAD_REQUEST,
        "after_id",
    )
    .await;
}

fn urlencode(s: &str) -> String {
    s.replace(':', "%3A").replace('+', "%2B")
}

#[tokio::test]
async fn single_detection_and_not_found() {
    let env = env().await;
    let (status, body) = json(&env.router, "/api/v1/detections/10").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], 10);
    assert_eq!(
        body["clip_path"],
        "2026-05-15/Black_capped_Chickadee/clip.wav"
    );
    assert_error(
        &env.router,
        "/api/v1/detections/999",
        StatusCode::NOT_FOUND,
        "999",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/detections/abc",
        StatusCode::BAD_REQUEST,
        "integer",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/nope",
        StatusCode::NOT_FOUND,
        "/api/v1/nope",
    )
    .await;
}

#[tokio::test]
async fn audio_supports_ranges_and_spectrogram_is_served() {
    let env = env().await;
    let (status, headers, body) = send(&env.router, "/api/v1/detections/10/audio", &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("audio/"),
        "{headers:?}"
    );
    assert_eq!(headers[header::ACCEPT_RANGES], "bytes");
    assert_eq!(body.len(), 44 + 2 * 48_000);
    assert!(headers.get(header::CONTENT_ENCODING).is_none());

    let (status, headers, body) = send(
        &env.router,
        "/api/v1/detections/10/audio",
        &[("range", "bytes=0-99"), ("accept-encoding", "gzip")],
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        headers[header::CONTENT_RANGE],
        format!("bytes 0-99/{}", 44 + 2 * 48_000)
    );
    assert_eq!(body.len(), 100);
    assert!(
        headers.get(header::CONTENT_ENCODING).is_none(),
        "audio is never compressed"
    );

    let (status, headers, body) =
        send(&env.router, "/api/v1/detections/10/spectrogram.png", &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "image/png");
    assert_eq!(&body[..4], b"\x89PNG");

    assert_error(
        &env.router,
        "/api/v1/detections/11/audio",
        StatusCode::NOT_FOUND,
        "no audio",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/detections/11/spectrogram.png",
        StatusCode::NOT_FOUND,
        "no spectrogram",
    )
    .await;
    assert_error(
        &env.router,
        "/api/v1/detections/999/audio",
        StatusCode::NOT_FOUND,
        "not found",
    )
    .await;

    // A purged file (row still pointing at it) is a 404, not a 500.
    std::fs::remove_file(
        env.state
            .clips_dir
            .join("2026-05-15/Black_capped_Chickadee/clip.wav"),
    )
    .unwrap();
    assert_error(
        &env.router,
        "/api/v1/detections/10/audio",
        StatusCode::NOT_FOUND,
        "no audio",
    )
    .await;
}

#[tokio::test]
async fn species_and_stats() {
    let env = env().await;
    let (status, body) = json(&env.router, "/api/v1/species").await;
    assert_eq!(status, StatusCode::OK);
    let items = body["items"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    let counts: Vec<u64> = items.iter().map(|s| s["count"].as_u64().unwrap()).collect();
    assert_eq!(counts, [17, 17, 16]);
    for field in [
        "scientific_name",
        "common_name",
        "first_seen",
        "last_seen",
        "max_confidence",
        "best_detection_id",
    ] {
        assert!(!items[0][field].is_null(), "missing {field}");
    }
    // Only detection 10 has a clip; its species reports it, the others report null.
    let with_clip = env.state.store.get(10).await.unwrap().unwrap();
    for s in items {
        let expected = if s["scientific_name"] == with_clip.scientific_name.as_str() {
            Value::from(10)
        } else {
            Value::Null
        };
        assert_eq!(s["best_clip_detection_id"], expected, "{s}");
    }

    // Daily stats: totals match the rows on that local date.
    let all = env
        .state
        .store
        .list(&DetectionQuery {
            limit: Some(1000),
            ..Default::default()
        })
        .await
        .unwrap();
    let date = all[0].local_date;
    let (status, daily) = json(&env.router, &format!("/api/v1/stats/daily?date={date}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(daily["date"], date.to_string());
    let total: u64 = daily["species"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["total"].as_u64().unwrap())
        .sum();
    assert_eq!(
        total as usize,
        all.iter().filter(|r| r.local_date == date).count()
    );
    assert_eq!(daily["species"][0]["by_hour"].as_array().unwrap().len(), 24);
    assert_eq!(
        json(&env.router, "/api/v1/stats/daily").await.0,
        StatusCode::OK,
        "defaults to today"
    );
    assert_error(
        &env.router,
        "/api/v1/stats/daily?date=15-05-2026",
        StatusCode::BAD_REQUEST,
        "YYYY-MM-DD",
    )
    .await;

    // Recent window: rows are 2 minutes apart ending at seeding time; the last hour holds 29.
    let (status, recent) = json(&env.router, "/api/v1/stats/recent?window=1h").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(recent["window"], "1h");
    let in_hour: u64 = recent["species"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["count"].as_u64().unwrap())
        .sum();
    let elapsed = (Utc::now() - env.seeded_at).num_seconds();
    assert!(elapsed < 100, "test too slow for exact window counts");
    assert_eq!(in_hour, 29);
    let (_, day) = json(&env.router, "/api/v1/stats/recent").await;
    assert_eq!(day["window"], "24h");
    assert_error(
        &env.router,
        "/api/v1/stats/recent?window=2d",
        StatusCode::BAD_REQUEST,
        "1h, 6h, 24h, 7d, 30d",
    )
    .await;
}

#[tokio::test]
async fn config_is_redacted_and_cors_allows_any_origin() {
    let env = env().await;
    let (status, headers, body) = send(
        &env.router,
        "/api/v1/config",
        &[("origin", "http://dashboard.local")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("rtsp://***@cam.local/stream"), "{text}");
    assert!(!text.contains("secret"));
    assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");
}

async fn read_events(body: &mut Body, want: usize, timeout: Duration) -> Vec<(i64, Value)> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut text = String::new();
    let mut events = Vec::new();
    while events.len() < want {
        let frame = tokio::time::timeout_at(deadline, body.frame())
            .await
            .expect("event within timeout");
        let Some(frame) = frame else { break };
        if let Ok(data) = frame.unwrap().into_data() {
            text.push_str(std::str::from_utf8(&data).unwrap());
        }
        while let Some(end) = text.find("\n\n") {
            let block: String = text.drain(..end + 2).collect();
            let mut id = None;
            let mut data = None;
            let mut is_detection = false;
            for line in block.lines() {
                if let Some(v) = line.strip_prefix("id: ") {
                    id = v.parse::<i64>().ok();
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data = serde_json::from_str::<Value>(v).ok();
                } else if line == "event: detection" {
                    is_detection = true;
                }
            }
            if let (true, Some(id), Some(data)) = (is_detection, id, data) {
                events.push((id, data));
            }
        }
    }
    events
}

#[tokio::test]
async fn stream_delivers_live_detections_within_a_second() {
    let env = env().await;
    let resp = env
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/stream")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()[header::CONTENT_TYPE], "text/event-stream");
    let mut body = resp.into_body();

    let state = env.state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut d = detection(0, Utc::now());
        d.id = Some(state.store.insert(&d).await.unwrap());
        state.detections.send(d).unwrap();
    });

    let started = Instant::now();
    let events = read_events(&mut body, 1, Duration::from_secs(1)).await;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, 51);
    assert_eq!(events[0].1["id"], 51);
    assert_eq!(events[0].1["common_name"], "Northern Cardinal");
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn stream_replays_after_last_event_id_then_ends_on_shutdown() {
    let env = env().await;
    let req = Request::builder()
        .uri("/api/v1/stream")
        .header("last-event-id", "45")
        .body(Body::empty())
        .unwrap();
    let mut body = env.router.clone().oneshot(req).await.unwrap().into_body();
    let events = read_events(&mut body, 5, Duration::from_secs(2)).await;
    assert_eq!(
        events.iter().map(|e| e.0).collect::<Vec<_>>(),
        [46, 47, 48, 49, 50]
    );

    // A live copy of an already replayed detection is not sent twice.
    let mut dup = detection(49, Utc::now());
    dup.id = Some(50);
    env.state.detections.send(dup).unwrap();

    env.state.shutdown.cancel();
    let end = tokio::time::timeout(Duration::from_secs(2), body.frame())
        .await
        .expect("stream closes on shutdown");
    assert!(end.is_none(), "no further events after shutdown");
}

#[tokio::test]
async fn served_over_a_real_socket_and_shuts_down() {
    let env = env().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(api::serve(env.state.clone(), listener));

    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    stream
        .write_all(b"GET /api/v1/health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("\"status\":\"ok\""));

    env.state.shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(3), server)
        .await
        .expect("graceful shutdown")
        .unwrap()
        .unwrap();
    let _ = PathBuf::new();
}

#[tokio::test]
async fn dashboard_assets_are_served() {
    let env = env().await;
    let (status, headers, body) = send(&env.router, "/", &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "text/html; charset=utf-8");
    let html = String::from_utf8(body).unwrap();
    assert!(
        html.contains("<title>Birdsong</title>")
            && html.contains("app.js")
            && html.contains("style.css")
    );

    for (path, mime, marker) in [
        ("/app.js", "text/javascript; charset=utf-8", "EventSource"),
        (
            "/style.css",
            "text/css; charset=utf-8",
            "prefers-color-scheme",
        ),
        ("/favicon.svg", "image/svg+xml", "<svg"),
    ] {
        let (status, headers, body) = send(&env.router, path, &[]).await;
        assert_eq!(status, StatusCode::OK, "{path}");
        assert_eq!(headers[header::CONTENT_TYPE], mime, "{path}");
        assert_eq!(headers[header::CACHE_CONTROL], "no-store", "{path}");
        assert!(String::from_utf8(body).unwrap().contains(marker), "{path}");
    }

    let (status, headers, _) = send(&env.router, "/app.js", &[("accept-encoding", "gzip")]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers[header::CONTENT_ENCODING],
        "gzip",
        "text assets are compressed"
    );
    assert_error(
        &env.router,
        "/missing.html",
        StatusCode::NOT_FOUND,
        "/missing.html",
    )
    .await;
}

#[tokio::test]
async fn prometheus_metrics() {
    let env = env().await;
    let (status, headers, body) = send(&env.router, "/metrics", &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers[header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .starts_with("text/plain; version=0.0.4"));
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("birdsong_chunks_processed_total 0"), "{text}");
    assert!(
        text.contains(
            r#"birdsong_build_info{version="0.1.0",model="birdnet-v2.4",station="Test station"} 1"#
        ),
        "{text}"
    );
    assert!(text.contains(r#"birdsong_species_detections{scientific_name="Cardinalis cardinalis",common_name="Northern Cardinal"} 17"#), "{text}");
    let clip_bytes: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("birdsong_clip_bytes "))
        .and_then(|v| v.parse().ok())
        .expect("clip bytes gauge");
    assert!(clip_bytes > 0);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (name, value) = line
            .rsplit_once(' ')
            .unwrap_or_else(|| panic!("bad line {line:?}"));
        assert!(name.starts_with("birdsong_"), "{line}");
        assert!(value.parse::<f64>().is_ok(), "{line}");
    }
}
