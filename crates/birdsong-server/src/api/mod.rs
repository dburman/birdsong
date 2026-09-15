//! HTTP API v1 (`BUILD_PLAN.md` §6, documented in `docs/API.md`) and the web dashboard at `/`.

mod detections;
mod error;
mod insights;
mod metrics;
mod params;
mod stream;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::http::header::{ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE};
use axum::http::{HeaderValue, Method, Uri};
use axum::routing::get;
use axum::Router;
use birdsong_core::{Config, Detection};
use birdsong_store::DetectionStore;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tower_http::compression::predicate::{DefaultPredicate, NotForContentType, Predicate};
use tower_http::compression::CompressionLayer;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;

pub use detections::Page;
pub use error::ApiError;

use crate::stats::PipelineStats;

/// Everything request handlers need. Cheap to clone.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<dyn DetectionStore>,
    /// Root that stored clip paths are relative to.
    pub clips_dir: PathBuf,
    pub config: Arc<Config>,
    pub stats: Arc<PipelineStats>,
    /// Stored detections as they happen (from [`crate::Pipeline::detections_sender`]).
    pub detections: broadcast::Sender<Detection>,
    pub model_id: String,
    pub started: Instant,
    /// Cancelling ends open event streams so graceful shutdown can finish.
    pub shutdown: CancellationToken,
}

/// The dashboard at `/` and all routes under `/api/v1`, with compression (not for audio or event streams), CORS and
/// request tracing. Unknown paths get a JSON 404.
pub fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/health", get(insights::health))
        .route("/detections", get(detections::list))
        .route("/detections/latest", get(detections::latest))
        .route("/detections/{id}", get(detections::get_one))
        .route("/detections/{id}/audio", get(detections::audio))
        .route(
            "/detections/{id}/spectrogram.png",
            get(detections::spectrogram),
        )
        .route("/species", get(insights::species))
        .route("/stats/daily", get(insights::stats_daily))
        .route("/stats/recent", get(insights::stats_recent))
        .route("/stream", get(stream::stream))
        .route("/config", get(insights::config));

    let compression = CompressionLayer::new()
        .compress_when(DefaultPredicate::new().and(NotForContentType::new("audio/")));
    let cors = cors_layer(&state.config.server.cors_allow_origins);

    Router::new()
        .route("/", get(ui::index))
        .route("/index.html", get(ui::index))
        .route("/app.js", get(ui::app_js))
        .route("/style.css", get(ui::style_css))
        .route("/favicon.svg", get(ui::favicon))
        .route("/metrics", get(metrics::metrics))
        .nest("/api/v1", api)
        .fallback(fallback)
        .layer(compression)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn fallback(uri: Uri) -> ApiError {
    ApiError::not_found(format!("no route for {}", uri.path()))
}

fn cors_layer(origins: &[String]) -> CorsLayer {
    let layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::HEAD, Method::OPTIONS])
        .allow_headers(Any)
        .expose_headers([CONTENT_RANGE, ACCEPT_RANGES, CONTENT_LENGTH]);
    if origins.iter().any(|o| o == "*") {
        return layer.allow_origin(Any);
    }
    let list: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|o| match HeaderValue::from_str(o) {
            Ok(v) => Some(v),
            Err(_) => {
                tracing::warn!(origin = %o, "ignoring invalid CORS origin");
                None
            }
        })
        .collect();
    layer.allow_origin(AllowOrigin::list(list))
}

/// Serve until `state.shutdown` is cancelled.
pub async fn serve(state: AppState, listener: TcpListener) -> std::io::Result<()> {
    let shutdown = state.shutdown.clone();
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await
}
