use axum::extract::{Query, State};
use axum::Json;
use birdsong_audio::redact_url;
use chrono::{TimeDelta, Utc};
use serde_json::{json, Value};

use super::error::{ApiError, ApiResult};
use super::params::{self, Params};
use super::AppState;

/// Allowed `window` values for `/stats/recent`.
pub const RECENT_WINDOWS: [(&str, i64); 5] = [
    ("1h", 3_600),
    ("6h", 21_600),
    ("24h", 86_400),
    ("7d", 604_800),
    ("30d", 2_592_000),
];

/// `GET /health`
pub async fn health(State(state): State<AppState>) -> Json<Value> {
    let stats = state.stats.snapshot();
    // Wall clock, not audio time: files decoded faster than real time stamp chunks in the future.
    let since_last = stats
        .last_processed_at
        .map(|t| (Utc::now() - t).num_milliseconds().max(0) as f64 / 1000.0);
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": state.started.elapsed().as_secs(),
        "station": state.config.station.name,
        "model_id": state.model_id,
        "last_chunk_at": stats.last_chunk_at,
        "seconds_since_last_chunk": since_last,
        "stats": stats,
    }))
}

/// `GET /species?since=`
pub async fn species(
    State(state): State<AppState>,
    Query(q): Query<Params>,
) -> ApiResult<Json<Value>> {
    let since = params::opt_timestamp(&q, "since")?;
    let items = state.store.species_summary(since).await?;
    Ok(Json(json!({ "since": since, "items": items })))
}

/// `GET /stats/daily?date=YYYY-MM-DD` (default: today in the station time zone)
pub async fn stats_daily(
    State(state): State<AppState>,
    Query(q): Query<Params>,
) -> ApiResult<Json<Value>> {
    let date = match params::opt_date(&q, "date")? {
        Some(d) => d,
        None => Utc::now()
            .with_timezone(&state.config.station.timezone)
            .date_naive(),
    };
    let stats = state.store.stats_daily(date).await?;
    serde_json::to_value(stats)
        .map(Json)
        .map_err(|e| ApiError::internal(e.to_string()))
}

/// `GET /stats/recent?window=24h`
pub async fn stats_recent(
    State(state): State<AppState>,
    Query(q): Query<Params>,
) -> ApiResult<Json<Value>> {
    let window = params::text(&q, "window").unwrap_or_else(|| "24h".into());
    let Some(&(name, seconds)) = RECENT_WINDOWS.iter().find(|(w, _)| *w == window) else {
        let allowed: Vec<&str> = RECENT_WINDOWS.iter().map(|(w, _)| *w).collect();
        return Err(ApiError::bad_request(format!(
            "window must be one of {}, got {window:?}",
            allowed.join(", ")
        )));
    };
    let until = Utc::now();
    let since = until - TimeDelta::seconds(seconds);
    let species = state.store.species_summary(Some(since)).await?;
    Ok(Json(
        json!({ "window": name, "since": since, "until": until, "species": species }),
    ))
}

/// `GET /config` with stream credentials redacted.
pub async fn config(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let mut cfg = (*state.config).clone();
    if cfg.birdweather.enabled() {
        cfg.birdweather.token = "***".into();
    }
    for source in &mut cfg.audio.sources {
        if let Some(url) = &source.url {
            source.url = Some(redact_url(url));
        }
    }
    serde_json::to_value(cfg)
        .map(Json)
        .map_err(|e| ApiError::internal(e.to_string()))
}
