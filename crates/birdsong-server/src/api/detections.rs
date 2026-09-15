use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::response::Response;
use axum::Json;
use birdsong_store::{safe_clip_path, DetectionQuery, DetectionRecord, Order};
use serde::Serialize;
use tower::ServiceExt;
use tower_http::services::ServeFile;

use super::error::{ApiError, ApiResult};
use super::params::{self, Params};
use super::AppState;

/// A page of results with cursors.
#[derive(Clone, Debug, Serialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    /// Highest id on this page; pass as `after_id` (with `order=asc`) to get newer rows.
    /// `null` when the page is empty: keep using your previous cursor.
    pub next_after_id: Option<i64>,
    /// Lowest id on a full page; pass as `before_id` (with `order=desc`) to page back in time.
    /// `null` when the page was not full.
    pub next_before_id: Option<i64>,
}

fn page(items: Vec<DetectionRecord>, limit: u32) -> Page<DetectionRecord> {
    let next_after_id = items.iter().map(|r| r.id).max();
    let full = items.len() as u64 >= u64::from(limit);
    let next_before_id = if full {
        items.iter().map(|r| r.id).min()
    } else {
        None
    };
    Page {
        items,
        next_after_id,
        next_before_id,
    }
}

fn query_from(q: &Params) -> ApiResult<DetectionQuery> {
    Ok(DetectionQuery {
        after_id: params::opt_i64(q, "after_id")?,
        before_id: params::opt_i64(q, "before_id")?,
        since: params::opt_timestamp(q, "since")?,
        until: params::opt_timestamp(q, "until")?,
        species: params::text(q, "species"),
        min_confidence: params::opt_confidence(q, "min_confidence")?,
        limit: params::opt_limit(q)?,
        order: params::order(q)?,
    })
}

/// `GET /detections`
pub async fn list(
    State(state): State<AppState>,
    Query(q): Query<Params>,
) -> ApiResult<Json<Page<DetectionRecord>>> {
    let query = query_from(&q)?;
    let items = state.store.list(&query).await?;
    Ok(Json(page(items, query.effective_limit())))
}

/// `GET /detections/latest` (newest first, default 20).
pub async fn latest(
    State(state): State<AppState>,
    Query(q): Query<Params>,
) -> ApiResult<Json<Page<DetectionRecord>>> {
    let query = DetectionQuery {
        limit: Some(params::opt_limit(&q)?.unwrap_or(20)),
        order: Order::Desc,
        ..Default::default()
    };
    let items = state.store.list(&query).await?;
    Ok(Json(page(items, query.effective_limit())))
}

async fn record(state: &AppState, raw_id: &str) -> ApiResult<DetectionRecord> {
    let id = params::id("detection id", raw_id)?;
    state
        .store
        .get(id)
        .await?
        .ok_or_else(|| ApiError::not_found(format!("detection {id} not found")))
}

/// `GET /detections/{id}`
pub async fn get_one(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<DetectionRecord>> {
    Ok(Json(record(&state, &id).await?))
}

#[derive(Clone, Copy)]
enum Asset {
    Audio,
    Spectrogram,
}

async fn serve_asset(
    state: AppState,
    raw_id: String,
    req: Request,
    asset: Asset,
) -> ApiResult<Response> {
    let rec = record(&state, &raw_id).await?;
    let (relative, what) = match asset {
        Asset::Audio => (rec.clip_path, "audio"),
        Asset::Spectrogram => (rec.spectrogram_path, "spectrogram"),
    };
    let missing = || {
        ApiError::not_found(format!(
            "no {what} for detection {} (never saved or already purged)",
            rec.id
        ))
    };
    let relative = relative.ok_or_else(missing)?;
    let path = safe_clip_path(&state.clips_dir, &relative).ok_or_else(missing)?;
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Err(missing());
    }
    let response = match ServeFile::new(&path).oneshot(req).await {
        Ok(response) => response,
        Err(never) => match never {},
    };
    Ok(response.map(Body::new))
}

/// `GET /detections/{id}/audio` (WAV, supports `Range`).
pub async fn audio(
    State(state): State<AppState>,
    Path(id): Path<String>,
    req: Request,
) -> ApiResult<Response> {
    serve_asset(state, id, req, Asset::Audio).await
}

/// `GET /detections/{id}/spectrogram.png`
pub async fn spectrogram(
    State(state): State<AppState>,
    Path(id): Path<String>,
    req: Request,
) -> ApiResult<Response> {
    serve_asset(state, id, req, Asset::Spectrogram).await
}
