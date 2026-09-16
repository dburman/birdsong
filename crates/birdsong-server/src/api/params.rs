//! Query parameter parsing with JSON 400 errors (axum's own rejections are plain text).

use std::collections::HashMap;

use birdsong_core::DetectionKind;
use birdsong_store::Order;
use chrono::{DateTime, NaiveDate, Utc};

use super::error::{ApiError, ApiResult};

pub type Params = HashMap<String, String>;

fn value<'a>(q: &'a Params, name: &str) -> Option<&'a str> {
    q.get(name).map(|v| v.trim()).filter(|v| !v.is_empty())
}

pub fn text(q: &Params, name: &str) -> Option<String> {
    value(q, name).map(str::to_string)
}

pub fn id(name: &str, raw: &str) -> ApiResult<i64> {
    raw.trim()
        .parse::<i64>()
        .map_err(|_| ApiError::bad_request(format!("{name} must be an integer, got {raw:?}")))
}

pub fn opt_i64(q: &Params, name: &str) -> ApiResult<Option<i64>> {
    value(q, name).map(|v| id(name, v)).transpose()
}

pub fn opt_timestamp(q: &Params, name: &str) -> ApiResult<Option<DateTime<Utc>>> {
    value(q, name)
        .map(|v| {
            DateTime::parse_from_rfc3339(v)
                .map(|t| t.with_timezone(&Utc))
                .map_err(|_| {
                    ApiError::bad_request(format!(
                        "{name} must be an RFC 3339 timestamp like 2026-05-01T06:00:00Z, got {v:?}"
                    ))
                })
        })
        .transpose()
}

pub fn opt_date(q: &Params, name: &str) -> ApiResult<Option<NaiveDate>> {
    value(q, name)
        .map(|v| {
            NaiveDate::parse_from_str(v, "%Y-%m-%d")
                .map_err(|_| ApiError::bad_request(format!("{name} must be YYYY-MM-DD, got {v:?}")))
        })
        .transpose()
}

pub fn opt_confidence(q: &Params, name: &str) -> ApiResult<Option<f32>> {
    value(q, name)
        .map(|v| match v.parse::<f32>() {
            Ok(c) if (0.0..=1.0).contains(&c) => Ok(c),
            _ => Err(ApiError::bad_request(format!(
                "{name} must be a number from 0 to 1, got {v:?}"
            ))),
        })
        .transpose()
}

pub fn opt_limit(q: &Params) -> ApiResult<Option<u32>> {
    value(q, "limit")
        .map(|v| {
            v.parse::<u32>().map_err(|_| {
                ApiError::bad_request(format!("limit must be a positive integer, got {v:?}"))
            })
        })
        .transpose()
}

/// `kind=animal|sound_event|all`; `default` when absent. `all` means no filter.
pub fn kind(q: &Params, default: Option<DetectionKind>) -> ApiResult<Option<DetectionKind>> {
    match value(q, "kind") {
        None => Ok(default),
        Some("all") => Ok(None),
        Some(v) => DetectionKind::parse(v).map(Some).ok_or_else(|| {
            ApiError::bad_request(format!(
                "kind must be animal, sound_event or all, got {v:?}"
            ))
        }),
    }
}

pub fn order(q: &Params) -> ApiResult<Order> {
    match value(q, "order") {
        None | Some("desc") => Ok(Order::Desc),
        Some("asc") => Ok(Order::Asc),
        Some(other) => Err(ApiError::bad_request(format!(
            "order must be asc or desc, got {other:?}"
        ))),
    }
}
