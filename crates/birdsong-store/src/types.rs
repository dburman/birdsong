use birdsong_core::{Config, Detection, DetectionKind};
use chrono::{DateTime, NaiveDate, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};

/// Page size when a query gives none.
pub const DEFAULT_LIMIT: u32 = 100;
/// Largest page a query may request.
pub const MAX_LIMIT: u32 = 1000;

/// Analysis settings recorded with every detection (BirdNET-Pi keeps the same columns).
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisParams {
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub sensitivity: f32,
    pub overlap_seconds: f32,
    pub min_confidence: f32,
}

/// How the store interprets and annotates detections.
#[derive(Clone, Debug, PartialEq)]
pub struct StoreOptions {
    /// Station time zone for `local_date` / `local_hour` (daily charts) and the week number.
    pub timezone: Tz,
    pub params: AnalysisParams,
}

impl StoreOptions {
    pub fn from_config(cfg: &Config) -> Self {
        let located = cfg.station.has_location();
        Self {
            timezone: cfg.station.timezone,
            params: AnalysisParams {
                latitude: located.then_some(cfg.station.latitude),
                longitude: located.then_some(cfg.station.longitude),
                sensitivity: cfg.detection.sensitivity,
                overlap_seconds: cfg.detection.overlap_seconds,
                min_confidence: cfg.detection.min_confidence,
            },
        }
    }
}

/// A stored detection, as returned by queries and the HTTP API.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectionRecord {
    pub id: i64,
    /// Start of the 3 s window, UTC (microsecond precision).
    pub detected_at: DateTime<Utc>,
    /// Calendar date in the station time zone.
    pub local_date: NaiveDate,
    /// Hour `0..=23` in the station time zone.
    pub local_hour: u32,
    /// BirdNET week `1..=48`.
    pub week: Option<u32>,
    pub scientific_name: String,
    pub common_name: String,
    pub confidence: f32,
    pub source_id: String,
    pub model_id: String,
    pub kind: DetectionKind,
    /// Relative to the clips directory; `None` when never saved or purged.
    pub clip_path: Option<String>,
    pub clip_bytes: Option<u64>,
    pub spectrogram_path: Option<String>,
}

impl DetectionRecord {
    pub fn to_detection(&self) -> Detection {
        Detection {
            id: Some(self.id),
            detected_at: self.detected_at,
            scientific_name: self.scientific_name.clone(),
            common_name: self.common_name.clone(),
            confidence: self.confidence,
            source_id: self.source_id.clone(),
            model_id: self.model_id.clone(),
            kind: self.kind,
            clip_path: self.clip_path.clone(),
        }
    }
}

/// Result ordering by id (equivalently, by insertion time).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Order {
    Asc,
    #[default]
    Desc,
}

/// Filters for [`crate::DetectionStore::list`]. All are optional and combine with AND.
///
/// To ingest reliably, page with `after_id` = highest id seen and `order = Asc`: ids are assigned
/// in commit order and never reused, so no row is skipped or repeated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DetectionQuery {
    /// Only ids strictly greater than this.
    pub after_id: Option<i64>,
    /// Only ids strictly less than this (for paging backwards with `Desc`).
    pub before_id: Option<i64>,
    /// `detected_at >= since`.
    pub since: Option<DateTime<Utc>>,
    /// `detected_at < until`.
    pub until: Option<DateTime<Utc>>,
    /// Exact scientific name.
    pub species: Option<String>,
    pub min_confidence: Option<f32>,
    /// Only animals or only sound events; both when `None`.
    pub kind: Option<DetectionKind>,
    /// Defaults to [`DEFAULT_LIMIT`]; clamped to `1..=MAX_LIMIT`.
    pub limit: Option<u32>,
    pub order: Order,
}

impl DetectionQuery {
    pub fn effective_limit(&self) -> u32 {
        self.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT)
    }
}

/// A saved clip attached to one or more detections from the same window.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClipInfo {
    pub clip_path: String,
    pub clip_bytes: u64,
    pub spectrogram_path: Option<String>,
}

/// One species' detections on one day, bucketed by local hour.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailySpecies {
    pub scientific_name: String,
    pub common_name: String,
    pub kind: DetectionKind,
    pub total: u32,
    pub by_hour: [u32; 24],
}

/// Data for the "today by hour" chart. Species sorted by total, most first.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyStats {
    pub date: NaiveDate,
    pub species: Vec<DailySpecies>,
}

/// Per-species aggregate over a time window. Sorted by count, most first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeciesSummary {
    pub scientific_name: String,
    /// Common name of the best detection.
    pub common_name: String,
    pub kind: DetectionKind,
    pub count: u64,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub max_confidence: f32,
    /// Highest-confidence detection (latest on ties).
    pub best_detection_id: i64,
    /// Highest-confidence detection that still has a clip.
    pub best_clip_detection_id: Option<i64>,
}

/// A distinct clip file referenced by the database.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredClip {
    pub clip_path: String,
    pub spectrogram_path: Option<String>,
    pub bytes: u64,
    /// Latest detection time among rows using this clip.
    pub detected_at: DateTime<Utc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PurgeReason {
    /// Older than `retention.clip_max_age_days`.
    Age,
    /// Oldest clip while the total exceeds `retention.clip_max_total_mb`.
    Size,
}

/// A clip the retention policy says to delete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipToPurge {
    pub clip_path: String,
    pub spectrogram_path: Option<String>,
    pub bytes: u64,
    pub detected_at: DateTime<Utc>,
    pub reason: PurgeReason,
}
