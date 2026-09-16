use birdsong_core::{Detection, DetectionKind};
use chrono::{DateTime, NaiveDate, Utc};

use crate::{ClipInfo, DailyStats, DetectionQuery, DetectionRecord, SpeciesSummary, StoreError};

/// Storage used by the pipeline and the HTTP API.
#[async_trait::async_trait]
pub trait DetectionStore: Send + Sync {
    /// Store detections atomically; returns their ids in the same order.
    async fn insert_many(&self, detections: &[Detection]) -> Result<Vec<i64>, StoreError>;

    /// Store one detection; returns its id.
    async fn insert(&self, detection: &Detection) -> Result<i64, StoreError> {
        let ids = self.insert_many(std::slice::from_ref(detection)).await?;
        ids.first().copied().ok_or_else(|| StoreError::Corrupt {
            column: "id",
            value: "no id returned".into(),
        })
    }

    /// Attach a clip to detections (one clip is shared by every detection in a window), or detach
    /// it with `None`.
    async fn set_clip(&self, ids: &[i64], clip: Option<&ClipInfo>) -> Result<(), StoreError>;

    async fn get(&self, id: i64) -> Result<Option<DetectionRecord>, StoreError>;

    async fn list(&self, query: &DetectionQuery) -> Result<Vec<DetectionRecord>, StoreError>;

    /// Detections on a station-local calendar date, per species per local hour; only `kind` when
    /// given.
    async fn stats_daily(
        &self,
        date: NaiveDate,
        kind: Option<DetectionKind>,
    ) -> Result<DailyStats, StoreError>;

    /// Per-species aggregates for detections at or after `since` (all time when `None`); only
    /// `kind` when given.
    async fn species_summary(
        &self,
        since: Option<DateTime<Utc>>,
        kind: Option<DetectionKind>,
    ) -> Result<Vec<SpeciesSummary>, StoreError>;

    /// Disk used by all distinct clip files (audio plus spectrogram).
    async fn total_clip_bytes(&self) -> Result<u64, StoreError>;
}
