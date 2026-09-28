#![forbid(unsafe_code)]
//! SQLite persistence for Birdsong.
//!
//! [`SqliteStore`] records detections, answers the API's list and chart queries, and plans clip
//! retention ([`plan_purge`]). The schema lives in `migrations/` and is applied on open.

mod error;
mod janitor;
mod retention;
mod review;
mod sqlite;
mod store;
mod timefmt;
mod types;

pub use error::StoreError;
pub use janitor::{safe_clip_path, Janitor, PurgeReport, ReconcileReport};
pub use retention::plan_purge;
pub use review::{suggest_threshold, Review, ReviewSummary, MIN_REVIEWS, TARGET_PRECISION};
pub use sqlite::SqliteStore;
pub use store::DetectionStore;
pub use types::{
    AnalysisParams, ClipInfo, ClipToPurge, DailySpecies, DailyStats, DetectionQuery,
    DetectionRecord, Order, PurgeReason, SpeciesSummary, StoreOptions, StoredClip, DEFAULT_LIMIT,
    MAX_LIMIT,
};
