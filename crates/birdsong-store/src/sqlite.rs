use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use birdsong_core::config::RetentionConfig;
use birdsong_core::{local_date_and_hour, week_of_year, Detection};
use chrono::{DateTime, NaiveDate, Utc};
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteRow, SqliteSynchronous,
};
use sqlx::{QueryBuilder, Row, Sqlite, SqlitePool};

use crate::retention::plan_purge;
use crate::timefmt::{format_date, format_ts, parse_date, parse_ts};
use crate::{
    ClipInfo, ClipToPurge, DailySpecies, DailyStats, DetectionQuery, DetectionRecord,
    DetectionStore, Order, SpeciesSummary, StoreError, StoreOptions, StoredClip,
};

const SELECT_DETECTIONS: &str =
    "SELECT id, detected_at_utc, local_date, local_hour, week, scientific_name, \
     common_name, confidence, source_id, model_id, clip_path, clip_bytes, spectrogram_path \
     FROM detections WHERE 1 = 1";

const INSERT_DETECTION: &str = "INSERT INTO detections (detected_at_utc, local_date, local_hour, \
     scientific_name, common_name, confidence, source_id, model_id, latitude, longitude, week, \
     sensitivity, overlap_seconds, min_confidence, clip_path) \
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

const DAILY_STATS: &str =
    "SELECT scientific_name, MAX(common_name) AS common_name, local_hour, COUNT(*) AS n \
     FROM detections WHERE local_date = ? GROUP BY scientific_name, local_hour";

const SPECIES_SUMMARY: &str = "WITH ranked AS ( \
       SELECT id, scientific_name, common_name, confidence, detected_at_utc, clip_path, \
         ROW_NUMBER() OVER (PARTITION BY scientific_name ORDER BY confidence DESC, id DESC) AS rn, \
         ROW_NUMBER() OVER (PARTITION BY scientific_name ORDER BY (clip_path IS NULL), confidence DESC, id DESC) AS rn_clip \
       FROM detections WHERE detected_at_utc >= ?) \
     SELECT scientific_name, \
       MAX(CASE WHEN rn = 1 THEN common_name END) AS common_name, \
       COUNT(*) AS n, MIN(detected_at_utc) AS first_seen, MAX(detected_at_utc) AS last_seen, \
       MAX(confidence) AS max_confidence, \
       MAX(CASE WHEN rn = 1 THEN id END) AS best_id, \
       MAX(CASE WHEN rn_clip = 1 AND clip_path IS NOT NULL THEN id END) AS best_clip_id \
     FROM ranked GROUP BY scientific_name ORDER BY n DESC, scientific_name";

const EXEMPT_CLIPS: &str = "SELECT DISTINCT clip_path FROM ( \
       SELECT clip_path, ROW_NUMBER() OVER (PARTITION BY local_date, scientific_name ORDER BY confidence DESC, id DESC) AS rn \
       FROM detections WHERE clip_path IS NOT NULL) \
     WHERE rn <= ?";

const STORED_CLIPS: &str =
    "SELECT clip_path, MAX(spectrogram_path) AS spectrogram_path, COALESCE(MAX(clip_bytes), 0) AS bytes, MAX(detected_at_utc) AS detected_at \
     FROM detections WHERE clip_path IS NOT NULL GROUP BY clip_path";

const TOTAL_CLIP_BYTES: &str = "SELECT COALESCE(SUM(bytes), 0) AS total FROM ( \
       SELECT MAX(COALESCE(clip_bytes, 0)) AS bytes FROM detections WHERE clip_path IS NOT NULL GROUP BY clip_path)";

/// SQLite-backed [`DetectionStore`].
///
/// WAL mode with one writer connection (so ids are assigned in commit order) and a small pool of
/// read-only connections that never block on the writer.
#[derive(Clone)]
pub struct SqliteStore {
    writer: SqlitePool,
    reader: SqlitePool,
    opts: StoreOptions,
    path: PathBuf,
}

impl SqliteStore {
    /// Open (creating if needed) the database at `path` and apply migrations.
    pub async fn open(path: &Path, opts: StoreOptions) -> Result<Self, StoreError> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| StoreError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
        }
        let base = SqliteConnectOptions::new()
            .filename(path)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(Duration::from_secs(5));
        let writer = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(base.clone().create_if_missing(true))
            .await?;
        sqlx::migrate!("./migrations").run(&writer).await?;
        let reader = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(base.read_only(true))
            .await?;
        tracing::info!(path = %path.display(), "database opened");
        Ok(Self {
            writer,
            reader,
            opts,
            path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn options(&self) -> &StoreOptions {
        &self.opts
    }

    /// Close every connection (flushes the WAL).
    pub async fn close(&self) {
        self.reader.close().await;
        self.writer.close().await;
    }

    /// Every distinct clip file the database references.
    pub async fn stored_clips(&self) -> Result<Vec<StoredClip>, StoreError> {
        let rows = sqlx::query(STORED_CLIPS).fetch_all(&self.reader).await?;
        rows.iter()
            .map(|r| {
                Ok(StoredClip {
                    clip_path: r.try_get("clip_path")?,
                    spectrogram_path: r.try_get("spectrogram_path")?,
                    bytes: r.try_get::<i64, _>("bytes")?.max(0) as u64,
                    detected_at: parse_ts(
                        "detected_at_utc",
                        &r.try_get::<String, _>("detected_at")?,
                    )?,
                })
            })
            .collect()
    }

    /// Clips holding one of the `n` highest-confidence detections of a species on a local day.
    pub async fn exempt_clip_paths(&self, n: u32) -> Result<HashSet<String>, StoreError> {
        if n == 0 {
            return Ok(HashSet::new());
        }
        let rows = sqlx::query(EXEMPT_CLIPS)
            .bind(n as i64)
            .fetch_all(&self.reader)
            .await?;
        rows.iter()
            .map(|r| {
                r.try_get::<String, _>("clip_path")
                    .map_err(StoreError::from)
            })
            .collect()
    }

    /// Apply the retention policy to the current database (see [`plan_purge`]). Deletes nothing.
    pub async fn clips_to_purge(
        &self,
        policy: &RetentionConfig,
        now: DateTime<Utc>,
    ) -> Result<Vec<ClipToPurge>, StoreError> {
        let clips = self.stored_clips().await?;
        let exempt = self
            .exempt_clip_paths(policy.keep_best_per_species_per_day)
            .await?;
        Ok(plan_purge(&clips, &exempt, policy, now))
    }

    /// Bytes of all distinct clips referenced by the database.
    pub async fn total_clip_bytes(&self) -> Result<u64, StoreError> {
        let total: i64 = sqlx::query(TOTAL_CLIP_BYTES)
            .fetch_one(&self.reader)
            .await?
            .try_get("total")?;
        Ok(total.max(0) as u64)
    }

    /// Detach a clip from every detection using it (after its file is deleted). Returns rows changed.
    pub async fn clear_clip(&self, clip_path: &str) -> Result<u64, StoreError> {
        let r = sqlx::query(
            "UPDATE detections SET clip_path = NULL, clip_bytes = NULL, spectrogram_path = NULL WHERE clip_path = ?",
        )
        .bind(clip_path)
        .execute(&self.writer)
        .await?;
        Ok(r.rows_affected())
    }

    /// Delete detection rows older than `cutoff`. Returns rows deleted.
    pub async fn delete_detections_before(&self, cutoff: DateTime<Utc>) -> Result<u64, StoreError> {
        let r = sqlx::query("DELETE FROM detections WHERE detected_at_utc < ?")
            .bind(format_ts(cutoff))
            .execute(&self.writer)
            .await?;
        Ok(r.rows_affected())
    }
}

fn record_from_row(r: &SqliteRow) -> Result<DetectionRecord, StoreError> {
    Ok(DetectionRecord {
        id: r.try_get("id")?,
        detected_at: parse_ts(
            "detected_at_utc",
            &r.try_get::<String, _>("detected_at_utc")?,
        )?,
        local_date: parse_date("local_date", &r.try_get::<String, _>("local_date")?)?,
        local_hour: r.try_get::<i64, _>("local_hour")?.clamp(0, 23) as u32,
        week: r
            .try_get::<Option<i64>, _>("week")?
            .map(|w| w.max(0) as u32),
        scientific_name: r.try_get("scientific_name")?,
        common_name: r.try_get("common_name")?,
        confidence: r.try_get::<f64, _>("confidence")? as f32,
        source_id: r.try_get("source_id")?,
        model_id: r.try_get("model_id")?,
        clip_path: r.try_get("clip_path")?,
        clip_bytes: r
            .try_get::<Option<i64>, _>("clip_bytes")?
            .map(|b| b.max(0) as u64),
        spectrogram_path: r.try_get("spectrogram_path")?,
    })
}

#[async_trait::async_trait]
impl DetectionStore for SqliteStore {
    async fn total_clip_bytes(&self) -> Result<u64, StoreError> {
        SqliteStore::total_clip_bytes(self).await
    }

    async fn insert_many(&self, detections: &[Detection]) -> Result<Vec<i64>, StoreError> {
        let p = &self.opts.params;
        let mut tx = self.writer.begin().await?;
        let mut ids = Vec::with_capacity(detections.len());
        for d in detections {
            let (local_date, local_hour) = local_date_and_hour(d.detected_at, self.opts.timezone);
            let r = sqlx::query(INSERT_DETECTION)
                .bind(format_ts(d.detected_at))
                .bind(format_date(local_date))
                .bind(local_hour as i64)
                .bind(&d.scientific_name)
                .bind(&d.common_name)
                .bind(d.confidence as f64)
                .bind(&d.source_id)
                .bind(&d.model_id)
                .bind(p.latitude)
                .bind(p.longitude)
                .bind(week_of_year(local_date) as i64)
                .bind(p.sensitivity as f64)
                .bind(p.overlap_seconds as f64)
                .bind(p.min_confidence as f64)
                .bind(&d.clip_path)
                .execute(&mut *tx)
                .await?;
            ids.push(r.last_insert_rowid());
        }
        tx.commit().await?;
        Ok(ids)
    }

    async fn set_clip(&self, ids: &[i64], clip: Option<&ClipInfo>) -> Result<(), StoreError> {
        let mut tx = self.writer.begin().await?;
        for id in ids {
            sqlx::query("UPDATE detections SET clip_path = ?, clip_bytes = ?, spectrogram_path = ? WHERE id = ?")
                .bind(clip.map(|c| c.clip_path.as_str()))
                .bind(clip.map(|c| c.clip_bytes.min(i64::MAX as u64) as i64))
                .bind(clip.and_then(|c| c.spectrogram_path.as_deref()))
                .bind(id)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn get(&self, id: i64) -> Result<Option<DetectionRecord>, StoreError> {
        let mut qb = QueryBuilder::<Sqlite>::new(SELECT_DETECTIONS);
        qb.push(" AND id = ").push_bind(id);
        let row = qb.build().fetch_optional(&self.reader).await?;
        row.as_ref().map(record_from_row).transpose()
    }

    async fn list(&self, q: &DetectionQuery) -> Result<Vec<DetectionRecord>, StoreError> {
        let mut qb = QueryBuilder::<Sqlite>::new(SELECT_DETECTIONS);
        if let Some(after) = q.after_id {
            qb.push(" AND id > ").push_bind(after);
        }
        if let Some(before) = q.before_id {
            qb.push(" AND id < ").push_bind(before);
        }
        if let Some(since) = q.since {
            qb.push(" AND detected_at_utc >= ")
                .push_bind(format_ts(since));
        }
        if let Some(until) = q.until {
            qb.push(" AND detected_at_utc < ")
                .push_bind(format_ts(until));
        }
        if let Some(species) = &q.species {
            qb.push(" AND scientific_name = ")
                .push_bind(species.clone());
        }
        if let Some(min) = q.min_confidence {
            qb.push(" AND confidence >= ").push_bind(min as f64);
        }
        qb.push(match q.order {
            Order::Asc => " ORDER BY id ASC",
            Order::Desc => " ORDER BY id DESC",
        });
        qb.push(" LIMIT ").push_bind(q.effective_limit() as i64);
        let rows = qb.build().fetch_all(&self.reader).await?;
        rows.iter().map(record_from_row).collect()
    }

    async fn stats_daily(&self, date: NaiveDate) -> Result<DailyStats, StoreError> {
        let rows = sqlx::query(DAILY_STATS)
            .bind(format_date(date))
            .fetch_all(&self.reader)
            .await?;
        let mut species: Vec<DailySpecies> = Vec::new();
        for r in &rows {
            let scientific_name: String = r.try_get("scientific_name")?;
            let hour = r.try_get::<i64, _>("local_hour")?.clamp(0, 23) as usize;
            let n = r.try_get::<i64, _>("n")?.max(0) as u32;
            let idx = match species
                .iter()
                .position(|s| s.scientific_name == scientific_name)
            {
                Some(i) => i,
                None => {
                    species.push(DailySpecies {
                        scientific_name,
                        common_name: r.try_get("common_name")?,
                        total: 0,
                        by_hour: [0; 24],
                    });
                    species.len() - 1
                }
            };
            species[idx].by_hour[hour] += n;
            species[idx].total += n;
        }
        species.sort_by(|a, b| {
            b.total
                .cmp(&a.total)
                .then_with(|| a.common_name.cmp(&b.common_name))
        });
        Ok(DailyStats { date, species })
    }

    async fn species_summary(
        &self,
        since: Option<DateTime<Utc>>,
    ) -> Result<Vec<SpeciesSummary>, StoreError> {
        let since = since.map(format_ts).unwrap_or_default();
        let rows = sqlx::query(SPECIES_SUMMARY)
            .bind(since)
            .fetch_all(&self.reader)
            .await?;
        rows.iter()
            .map(|r| {
                Ok(SpeciesSummary {
                    scientific_name: r.try_get("scientific_name")?,
                    common_name: r.try_get("common_name")?,
                    count: r.try_get::<i64, _>("n")?.max(0) as u64,
                    first_seen: parse_ts(
                        "detected_at_utc",
                        &r.try_get::<String, _>("first_seen")?,
                    )?,
                    last_seen: parse_ts("detected_at_utc", &r.try_get::<String, _>("last_seen")?)?,
                    max_confidence: r.try_get::<f64, _>("max_confidence")? as f32,
                    best_detection_id: r.try_get("best_id")?,
                    best_clip_detection_id: r.try_get("best_clip_id")?,
                })
            })
            .collect()
    }
}
