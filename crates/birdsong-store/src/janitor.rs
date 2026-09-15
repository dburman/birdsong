//! Retention enforcement: deletes clip files and keeps the database in step with the filesystem.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use birdsong_core::config::RetentionConfig;
use chrono::{DateTime, TimeDelta, Utc};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::{PurgeReason, SqliteStore, StoreError};

/// Join a stored relative clip path onto the clips directory, refusing anything that could escape
/// it (absolute paths, `..`, `.`, prefixes). `None` means "do not touch the filesystem".
pub fn safe_clip_path(clips_dir: &Path, relative: &str) -> Option<PathBuf> {
    let rel = Path::new(relative);
    let normal =
        !relative.is_empty() && rel.components().all(|c| matches!(c, Component::Normal(_)));
    normal.then(|| clips_dir.join(rel))
}

/// Outcome of one retention pass.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PurgeReport {
    pub clips_deleted_age: u64,
    pub clips_deleted_size: u64,
    /// Clips removed because their detection rows exceeded `detection_rows_max_age_days`.
    pub clips_deleted_row_age: u64,
    pub bytes_freed: u64,
    pub rows_deleted: u64,
    pub dirs_removed: u64,
    pub errors: u64,
}

/// Outcome of a startup consistency check.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ReconcileReport {
    /// Rows whose clip file was missing (or whose path was unsafe), now detached.
    pub rows_cleared: u64,
    /// Files under the clips directory that no row referenced, now deleted.
    pub orphan_files_deleted: u64,
    pub dirs_removed: u64,
    pub errors: u64,
}

/// Applies [`RetentionConfig`] to the clips directory and database.
#[derive(Clone)]
pub struct Janitor {
    store: SqliteStore,
    clips_dir: PathBuf,
    policy: RetentionConfig,
}

async fn remove_file_if_present(path: &Path) -> std::io::Result<()> {
    match tokio::fs::remove_file(path).await {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Remove each directory if empty, then its parents, stopping at `root` (never removed).
async fn remove_empty_dirs(root: &Path, dirs: impl IntoIterator<Item = PathBuf>) -> u64 {
    let mut dirs: Vec<PathBuf> = dirs.into_iter().collect();
    dirs.sort_by_key(|d| std::cmp::Reverse(d.components().count()));
    dirs.dedup();
    let mut removed = 0;
    for start in dirs {
        let mut dir = start;
        while dir != root && dir.starts_with(root) {
            if tokio::fs::remove_dir(&dir).await.is_err() {
                break; // not empty, already gone, or not permitted
            }
            removed += 1;
            match dir.parent() {
                Some(parent) => dir = parent.to_path_buf(),
                None => break,
            }
        }
    }
    removed
}

/// Every file and directory below `root` (not following directory symlinks). Directories are
/// listed parents first.
fn walk(root: &Path) -> std::io::Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && dir == root => {
                return Ok((files, dirs))
            }
            Err(e) => return Err(e),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                dirs.push(path.clone());
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    Ok((files, dirs))
}

impl Janitor {
    pub fn new(store: SqliteStore, clips_dir: impl Into<PathBuf>, policy: RetentionConfig) -> Self {
        Self {
            store,
            clips_dir: clips_dir.into(),
            policy,
        }
    }

    pub fn clips_dir(&self) -> &Path {
        &self.clips_dir
    }

    /// Delete a clip's WAV and spectrogram, then detach it from its rows. File deletion comes
    /// first so a crash in between leaves an orphan file (cleaned by [`Janitor::reconcile`]),
    /// never a row pointing at nothing. Returns whether the clip was detached.
    async fn delete_clip(
        &self,
        clip_path: &str,
        spectrogram_path: Option<&str>,
        touched: &mut HashSet<PathBuf>,
        errors: &mut u64,
    ) -> Result<bool, StoreError> {
        let mut files_ok = true;
        for rel in std::iter::once(clip_path).chain(spectrogram_path) {
            let Some(path) = safe_clip_path(&self.clips_dir, rel) else {
                tracing::warn!(
                    path = rel,
                    "refusing to delete a clip path outside the clips directory"
                );
                *errors += 1;
                continue;
            };
            match remove_file_if_present(&path).await {
                Ok(()) => {
                    if let Some(parent) = path.parent() {
                        touched.insert(parent.to_path_buf());
                    }
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "cannot delete clip file");
                    *errors += 1;
                    files_ok = false;
                }
            }
        }
        if files_ok {
            self.store.clear_clip(clip_path).await?;
        }
        Ok(files_ok)
    }

    /// One retention pass (BUILD_PLAN Step 7): age limit, size cap, best-per-species exemption,
    /// optional row age limit, empty directory cleanup.
    pub async fn run_once(&self, now: DateTime<Utc>) -> Result<PurgeReport, StoreError> {
        let mut report = PurgeReport::default();
        let mut touched = HashSet::new();

        for clip in self.store.clips_to_purge(&self.policy, now).await? {
            if self
                .delete_clip(
                    &clip.clip_path,
                    clip.spectrogram_path.as_deref(),
                    &mut touched,
                    &mut report.errors,
                )
                .await?
            {
                report.bytes_freed += clip.bytes;
                match clip.reason {
                    PurgeReason::Age => report.clips_deleted_age += 1,
                    PurgeReason::Size => report.clips_deleted_size += 1,
                }
            }
        }

        if self.policy.detection_rows_max_age_days > 0 {
            let cutoff = now - TimeDelta::days(self.policy.detection_rows_max_age_days as i64);
            let old_clips = self
                .store
                .stored_clips()
                .await?
                .into_iter()
                .filter(|c| c.detected_at < cutoff);
            for clip in old_clips {
                if self
                    .delete_clip(
                        &clip.clip_path,
                        clip.spectrogram_path.as_deref(),
                        &mut touched,
                        &mut report.errors,
                    )
                    .await?
                {
                    report.bytes_freed += clip.bytes;
                    report.clips_deleted_row_age += 1;
                }
            }
            report.rows_deleted = self.store.delete_detections_before(cutoff).await?;
        }

        report.dirs_removed = remove_empty_dirs(&self.clips_dir, touched).await;
        Ok(report)
    }

    /// Make the database and the clips directory agree. Call before the pipeline starts writing.
    pub async fn reconcile(&self) -> Result<ReconcileReport, StoreError> {
        let mut report = ReconcileReport::default();
        let mut referenced = HashSet::new();
        for clip in self.store.stored_clips().await? {
            let wav = safe_clip_path(&self.clips_dir, &clip.clip_path);
            let present = match &wav {
                Some(path) => tokio::fs::try_exists(path).await.unwrap_or(false),
                None => {
                    tracing::warn!(path = %clip.clip_path, "unsafe clip path in database; detaching");
                    report.errors += 1;
                    false
                }
            };
            match wav {
                Some(path) if present => {
                    referenced.insert(path);
                    if let Some(png) = clip
                        .spectrogram_path
                        .as_deref()
                        .and_then(|s| safe_clip_path(&self.clips_dir, s))
                    {
                        referenced.insert(png);
                    }
                }
                _ => report.rows_cleared += self.store.clear_clip(&clip.clip_path).await?,
            }
        }

        let root = self.clips_dir.clone();
        let walked = tokio::task::spawn_blocking(move || walk(&root))
            .await
            .map_err(|e| StoreError::Io {
                path: self.clips_dir.clone(),
                source: std::io::Error::other(e.to_string()),
            })?;
        let (files, dirs) = walked.map_err(|source| StoreError::Io {
            path: self.clips_dir.clone(),
            source,
        })?;
        for file in files.into_iter().filter(|f| !referenced.contains(f)) {
            match remove_file_if_present(&file).await {
                Ok(()) => report.orphan_files_deleted += 1,
                Err(e) => {
                    tracing::warn!(path = %file.display(), error = %e, "cannot delete orphan file");
                    report.errors += 1;
                }
            }
        }
        report.dirs_removed = remove_empty_dirs(&self.clips_dir, dirs).await;
        Ok(report)
    }

    /// Run a pass immediately and then every `purge_interval_minutes` until cancelled.
    pub async fn run(self, cancel: CancellationToken) {
        let minutes = u64::from(self.policy.purge_interval_minutes.max(1));
        let mut ticker = tokio::time::interval(Duration::from_secs(minutes * 60));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = ticker.tick() => match self.run_once(Utc::now()).await {
                    Ok(r) => tracing::info!(
                        deleted_age = r.clips_deleted_age,
                        deleted_size = r.clips_deleted_size,
                        deleted_row_age = r.clips_deleted_row_age,
                        freed_mb = %format_args!("{:.1}", r.bytes_freed as f64 / 1_048_576.0),
                        rows_deleted = r.rows_deleted,
                        dirs_removed = r.dirs_removed,
                        errors = r.errors,
                        "retention pass"
                    ),
                    Err(e) => tracing::error!(error = %e, "retention pass failed"),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_paths_are_refused() {
        let root = Path::new("/data/clips");
        assert_eq!(
            safe_clip_path(root, "2026-05-01/A/x.wav"),
            Some(PathBuf::from("/data/clips/2026-05-01/A/x.wav"))
        );
        for bad in ["", "../x.wav", "a/../../x.wav", "/etc/passwd", "./x.wav"] {
            assert_eq!(safe_clip_path(root, bad), None, "{bad:?}");
        }
    }
}
