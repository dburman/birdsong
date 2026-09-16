//! Janitor against a real database and a synthetic clips tree.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use birdsong_core::config::RetentionConfig;
use birdsong_core::Detection;
use birdsong_store::{
    AnalysisParams, ClipInfo, DetectionStore, Janitor, PurgeReport, ReconcileReport, SqliteStore,
    StoreOptions,
};
use chrono::{DateTime, TimeDelta, TimeZone, Utc};

const MIB: usize = 1024 * 1024;

struct Env {
    _dir: tempfile::TempDir,
    root: PathBuf,
    clips: PathBuf,
    store: SqliteStore,
}

async fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let opts = StoreOptions {
        timezone: chrono_tz::UTC,
        params: AnalysisParams {
            latitude: None,
            longitude: None,
            sensitivity: 1.25,
            overlap_seconds: 0.0,
            min_confidence: 0.7,
        },
    };
    let store = SqliteStore::open(&root.join("birdsong.sqlite"), opts)
        .await
        .unwrap();
    Env {
        clips: root.join("clips"),
        root,
        _dir: dir,
        store,
    }
}

fn now() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 9, 15, 12, 0, 0).unwrap()
}

fn write(path: &Path, bytes: usize) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, vec![0u8; bytes]).unwrap();
}

/// Insert a detection with a clip (and spectrogram) file of `bytes`.
async fn add(env: &Env, sci: &str, conf: f32, at: DateTime<Utc>, rel: &str, bytes: usize) -> i64 {
    let id = env
        .store
        .insert(&Detection {
            id: None,
            detected_at: at,
            scientific_name: sci.into(),
            common_name: sci.into(),
            confidence: conf,
            source_id: "mic0".into(),
            model_id: "test".into(),
            kind: birdsong_core::DetectionKind::Animal,
            clip_path: None,
        })
        .await
        .unwrap();
    let png = rel.replace(".wav", ".png");
    write(&env.clips.join(rel), bytes);
    write(&env.clips.join(&png), 10);
    env.store
        .set_clip(
            &[id],
            Some(&ClipInfo {
                clip_path: rel.into(),
                clip_bytes: bytes as u64,
                spectrogram_path: Some(png),
            }),
        )
        .await
        .unwrap();
    id
}

fn exists(env: &Env, rel: &str) -> bool {
    env.clips.join(rel).exists()
}

#[tokio::test]
async fn purge_applies_age_size_and_exemptions() {
    let env = env().await;
    let days = |d: i64| now() - TimeDelta::days(d);
    let old_best = add(&env, "A", 0.95, days(20), "d20/A/best.wav", MIB).await;
    let old_other = add(
        &env,
        "A",
        0.80,
        days(20) + TimeDelta::minutes(5),
        "d20/A_other/other.wav",
        MIB,
    )
    .await;
    add(&env, "B", 0.90, days(3), "d3/B/r1.wav", MIB).await;
    let r2 = add(
        &env,
        "B",
        0.80,
        days(3) + TimeDelta::minutes(1),
        "d3/B/r2.wav",
        MIB,
    )
    .await;
    add(&env, "C", 0.90, days(2), "d2/C/r3.wav", MIB).await;

    let policy = RetentionConfig {
        clip_max_age_days: 14,
        clip_max_total_mb: 3,
        keep_best_per_species_per_day: 1,
        detection_rows_max_age_days: 0,
        purge_interval_minutes: 30,
    };
    let janitor = Janitor::new(env.store.clone(), &env.clips, policy);
    let report = janitor.run_once(now()).await.unwrap();
    assert_eq!(
        report,
        PurgeReport {
            clips_deleted_age: 1,
            clips_deleted_size: 1,
            bytes_freed: 2 * MIB as u64,
            dirs_removed: 1, // d20/A_other; d20 still holds A/
            ..Default::default()
        }
    );

    assert!(
        exists(&env, "d20/A/best.wav") && exists(&env, "d20/A/best.png"),
        "exempt: best of the day"
    );
    assert!(!exists(&env, "d20/A_other/other.wav") && !exists(&env, "d20/A_other/other.png"));
    assert!(
        !env.clips.join("d20/A_other").exists(),
        "empty directory removed"
    );
    assert!(
        !exists(&env, "d3/B/r2.wav"),
        "oldest non-exempt clip removed for size"
    );
    assert!(exists(&env, "d3/B/r1.wav") && exists(&env, "d2/C/r3.wav"));

    let other = env.store.get(old_other).await.unwrap().unwrap();
    assert_eq!(
        (other.clip_path, other.spectrogram_path),
        (None, None),
        "row kept, clip detached"
    );
    assert!(env
        .store
        .get(r2)
        .await
        .unwrap()
        .unwrap()
        .clip_path
        .is_none());
    assert!(env
        .store
        .get(old_best)
        .await
        .unwrap()
        .unwrap()
        .clip_path
        .is_some());
    assert_eq!(env.store.total_clip_bytes().await.unwrap(), 3 * MIB as u64);

    assert_eq!(
        janitor.run_once(now()).await.unwrap(),
        PurgeReport::default(),
        "second pass is a no-op"
    );
    assert_eq!(
        janitor.reconcile().await.unwrap(),
        ReconcileReport::default(),
        "database and files agree"
    );
}

#[tokio::test]
async fn row_age_limit_removes_rows_and_their_clips() {
    let env = env().await;
    let old = add(
        &env,
        "A",
        0.95,
        now() - TimeDelta::days(20),
        "old/A/best.wav",
        100,
    )
    .await;
    let recent = add(
        &env,
        "A",
        0.90,
        now() - TimeDelta::days(1),
        "new/A/x.wav",
        100,
    )
    .await;
    let policy = RetentionConfig {
        clip_max_age_days: 0,
        clip_max_total_mb: 0,
        keep_best_per_species_per_day: 1,
        detection_rows_max_age_days: 10,
        purge_interval_minutes: 30,
    };
    let report = Janitor::new(env.store.clone(), &env.clips, policy)
        .run_once(now())
        .await
        .unwrap();
    assert_eq!(
        report.clips_deleted_row_age, 1,
        "exemption does not outlive the row"
    );
    assert_eq!(report.rows_deleted, 1);
    assert_eq!(report.dirs_removed, 2, "old/A and old");
    assert!(env.store.get(old).await.unwrap().is_none());
    assert!(!exists(&env, "old/A/best.wav"));
    assert!(env
        .store
        .get(recent)
        .await
        .unwrap()
        .unwrap()
        .clip_path
        .is_some());
}

#[tokio::test]
async fn reconcile_fixes_both_directions_and_stays_inside_clips_dir() {
    let env = env().await;
    let t = now() - TimeDelta::days(1);
    let kept = add(&env, "A", 0.9, t, "d/A/kept.wav", 100).await;
    let missing = add(&env, "B", 0.9, t, "d/B/missing.wav", 100).await;
    std::fs::remove_file(env.clips.join("d/B/missing.wav")).unwrap();

    // A row pointing outside the clips directory, with a real file there that must survive.
    let outside = env.root.join("outside.wav");
    write(&outside, 5);
    let evil = env
        .store
        .insert(&Detection {
            id: None,
            detected_at: t,
            scientific_name: "C".into(),
            common_name: "C".into(),
            confidence: 0.9,
            source_id: "mic0".into(),
            model_id: "test".into(),
            kind: birdsong_core::DetectionKind::Animal,
            clip_path: None,
        })
        .await
        .unwrap();
    env.store
        .set_clip(
            &[evil],
            Some(&ClipInfo {
                clip_path: "../outside.wav".into(),
                clip_bytes: 5,
                spectrogram_path: None,
            }),
        )
        .await
        .unwrap();

    write(&env.clips.join("x/orphan.wav"), 10);
    write(&env.clips.join("x/y/half-written.wav.tmp"), 10);

    let report = Janitor::new(env.store.clone(), &env.clips, RetentionConfig::default())
        .reconcile()
        .await
        .unwrap();
    assert_eq!(
        report,
        ReconcileReport {
            rows_cleared: 2,
            orphan_files_deleted: 3,
            dirs_removed: 3,
            errors: 1
        },
        "orphans: x/orphan.wav, the .tmp, and d/B/missing.png; dirs: x/y, x, and the now empty d/B"
    );
    assert!(
        outside.exists(),
        "never touches files outside the clips directory"
    );
    assert!(exists(&env, "d/A/kept.wav") && exists(&env, "d/A/kept.png"));
    assert!(env
        .store
        .get(kept)
        .await
        .unwrap()
        .unwrap()
        .clip_path
        .is_some());
    assert!(env
        .store
        .get(missing)
        .await
        .unwrap()
        .unwrap()
        .clip_path
        .is_none());
    assert!(env
        .store
        .get(evil)
        .await
        .unwrap()
        .unwrap()
        .clip_path
        .is_none());
    assert!(!env.clips.join("x").exists());
    assert!(!env.clips.join("d/B").exists() && env.clips.join("d/A").exists());
}

#[tokio::test]
async fn reconcile_without_clips_directory_is_fine() {
    let env = env().await;
    let report = Janitor::new(env.store.clone(), &env.clips, RetentionConfig::default())
        .reconcile()
        .await
        .unwrap();
    assert_eq!(report, ReconcileReport::default());
}
