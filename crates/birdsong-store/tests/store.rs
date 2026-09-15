//! SqliteStore against a real database file in a temp directory.
#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::sync::Arc;

use birdsong_core::config::RetentionConfig;
use birdsong_core::Detection;
use birdsong_store::{
    AnalysisParams, ClipInfo, DetectionQuery, DetectionStore, Order, PurgeReason, SqliteStore,
    StoreOptions,
};
use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};

fn opts() -> StoreOptions {
    StoreOptions {
        timezone: chrono_tz::America::New_York,
        params: AnalysisParams {
            latitude: Some(42.36),
            longitude: Some(-71.06),
            sensitivity: 1.25,
            overlap_seconds: 0.0,
            min_confidence: 0.7,
        },
    }
}

async fn open() -> (tempfile::TempDir, SqliteStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(&dir.path().join("sub/birdsong.sqlite"), opts())
        .await
        .unwrap();
    (dir, store)
}

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
}

fn det(sci: &str, common: &str, conf: f32, at: DateTime<Utc>) -> Detection {
    Detection {
        id: None,
        detected_at: at,
        scientific_name: sci.into(),
        common_name: common.into(),
        confidence: conf,
        source_id: "mic0".into(),
        model_id: "birdnet-v2.4".into(),
        clip_path: None,
    }
}

fn clip(path: &str, bytes: u64) -> ClipInfo {
    ClipInfo {
        clip_path: path.into(),
        clip_bytes: bytes,
        spectrogram_path: Some(format!("{path}.png")),
    }
}

#[tokio::test]
async fn insert_get_round_trip() {
    let (_dir, store) = open().await;
    // 2026-05-15 10:00:03.123456789 UTC = 06:00 EDT; nanoseconds are truncated to micros.
    let at = utc(2026, 5, 15, 10, 0) + TimeDelta::nanoseconds(3_123_456_789);
    let id = store
        .insert(&det(
            "Poecile atricapillus",
            "Black-capped Chickadee",
            0.87,
            at,
        ))
        .await
        .unwrap();
    let r = store.get(id).await.unwrap().expect("stored");
    assert_eq!(r.id, id);
    assert_eq!(
        r.detected_at,
        utc(2026, 5, 15, 10, 0) + TimeDelta::microseconds(3_123_456)
    );
    assert_eq!(r.local_date, NaiveDate::from_ymd_opt(2026, 5, 15).unwrap());
    assert_eq!(r.local_hour, 6);
    assert_eq!(r.week, Some(19)); // May (5) → (5-1)*4 + week 3 of the month
    assert_eq!(r.common_name, "Black-capped Chickadee");
    assert!((r.confidence - 0.87).abs() < 1e-6);
    assert_eq!(r.clip_path, None);
    assert_eq!(r.to_detection().id, Some(id));
    assert!(store.get(id + 1000).await.unwrap().is_none());

    store
        .set_clip(
            &[id],
            Some(&clip("2026-05-15/Black_capped_Chickadee/a.wav", 576_044)),
        )
        .await
        .unwrap();
    let r = store.get(id).await.unwrap().unwrap();
    assert_eq!(
        r.clip_path.as_deref(),
        Some("2026-05-15/Black_capped_Chickadee/a.wav")
    );
    assert_eq!(r.clip_bytes, Some(576_044));
    assert_eq!(
        r.spectrogram_path.as_deref(),
        Some("2026-05-15/Black_capped_Chickadee/a.wav.png")
    );
    store.set_clip(&[id], None).await.unwrap();
    assert_eq!(store.get(id).await.unwrap().unwrap().clip_path, None);
}

#[tokio::test]
async fn insert_many_is_ordered_and_reopen_keeps_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("birdsong.sqlite");
    let store = SqliteStore::open(&path, opts()).await.unwrap();
    let at = utc(2026, 6, 1, 12, 0);
    let ids = store
        .insert_many(&[
            det("A a", "a", 0.9, at),
            det("B b", "b", 0.8, at),
            det("C c", "c", 0.75, at),
        ])
        .await
        .unwrap();
    assert_eq!(ids.len(), 3);
    assert!(ids[0] < ids[1] && ids[1] < ids[2]);
    store.close().await;

    let store = SqliteStore::open(&path, opts()).await.unwrap(); // migrations already applied
    let all = store
        .list(&DetectionQuery {
            order: Order::Asc,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(all.iter().map(|r| r.id).collect::<Vec<_>>(), ids);
}

#[tokio::test]
async fn list_filters() {
    let (_dir, store) = open().await;
    let t0 = utc(2026, 6, 1, 12, 0);
    for i in 0..10 {
        let (sci, common) = if i % 2 == 0 {
            ("A a", "a")
        } else {
            ("B b", "b")
        };
        store
            .insert(&det(
                sci,
                common,
                0.7 + i as f32 * 0.02,
                t0 + TimeDelta::minutes(i),
            ))
            .await
            .unwrap();
    }
    let ids = |v: &[birdsong_store::DetectionRecord]| v.iter().map(|r| r.id).collect::<Vec<_>>();

    let desc = store.list(&DetectionQuery::default()).await.unwrap();
    assert_eq!(
        ids(&desc),
        (1..=10).rev().collect::<Vec<_>>(),
        "default: newest first"
    );

    let q = DetectionQuery {
        species: Some("A a".into()),
        order: Order::Asc,
        ..Default::default()
    };
    assert_eq!(ids(&store.list(&q).await.unwrap()), [1, 3, 5, 7, 9]);

    let q = DetectionQuery {
        since: Some(t0 + TimeDelta::minutes(3)),
        until: Some(t0 + TimeDelta::minutes(6)),
        order: Order::Asc,
        ..Default::default()
    };
    assert_eq!(
        ids(&store.list(&q).await.unwrap()),
        [4, 5, 6],
        "since inclusive, until exclusive"
    );

    let q = DetectionQuery {
        min_confidence: Some(0.85),
        order: Order::Asc,
        ..Default::default()
    };
    assert_eq!(ids(&store.list(&q).await.unwrap()), [9, 10]);

    let q = DetectionQuery {
        after_id: Some(7),
        order: Order::Asc,
        limit: Some(2),
        ..Default::default()
    };
    assert_eq!(ids(&store.list(&q).await.unwrap()), [8, 9]);

    let q = DetectionQuery {
        before_id: Some(4),
        ..Default::default()
    };
    assert_eq!(ids(&store.list(&q).await.unwrap()), [3, 2, 1]);

    let q = DetectionQuery {
        limit: Some(0),
        ..Default::default()
    };
    assert_eq!(
        store.list(&q).await.unwrap().len(),
        1,
        "limit clamped to at least 1"
    );
    assert_eq!(
        DetectionQuery {
            limit: Some(50_000),
            ..Default::default()
        }
        .effective_limit(),
        1000
    );
}

#[tokio::test]
async fn cursor_pagination_is_stable_under_concurrent_inserts() {
    let (_dir, store) = open().await;
    let store = Arc::new(store);
    const TOTAL: usize = 300;

    let writer = {
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            let t0 = utc(2026, 6, 1, 12, 0);
            for i in 0..TOTAL / 3 {
                let at = t0 + TimeDelta::seconds(3 * i as i64);
                store
                    .insert_many(&[
                        det("A a", "a", 0.9, at),
                        det("B b", "b", 0.8, at),
                        det("C c", "c", 0.7, at),
                    ])
                    .await
                    .unwrap();
                tokio::task::yield_now().await;
            }
        })
    };

    let mut seen: Vec<i64> = Vec::new();
    let mut cursor = 0;
    loop {
        let writer_done = writer.is_finished();
        let page = store
            .list(&DetectionQuery {
                after_id: Some(cursor),
                order: Order::Asc,
                limit: Some(17),
                ..Default::default()
            })
            .await
            .unwrap();
        if page.is_empty() {
            if writer_done {
                break;
            }
            tokio::task::yield_now().await;
            continue;
        }
        for r in &page {
            assert!(
                r.id > cursor,
                "ids must strictly increase: {} after {cursor}",
                r.id
            );
            cursor = r.id;
            seen.push(r.id);
        }
    }
    writer.await.unwrap();
    assert_eq!(seen.len(), TOTAL, "every row delivered exactly once");
    assert_eq!(seen.iter().collect::<HashSet<_>>().len(), TOTAL);
}

#[tokio::test]
async fn daily_stats_across_dst_changes() {
    let (_dir, store) = open().await;
    // America/New_York springs forward 2026-03-08 02:00 EST → 03:00 EDT.
    store
        .insert(&det("A a", "Alpha", 0.9, utc(2026, 3, 8, 4, 30)))
        .await
        .unwrap(); // 03-07 23:30 EST
    store
        .insert(&det("A a", "Alpha", 0.9, utc(2026, 3, 8, 6, 30)))
        .await
        .unwrap(); // 01:30 EST
    store
        .insert(&det("A a", "Alpha", 0.9, utc(2026, 3, 8, 7, 30)))
        .await
        .unwrap(); // 03:30 EDT
    store
        .insert(&det("B b", "Beta", 0.9, utc(2026, 3, 8, 7, 45)))
        .await
        .unwrap(); // 03:45 EDT
                   // Falls back 2026-11-01 02:00 EDT → 01:00 EST: 01:30 happens twice.
    store
        .insert(&det("C c", "Gamma", 0.9, utc(2026, 11, 1, 5, 30)))
        .await
        .unwrap(); // 01:30 EDT
    store
        .insert(&det("C c", "Gamma", 0.9, utc(2026, 11, 1, 6, 30)))
        .await
        .unwrap(); // 01:30 EST

    let spring = store
        .stats_daily(NaiveDate::from_ymd_opt(2026, 3, 8).unwrap())
        .await
        .unwrap();
    assert_eq!(spring.species.len(), 2);
    assert_eq!(spring.species[0].common_name, "Alpha");
    assert_eq!(spring.species[0].total, 2);
    assert_eq!(spring.species[0].by_hour[1], 1);
    assert_eq!(
        spring.species[0].by_hour[2], 0,
        "02:00-03:00 does not exist that day"
    );
    assert_eq!(spring.species[0].by_hour[3], 1);
    assert_eq!(spring.species[1].common_name, "Beta");
    assert_eq!(spring.species[1].by_hour[3], 1);

    let eve = store
        .stats_daily(NaiveDate::from_ymd_opt(2026, 3, 7).unwrap())
        .await
        .unwrap();
    assert_eq!(eve.species.len(), 1);
    assert_eq!(
        eve.species[0].by_hour[23], 1,
        "04:30 UTC belongs to the previous local day"
    );

    let fall = store
        .stats_daily(NaiveDate::from_ymd_opt(2026, 11, 1).unwrap())
        .await
        .unwrap();
    assert_eq!(fall.species[0].total, 2);
    assert_eq!(fall.species[0].by_hour[1], 2, "both 01:30s land in hour 1");

    let empty = store
        .stats_daily(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .await
        .unwrap();
    assert!(empty.species.is_empty());
}

#[tokio::test]
async fn species_summary_aggregates() {
    let (_dir, store) = open().await;
    let t0 = utc(2026, 6, 1, 12, 0);
    let a1 = store.insert(&det("A a", "Alpha", 0.80, t0)).await.unwrap();
    let a2 = store
        .insert(&det("A a", "Alpha", 0.95, t0 + TimeDelta::minutes(1)))
        .await
        .unwrap();
    let a3 = store
        .insert(&det("A a", "Alpha", 0.90, t0 + TimeDelta::minutes(2)))
        .await
        .unwrap();
    let b1 = store
        .insert(&det("B b", "Beta", 0.75, t0 + TimeDelta::minutes(3)))
        .await
        .unwrap();
    store
        .set_clip(&[a1], Some(&clip("a1.wav", 10)))
        .await
        .unwrap();
    store
        .set_clip(&[a3], Some(&clip("a3.wav", 10)))
        .await
        .unwrap();

    let all = store.species_summary(None).await.unwrap();
    assert_eq!(all.len(), 2);
    let a = &all[0];
    assert_eq!(
        (a.scientific_name.as_str(), a.common_name.as_str(), a.count),
        ("A a", "Alpha", 3)
    );
    assert_eq!(a.first_seen, t0);
    assert_eq!(a.last_seen, t0 + TimeDelta::minutes(2));
    assert!((a.max_confidence - 0.95).abs() < 1e-6);
    assert_eq!(a.best_detection_id, a2);
    assert_eq!(
        a.best_clip_detection_id,
        Some(a3),
        "best among those with clips"
    );
    assert_eq!(all[1].best_detection_id, b1);
    assert_eq!(all[1].best_clip_detection_id, None);

    let recent = store
        .species_summary(Some(t0 + TimeDelta::minutes(2)))
        .await
        .unwrap();
    assert_eq!(
        recent
            .iter()
            .map(|s| (s.common_name.as_str(), s.count))
            .collect::<Vec<_>>(),
        [("Alpha", 1), ("Beta", 1)]
    );
}

#[tokio::test]
async fn retention_plan_and_clear() {
    let (_dir, store) = open().await;
    let now = utc(2026, 9, 15, 12, 0);
    let mb = 1024 * 1024;
    let day = |d: i64| now - TimeDelta::days(d);

    // Same species, same local day 20 days ago: the 0.95 one is the day's best (exempt).
    let old_best = store
        .insert(&det("A a", "Alpha", 0.95, day(20)))
        .await
        .unwrap();
    let old_other = store
        .insert(&det("A a", "Alpha", 0.80, day(20) + TimeDelta::minutes(5)))
        .await
        .unwrap();
    // Two species sharing one clip (same window) 3 days ago.
    let shared = store
        .insert_many(&[
            det("B b", "Beta", 0.9, day(3)),
            det("C c", "Gamma", 0.8, day(3)),
        ])
        .await
        .unwrap();
    // Recent clips, one species per day so each is its day's best.
    let recent1 = store
        .insert(&det("A a", "Alpha", 0.72, day(2)))
        .await
        .unwrap();
    let recent2 = store
        .insert(&det("A a", "Alpha", 0.71, day(2) + TimeDelta::minutes(1)))
        .await
        .unwrap();

    store
        .set_clip(&[old_best], Some(&clip("old_best.wav", 2 * mb)))
        .await
        .unwrap();
    store
        .set_clip(&[old_other], Some(&clip("old_other.wav", 2 * mb)))
        .await
        .unwrap();
    store
        .set_clip(&shared, Some(&clip("shared.wav", 3 * mb)))
        .await
        .unwrap();
    store
        .set_clip(&[recent1], Some(&clip("recent1.wav", 3 * mb)))
        .await
        .unwrap();
    store
        .set_clip(&[recent2], Some(&clip("recent2.wav", 3 * mb)))
        .await
        .unwrap();

    assert_eq!(
        store.total_clip_bytes().await.unwrap(),
        13 * mb,
        "shared clip counted once"
    );
    assert_eq!(store.stored_clips().await.unwrap().len(), 5);
    let exempt = store.exempt_clip_paths(1).await.unwrap();
    assert!(
        exempt.contains("old_best.wav")
            && exempt.contains("shared.wav")
            && exempt.contains("recent1.wav")
    );
    assert!(!exempt.contains("old_other.wav") && !exempt.contains("recent2.wav"));

    let policy = RetentionConfig {
        clip_max_age_days: 14,
        clip_max_total_mb: 8,
        keep_best_per_species_per_day: 1,
        ..RetentionConfig::default()
    };
    let plan = store.clips_to_purge(&policy, now).await.unwrap();
    let got: Vec<_> = plan
        .iter()
        .map(|c| (c.clip_path.as_str(), c.reason))
        .collect();
    // Age removes old_other (2 MB) → 11 MB left; only recent2 is non-exempt → 8 MB.
    assert_eq!(
        got,
        [
            ("old_other.wav", PurgeReason::Age),
            ("recent2.wav", PurgeReason::Size)
        ]
    );

    for c in &plan {
        assert_eq!(store.clear_clip(&c.clip_path).await.unwrap(), 1);
    }
    assert_eq!(store.total_clip_bytes().await.unwrap(), 8 * mb);
    assert!(
        store.clips_to_purge(&policy, now).await.unwrap().is_empty(),
        "plan is idempotent"
    );
    let row = store
        .get(old_other)
        .await
        .unwrap()
        .expect("rows survive clip purge");
    assert_eq!(
        (row.clip_path, row.clip_bytes, row.spectrogram_path),
        (None, None, None)
    );
    assert_eq!(
        store.clear_clip("shared.wav").await.unwrap(),
        2,
        "every row using the clip"
    );

    assert_eq!(store.delete_detections_before(day(10)).await.unwrap(), 2);
    assert!(store.get(old_best).await.unwrap().is_none());
    let next = store.insert(&det("D d", "Delta", 0.9, now)).await.unwrap();
    assert!(next > recent2, "ids are never reused after deletes");
}
