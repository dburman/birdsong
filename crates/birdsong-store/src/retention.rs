use std::collections::HashSet;

use birdsong_core::config::RetentionConfig;
use chrono::{DateTime, TimeDelta, Utc};

use crate::{ClipToPurge, PurgeReason, StoredClip};

/// Decide which clips to delete. Pure: no I/O.
///
/// 1. Clips in `exempt` (best per species per day) are never selected.
/// 2. Clips older than `clip_max_age_days` (when non-zero) are selected, reason `Age`.
/// 3. While the remaining total (exempt clips included) exceeds `clip_max_total_mb` (when
///    non-zero), the oldest non-exempt clips are selected, reason `Size`. If exempt clips alone
///    exceed the cap, the cap is not met.
///
/// Result is oldest first.
pub fn plan_purge(
    clips: &[StoredClip],
    exempt: &HashSet<String>,
    policy: &RetentionConfig,
    now: DateTime<Utc>,
) -> Vec<ClipToPurge> {
    let mut sorted: Vec<&StoredClip> = clips.iter().collect();
    sorted.sort_by(|a, b| {
        a.detected_at
            .cmp(&b.detected_at)
            .then_with(|| a.clip_path.cmp(&b.clip_path))
    });

    let cutoff = (policy.clip_max_age_days > 0)
        .then(|| now - TimeDelta::days(policy.clip_max_age_days as i64));
    let purge = |c: &StoredClip, reason| ClipToPurge {
        clip_path: c.clip_path.clone(),
        spectrogram_path: c.spectrogram_path.clone(),
        bytes: c.bytes,
        detected_at: c.detected_at,
        reason,
    };

    let mut out = Vec::new();
    let mut kept = Vec::new();
    for clip in sorted {
        let is_exempt = exempt.contains(&clip.clip_path);
        match cutoff {
            Some(cut) if clip.detected_at < cut && !is_exempt => {
                out.push(purge(clip, PurgeReason::Age))
            }
            _ => kept.push(clip),
        }
    }

    if policy.clip_max_total_mb > 0 {
        let cap = policy.clip_max_total_mb.saturating_mul(1024 * 1024);
        let mut total: u64 = kept.iter().map(|c| c.bytes).sum();
        for clip in kept {
            if total <= cap {
                break;
            }
            if exempt.contains(&clip.clip_path) {
                continue;
            }
            total -= clip.bytes;
            out.push(purge(clip, PurgeReason::Size));
        }
    }
    out.sort_by(|a, b| {
        a.detected_at
            .cmp(&b.detected_at)
            .then_with(|| a.clip_path.cmp(&b.clip_path))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    const MB: u64 = 1024 * 1024;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 15, 12, 0, 0).unwrap()
    }

    fn clip(name: &str, days_ago: i64, mb: u64) -> StoredClip {
        StoredClip {
            clip_path: name.into(),
            spectrogram_path: None,
            bytes: mb * MB,
            detected_at: now() - TimeDelta::days(days_ago),
        }
    }

    fn policy(age: u32, cap_mb: u64) -> RetentionConfig {
        RetentionConfig {
            clip_max_age_days: age,
            clip_max_total_mb: cap_mb,
            ..RetentionConfig::default()
        }
    }

    fn names(p: &[ClipToPurge]) -> Vec<(&str, PurgeReason)> {
        p.iter().map(|c| (c.clip_path.as_str(), c.reason)).collect()
    }

    #[test]
    fn age_rule_respects_exemptions() {
        let clips = [
            clip("old", 20, 1),
            clip("old_best", 30, 1),
            clip("new", 1, 1),
        ];
        let exempt = HashSet::from(["old_best".to_string()]);
        let p = plan_purge(&clips, &exempt, &policy(14, 0), now());
        assert_eq!(names(&p), [("old", PurgeReason::Age)]);
    }

    #[test]
    fn size_rule_deletes_oldest_non_exempt_first() {
        let clips = [
            clip("a", 5, 4),
            clip("b", 4, 4),
            clip("c", 3, 4),
            clip("d", 2, 4),
        ]; // 16 MB
        let exempt = HashSet::from(["a".to_string()]);
        let p = plan_purge(&clips, &exempt, &policy(0, 9), now());
        // 16 → drop b (12) → drop c (8 <= 9). a is exempt, d is newest.
        assert_eq!(
            names(&p),
            [("b", PurgeReason::Size), ("c", PurgeReason::Size)]
        );
    }

    #[test]
    fn age_then_size() {
        let clips = [clip("ancient", 40, 10), clip("x", 3, 6), clip("y", 2, 6)];
        let p = plan_purge(&clips, &HashSet::new(), &policy(14, 10), now());
        assert_eq!(
            names(&p),
            [("ancient", PurgeReason::Age), ("x", PurgeReason::Size)]
        );
    }

    #[test]
    fn exempt_clips_can_exceed_cap_and_zero_disables() {
        let clips = [clip("a", 1, 50), clip("b", 1, 50)];
        let exempt = HashSet::from(["a".to_string(), "b".to_string()]);
        assert!(plan_purge(&clips, &exempt, &policy(0, 10), now()).is_empty());
        assert!(plan_purge(&clips, &HashSet::new(), &policy(0, 0), now()).is_empty());
        assert!(plan_purge(&[], &HashSet::new(), &policy(14, 10), now()).is_empty());
    }
}
