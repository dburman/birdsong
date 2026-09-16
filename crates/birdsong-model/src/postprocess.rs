//! From raw logits to detections, matching BirdNET-Pi (`BUILD_PLAN.md` §7, `docs/DECISIONS.md` #6, #8).

use birdsong_core::config::DetectionConfig;
use birdsong_core::Detection;
use chrono::{DateTime, Utc};

use crate::{Labels, SpeciesFilter};

/// BirdNET-Pi's sensitivity mapping: user value `0.5..=1.5` → sigmoid slope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmoidSensitivity {
    slope: f32,
}

impl SigmoidSensitivity {
    /// `slope = clamp(1 - (sensitivity - 1), 0.5, 1.5)`; the default 1.25 gives 0.75.
    pub fn from_user_value(sensitivity: f32) -> Self {
        Self {
            slope: (1.0 - (sensitivity - 1.0)).clamp(0.5, 1.5),
        }
    }

    pub fn slope(&self) -> f32 {
        self.slope
    }

    /// `1 / (1 + exp(-slope × logit))`.
    pub fn apply(&self, logit: f32) -> f32 {
        1.0 / (1.0 + (-self.slope * logit).exp())
    }
}

/// Post-processing knobs, derived from [`DetectionConfig`].
#[derive(Clone, Debug, PartialEq)]
pub struct PostprocessConfig {
    pub min_confidence: f32,
    pub sensitivity: SigmoidSensitivity,
    pub top_n_per_chunk: usize,
    pub privacy_filter: bool,
    /// BirdNET-Pi: `max(10, int(6000 × privacy_threshold / 100))` ranks are checked for `Human`.
    pub human_rank_cutoff: usize,
}

impl PostprocessConfig {
    pub fn from_detection_config(d: &DetectionConfig) -> Self {
        Self {
            min_confidence: d.min_confidence,
            sensitivity: SigmoidSensitivity::from_user_value(d.sensitivity),
            top_n_per_chunk: d.top_n_per_chunk,
            privacy_filter: d.privacy_filter,
            human_rank_cutoff: human_rank_cutoff(d.privacy_threshold),
        }
    }
}

/// BirdNET-Pi `filter_humans`: `max(10, int(6000 * priv_thresh / 100.0))`.
pub fn human_rank_cutoff(privacy_threshold_percent: f32) -> usize {
    ((6000.0 * privacy_threshold_percent / 100.0) as usize).max(10)
}

/// Where and when a chunk came from.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkContext {
    pub start_at: DateTime<Utc>,
    pub source_id: String,
    pub model_id: String,
}

/// Result of analysing one chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkAnalysis {
    pub start_at: DateTime<Utc>,
    pub source_id: String,
    /// Detections that passed every filter, best first. Empty when `masked`.
    pub detections: Vec<Detection>,
    /// A `Human*` class ranked within the privacy cutoff (before species filtering).
    pub human_present: bool,
    /// Blanked by the privacy rule (this chunk or a neighbour had a human). No clip must be saved.
    pub masked: bool,
}

/// Class indices sorted by logit, best first.
fn ranking(logits: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx
}

/// Top `n` classes with confidences, ignoring every filter (for the `analyze` CLI and debugging).
pub fn top_scores(logits: &[f32], sensitivity: SigmoidSensitivity, n: usize) -> Vec<(usize, f32)> {
    ranking(logits)
        .into_iter()
        .take(n)
        .map(|i| (i, sensitivity.apply(logits[i])))
        .collect()
}

/// Sigmoid, human check, species filter, top-N, confidence threshold — for one chunk.
///
/// `logits.len()` must equal `labels.len()`. The neighbour part of the privacy rule is applied
/// separately by [`NeighbourMask`]; here `masked` is set only for the chunk's own human presence.
pub fn analyze_chunk(
    logits: &[f32],
    labels: &Labels,
    filter: &SpeciesFilter,
    cfg: &PostprocessConfig,
    ctx: &ChunkContext,
) -> ChunkAnalysis {
    analyze_chunk_with(logits, labels, filter, cfg, ctx, &|_| cfg.min_confidence)
}

/// As [`analyze_chunk`], but with a per-class confidence threshold, so
/// [`crate::DynamicThresholds`] can lower the bar for species heard clearly a moment ago.
pub fn analyze_chunk_with(
    logits: &[f32],
    labels: &Labels,
    filter: &SpeciesFilter,
    cfg: &PostprocessConfig,
    ctx: &ChunkContext,
    threshold_for: &dyn Fn(usize) -> f32,
) -> ChunkAnalysis {
    debug_assert_eq!(logits.len(), labels.len());
    let ranked = ranking(logits);

    let human_present = cfg.privacy_filter
        && ranked
            .iter()
            .take(cfg.human_rank_cutoff)
            .any(|&i| labels.get(i).is_some_and(Label_is_human));

    let mut detections = Vec::new();
    if !human_present {
        for &i in ranked
            .iter()
            .filter(|&&i| filter.is_allowed(i))
            .take(cfg.top_n_per_chunk)
        {
            let confidence = cfg.sensitivity.apply(logits[i]);
            if confidence < threshold_for(i) {
                // Thresholds can differ per species, so a lower-ranked class may still pass.
                continue;
            }
            let label = &labels.get(i).expect("index within labels");
            detections.push(Detection {
                id: None,
                detected_at: ctx.start_at,
                scientific_name: label.scientific.clone(),
                common_name: label.common.clone(),
                confidence,
                source_id: ctx.source_id.clone(),
                model_id: ctx.model_id.clone(),
                clip_path: None,
            });
        }
    }

    ChunkAnalysis {
        start_at: ctx.start_at,
        source_id: ctx.source_id.clone(),
        detections,
        human_present,
        masked: human_present,
    }
}

#[allow(non_snake_case)]
fn Label_is_human(l: &crate::Label) -> bool {
    l.is_human()
}

/// BirdNET-Pi blanks the chunks *next to* a human chunk as well. On a live stream the next chunk
/// is not known yet, so this holds each analysis back by one chunk (3 s of latency) and releases
/// it once its successor has been seen. Call [`NeighbourMask::push_missing`] when a chunk was lost
/// (dropped under load): it may have contained speech, so both of its neighbours are blanked.
/// Call [`NeighbourMask::flush`] at end of stream and [`NeighbourMask::reset`] after a gap.
#[derive(Debug, Default)]
pub struct NeighbourMask {
    pending: Option<ChunkAnalysis>,
    /// Whether the chunk before `pending` had (or may have had) a human.
    before_pending_human: bool,
    /// Whether the most recent chunk seen, pushed or missing, had (or may have had) a human.
    last_human: bool,
}

impl NeighbourMask {
    pub fn new() -> Self {
        Self::default()
    }

    fn blank(mut analysis: ChunkAnalysis) -> ChunkAnalysis {
        analysis.masked = true;
        analysis.detections.clear();
        analysis
    }

    /// Feed the newest analysis; returns the previous one, masked if any neighbour had a human.
    pub fn push(&mut self, current: ChunkAnalysis) -> Option<ChunkAnalysis> {
        let blank_previous = self.before_pending_human || current.human_present;
        let released = self.pending.take().map(|prev| {
            if blank_previous {
                Self::blank(prev)
            } else {
                prev
            }
        });
        self.before_pending_human = self.last_human;
        self.last_human = current.human_present;
        self.pending = Some(current);
        released
    }

    /// A chunk between the previous and the next one was lost. Returns the held analysis, blanked.
    pub fn push_missing(&mut self) -> Option<ChunkAnalysis> {
        let released = self.pending.take().map(Self::blank);
        self.before_pending_human = false;
        self.last_human = true;
        released
    }

    /// Release the last held analysis (end of stream).
    pub fn flush(&mut self) -> Option<ChunkAnalysis> {
        let before_human = self.before_pending_human;
        let last = self
            .pending
            .take()
            .map(|a| if before_human { Self::blank(a) } else { a });
        self.before_pending_human = false;
        self.last_human = false;
        last
    }

    /// Drop state after a discontinuity, releasing the held analysis first.
    pub fn reset(&mut self) -> Option<ChunkAnalysis> {
        let out = self.flush();
        *self = Self::default();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn labels() -> Labels {
        Labels::parse("A_a\nB_b\nC_c\nD_d\nHuman vocal_Human vocal\n").unwrap()
    }

    fn cfg() -> PostprocessConfig {
        PostprocessConfig {
            min_confidence: 0.7,
            sensitivity: SigmoidSensitivity::from_user_value(1.0), // slope 1.0
            top_n_per_chunk: 2,
            privacy_filter: true,
            human_rank_cutoff: 2,
        }
    }

    fn ctx() -> ChunkContext {
        ChunkContext {
            start_at: Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap(),
            source_id: "mic0".into(),
            model_id: "test".into(),
        }
    }

    #[test]
    fn sigmoid_and_sensitivity() {
        let s = SigmoidSensitivity::from_user_value(1.25);
        assert!((s.slope() - 0.75).abs() < 1e-6);
        assert_eq!(SigmoidSensitivity::from_user_value(1.0).apply(0.0), 0.5);
        assert!(SigmoidSensitivity::from_user_value(0.5).slope() <= 1.5);
        assert!(SigmoidSensitivity::from_user_value(9.0).slope() >= 0.5);
        let one = SigmoidSensitivity::from_user_value(1.0);
        assert!(one.apply(2.0) > one.apply(1.0) && one.apply(1.0) > one.apply(-1.0));
        assert!((one.apply(50.0) - 1.0).abs() < 1e-6);
        // Higher user sensitivity → flatter slope → same logit gives a confidence closer to 0.5.
        assert!(SigmoidSensitivity::from_user_value(1.5).apply(2.0) < one.apply(2.0));
    }

    #[test]
    fn cutoff_matches_birdnet_pi() {
        assert_eq!(human_rank_cutoff(0.0), 10);
        assert_eq!(human_rank_cutoff(0.1), 10);
        assert_eq!(human_rank_cutoff(1.0), 60);
        assert_eq!(human_rank_cutoff(100.0), 6000);
    }

    #[test]
    fn selection_respects_filter_topn_and_threshold() {
        let l = labels();
        // logits: A=3 (0.95), B=2 (0.88), C=1 (0.73), D=0 (0.5), Human=-5
        let logits = [3.0, 2.0, 1.0, 0.0, -5.0];
        let all = SpeciesFilter::allow_all(5);
        let r = analyze_chunk(&logits, &l, &all, &cfg(), &ctx());
        assert!(!r.human_present && !r.masked);
        let names: Vec<_> = r
            .detections
            .iter()
            .map(|d| d.common_name.as_str())
            .collect();
        assert_eq!(names, ["a", "b"]); // top_n = 2
        assert!(r.detections[0].confidence > r.detections[1].confidence);
        assert_eq!(r.detections[0].source_id, "mic0");

        // Disallow A: top-2 of the allowed become B and C.
        let f = SpeciesFilter::allow_all(5).exclude(&[0]);
        let r = analyze_chunk(&logits, &l, &f, &cfg(), &ctx());
        let names: Vec<_> = r
            .detections
            .iter()
            .map(|d| d.common_name.as_str())
            .collect();
        assert_eq!(names, ["b", "c"]);

        // Raise threshold so only A passes.
        let mut c = cfg();
        c.min_confidence = 0.9;
        let r = analyze_chunk(&logits, &l, &all, &c, &ctx());
        assert_eq!(r.detections.len(), 1);
        assert_eq!(r.detections[0].scientific_name, "A");
    }

    #[test]
    fn human_in_top_ranks_masks_chunk() {
        let l = labels();
        let all = SpeciesFilter::allow_all(5);
        // Human ranks 2nd (cutoff 2) → masked, even though A would pass.
        let r = analyze_chunk(&[3.0, 0.0, 0.0, 0.0, 2.5], &l, &all, &cfg(), &ctx());
        assert!(r.human_present && r.masked && r.detections.is_empty());
        // Human ranks 3rd → not within cutoff.
        let r = analyze_chunk(&[3.0, 2.9, 0.0, 0.0, 2.5], &l, &all, &cfg(), &ctx());
        assert!(!r.human_present && r.detections.len() == 2);
        // Filter off → never masked.
        let mut c = cfg();
        c.privacy_filter = false;
        let r = analyze_chunk(&[3.0, 0.0, 0.0, 0.0, 2.5], &l, &all, &c, &ctx());
        assert!(!r.human_present && !r.masked);
    }

    #[test]
    fn top_scores_ignores_filters() {
        let s = SigmoidSensitivity::from_user_value(1.0);
        let t = top_scores(&[0.0, 5.0, -1.0], s, 2);
        assert_eq!(t[0].0, 1);
        assert_eq!(t[1].0, 0);
        assert!(t[0].1 > 0.99);
    }

    fn chunk(n: u32, human: bool) -> ChunkAnalysis {
        let start_at = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap()
            + chrono::Duration::seconds(3 * n as i64);
        let det = Detection {
            id: None,
            detected_at: start_at,
            scientific_name: "A".into(),
            common_name: "a".into(),
            confidence: 0.9,
            source_id: "mic0".into(),
            model_id: "test".into(),
            clip_path: None,
        };
        ChunkAnalysis {
            start_at,
            source_id: "mic0".into(),
            detections: if human { vec![] } else { vec![det] },
            human_present: human,
            masked: human,
        }
    }

    #[test]
    fn neighbour_mask_blanks_both_sides() {
        let mut m = NeighbourMask::new();
        // chunks: 0 bird, 1 bird, 2 HUMAN, 3 bird, 4 bird, 5 bird
        assert!(m.push(chunk(0, false)).is_none());
        let c0 = m.push(chunk(1, false)).unwrap(); // releases 0; neighbour 1 is clean
        assert!(!c0.masked && c0.detections.len() == 1);
        let c1 = m.push(chunk(2, true)).unwrap(); // releases 1; next is human → masked
        assert!(c1.masked && c1.detections.is_empty());
        let c2 = m.push(chunk(3, false)).unwrap(); // the human chunk itself
        assert!(c2.masked && c2.human_present);
        let c3 = m.push(chunk(4, false)).unwrap(); // previous was human → masked
        assert!(c3.masked && c3.detections.is_empty());
        let c4 = m.push(chunk(5, false)).unwrap(); // clean on both sides
        assert!(!c4.masked && c4.detections.len() == 1);
        let c5 = m.flush().unwrap();
        assert!(!c5.masked && c5.detections.len() == 1);
        assert!(m.flush().is_none());
    }

    #[test]
    fn neighbour_mask_flush_after_human() {
        let mut m = NeighbourMask::new();
        m.push(chunk(0, true));
        let c0 = m.push(chunk(1, false)).unwrap();
        assert!(c0.masked);
        let c1 = m.flush().unwrap();
        assert!(
            c1.masked,
            "chunk after a human must be masked even at end of stream"
        );
        // reset releases and clears
        m.push(chunk(2, false));
        let c2 = m.reset().unwrap();
        assert!(!c2.masked);
        assert!(m.push(chunk(3, false)).is_none());
    }

    #[test]
    fn missing_chunk_blanks_both_neighbours() {
        let mut m = NeighbourMask::new();
        assert!(m.push(chunk(0, false)).is_none());
        let c0 = m.push_missing().unwrap(); // chunk 1 was dropped
        assert!(
            c0.masked && c0.detections.is_empty(),
            "chunk before the gap is blanked"
        );
        assert!(m.push(chunk(2, false)).is_none());
        let c2 = m.push(chunk(3, false)).unwrap();
        assert!(
            c2.masked && c2.detections.is_empty(),
            "chunk after the gap is blanked"
        );
        let c3 = m.flush().unwrap();
        assert!(
            !c3.masked && c3.detections.len() == 1,
            "further chunks are unaffected"
        );
        // A missing chunk with nothing pending still blanks the next one.
        assert!(m.push_missing().is_none());
        m.push(chunk(5, false));
        assert!(m.flush().unwrap().masked);
    }
}
