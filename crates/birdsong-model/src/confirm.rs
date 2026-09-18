//! Optional detection-quality filters, both off by default:
//!
//! * [`DynamicThresholds`] lowers a species' confidence threshold for a while after it has been
//!   heard clearly, so quieter calls of a bird that is definitely present still count. Modelled on
//!   BirdNET-Go: each detection above the trigger steps the threshold to 75 %, 50 % then 25 % of
//!   the configured one, never below a floor, and the effect expires.
//! * [`Confirmer`] holds a species back until it has been detected several times within a window,
//!   which removes one-off false positives. The hits that led to a confirmation are then stored
//!   too, so nothing is lost.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, TimeDelta, Utc};

use crate::ChunkAnalysis;

/// Threshold multiplier after 1, 2 and 3 or more clear detections (BirdNET-Go's levels).
const LEVEL_MULTIPLIERS: [f32; 3] = [0.75, 0.50, 0.25];

#[derive(Clone, Copy, Debug)]
struct SpeciesState {
    /// Clear detections seen while the effect was valid, capped at the last level.
    level: usize,
    expires_at: DateTime<Utc>,
}

/// Per-species thresholds that fall after clear detections and expire.
#[derive(Clone, Debug)]
pub struct DynamicThresholds {
    base: f32,
    trigger: f32,
    floor: f32,
    valid_for: TimeDelta,
    species: HashMap<String, SpeciesState>,
}

impl DynamicThresholds {
    /// `base` is `detection.min_confidence`; `trigger` the confidence that counts as clear;
    /// `floor` the lowest threshold allowed; `valid_for` how long the effect lasts.
    pub fn new(base: f32, trigger: f32, floor: f32, valid_for: TimeDelta) -> Self {
        Self {
            base,
            trigger,
            floor: floor.min(base),
            valid_for,
            species: HashMap::new(),
        }
    }

    /// The threshold for a species at `at`, after dropping any expired effect.
    pub fn threshold_for(&self, species: &str, at: DateTime<Utc>) -> f32 {
        match self.species.get(species) {
            Some(state) if at < state.expires_at => {
                let multiplier =
                    LEVEL_MULTIPLIERS[(state.level - 1).min(LEVEL_MULTIPLIERS.len() - 1)];
                (self.base * multiplier).max(self.floor)
            }
            _ => self.base,
        }
    }

    /// Record a stored detection. Only confidences above the trigger lower the threshold.
    pub fn observe(&mut self, species: &str, confidence: f32, at: DateTime<Utc>) {
        if confidence <= self.trigger {
            return;
        }
        let expires_at = at + self.valid_for;
        match self.species.get_mut(species) {
            Some(state) if at < state.expires_at => {
                state.level = (state.level + 1).min(LEVEL_MULTIPLIERS.len());
                state.expires_at = expires_at;
            }
            _ => {
                self.species.insert(
                    species.to_string(),
                    SpeciesState {
                        level: 1,
                        expires_at,
                    },
                );
            }
        }
    }

    /// Species whose threshold is currently lowered.
    pub fn active(&self, at: DateTime<Utc>) -> usize {
        self.species.values().filter(|s| at < s.expires_at).count()
    }
}

#[derive(Debug)]
struct Held {
    analysis: ChunkAnalysis,
    /// Indexes into `analysis.detections` still waiting for confirmation.
    undecided: Vec<usize>,
}

/// Holds detections until their species has been heard `min_detections` times within `window`.
///
/// Analyses are released in order. One waiting analysis delays the ones behind it, by at most
/// `window`; with the default settings the feature is off and nothing is delayed.
#[derive(Debug)]
pub struct Confirmer {
    min_detections: usize,
    window: TimeDelta,
    held: Vec<Held>,
    /// Recent hit times per species, oldest first.
    hits: HashMap<String, Vec<DateTime<Utc>>>,
    /// Species already confirmed, with the time their confirmation lapses.
    confirmed_until: HashMap<String, DateTime<Utc>>,
    /// Species stored immediately, without confirmation.
    exempt: HashSet<String>,
    discarded: u64,
}

impl Confirmer {
    pub fn new(min_detections: usize, window: TimeDelta) -> Self {
        Self {
            min_detections: min_detections.max(1),
            window,
            held: Vec::new(),
            hits: HashMap::new(),
            confirmed_until: HashMap::new(),
            exempt: HashSet::new(),
            discarded: 0,
        }
    }

    /// Species (scientific names) that never wait for confirmation.
    pub fn with_exempt(mut self, species: impl IntoIterator<Item = String>) -> Self {
        self.exempt.extend(species);
        self
    }

    /// Detections dropped because their species was never confirmed.
    pub fn discarded(&self) -> u64 {
        self.discarded
    }

    /// Feed one analysis; returns the analyses that are now settled, in order.
    pub fn push(&mut self, analysis: ChunkAnalysis) -> Vec<ChunkAnalysis> {
        let now = analysis.start_at;
        let mut undecided = Vec::new();
        for (index, detection) in analysis.detections.iter().enumerate() {
            let species = detection.scientific_name.clone();
            if self.exempt.contains(&species) {
                continue; // stored immediately
            }
            if self
                .confirmed_until
                .get(&species)
                .is_some_and(|until| now <= *until)
            {
                self.confirmed_until.insert(species, now + self.window);
                continue; // already confirmed: store immediately
            }
            let hits = self.hits.entry(species.clone()).or_default();
            hits.retain(|t| now - *t <= self.window);
            hits.push(now);
            if hits.len() >= self.min_detections {
                self.hits.remove(&species);
                self.confirmed_until
                    .insert(species.clone(), now + self.window);
                self.confirm_held(&species);
            } else {
                undecided.push(index);
            }
        }
        self.held.push(Held {
            analysis,
            undecided,
        });
        self.release(now)
    }

    /// End of stream: release what is settled and drop everything still unconfirmed.
    pub fn flush(&mut self) -> Vec<ChunkAnalysis> {
        let mut out = Vec::new();
        for mut held in std::mem::take(&mut self.held) {
            self.discarded += held.undecided.len() as u64;
            remove_indexes(&mut held.analysis.detections, &held.undecided);
            out.push(held.analysis);
        }
        self.hits.clear();
        self.confirmed_until.clear();
        out
    }

    /// Mark a species' waiting detections as confirmed.
    fn confirm_held(&mut self, species: &str) {
        for held in &mut self.held {
            held.undecided
                .retain(|&i| held.analysis.detections[i].scientific_name != species);
        }
    }

    /// Release settled analyses from the front; drop detections whose window has passed.
    fn release(&mut self, now: DateTime<Utc>) -> Vec<ChunkAnalysis> {
        let mut out = Vec::new();
        while let Some(held) = self.held.first_mut() {
            if !held.undecided.is_empty() {
                let expired = now - held.analysis.start_at > self.window;
                if !expired {
                    break;
                }
                self.discarded += held.undecided.len() as u64;
                let undecided = std::mem::take(&mut held.undecided);
                remove_indexes(&mut held.analysis.detections, &undecided);
            }
            out.push(self.held.remove(0).analysis);
        }
        out
    }
}

fn remove_indexes<T>(items: &mut Vec<T>, remove: &[usize]) {
    let mut index = 0;
    items.retain(|_| {
        let keep = !remove.contains(&index);
        index += 1;
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use birdsong_core::Detection;
    use chrono::TimeZone;

    fn t(secs: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 15, 6, 0, 0).unwrap() + TimeDelta::milliseconds(secs * 1000)
    }

    fn analysis(at: DateTime<Utc>, species: &[(&str, f32)]) -> ChunkAnalysis {
        ChunkAnalysis {
            start_at: at,
            source_id: "mic0".into(),
            detections: species
                .iter()
                .map(|(name, confidence)| Detection {
                    id: None,
                    detected_at: at,
                    scientific_name: (*name).into(),
                    common_name: (*name).into(),
                    confidence: *confidence,
                    source_id: "mic0".into(),
                    model_id: "test".into(),
                    kind: birdsong_core::DetectionKind::Animal,
                    clip_path: None,
                })
                .collect(),
            human_present: false,
            masked: false,
        }
    }

    fn names(analyses: &[ChunkAnalysis]) -> Vec<Vec<&str>> {
        analyses
            .iter()
            .map(|a| {
                a.detections
                    .iter()
                    .map(|d| d.scientific_name.as_str())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn thresholds_step_down_and_expire() {
        let mut dt = DynamicThresholds::new(0.7, 0.9, 0.2, TimeDelta::hours(24));
        assert_eq!(dt.threshold_for("A", t(0)), 0.7);

        dt.observe("A", 0.85, t(0));
        assert_eq!(
            dt.threshold_for("A", t(1)),
            0.7,
            "below the trigger changes nothing"
        );

        dt.observe("A", 0.95, t(1));
        assert!((dt.threshold_for("A", t(2)) - 0.525).abs() < 1e-6, "75 %");
        dt.observe("A", 0.95, t(2));
        assert!((dt.threshold_for("A", t(3)) - 0.35).abs() < 1e-6, "50 %");
        dt.observe("A", 0.95, t(3));
        assert!(
            (dt.threshold_for("A", t(4)) - 0.2).abs() < 1e-6,
            "25 % would be 0.175, floored at 0.2"
        );
        dt.observe("A", 0.95, t(4));
        assert!(
            (dt.threshold_for("A", t(5)) - 0.2).abs() < 1e-6,
            "never below the floor"
        );

        assert_eq!(
            dt.threshold_for("B", t(5)),
            0.7,
            "other species are unaffected"
        );
        assert_eq!(dt.active(t(5)), 1);
        let later = t(4) + TimeDelta::hours(25);
        assert_eq!(dt.threshold_for("A", later), 0.7, "expired");
        assert_eq!(dt.active(later), 0);
    }

    #[test]
    fn floor_never_exceeds_the_base_threshold() {
        let mut dt = DynamicThresholds::new(0.3, 0.9, 0.5, TimeDelta::hours(1));
        dt.observe("A", 0.95, t(0));
        assert!(dt.threshold_for("A", t(1)) <= 0.3);
    }

    #[test]
    fn single_hits_are_dropped_and_repeats_are_kept() {
        let mut c = Confirmer::new(2, TimeDelta::seconds(15));
        assert!(
            c.push(analysis(t(0), &[("A", 0.8)])).is_empty(),
            "held, waiting for a second hit"
        );

        // A second hit within the window confirms it and releases both windows in order.
        let released = c.push(analysis(t(3), &[("A", 0.7)]));
        assert_eq!(names(&released), [vec!["A"], vec!["A"]]);

        // While confirmed, further hits pass straight through.
        assert_eq!(names(&c.push(analysis(t(6), &[("A", 0.6)]))), [vec!["A"]]);

        // A different species alone is dropped once its window passes.
        assert!(c.push(analysis(t(9), &[("B", 0.9)])).is_empty());
        let released = c.push(analysis(t(30), &[("C", 0.9)]));
        assert_eq!(
            names(&released),
            [Vec::<&str>::new()],
            "B expired unconfirmed, its window is empty"
        );
        assert_eq!(c.discarded(), 1);
        assert_eq!(
            names(&c.flush()),
            [Vec::<&str>::new()],
            "C never confirmed either"
        );
        assert_eq!(c.discarded(), 2);
    }

    #[test]
    fn a_waiting_species_does_not_lose_the_others_in_its_window() {
        let mut c = Confirmer::new(2, TimeDelta::seconds(15));
        c.push(analysis(t(0), &[("A", 0.9)])); // held
        assert!(
            c.push(analysis(t(3), &[("B", 0.9)])).is_empty(),
            "B waits behind A"
        );
        // B confirms: its own window is still blocked by A's, which confirms next.
        let released = c.push(analysis(t(5), &[("B", 0.8), ("A", 0.8)]));
        assert_eq!(
            names(&released),
            [vec!["A"], vec!["B"], vec!["B", "A"]],
            "order preserved"
        );
        assert_eq!(c.discarded(), 0);
    }

    #[test]
    fn three_hits_required() {
        let mut c = Confirmer::new(3, TimeDelta::seconds(15));
        c.push(analysis(t(0), &[("A", 0.9)]));
        c.push(analysis(t(2), &[("A", 0.9)]));
        let released = c.push(analysis(t(4), &[("A", 0.9)]));
        assert_eq!(names(&released), [vec!["A"], vec!["A"], vec!["A"]]);
    }

    #[test]
    fn exempt_species_skip_confirmation() {
        let mut c =
            Confirmer::new(2, TimeDelta::seconds(15)).with_exempt(["Strix varia".to_string()]);
        let released = c.push(analysis(t(0), &[("Strix varia", 0.8), ("A", 0.9)]));
        assert!(released.is_empty(), "the window waits for A");
        let released = c.push(analysis(t(20), &[("Strix varia", 0.7)]));
        assert_eq!(
            names(&released),
            [vec!["Strix varia"], vec!["Strix varia"]],
            "the owl was kept although it called once per window; A was not confirmed"
        );
        assert_eq!(c.discarded(), 1);
    }

    #[test]
    fn one_detection_required_is_a_pass_through() {
        let mut c = Confirmer::new(1, TimeDelta::seconds(15));
        assert_eq!(names(&c.push(analysis(t(0), &[("A", 0.5)]))), [vec!["A"]]);
        assert_eq!(c.discarded(), 0);
    }
}
