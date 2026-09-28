//! Reviewed detections: what a person said about them, and the threshold that follows.

use serde::{Deserialize, Serialize};

/// A person's verdict on a detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Review {
    Correct,
    Wrong,
}

impl Review {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Wrong => "wrong",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "correct" => Some(Self::Correct),
            "wrong" => Some(Self::Wrong),
            _ => None,
        }
    }
}

/// Precision a suggested threshold must reach among reviewed detections above it.
pub const TARGET_PRECISION: f32 = 0.9;
/// Reviewed detections needed at or above a threshold before it is suggested.
pub const MIN_REVIEWS: usize = 5;

/// Reviews of one species.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReviewSummary {
    pub scientific_name: String,
    pub common_name: String,
    pub reviewed: u64,
    pub correct: u64,
    pub wrong: u64,
    /// `correct / reviewed`, to 3 decimals.
    pub precision: f64,
    /// The lowest confidence at which the reviewed detections of this species are at least
    /// [`TARGET_PRECISION`] correct (over at least [`MIN_REVIEWS`] of them), when some were wrong;
    /// a value for `detection.species_min_confidence`. `None` when every review was correct, or
    /// there are not yet enough reviews to say.
    pub suggested_min_confidence: Option<f64>,
}

/// See [`ReviewSummary::suggested_min_confidence`]. `reviews` are (confidence, correct).
pub fn suggest_threshold(reviews: &[(f32, bool)]) -> Option<f32> {
    if reviews.iter().all(|(_, correct)| *correct) {
        return None;
    }
    let mut sorted = reviews.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Try each reviewed confidence as the threshold, lowest first.
    for i in 0..sorted.len() {
        let above = &sorted[i..];
        if above.len() < MIN_REVIEWS {
            return None;
        }
        let correct = above.iter().filter(|(_, c)| *c).count();
        if correct as f32 >= TARGET_PRECISION * above.len() as f32 {
            // Round down to 2 decimals so the reviewed detection at the threshold still passes.
            return Some((sorted[i].0 * 100.0).floor() / 100.0);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_from_reviews() {
        let all_right = [(0.5, true), (0.6, true), (0.9, true)];
        assert_eq!(suggest_threshold(&all_right), None, "nothing to fix");

        // Wrong answers at low confidence: the threshold moves above them.
        let mut reviews = vec![(0.51, false), (0.53, false), (0.55, true), (0.58, false)];
        reviews.extend([
            (0.61, true),
            (0.66, true),
            (0.7, true),
            (0.8, true),
            (0.9, true),
        ]);
        assert_eq!(suggest_threshold(&reviews), Some(0.61));

        assert_eq!(
            suggest_threshold(&[(0.5, false), (0.6, true), (0.7, true)]),
            None,
            "too few reviews"
        );
        let mostly_wrong = [
            (0.5, false),
            (0.6, false),
            (0.7, false),
            (0.8, false),
            (0.9, true),
        ];
        assert_eq!(
            suggest_threshold(&mostly_wrong),
            None,
            "no threshold is good enough yet"
        );
        assert_eq!(Review::parse("wrong"), Some(Review::Wrong));
        assert_eq!(Review::Correct.as_str(), "correct");
    }
}
