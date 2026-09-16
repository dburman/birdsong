use std::collections::HashMap;
use std::path::Path;

use crate::ModelError;

/// First line of Perch v2's full label file; not a class.
pub const PERCH_LABELS_HEADER: &str = "inat2024_fsd50k";

/// Perch v2 sound-event classes (FSD50K) that the privacy filter treats as people: voices, as
/// BirdNET's `Human vocal`, and body and activity sounds, as its `Human non-vocal`.
/// `Speech_synthesizer` is deliberately not included.
pub const PERCH_HUMAN_CLASSES: &[&str] = &[
    "Breathing",
    "Burping_and_eructation",
    "Chatter",
    "Cheering",
    "Chewing_and_mastication",
    "Child_speech_and_kid_speaking",
    "Chuckle_and_chortle",
    "Clapping",
    "Conversation",
    "Cough",
    "Crowd",
    "Crying_and_sobbing",
    "Fart",
    "Female_singing",
    "Female_speech_and_woman_speaking",
    "Finger_snapping",
    "Gasp",
    "Giggle",
    "Hands",
    "Human_group_actions",
    "Human_voice",
    "Laughter",
    "Male_singing",
    "Male_speech_and_man_speaking",
    "Respiratory_sounds",
    "Run",
    "Screaming",
    "Shout",
    "Sigh",
    "Singing",
    "Sneeze",
    "Speech",
    "Walk_and_footsteps",
    "Whispering",
    "Yell",
];

/// One classifier output class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    /// The raw line: `Scientific name_Common name` for BirdNET, the class name for Perch.
    pub raw: String,
    pub scientific: String,
    pub common: String,
    /// Counts as a person for the privacy filter.
    pub human: bool,
}

impl Label {
    /// BirdNET-Pi's privacy rule: any BirdNET label containing `Human` (e.g.
    /// `Human vocal_Human vocal`); for Perch, one of [`PERCH_HUMAN_CLASSES`].
    pub fn is_human(&self) -> bool {
        self.human
    }
}

/// The classifier's label list; index = class index = line number.
#[derive(Clone, Debug)]
pub struct Labels {
    entries: Vec<Label>,
    by_scientific: HashMap<String, usize>,
}

impl Labels {
    /// Load a BirdNET labels file (one `Scientific name_Common name` per line).
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(path).map_err(|e| ModelError::io(path, e))?;
        Self::parse(&text).map_err(|message| ModelError::Labels {
            path: path.to_path_buf(),
            message,
        })
    }

    /// Parse label text. Blank lines are not allowed (they would shift every index).
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let raw = line.trim_end_matches('\r');
            if raw.trim().is_empty() {
                return Err(format!("line {} is empty", i + 1));
            }
            let (scientific, common) = match raw.split_once('_') {
                Some((s, c)) => (s.to_string(), c.to_string()),
                None => (raw.to_string(), raw.to_string()),
            };
            entries.push(Label {
                raw: raw.to_string(),
                scientific,
                common,
                human: raw.contains("Human"),
            });
        }
        Self::from_entries(entries)
    }

    /// Load a Perch v2 labels file, taking common names from `common_names` (a BirdNET labels
    /// file) where the scientific name matches.
    pub fn load_perch(path: &Path, common_names: Option<&Labels>) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(path).map_err(|e| ModelError::io(path, e))?;
        Self::parse_perch(&text, common_names).map_err(|message| ModelError::Labels {
            path: path.to_path_buf(),
            message,
        })
    }

    /// Parse Perch v2 label text: one class per line, species as `Genus species`, sound events
    /// (FSD50K) as `Words_with_underscores`. The full model's header line is skipped.
    pub fn parse_perch(text: &str, common_names: Option<&Labels>) -> Result<Self, String> {
        let mut lines = text.lines().enumerate().peekable();
        if lines
            .peek()
            .is_some_and(|(_, l)| l.trim() == PERCH_LABELS_HEADER)
        {
            lines.next();
        }
        let mut entries = Vec::new();
        for (i, line) in lines {
            let raw = line.trim();
            if raw.is_empty() {
                return Err(format!("line {} is empty", i + 1));
            }
            let common = if raw.contains(' ') {
                common_names
                    .and_then(|c| c.index_of_scientific(raw).and_then(|j| c.get(j)))
                    .map_or_else(|| raw.to_string(), |l| l.common.clone())
            } else {
                raw.replace('_', " ")
            };
            entries.push(Label {
                raw: raw.to_string(),
                scientific: raw.to_string(),
                common,
                human: PERCH_HUMAN_CLASSES.contains(&raw),
            });
        }
        Self::from_entries(entries)
    }

    fn from_entries(entries: Vec<Label>) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("no labels".into());
        }
        let mut by_scientific = HashMap::new();
        for (i, l) in entries.iter().enumerate() {
            by_scientific.entry(l.scientific.clone()).or_insert(i);
        }
        Ok(Self {
            entries,
            by_scientific,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&Label> {
        self.entries.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Label> {
        self.entries.iter()
    }

    /// Class index for a scientific name (exact match).
    pub fn index_of_scientific(&self, scientific: &str) -> Option<usize> {
        self.by_scientific.get(scientific).copied()
    }

    /// Resolve a list of scientific names to indices, failing on the first unknown one.
    pub fn indices_for(&self, names: &[String], context: &str) -> Result<Vec<usize>, ModelError> {
        names
            .iter()
            .map(|n| {
                self.index_of_scientific(n)
                    .ok_or_else(|| ModelError::UnknownSpecies {
                        name: n.clone(),
                        context: context.to_string(),
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scientific_and_common() {
        let l = Labels::parse(
            "Poecile atricapillus_Black-capped Chickadee\nHuman vocal_Human vocal\nNoise_Noise\n",
        )
        .unwrap();
        assert_eq!(l.len(), 3);
        assert_eq!(l.get(0).unwrap().scientific, "Poecile atricapillus");
        assert_eq!(l.get(0).unwrap().common, "Black-capped Chickadee");
        assert!(!l.get(0).unwrap().is_human());
        assert!(l.get(1).unwrap().is_human());
        assert_eq!(l.index_of_scientific("Noise"), Some(2));
        assert_eq!(l.index_of_scientific("Nope"), None);
        assert!(l.indices_for(&["Nope".into()], "test").is_err());
    }

    #[test]
    fn parses_perch_labels() {
        let common = Labels::parse("Poecile atricapillus_Black-capped Chickadee\n").unwrap();
        let text = "inat2024_fsd50k\nPoecile atricapillus\nAlces alces\nMale_speech_and_man_speaking\nSpeech_synthesizer\nCar_passing_by\n";
        let l = Labels::parse_perch(text, Some(&common)).unwrap();
        assert_eq!(l.len(), 5, "header skipped");
        assert_eq!(l.get(0).unwrap().common, "Black-capped Chickadee");
        assert_eq!(
            l.get(1).unwrap().common,
            "Alces alces",
            "no common name known"
        );
        assert_eq!(l.get(2).unwrap().scientific, "Male_speech_and_man_speaking");
        assert_eq!(l.get(2).unwrap().common, "Male speech and man speaking");
        let human: Vec<_> = l.iter().map(Label::is_human).collect();
        assert_eq!(human, [false, false, true, false, false]);
        assert_eq!(l.index_of_scientific("Car_passing_by"), Some(4));
        assert_eq!(
            Labels::parse_perch("Alces alces\n", None).unwrap().len(),
            1,
            "regional files have no header"
        );
        assert!(Labels::parse_perch("A b\n\nC d\n", None).is_err());
    }

    #[test]
    fn perch_human_classes_are_real_perch_labels() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/perch/perch_v2_labels.txt");
        let Ok(l) = Labels::load_perch(&path, None) else {
            eprintln!("skipping: {} not present", path.display());
            return;
        };
        assert_eq!(l.len(), 14_795);
        for name in PERCH_HUMAN_CLASSES {
            assert!(l.index_of_scientific(name).is_some(), "{name}");
        }
    }

    #[test]
    fn rejects_blank_lines() {
        assert!(Labels::parse("A_a\n\nB_b\n").is_err());
        assert!(Labels::parse("").is_err());
    }

    /// The real BirdNET V2.4 file, when present.
    #[test]
    fn real_labels_file() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/labels/en_us.txt");
        let Ok(l) = Labels::load(&path) else {
            eprintln!("skipping: {} not present", path.display());
            return;
        };
        assert_eq!(l.len(), 6522);
        assert_eq!(l.get(4771).unwrap().common, "Black-capped Chickadee");
        assert_eq!(l.index_of_scientific("Human vocal"), Some(2819)); // 0-based (line 2820)
        assert_eq!(l.iter().filter(|x| x.is_human()).count(), 3);
    }
}
