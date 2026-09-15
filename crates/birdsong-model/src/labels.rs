use std::collections::HashMap;
use std::path::Path;

use crate::ModelError;

/// One classifier output class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    /// The raw line, `Scientific name_Common name`.
    pub raw: String,
    pub scientific: String,
    pub common: String,
}

impl Label {
    /// BirdNET-Pi's privacy rule: any label containing `Human` (e.g. `Human vocal_Human vocal`).
    pub fn is_human(&self) -> bool {
        self.raw.contains("Human")
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
        let mut by_scientific = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            let raw = line.trim_end_matches('\r');
            if raw.trim().is_empty() {
                return Err(format!("line {} is empty", i + 1));
            }
            let (scientific, common) = match raw.split_once('_') {
                Some((s, c)) => (s.to_string(), c.to_string()),
                None => (raw.to_string(), raw.to_string()),
            };
            by_scientific.entry(scientific.clone()).or_insert(i);
            entries.push(Label {
                raw: raw.to_string(),
                scientific,
                common,
            });
        }
        if entries.is_empty() {
            return Err("no labels".into());
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
