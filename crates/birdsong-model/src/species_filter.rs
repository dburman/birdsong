use std::path::Path;

use crate::{Labels, MetaModel, ModelError};

/// Which class indices may be reported. Always sized to the label list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeciesFilter {
    allowed: Vec<bool>,
}

impl SpeciesFilter {
    pub fn allow_all(num_classes: usize) -> Self {
        Self {
            allowed: vec![true; num_classes],
        }
    }

    pub fn allow_none(num_classes: usize) -> Self {
        Self {
            allowed: vec![false; num_classes],
        }
    }

    /// Allowed = classes whose meta-model probability is at least `threshold`.
    pub fn from_meta_model(
        meta: &MetaModel,
        latitude: f64,
        longitude: f64,
        week: i32,
        threshold: f32,
    ) -> Result<Self, ModelError> {
        let probs = meta.predict(latitude, longitude, week)?;
        Ok(Self {
            allowed: probs.iter().map(|&p| p >= threshold).collect(),
        })
    }

    /// For a classifier whose labels differ from the location model's (Perch): `map[i]` is the
    /// location-model index of class `i`, found by scientific name. Mapped classes are allowed when
    /// their probability is at least `threshold`; unmapped species follow `allow_unmapped`; classes
    /// that are not species (sound events) are always allowed.
    pub fn from_mapped_scores(
        probs: &[f32],
        map: &[Option<usize>],
        threshold: f32,
        allow_unmapped: bool,
        is_species: impl Fn(usize) -> bool,
    ) -> Self {
        Self {
            allowed: map
                .iter()
                .enumerate()
                .map(|(i, m)| match m {
                    Some(j) => probs.get(*j).is_some_and(|&p| p >= threshold),
                    None if is_species(i) => allow_unmapped,
                    None => true,
                })
                .collect(),
        }
    }

    /// Allowed = species listed in a text file, one `Scientific name` or `Scientific name_Common name`
    /// per line (`#` comments and blank lines ignored).
    ///
    /// Names the label file does not know are skipped with a warning rather than rejected:
    /// species lists are often produced by newer tools whose taxonomy has moved on (for example
    /// `Astur cooperii` for what BirdNET V2.4 still calls `Accipiter cooperii`). A file that
    /// matches nothing at all is an error, since it would silence every detection.
    pub fn from_list_file(path: &Path, labels: &Labels) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(path).map_err(|e| ModelError::io(path, e))?;
        Self::from_list_text(&text, labels, &path.display().to_string())
    }

    pub fn from_list_text(text: &str, labels: &Labels, context: &str) -> Result<Self, ModelError> {
        let mut f = Self::allow_none(labels.len());
        let mut unknown = Vec::new();
        let mut listed = 0usize;
        for line in text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            listed += 1;
            let scientific = line.split_once('_').map_or(line, |(s, _)| s);
            match labels.index_of_scientific(scientific) {
                Some(i) => f.allowed[i] = true,
                None => unknown.push(scientific.to_string()),
            }
        }
        if !unknown.is_empty() {
            tracing::warn!(
                context,
                skipped = unknown.len(),
                names = ?unknown,
                "species list names not in the label file were skipped"
            );
        }
        if listed > 0 && f.num_allowed() == 0 {
            return Err(ModelError::Config(format!(
                "species list {context}: none of its {listed} names match the label file"
            )));
        }
        Ok(f)
    }

    /// Force-allow these class indices (config `include_species`).
    pub fn include(mut self, indices: &[usize]) -> Self {
        for &i in indices {
            if let Some(a) = self.allowed.get_mut(i) {
                *a = true;
            }
        }
        self
    }

    /// Force-deny these class indices (config `exclude_species`). Applied last, so it wins.
    pub fn exclude(mut self, indices: &[usize]) -> Self {
        for &i in indices {
            if let Some(a) = self.allowed.get_mut(i) {
                *a = false;
            }
        }
        self
    }

    pub fn is_allowed(&self, index: usize) -> bool {
        self.allowed.get(index).copied().unwrap_or(false)
    }

    pub fn num_allowed(&self) -> usize {
        self.allowed.iter().filter(|&&a| a).count()
    }

    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Labels {
        Labels::parse("A_a\nB_b\nC_c\nHuman vocal_Human vocal\n").unwrap()
    }

    #[test]
    fn mapped_scores() {
        // Classes: mapped A (score 0.5), mapped B (0.01), unmapped species, sound event.
        let map = [Some(1), Some(0), None, None];
        let probs = [0.01, 0.5];
        let species = |i: usize| i < 3;
        let f = SpeciesFilter::from_mapped_scores(&probs, &map, 0.03, true, species);
        let allowed: Vec<_> = (0..4).map(|i| f.is_allowed(i)).collect();
        assert_eq!(allowed, [true, false, true, true]);
        let f = SpeciesFilter::from_mapped_scores(&probs, &map, 0.03, false, species);
        let allowed: Vec<_> = (0..4).map(|i| f.is_allowed(i)).collect();
        assert_eq!(
            allowed,
            [true, false, false, true],
            "sound events are never blocked"
        );
    }

    #[test]
    fn list_include_exclude() {
        let l = labels();
        let f = SpeciesFilter::from_list_text("# comment\nA_a\n\nC\n", &l, "test").unwrap();
        assert!(f.is_allowed(0) && !f.is_allowed(1) && f.is_allowed(2));
        let f = f.include(&[1]).exclude(&[0, 99]);
        assert!(!f.is_allowed(0) && f.is_allowed(1) && f.is_allowed(2));
        assert_eq!(f.num_allowed(), 2);
        assert!(!f.is_allowed(99));
        // Unknown names are skipped, unless nothing matches at all.
        let f = SpeciesFilter::from_list_text("Zzz_zzz\nA_a", &l, "test").unwrap();
        assert_eq!(f.num_allowed(), 1);
        assert!(SpeciesFilter::from_list_text("Zzz_zzz", &l, "test").is_err());
        assert_eq!(
            SpeciesFilter::from_list_text("", &l, "test")
                .unwrap()
                .num_allowed(),
            0
        );
    }

    #[test]
    fn allow_all_then_exclude() {
        let f = SpeciesFilter::allow_all(4).exclude(&[3]);
        assert_eq!(f.num_allowed(), 3);
        assert!(!f.is_allowed(3));
    }
}
