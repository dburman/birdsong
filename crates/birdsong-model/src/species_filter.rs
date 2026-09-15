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

    /// Allowed = species listed in a text file, one `Scientific name` or `Scientific name_Common name`
    /// per line (`#` comments and blank lines ignored). Unknown names are an error.
    pub fn from_list_file(path: &Path, labels: &Labels) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(path).map_err(|e| ModelError::io(path, e))?;
        Self::from_list_text(&text, labels, &path.display().to_string())
    }

    pub fn from_list_text(text: &str, labels: &Labels, context: &str) -> Result<Self, ModelError> {
        let names: Vec<String> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.split_once('_').map_or(l, |(s, _)| s).to_string())
            .collect();
        let mut f = Self::allow_none(labels.len());
        for i in labels.indices_for(&names, context)? {
            f.allowed[i] = true;
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
    fn list_include_exclude() {
        let l = labels();
        let f = SpeciesFilter::from_list_text("# comment\nA_a\n\nC\n", &l, "test").unwrap();
        assert!(f.is_allowed(0) && !f.is_allowed(1) && f.is_allowed(2));
        let f = f.include(&[1]).exclude(&[0, 99]);
        assert!(!f.is_allowed(0) && f.is_allowed(1) && f.is_allowed(2));
        assert_eq!(f.num_allowed(), 2);
        assert!(!f.is_allowed(99));
        assert!(SpeciesFilter::from_list_text("Zzz_zzz", &l, "test").is_err());
    }

    #[test]
    fn allow_all_then_exclude() {
        let f = SpeciesFilter::allow_all(4).exclude(&[3]);
        assert_eq!(f.num_allowed(), 3);
        assert!(!f.is_allowed(3));
    }
}
