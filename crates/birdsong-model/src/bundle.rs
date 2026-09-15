use birdsong_core::Config;

use crate::{
    Classifier, Labels, MetaModel, ModelError, PostprocessConfig, SpeciesFilter, TractClassifier,
};

/// Where a bundle's species filter comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeciesFilterKind {
    /// Every species allowed (no location, or no meta model / species list configured).
    None,
    /// BirdNET's location/week model.
    LocationModel,
    /// A static species list file.
    SpeciesList,
}

impl SpeciesFilterKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LocationModel => "location-model",
            Self::SpeciesList => "species-list",
        }
    }
}

/// Everything the pipeline needs to turn audio into detections, loaded from [`Config`].
pub struct ModelBundle {
    pub classifier: Box<dyn Classifier>,
    pub labels: Labels,
    pub postprocess: PostprocessConfig,
    meta_model: Option<MetaModel>,
    static_list: Option<SpeciesFilter>,
    location: Option<(f64, f64)>,
    species_filter_threshold: f32,
    include: Vec<usize>,
    exclude: Vec<usize>,
}

impl ModelBundle {
    /// Load the classifier, labels, and (if configured) the meta model or species list.
    ///
    /// `model.threads` is currently unused: tract runs single-threaded and the pipeline decides
    /// how many inference workers to run.
    pub fn load(cfg: &Config) -> Result<Self, ModelError> {
        let classifier = TractClassifier::load(&cfg.model.classifier_path())?;
        let labels = Labels::load(&cfg.model.labels_path())?;
        if labels.len() != classifier.num_classes() {
            return Err(ModelError::Config(format!(
                "labels file has {} entries but the classifier has {} outputs",
                labels.len(),
                classifier.num_classes()
            )));
        }
        let include =
            labels.indices_for(&cfg.detection.include_species, "detection.include_species")?;
        let exclude =
            labels.indices_for(&cfg.detection.exclude_species, "detection.exclude_species")?;

        let static_list = match &cfg.model.species_list {
            Some(path) => Some(SpeciesFilter::from_list_file(path, &labels)?),
            None => None,
        };
        let location = cfg
            .station
            .has_location()
            .then_some((cfg.station.latitude, cfg.station.longitude));
        let meta_model = match (&static_list, location, cfg.model.meta_model_path()) {
            (None, Some(_), Some(path)) => Some(MetaModel::load(&path)?),
            (None, Some(_), None) => {
                tracing::warn!(
                    "station location set but model.meta_model is not: no location filter"
                );
                None
            }
            (None, None, _) => {
                tracing::info!("no station location: species filter disabled");
                None
            }
            (Some(_), _, _) => {
                tracing::info!("using model.species_list; meta model not loaded");
                None
            }
        };

        Ok(Self {
            classifier: Box::new(classifier),
            labels,
            postprocess: PostprocessConfig::from_detection_config(&cfg.detection),
            meta_model,
            static_list,
            location,
            species_filter_threshold: cfg.detection.species_filter_threshold,
            include,
            exclude,
        })
    }

    /// The species filter for a given BirdNET week (`1..=48`, or `-1` for year-round), with the
    /// configured include/exclude lists applied. Cheap; the pipeline calls it when the week changes.
    pub fn species_filter_for_week(&self, week: i32) -> Result<SpeciesFilter, ModelError> {
        let base = match (&self.static_list, &self.meta_model, self.location) {
            (Some(list), _, _) => list.clone(),
            (None, Some(meta), Some((lat, lon))) => {
                SpeciesFilter::from_meta_model(meta, lat, lon, week, self.species_filter_threshold)?
            }
            _ => SpeciesFilter::allow_all(self.labels.len()),
        };
        Ok(base.include(&self.include).exclude(&self.exclude))
    }

    /// `true` when detections are restricted by location or a species list.
    pub fn has_species_filter(&self) -> bool {
        self.species_filter_kind() != SpeciesFilterKind::None
    }

    pub fn species_filter_kind(&self) -> SpeciesFilterKind {
        match (&self.static_list, &self.meta_model, self.location) {
            (Some(_), _, _) => SpeciesFilterKind::SpeciesList,
            (None, Some(_), Some(_)) => SpeciesFilterKind::LocationModel,
            _ => SpeciesFilterKind::None,
        }
    }

    /// Threshold applied to location-model scores.
    pub fn species_filter_threshold(&self) -> f32 {
        self.species_filter_threshold
    }

    /// Raw location-model occurrence scores per class for a week, when the location model is the
    /// active filter; `None` otherwise.
    pub fn location_scores_for_week(&self, week: i32) -> Result<Option<Vec<f32>>, ModelError> {
        match (&self.static_list, &self.meta_model, self.location) {
            (None, Some(meta), Some((lat, lon))) => meta.predict(lat, lon, week).map(Some),
            _ => Ok(None),
        }
    }
}
