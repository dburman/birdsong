use std::collections::HashSet;

use birdsong_core::config::{ModelKind, UnmappedSpecies};
use birdsong_core::{Config, DetectionKind};

use crate::{
    Classifier, Labels, MetaModel, ModelError, PerchClassifier, PostprocessConfig, SpeciesFilter,
    TractClassifier,
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
    /// Perch: location-model index for each class; `None` for BirdNET, whose classes are the
    /// location model's.
    meta_map: Option<Vec<Option<usize>>>,
    /// The location model has no year-round mode (the Geomodel): use the maximum over 48 weeks.
    year_round_by_max: bool,
    /// Classes known to be birds (scientific names), when the labels allow telling; `None` for
    /// BirdNET, whose uploads are not restricted.
    birds: Option<HashSet<String>>,
    allow_unmapped: bool,
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
        let mut birdnet_labels = None;
        let (classifier, labels): (Box<dyn Classifier>, Labels) = match cfg.model.kind {
            ModelKind::BirdnetV24 => (
                Box::new(TractClassifier::load(&cfg.model.classifier_path())?),
                Labels::load(&cfg.model.labels_path())?,
            ),
            ModelKind::PerchV2 => {
                birdnet_labels = cfg
                    .model
                    .common_names_path()
                    .map(|p| Labels::load(&p))
                    .transpose()?;
                (
                    Box::new(PerchClassifier::load(&cfg.model.classifier_path())?),
                    Labels::load_perch(&cfg.model.labels_path(), birdnet_labels.as_ref())?,
                )
            }
        };
        if labels.len() != classifier.num_classes() {
            return Err(ModelError::Config(format!(
                "labels file has {} entries but the classifier has {} outputs",
                labels.len(),
                classifier.num_classes()
            )));
        }
        let location_labels = cfg
            .model
            .meta_model_labels_path()
            .map(|p| Labels::load_location(&p))
            .transpose()?;
        let mut labels = labels;
        if let (ModelKind::PerchV2, Some(location_labels)) = (cfg.model.kind, &location_labels) {
            labels.fill_common_names(location_labels);
        }
        labels.indices_for(
            &cfg.detection.confirmation_exempt_species,
            "detection.confirmation_exempt_species",
        )?;
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
        let birdnet = cfg.model.kind == ModelKind::BirdnetV24;
        let meta_model = match (&static_list, location, cfg.model.meta_model_path()) {
            (None, Some(_), Some(_))
                if !birdnet && birdnet_labels.is_none() && location_labels.is_none() =>
            {
                tracing::warn!(
                    "the Perch location filter needs model.meta_model_labels or model.common_names \
                     to map species; no location filter"
                );
                None
            }
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

        // BirdNET's own location model scores BirdNET's classes in order. Any other pairing (a
        // Geomodel, or Perch with BirdNET's model) is matched by scientific name.
        let target = location_labels.as_ref().or(if birdnet {
            None
        } else {
            birdnet_labels.as_ref()
        });
        let meta_map = match (&meta_model, target) {
            (Some(_), Some(target)) => {
                let map = labels.map_to(target);
                let species = (0..labels.len())
                    .filter(|&i| is_species(&labels, i))
                    .count();
                let mapped = map.iter().filter(|m| m.is_some()).count();
                tracing::info!(
                    species,
                    mapped,
                    location_model_classes = target.len(),
                    unmapped = cfg.model.location_filter_unmapped.as_str(),
                    "location filter matches species by scientific name"
                );
                Some(map)
            }
            _ => None,
        };
        if let Some(meta) = &meta_model {
            let expected = target.map_or(labels.len(), Labels::len);
            let got = meta.predict(0.0, 0.0, 1)?.len();
            if got != expected {
                return Err(ModelError::Config(format!(
                    "the location model has {got} outputs but its labels have {expected} entries \
                     (set model.meta_model_labels for a location model other than BirdNET's)"
                )));
            }
        }

        let birds = (!birdnet)
            .then(|| bird_species(&labels, location_labels.as_ref(), birdnet_labels.as_ref()))
            .flatten();

        let mut postprocess = PostprocessConfig::from_detection_config(&cfg.detection);
        postprocess.softmax = !birdnet;

        Ok(Self {
            classifier,
            labels,
            postprocess,
            meta_model,
            meta_map,
            year_round_by_max: location_labels.is_some(),
            birds,
            allow_unmapped: cfg.model.location_filter_unmapped == UnmappedSpecies::Allow,
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
            (None, Some(meta), Some((lat, lon))) => match &self.meta_map {
                None => SpeciesFilter::from_meta_model(
                    meta,
                    lat,
                    lon,
                    week,
                    self.species_filter_threshold,
                )?,
                Some(map) => SpeciesFilter::from_mapped_scores(
                    &self.meta_scores(meta, lat, lon, week)?,
                    map,
                    self.species_filter_threshold,
                    self.allow_unmapped,
                    |i| is_species(&self.labels, i),
                ),
            },
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

    /// Location-model occurrence scores per class for a week, when the location model is the
    /// active filter; `None` otherwise. For Perch, classes the location model cannot score are
    /// `NaN`.
    pub fn location_scores_for_week(&self, week: i32) -> Result<Option<Vec<f32>>, ModelError> {
        match (&self.static_list, &self.meta_model, self.location) {
            (None, Some(meta), Some((lat, lon))) => {
                let probs = self.meta_scores(meta, lat, lon, week)?;
                Ok(Some(match &self.meta_map {
                    None => probs,
                    Some(map) => map
                        .iter()
                        .map(|m| m.and_then(|j| probs.get(j).copied()).unwrap_or(f32::NAN))
                        .collect(),
                }))
            }
            _ => Ok(None),
        }
    }
}

impl ModelBundle {
    /// Location-model scores for a week; year-round is the maximum over the 48 weeks for models
    /// without a year-round mode.
    fn meta_scores(
        &self,
        meta: &MetaModel,
        lat: f64,
        lon: f64,
        week: i32,
    ) -> Result<Vec<f32>, ModelError> {
        if !(self.year_round_by_max && week == birdsong_core::YEAR_ROUND_WEEK) {
            return meta.predict(lat, lon, week);
        }
        let mut max = meta.predict(lat, lon, 1)?;
        for w in 2..=48 {
            for (m, p) in max.iter_mut().zip(meta.predict(lat, lon, w)?) {
                *m = m.max(p);
            }
        }
        Ok(max)
    }
}

impl ModelBundle {
    /// Scientific names of the classes that are birds, for uploads restricted to birds.
    pub fn bird_species(&self) -> Option<&HashSet<String>> {
        self.birds.as_ref()
    }
}

/// Which species are birds: the Geomodel's species codes are eBird codes for birds and numeric
/// iNaturalist ids for other animals. Species it does not know count as birds when BirdNET's
/// labels list them (mostly birds under older names). `None` when neither list is available.
fn bird_species(
    labels: &Labels,
    geomodel: Option<&Labels>,
    birdnet: Option<&Labels>,
) -> Option<HashSet<String>> {
    if geomodel.is_none() && birdnet.is_none() {
        return None;
    }
    let is_bird = |scientific: &str| match geomodel
        .and_then(|g| g.index_of_scientific(scientific).and_then(|i| g.get(i)))
    {
        Some(label) => label
            .raw
            .split('\t')
            .next()
            .is_some_and(|code| !code.is_empty() && !code.chars().all(|c| c.is_ascii_digit())),
        None => birdnet.is_some_and(|b| b.index_of_scientific(scientific).is_some()),
    };
    Some(
        labels
            .iter()
            .filter(|l| l.scientific.contains(' ') && l.kind() == DetectionKind::Animal)
            .filter(|l| is_bird(&l.scientific))
            .map(|l| l.scientific.clone())
            .collect(),
    )
}

/// Perch species are `Genus species`; sound events have no space.
fn is_species(labels: &Labels, i: usize) -> bool {
    labels.get(i).is_some_and(|l| l.scientific.contains(' '))
}
