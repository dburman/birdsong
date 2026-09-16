#![forbid(unsafe_code)]
//! Classifier inference for Birdsong.
//!
//! Pipeline for one window (3 s for BirdNET V2.4, 5 s for Perch v2): [`Classifier::predict`] →
//! logits → [`analyze_chunk`] (sigmoid with sensitivity, or softmax for Perch; species filter,
//! top-N, confidence threshold, human-voice rank check) →
//! [`NeighbourMask`] (BirdNET-Pi's privacy rule that also blanks adjacent chunks) → detections.
//!
//! [`ModelBundle::load`] wires everything from a [`birdsong_core::Config`].

mod bundle;
mod classifier;
mod confirm;
mod error;
mod labels;
pub mod mel;
mod meta;
mod perch;
mod postprocess;
pub mod resample;
mod species_filter;

pub use bundle::{ModelBundle, SpeciesFilterKind};
pub use classifier::{Classifier, TractClassifier, BIRDNET_V24_MODEL_ID};
pub use confirm::{Confirmer, DynamicThresholds};
pub use error::ModelError;
pub use labels::{
    Label, Labels, BIRDNET_SOUND_EVENTS, PERCH_ANIMAL_EVENTS, PERCH_HUMAN_CLASSES,
    PERCH_LABELS_HEADER,
};
pub use meta::MetaModel;
pub use perch::{PerchClassifier, PERCH_V2_MODEL_ID, PERCH_WINDOW_SECONDS};
pub use postprocess::{
    analyze_chunk, analyze_chunk_with, softmax, top_scores, ChunkAnalysis, ChunkContext,
    NeighbourMask, PostprocessConfig, SigmoidSensitivity,
};
pub use species_filter::SpeciesFilter;
