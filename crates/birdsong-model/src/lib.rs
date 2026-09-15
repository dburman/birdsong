#![forbid(unsafe_code)]
//! Classifier inference for Birdsong.
//!
//! Pipeline for one 3 s chunk: [`Classifier::predict`] → logits → [`analyze_chunk`] (sigmoid with
//! sensitivity, species filter, top-N, confidence threshold, human-voice rank check) →
//! [`NeighbourMask`] (BirdNET-Pi's privacy rule that also blanks adjacent chunks) → detections.
//!
//! [`ModelBundle::load`] wires everything from a [`birdsong_core::Config`].

mod bundle;
mod classifier;
mod error;
mod labels;
pub mod mel;
mod meta;
mod postprocess;
mod species_filter;

pub use bundle::{ModelBundle, SpeciesFilterKind};
pub use classifier::{Classifier, TractClassifier, BIRDNET_V24_MODEL_ID};
pub use error::ModelError;
pub use labels::{Label, Labels};
pub use meta::MetaModel;
pub use postprocess::{
    analyze_chunk, top_scores, ChunkAnalysis, ChunkContext, NeighbourMask, PostprocessConfig,
    SigmoidSensitivity,
};
pub use species_filter::SpeciesFilter;
