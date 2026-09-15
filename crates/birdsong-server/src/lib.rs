#![forbid(unsafe_code)]
//! The Birdsong application: the detection pipeline, and later the HTTP API and web UI.

pub mod pipeline;
mod queue;
pub mod stats;

pub use pipeline::{Pipeline, PipelineOptions, PipelineSummary, SourceSpec};
pub use queue::Backpressure;
pub use stats::{PipelineStats, StatsSnapshot};
