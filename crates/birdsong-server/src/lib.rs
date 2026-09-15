#![forbid(unsafe_code)]
//! The Birdsong application: the detection pipeline, offline analysis tools, and the HTTP API.

pub mod analyze;
pub mod api;
pub mod clips;
pub mod pipeline;
mod queue;
pub mod stats;

pub use pipeline::{Pipeline, PipelineOptions, PipelineSummary, SourceSpec};
pub use queue::Backpressure;
pub use stats::{PipelineStats, StatsSnapshot};
