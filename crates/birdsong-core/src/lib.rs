#![forbid(unsafe_code)]
//! Configuration, domain types and small helpers shared by every Birdsong crate.
//!
//! Nothing here does I/O except [`Config::load`].

pub mod config;
mod error;
mod names;
mod time;
mod types;

pub use config::Config;
pub use error::ConfigError;
pub use names::sanitize_name;
pub use time::{local_date_and_hour, week_of_year, YEAR_ROUND_WEEK};
pub use types::{Detection, CHUNK_SAMPLES, CHUNK_SECONDS, SAMPLE_RATE_HZ};
