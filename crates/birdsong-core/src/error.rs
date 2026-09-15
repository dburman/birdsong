/// Errors produced while loading or validating configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The file or environment could not be read or parsed.
    #[error("failed to read configuration: {0}")]
    Load(#[from] config::ConfigError),
    /// The configuration parsed but a value is out of range or inconsistent.
    #[error("invalid configuration: {0}")]
    Invalid(String),
}
