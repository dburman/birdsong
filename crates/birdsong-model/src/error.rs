use std::path::PathBuf;

/// Errors from loading models and labels or running inference.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// tract failed to load, optimise or run a model.
    #[error("model {path}: {source}")]
    Model {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },
    #[error("labels file {path}: {message}")]
    Labels { path: PathBuf, message: String },
    #[error("classifier expects {expected} samples, got {got}")]
    BadInput { expected: usize, got: usize },
    #[error("unknown species {name:?} in {context}")]
    UnknownSpecies { name: String, context: String },
    #[error("{0}")]
    Config(String),
}

impl ModelError {
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
    pub(crate) fn model(path: &std::path::Path, source: anyhow::Error) -> Self {
        Self::Model {
            path: path.to_path_buf(),
            source,
        }
    }
}
