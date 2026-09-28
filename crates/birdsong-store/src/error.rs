use std::path::PathBuf;

/// Errors from the detection store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("database migration failed: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// A stored value could not be decoded (hand-edited or corrupt database).
    #[error("corrupt value in column {column}: {value:?}")]
    Corrupt { column: &'static str, value: String },
}
