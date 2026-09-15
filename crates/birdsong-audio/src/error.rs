use std::path::PathBuf;

/// Errors from audio capture and WAV I/O.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("WAV file {path}: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },
    #[error("{path}: unsupported audio: {message}")]
    Unsupported { path: PathBuf, message: String },
    #[error("ffmpeg not found at {0:?}; install ffmpeg or set audio.ffmpeg_path")]
    FfmpegNotFound(PathBuf),
    #[error("failed to start ffmpeg: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("ffmpeg exited with {status}: {stderr}")]
    FfmpegFailed { status: String, stderr: String },
    #[error("audio source {id:?}: {message}")]
    Config { id: String, message: String },
    #[error("cannot encode {path}: {message}")]
    Encode { path: PathBuf, message: String },
    #[error("FLAC encoding failed: {0}")]
    FlacEncode(String),
    #[error("background task failed: {0}")]
    Task(String),
}
