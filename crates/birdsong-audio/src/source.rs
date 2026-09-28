use birdsong_core::config::{AudioConfig, AudioSourceConfig};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{AudioError, AudioFrame, FfmpegOptions, FfmpegSource};

/// Something that produces mono 48 kHz audio frames until cancelled or exhausted.
#[async_trait::async_trait]
pub trait AudioSource: Send {
    /// The configured source id (`"mic0"`).
    fn id(&self) -> &str;

    /// Push frames into `tx` until `cancel` fires, the receiver is dropped, or (for finite
    /// sources) the input ends. Live sources restart their input on failure instead of returning.
    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<AudioFrame>,
        cancel: CancellationToken,
    ) -> Result<(), AudioError>;
}

/// Build the capture source for one configured input. Every kind currently goes through ffmpeg.
pub fn source_from_config(
    src: &AudioSourceConfig,
    audio: &AudioConfig,
) -> Result<Box<dyn AudioSource>, AudioError> {
    let opts = FfmpegOptions {
        ffmpeg_path: audio.ffmpeg_path.clone(),
        ..FfmpegOptions::default()
    };
    Ok(Box::new(FfmpegSource::new(src.clone(), opts)?))
}
