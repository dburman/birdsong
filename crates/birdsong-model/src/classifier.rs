use std::path::Path;

use birdsong_core::CHUNK_SAMPLES;
use tract_onnx::prelude::*;

use crate::mel::BirdnetV24Frontend;
use crate::ModelError;

/// `model_id` recorded on every detection made with [`TractClassifier`].
pub const BIRDNET_V24_MODEL_ID: &str = "birdnet-v2.4";

/// A sound classifier: 3 s of 48 kHz mono audio in, one logit per class out.
pub trait Classifier: Send {
    fn model_id(&self) -> &str;
    fn num_classes(&self) -> usize;
    /// `samples.len()` must equal [`CHUNK_SAMPLES`]. Returns one logit per class.
    fn predict(&mut self, samples: &[f32]) -> Result<Vec<f32>, ModelError>;
}

type Runnable = std::sync::Arc<TypedRunnableModel>;

/// BirdNET V2.4 as a Rust mel frontend plus the headless CNN run by tract (see `docs/MODEL.md`).
pub struct TractClassifier {
    frontend: BirdnetV24Frontend,
    model: Runnable,
    num_classes: usize,
}

impl TractClassifier {
    /// Load `birdnet-v2.4-headless.onnx`.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let frontend = BirdnetV24Frontend::new();
        let [h, w, c] = frontend.output_shape(CHUNK_SAMPLES);
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .and_then(|m| m.with_input_fact(0, f32::fact([1, h, w, c]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| ModelError::model(path, e))?;
        let num_classes = model
            .model()
            .output_fact(0)
            .ok()
            .and_then(|f| f.shape.as_concrete().map(|s| s.iter().product()))
            .ok_or_else(|| {
                ModelError::model(path, anyhow::anyhow!("cannot determine output size"))
            })?;
        tracing::info!(path = %path.display(), num_classes, "classifier loaded");
        Ok(Self {
            frontend,
            model,
            num_classes,
        })
    }
}

impl Classifier for TractClassifier {
    fn model_id(&self) -> &str {
        BIRDNET_V24_MODEL_ID
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn predict(&mut self, samples: &[f32]) -> Result<Vec<f32>, ModelError> {
        if samples.len() != CHUNK_SAMPLES {
            return Err(ModelError::BadInput {
                expected: CHUNK_SAMPLES,
                got: samples.len(),
            });
        }
        let [h, w, c] = self.frontend.output_shape(CHUNK_SAMPLES);
        let spec = self.frontend.compute(samples);
        let run = || -> TractResult<Vec<f32>> {
            let input = Tensor::from_shape(&[1, h, w, c], &spec)?;
            let out = self.model.run(tvec!(input.into()))?;
            Ok(out[0].try_as_plain()?.as_slice::<f32>()?.to_vec())
        };
        run().map_err(|e| ModelError::Model {
            path: "<classifier>".into(),
            source: e,
        })
    }
}
