//! Google Perch v2 run by tract: 5 s windows, resampled to 32 kHz inside the classifier.
//!
//! Uses the ONNX export with the in-graph DFT replaced by matrix multiplication (`*_no_dft_fp32`),
//! which tract can run. See `docs/MODEL.md` for provenance, outputs and checksums.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use birdsong_core::SAMPLE_RATE_HZ;
use tract_onnx::prelude::*;

use crate::resample::Resampler48To32;
use crate::{Classifier, ModelError};

/// `model_id` recorded on every detection made with [`PerchClassifier`].
pub const PERCH_V2_MODEL_ID: &str = "perch-v2";
/// Perch analyses 5 s windows.
pub const PERCH_WINDOW_SECONDS: f32 = 5.0;
/// Model input: 5 s at 32 kHz.
const INPUT_SAMPLES: usize = 160_000;
/// Outputs are embedding, spatial embedding, spectrogram, then the class logits.
const LOGITS_OUTPUT: usize = 3;

/// Perch v2 (full model or a regional slice).
pub struct PerchClassifier {
    model: Arc<TypedRunnableModel>,
    resampler: Resampler48To32,
    num_classes: usize,
}

impl PerchClassifier {
    /// Load a `perch_v2*_no_dft_fp32.onnx` file.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .and_then(|m| m.into_typed())
            .and_then(|m| {
                // The batch size is a symbol used by the input and, in the full model, by internal
                // reshapes too, so it is substituted everywhere rather than on the input alone.
                let batch = m.symbols.sym("batch");
                m.set_symbols(&HashMap::from([(batch, TDim::Val(1))]))
            })
            .and_then(|m| {
                let input = m.input_fact(0)?.shape.as_concrete().map(<[usize]>::to_vec);
                anyhow::ensure!(
                    input.as_deref() == Some(&[1, INPUT_SAMPLES][..]),
                    "expected input shape [1, {INPUT_SAMPLES}], found {:?}",
                    m.input_fact(0)?.shape
                );
                Ok(m)
            })
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| ModelError::model(path, e))?;
        let outputs = model
            .model()
            .output_outlets()
            .map_err(|e| ModelError::model(path, e))?
            .len();
        if outputs <= LOGITS_OUTPUT {
            return Err(ModelError::model(
                path,
                anyhow::anyhow!(
                    "expected at least {} outputs (Perch v2 no-DFT export), found {outputs}",
                    LOGITS_OUTPUT + 1
                ),
            ));
        }
        let mut classifier = Self {
            model,
            resampler: Resampler48To32::new(),
            num_classes: 0,
        };
        // Output shapes are not always concrete after optimisation; one silent window settles it.
        classifier.num_classes = classifier
            .run(&vec![0.0; INPUT_SAMPLES])
            .map_err(|e| ModelError::model(path, e))?
            .len();
        tracing::info!(
            path = %path.display(),
            num_classes = classifier.num_classes,
            "Perch classifier loaded"
        );
        Ok(classifier)
    }

    fn run(&self, input_32k: &[f32]) -> TractResult<Vec<f32>> {
        let input = Tensor::from_shape(&[1, INPUT_SAMPLES], input_32k)?;
        let out = self.model.run(tvec!(input.into()))?;
        Ok(out[LOGITS_OUTPUT]
            .try_as_plain()?
            .as_slice::<f32>()?
            .to_vec())
    }
}

impl Classifier for PerchClassifier {
    fn model_id(&self) -> &str {
        PERCH_V2_MODEL_ID
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn window_seconds(&self) -> f32 {
        PERCH_WINDOW_SECONDS
    }

    fn predict(&mut self, samples: &[f32]) -> Result<Vec<f32>, ModelError> {
        let expected = self.window_samples();
        if samples.len() != expected {
            return Err(ModelError::BadInput {
                expected,
                got: samples.len(),
            });
        }
        debug_assert_eq!(SAMPLE_RATE_HZ, 48_000);
        let resampled = self.resampler.process(samples);
        self.run(&resampled).map_err(|e| ModelError::Model {
            path: "<perch>".into(),
            source: e,
        })
    }
}
