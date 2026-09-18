use std::collections::HashMap;
use std::path::Path;

use tract_onnx::prelude::*;

use crate::ModelError;

type Runnable = std::sync::Arc<TypedRunnableModel>;

/// Fix the batch size to 1. A model that names it `batch` may also use it inside the graph (the
/// Geomodel does), so it is substituted everywhere; otherwise only the input shape is set.
fn pin_batch(model: InferenceModel) -> TractResult<TypedModel> {
    let substituted = model.clone().into_typed().and_then(|m| {
        let batch = m.symbols.sym("batch");
        m.set_symbols(&HashMap::from([(batch, TDim::Val(1))]))
    });
    if let Ok(typed) = substituted {
        if typed.input_fact(0)?.shape.as_concrete() == Some(&[1, 3][..]) {
            return Ok(typed);
        }
    }
    model
        .with_input_fact(0, f32::fact([1, 3]).into())?
        .into_typed()
}

/// BirdNET's location/week model: `[latitude, longitude, week]` → occurrence probability per class.
pub struct MetaModel {
    model: Runnable,
}

impl MetaModel {
    /// Load `meta-model.onnx` or another location model with input `[lat, lon, week]`, such as the
    /// BirdNET Geomodel (see `docs/MODEL.md`).
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .and_then(pin_batch)
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| ModelError::model(path, e))?;
        tracing::info!(path = %path.display(), "meta model loaded");
        Ok(Self { model })
    }

    /// `week` is `1..=48` or [`birdsong_core::YEAR_ROUND_WEEK`] (`-1`).
    pub fn predict(
        &self,
        latitude: f64,
        longitude: f64,
        week: i32,
    ) -> Result<Vec<f32>, ModelError> {
        let run = || -> TractResult<Vec<f32>> {
            let input =
                Tensor::from_shape(&[1, 3], &[latitude as f32, longitude as f32, week as f32])?;
            let out = self.model.run(tvec!(input.into()))?;
            Ok(out[0].try_as_plain()?.as_slice::<f32>()?.to_vec())
        };
        run().map_err(|e| ModelError::Model {
            path: "<meta-model>".into(),
            source: e,
        })
    }
}
