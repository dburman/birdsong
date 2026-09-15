//! Spike golden test: the tf2onnx-converted BirdNET V2.4 meta (location/week) model run in tract
//! must reproduce the TFLite reference probabilities from tools/convert_model/export_meta_reference.py.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use tract_onnx::prelude::*;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn meta_v24_matches_tflite_golden() -> anyhow::Result<()> {
    let model_path = repo_root().join("models/meta-model.onnx");
    let golden_path = repo_root().join("tools/fixtures/golden/meta_v24.json");
    if !model_path.exists() || !golden_path.exists() {
        eprintln!("skipping: model or golden file not present");
        return Ok(());
    }
    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .with_input_fact(0, f32::fact([1, 3]).into())?
        .into_optimized()?
        .into_runnable()?;
    let golden: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&golden_path)?)?;
    let threshold = golden["threshold_default"].as_f64().unwrap() as f32;
    for case in golden["cases"].as_array().unwrap() {
        let (lat, lon, week) = (
            case["lat"].as_f64().unwrap() as f32,
            case["lon"].as_f64().unwrap() as f32,
            case["week"].as_f64().unwrap() as f32,
        );
        let input = Tensor::from_shape(&[1, 3], &[lat, lon, week])?;
        let out = model.run(tvec!(input.into()))?;
        let probs = out[0].try_as_plain()?.as_slice::<f32>()?.to_vec();
        let reference: Vec<f32> = case["probabilities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(probs.len(), reference.len());
        let max_err = probs
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let ours_allowed = probs.iter().filter(|&&p| p >= threshold).count();
        let ref_allowed = case["allowed_at_0_03"].as_u64().unwrap() as usize;
        eprintln!(
            "{}: max|dp|={max_err:.5} allowed ours={ours_allowed} ref={ref_allowed}",
            case["name"]
        );
        assert!(
            max_err < 1e-3,
            "{}: probability mismatch {max_err}",
            case["name"]
        );
        assert_eq!(
            ours_allowed, ref_allowed,
            "{}: allowed-species count differs",
            case["name"]
        );
    }
    Ok(())
}
