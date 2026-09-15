//! Golden test: [`MetaModel`] (tf2onnx-converted BirdNET V2.4 location model in tract) must
//! reproduce the TFLite reference from tools/convert_model/export_meta_reference.py.
#![forbid(unsafe_code)]

use std::path::PathBuf;

use birdsong_model::{MetaModel, SpeciesFilter};

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
    let meta = MetaModel::load(&model_path)?;
    let golden: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&golden_path)?)?;
    let threshold = golden["threshold_default"].as_f64().unwrap() as f32;
    for case in golden["cases"].as_array().unwrap() {
        let (lat, lon, week) = (
            case["lat"].as_f64().unwrap(),
            case["lon"].as_f64().unwrap(),
            case["week"].as_i64().unwrap() as i32,
        );
        let probs = meta.predict(lat, lon, week)?;
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
        let filter = SpeciesFilter::from_meta_model(&meta, lat, lon, week, threshold)?;
        let ref_allowed = case["allowed_at_0_03"].as_u64().unwrap() as usize;
        eprintln!(
            "{}: max|dp|={max_err:.5} allowed ours={} ref={ref_allowed}",
            case["name"],
            filter.num_allowed()
        );
        assert!(
            max_err < 1e-3,
            "{}: probability mismatch {max_err}",
            case["name"]
        );
        assert_eq!(
            filter.num_allowed(),
            ref_allowed,
            "{}: allowed-species count differs",
            case["name"]
        );
    }
    Ok(())
}
