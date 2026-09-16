//! Perch v2 on the fixture through Birdsong's own resampler. Skips when the regional model from
//! `scripts/fetch-perch.sh` is absent.
#![forbid(unsafe_code)]

use std::path::PathBuf;

use birdsong_core::Config;
use birdsong_model::{top_scores, ModelBundle, PERCH_V2_MODEL_ID};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const ONNX: &str = "perch/perch_v2_north-america-east_no_dft_fp32.onnx";

fn fixture_48k() -> Vec<f32> {
    let mut reader =
        hound::WavReader::open(repo_root().join("tools/fixtures/soundscape_15s.wav")).unwrap();
    assert_eq!(reader.spec().sample_rate, 48_000);
    reader
        .samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect()
}

#[test]
fn perch_hears_the_chickadee_in_the_first_window() {
    let models = repo_root().join("models");
    if !models.join(ONNX).exists() {
        eprintln!("skipping: {ONNX} not present (scripts/fetch-perch.sh)");
        return;
    }
    let cfg = Config::from_toml(&format!(
        r#"
[[audio.sources]]
id = "file0"
kind = "file"
path = "unused.wav"
[model]
dir = {dir:?}
kind = "perch-v2"
classifier = "{ONNX}"
labels = "perch/perch_v2_north-america-east_labels.txt"
common_names = "labels/en_us.txt"
"#,
        dir = models.display().to_string(),
    ))
    .unwrap();
    let mut bundle = ModelBundle::load(&cfg).unwrap();
    assert_eq!(bundle.classifier.model_id(), PERCH_V2_MODEL_ID);
    assert_eq!(bundle.classifier.window_samples(), 240_000);
    assert_eq!(bundle.classifier.num_classes(), 999);
    assert_eq!(bundle.labels.len(), 999);
    assert!(bundle.postprocess.softmax);
    assert!(
        !bundle.has_species_filter(),
        "the location model is BirdNET-only"
    );

    let audio = fixture_48k();
    let logits = bundle.classifier.predict(&audio[..240_000]).unwrap();
    let top = top_scores(&logits, &bundle.postprocess, 3);
    let (best, confidence) = top[0];
    let label = bundle.labels.get(best).unwrap();
    assert_eq!(label.scientific, "Poecile atricapillus", "{top:?}");
    assert_eq!(label.common, "Black-capped Chickadee");
    // 0.698 with ffmpeg's resampler; ours should agree closely.
    assert!((0.6..0.8).contains(&confidence), "confidence {confidence}");

    assert!(bundle.classifier.predict(&audio[..144_000]).is_err());
}

/// The full 14 795-class model uses a symbolic batch size inside the graph as well. Run with
/// `cargo test --release -p birdsong-model --test perch_v2 -- --ignored` after
/// `scripts/fetch-perch.sh full`.
#[test]
#[ignore = "needs the 413 MB full model; slow in debug builds"]
fn full_perch_model_loads_and_hears_the_chickadee() {
    let models = repo_root().join("models");
    let cfg = Config::from_toml(&format!(
        r#"
[[audio.sources]]
id = "file0"
kind = "file"
path = "unused.wav"
[model]
dir = {dir:?}
kind = "perch-v2"
classifier = "perch/perch_v2_no_dft_fp32.onnx"
labels = "perch/perch_v2_labels.txt"
common_names = "labels/en_us.txt"
"#,
        dir = models.display().to_string(),
    ))
    .unwrap();
    let mut bundle = ModelBundle::load(&cfg).unwrap();
    assert_eq!(bundle.classifier.num_classes(), 14_795);
    assert_eq!(bundle.labels.len(), 14_795);
    let logits = bundle
        .classifier
        .predict(&fixture_48k()[..240_000])
        .unwrap();
    let (best, _) = top_scores(&logits, &bundle.postprocess, 1)[0];
    assert_eq!(
        bundle.labels.get(best).unwrap().scientific,
        "Poecile atricapillus"
    );
}
