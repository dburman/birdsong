//! Golden test: [`TractClassifier`] (Rust mel frontend + headless BirdNET V2.4 ONNX in tract)
//! must reproduce the TFLite reference logits recorded by tools/convert_model/export_reference.py.
//!
//! Skips when the model is absent, since `models/` is not committed. `soundscape_15s` (5 chunks)
//! is committed; the full 2-minute `soundscape` is optional and local.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use birdsong_model::{Classifier, TractClassifier};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn headless_v24_matches_tflite_golden() -> anyhow::Result<()> {
    let model_path = repo_root().join("models/birdnet-v2.4-headless.onnx");
    if !model_path.exists() {
        eprintln!("skipping: {} not present", model_path.display());
        return Ok(());
    }
    let t0 = Instant::now();
    let mut classifier = TractClassifier::load(&model_path)?;
    eprintln!("model load+optimise: {:?}", t0.elapsed());
    assert_eq!(classifier.num_classes(), 6522);
    assert_eq!(classifier.model_id(), "birdnet-v2.4");
    assert!(
        classifier.predict(&[0.0; 10]).is_err(),
        "wrong length must be rejected"
    );

    let mut ran = 0;
    for stem in ["soundscape_15s", "soundscape"] {
        let wav_path = repo_root().join(format!("tools/fixtures/{stem}.wav"));
        let golden_path = repo_root().join(format!("tools/fixtures/golden/{stem}.json"));
        if wav_path.exists() && golden_path.exists() {
            eprintln!("== fixture {stem} ==");
            check_fixture(&mut classifier, &wav_path, &golden_path)?;
            ran += 1;
        }
    }
    assert!(ran > 0, "no golden fixtures found under tools/fixtures");
    Ok(())
}

fn check_fixture(
    classifier: &mut dyn Classifier,
    wav_path: &Path,
    golden_path: &Path,
) -> anyhow::Result<()> {
    let mut reader = hound::WavReader::open(wav_path)?;
    assert_eq!(reader.spec().sample_rate, 48_000);
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    let golden: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(golden_path)?)?;

    let mut worst = 0.0f32;
    let mut top1_mismatches = 0;
    let mut n = 0u32;
    let mut total = std::time::Duration::ZERO;
    for chunk in golden["chunks"].as_array().unwrap() {
        let start = chunk["start_sample"].as_u64().unwrap() as usize;
        let mut audio = samples[start..(start + 144_000).min(samples.len())].to_vec();
        audio.resize(144_000, 0.0);

        let t = Instant::now();
        let logits = classifier.predict(&audio)?;
        total += t.elapsed();

        let reference: Vec<f32> = chunk["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(logits.len(), reference.len(), "class count");
        let max_err = logits
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_err);
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        if argmax(&logits) != argmax(&reference) {
            top1_mismatches += 1;
        }
        n += 1;
        eprintln!(
            "chunk {:2}: max|dlogit|={max_err:.4} top1 ours={} ref={}",
            chunk["chunk_index"],
            argmax(&logits),
            argmax(&reference)
        );
    }
    eprintln!("{n} chunks: worst max|dlogit|={worst:.4}, top1 mismatches={top1_mismatches}");
    eprintln!("mean predict (frontend + network) {:?}/chunk", total / n);
    assert!(worst < 5e-2, "logit mismatch too large: {worst}");
    assert_eq!(top1_mismatches, 0, "top-1 class differs from reference");
    Ok(())
}
