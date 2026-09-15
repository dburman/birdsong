//! End-to-end: `ModelBundle::load` from a real config, then audio → detections.
//! Skips when `models/` is absent.
#![forbid(unsafe_code)]

use std::path::PathBuf;

use birdsong_core::{week_of_year, Config};
use birdsong_model::{analyze_chunk, ChunkContext, ModelBundle};
use chrono::{TimeZone, Utc};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn config(models_dir: &std::path::Path, species_list: Option<&std::path::Path>) -> Config {
    let list = species_list
        .map(|p| format!("species_list = {:?}", p.display().to_string()))
        .unwrap_or_default();
    Config::from_toml(&format!(
        r#"
[station]
latitude = 42.36
longitude = -71.06
timezone = "America/New_York"
[[audio.sources]]
id = "mic0"
kind = "file"
path = "/tmp/x.wav"
[detection]
exclude_species = ["Engine"]
[model]
dir = {:?}
{list}
"#,
        models_dir.display().to_string()
    ))
    .unwrap()
}

#[test]
fn bundle_detects_chickadee_in_fixture() -> anyhow::Result<()> {
    let models_dir = repo_root().join("models");
    if !models_dir.join("birdnet-v2.4-headless.onnx").exists() {
        eprintln!("skipping: models not present");
        return Ok(());
    }
    let cfg = config(&models_dir, None);
    let mut bundle = ModelBundle::load(&cfg)?;
    assert!(bundle.has_species_filter());

    let wav = repo_root().join("tools/fixtures/soundscape_15s.wav");
    let mut reader = hound::WavReader::open(&wav)?;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let logits = bundle.classifier.predict(&samples[..144_000])?;

    let start_at = Utc.with_ymd_and_hms(2026, 5, 15, 10, 0, 0).unwrap();
    let week = week_of_year(start_at.date_naive()) as i32;
    let filter = bundle.species_filter_for_week(week)?;
    assert!(
        filter.num_allowed() > 50 && filter.num_allowed() < 1000,
        "Boston in May: {}",
        filter.num_allowed()
    );
    assert!(
        !filter.is_allowed(bundle.labels.index_of_scientific("Engine").unwrap()),
        "excluded"
    );

    let ctx = ChunkContext {
        start_at,
        source_id: "mic0".into(),
        model_id: bundle.classifier.model_id().into(),
    };
    let r = analyze_chunk(&logits, &bundle.labels, &filter, &bundle.postprocess, &ctx);
    assert!(!r.masked);
    assert_eq!(r.detections[0].common_name, "Black-capped Chickadee");
    // golden logit 1.477 × slope 0.75 → sigmoid ≈ 0.752
    assert!(
        (r.detections[0].confidence - 0.752).abs() < 0.01,
        "{}",
        r.detections[0].confidence
    );
    assert_eq!(r.detections[0].model_id, "birdnet-v2.4");
    Ok(())
}

#[test]
fn bundle_with_static_species_list() -> anyhow::Result<()> {
    let models_dir = repo_root().join("models");
    if !models_dir.join("birdnet-v2.4-headless.onnx").exists() {
        eprintln!("skipping: models not present");
        return Ok(());
    }
    let list = repo_root().join("tools/fixtures/soundscape_species_list.txt");
    let cfg = config(&models_dir, Some(&list));
    let bundle = ModelBundle::load(&cfg)?;
    let filter = bundle.species_filter_for_week(20)?;
    // The fixture comes from a newer BirdNET-Analyzer whose taxonomy differs slightly from the
    // V2.4 labels (e.g. Astur cooperii); only names the label file knows count.
    let text = std::fs::read_to_string(&list)?;
    let names: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let known = names
        .iter()
        .filter(|l| {
            bundle
                .labels
                .index_of_scientific(l.split_once('_').map_or(l, |(s, _)| s))
                .is_some()
        })
        .count();
    assert!(
        known > names.len() / 2,
        "most fixture names should be known: {known}/{}",
        names.len()
    );
    assert_eq!(filter.num_allowed(), known);
    Ok(())
}

#[test]
fn unknown_include_species_is_an_error() {
    let models_dir = repo_root().join("models");
    if !models_dir.join("birdnet-v2.4-headless.onnx").exists() {
        return;
    }
    let mut cfg = config(&models_dir, None);
    cfg.detection.include_species = vec!["Dinornis maximus".into()];
    let err = match ModelBundle::load(&cfg) {
        Ok(_) => panic!("unknown species must fail"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("Dinornis"), "{err}");
}
