//! `birdsong analyze` and `birdsong species-list` as real processes. Need `models/`.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn models() -> Option<PathBuf> {
    let dir = repo_root().join("models");
    dir.join("birdnet-v2.4-headless.onnx")
        .exists()
        .then_some(dir)
}

fn birdsong(args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_birdsong"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .unwrap();
    out
}

fn stdout_json(out: &Output) -> serde_json::Value {
    assert!(
        out.status.success(),
        "exit {}; stderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("stdout is JSON")
}

#[test]
fn analyze_json_matches_golden_logits() {
    let Some(models) = models() else {
        eprintln!("skipping: models not present");
        return;
    };
    let fixture = repo_root().join("tools/fixtures/soundscape_15s.wav");
    let out = birdsong(&[
        "analyze",
        fixture.to_str().unwrap(),
        "--models",
        models.to_str().unwrap(),
        "--json",
        "--top",
        "5",
    ]);
    let report = stdout_json(&out);
    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join("tools/fixtures/golden/soundscape_15s.json"))
            .unwrap(),
    )
    .unwrap();

    assert_eq!(report["model_id"], "birdnet-v2.4");
    assert_eq!(
        report["species_filter"], "none",
        "no location without a config"
    );
    let slope = report["sigmoid_slope"].as_f64().unwrap();
    assert!((slope - 0.75).abs() < 1e-6, "default sensitivity 1.25");

    let chunks = report["chunks"].as_array().unwrap();
    let golden_chunks = golden["chunks"].as_array().unwrap();
    assert_eq!(chunks.len(), golden_chunks.len());
    for (ours, gold) in chunks.iter().zip(golden_chunks) {
        let k = ours["index"].as_u64().unwrap();
        assert_eq!(ours["start_seconds"].as_f64().unwrap(), 3.0 * k as f64);
        let top = ours["top"].as_array().unwrap();
        let gold_top = gold["top5"].as_array().unwrap();
        assert_eq!(top.len(), 5);
        for (t, g) in top.iter().zip(gold_top) {
            assert_eq!(t["class_index"], g["class_index"], "chunk {k} ranking");
            let logit = t["logit"].as_f64().unwrap();
            assert!(
                (logit - g["logit"].as_f64().unwrap()).abs() < 0.05,
                "chunk {k} logit"
            );
            let expected = 1.0 / (1.0 + (-slope * logit).exp());
            assert!(
                (t["confidence"].as_f64().unwrap() - expected).abs() < 1e-4,
                "chunk {k} confidence"
            );
        }
    }
}

#[test]
fn analyze_text_with_location_reports_chickadee() {
    let Some(models) = models() else { return };
    let fixture = repo_root().join("tools/fixtures/soundscape_15s.wav");
    let out = birdsong(&[
        "analyze",
        fixture.to_str().unwrap(),
        "--models",
        models.to_str().unwrap(),
        "--lat",
        "42.36",
        "--lon",
        "-71.06",
        "--date",
        "2026-05-15",
    ]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("week 19"), "{text}");
    assert!(text.contains("species filter: location-model"), "{text}");
    assert!(
        text.contains("REPORTED") && text.contains("Black-capped Chickadee"),
        "{text}"
    );
    assert!(text.contains("summary:"), "{text}");
}

#[test]
fn species_list_matches_location_golden() {
    let Some(models) = models() else { return };
    let out = birdsong(&[
        "species-list",
        "--models",
        models.to_str().unwrap(),
        "--lat",
        "42.36",
        "--lon",
        "-71.06",
        "--week",
        "20",
        "--json",
    ]);
    let report = stdout_json(&out);
    assert_eq!(report["species_filter"], "location-model");
    assert_eq!(
        report["count"], 126,
        "same as tools/fixtures/golden/meta_v24.json boston_week20"
    );
    let species = report["species"].as_array().unwrap();
    let scores: Vec<f64> = species
        .iter()
        .map(|s| s["score"].as_f64().unwrap())
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "sorted by score");
    assert!(scores.iter().all(|&s| s >= 0.03 - 1e-6));

    let year_round = stdout_json(&birdsong(&[
        "species-list",
        "--models",
        models.to_str().unwrap(),
        "--lat",
        "42.36",
        "--lon",
        "-71.06",
        "--week",
        "-1",
        "--json",
    ]));
    assert_eq!(year_round["count"], 236, "golden boston_yearround");
}

#[test]
fn species_list_without_location_fails_clearly() {
    let Some(models) = models() else { return };
    let out = birdsong(&[
        "species-list",
        "--models",
        models.to_str().unwrap(),
        "--week",
        "20",
    ]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("species filter disabled"), "{err}");
    let out = birdsong(&[
        "species-list",
        "--models",
        models.to_str().unwrap(),
        "--lat",
        "1",
        "--lon",
        "2",
        "--week",
        "60",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("week must be"));
}
