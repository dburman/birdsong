//! `birdsong run --exit-on-eof --fast-files` as a real process. Needs models and ffmpeg.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use birdsong_store::{DetectionQuery, DetectionStore, SqliteStore, StoreOptions};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[tokio::test(flavor = "multi_thread")]
async fn run_command_processes_a_file_and_exits() {
    if !repo_root()
        .join("models/birdnet-v2.4-headless.onnx")
        .exists()
        || !ffmpeg_available()
    {
        eprintln!("skipping: needs models/ and ffmpeg");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("birdsong.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
[station]
latitude = 42.36
longitude = -71.06
timezone = "America/New_York"
[[audio.sources]]
id = "file0"
kind = "file"
path = {fixture:?}
[model]
kind = "birdnet-v2.4"
dir = {models:?}
[storage]
data_dir = {data:?}
[server]
bind = "127.0.0.1:0"
"#,
            fixture = repo_root()
                .join("tools/fixtures/soundscape_15s.wav")
                .display()
                .to_string(),
            models = repo_root().join("models").display().to_string(),
            data = dir.path().join("data").display().to_string(),
        ),
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_birdsong"))
        .args(["run", "--config"])
        .arg(&config_path)
        .args(["--exit-on-eof", "--fast-files"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("birdsong run did not exit within 120 s");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut stderr = String::new();
    std::io::Read::read_to_string(child.stderr.as_mut().unwrap(), &mut stderr).unwrap();
    assert!(status.success(), "exit {status}; stderr:\n{stderr}");
    assert!(stderr.contains("HTTP API listening"), "stderr:\n{stderr}");
    assert!(
        stderr.contains("detection") && stderr.contains("Black-capped Chickadee"),
        "stderr:\n{stderr}"
    );

    let cfg = birdsong_core::Config::load(Some(&config_path)).unwrap();
    let store = SqliteStore::open(
        &cfg.storage.database_path(),
        StoreOptions::from_config(&cfg),
    )
    .await
    .unwrap();
    let rows = store
        .list(&DetectionQuery {
            species: Some("Poecile atricapillus".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!rows.is_empty());
    assert_eq!(rows[0].source_id, "file0");
}

#[test]
fn run_fails_clearly_when_models_are_missing() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("birdsong.toml");
    std::fs::write(
        &config_path,
        format!(
            "[[audio.sources]]\nid = \"file0\"\nkind = \"file\"\npath = \"/nonexistent.wav\"\n[model]\nkind = \"birdnet-v2.4\"\ndir = \"/nonexistent-models\"\n[storage]\ndata_dir = {:?}\n[server]\nbind = \"127.0.0.1:0\"\n",
            dir.path().join("data").display().to_string()
        ),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_birdsong"))
        .args(["run", "--config"])
        .arg(&config_path)
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("loading models") && stderr.contains("/nonexistent-models"),
        "stderr:\n{stderr}"
    );
}
