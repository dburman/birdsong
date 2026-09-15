//! Configuration schema (see `BUILD_PLAN.md` §4), loading and validation.
//!
//! Sources, lowest precedence first: built-in defaults, the TOML file, then environment
//! variables `BIRDSONG__<SECTION>__<KEY>` (e.g. `BIRDSONG__DETECTION__MIN_CONFIDENCE=0.8`).
//! Array entries such as `audio.sources` can only be set in the file.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use chrono_tz::Tz;
use serde::{Deserialize, Serialize};

use crate::ConfigError;

/// Prefix for environment overrides.
pub const ENV_PREFIX: &str = "BIRDSONG";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub station: StationConfig,
    pub audio: AudioConfig,
    pub detection: DetectionConfig,
    pub model: ModelConfig,
    pub storage: StorageConfig,
    pub retention: RetentionConfig,
    pub server: ServerConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StationConfig {
    pub name: String,
    /// Set both latitude and longitude to 0 to disable the location filter.
    pub latitude: f64,
    pub longitude: f64,
    /// IANA time zone used for daily charts and clip folders.
    pub timezone: Tz,
}

impl Default for StationConfig {
    fn default() -> Self {
        Self {
            name: "Birdsong".into(),
            latitude: 0.0,
            longitude: 0.0,
            timezone: Tz::UTC,
        }
    }
}

impl StationConfig {
    /// `true` when a real location is configured (both coordinates non-zero).
    pub fn has_location(&self) -> bool {
        self.latitude != 0.0 || self.longitude != 0.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AudioConfig {
    /// ffmpeg executable used for capture (name on `PATH` or absolute path).
    pub ffmpeg_path: PathBuf,
    /// Seconds of recent audio kept per source for clip extraction.
    pub ring_buffer_seconds: f32,
    pub sources: Vec<AudioSourceConfig>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            ffmpeg_path: PathBuf::from("ffmpeg"),
            ring_buffer_seconds: 90.0,
            sources: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioSourceKind {
    /// ALSA capture device (USB microphone / sound card).
    Alsa,
    /// Network stream (RTSP/RTMP/HTTP, anything ffmpeg can open).
    Rtsp,
    /// Replay a WAV file (testing).
    File,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSourceConfig {
    /// Short unique id recorded with every detection (`"mic0"`).
    pub id: String,
    pub kind: AudioSourceKind,
    /// ALSA device name, e.g. `"hw:1,0"` (kind = alsa).
    #[serde(default)]
    pub device: Option<String>,
    /// Stream URL (kind = rtsp).
    #[serde(default)]
    pub url: Option<String>,
    /// WAV file path (kind = file).
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Gain applied to the captured signal in dB.
    #[serde(default)]
    pub gain_db: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DetectionConfig {
    /// BirdNET-Pi `CONFIDENCE`.
    pub min_confidence: f32,
    /// BirdNET-Pi `SENSITIVITY`, valid `0.5..=1.5`.
    pub sensitivity: f32,
    /// BirdNET-Pi `OVERLAP`, valid `0.0..3.0`.
    pub overlap_seconds: f32,
    /// Location-filter threshold on the meta model output (BirdNET `SF_THRESH`).
    pub species_filter_threshold: f32,
    /// How many classes per chunk may become detections.
    pub top_n_per_chunk: usize,
    /// Scientific names always allowed (bypass the location filter).
    pub include_species: Vec<String>,
    /// Scientific names never reported.
    pub exclude_species: Vec<String>,
    /// BirdNET-Pi's human-voice mask (see BUILD_PLAN §7.6).
    pub privacy_filter: bool,
    /// Percent `0..=100`; higher looks deeper into the ranking for human classes.
    pub privacy_threshold: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            sensitivity: 1.25,
            overlap_seconds: 0.0,
            species_filter_threshold: 0.03,
            top_n_per_chunk: 3,
            include_species: Vec::new(),
            exclude_species: Vec::new(),
            privacy_filter: true,
            privacy_threshold: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    pub dir: PathBuf,
    /// Headless classifier ONNX, relative to `dir` (see `docs/MODEL.md`).
    pub classifier: String,
    /// Labels file, relative to `dir`.
    pub labels: String,
    /// Location/week model ONNX, relative to `dir`. `None` disables the location filter.
    pub meta_model: Option<String>,
    /// Precomputed allowed-species list; when set, `meta_model` is ignored.
    pub species_list: Option<PathBuf>,
    /// Inference threads; `0` = number of CPUs minus one, minimum one.
    pub threads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("/models"),
            classifier: "birdnet-v2.4-headless.onnx".into(),
            labels: "labels/en_us.txt".into(),
            meta_model: Some("meta-model.onnx".into()),
            species_list: None,
            threads: 0,
        }
    }
}

impl ModelConfig {
    pub fn classifier_path(&self) -> PathBuf {
        self.dir.join(&self.classifier)
    }
    pub fn labels_path(&self) -> PathBuf {
        self.dir.join(&self.labels)
    }
    pub fn meta_model_path(&self) -> Option<PathBuf> {
        self.meta_model.as_ref().map(|m| self.dir.join(m))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClipFormat {
    Wav,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StorageConfig {
    /// SQLite database and clips live here.
    pub data_dir: PathBuf,
    /// BirdNET-Pi `EXTRACTION_LENGTH`; the clip is centred on the 3 s chunk. Minimum 3.
    pub clip_seconds: f32,
    pub clip_format: ClipFormat,
    /// Generate a PNG spectrogram next to each clip.
    pub spectrograms: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("/data"),
            clip_seconds: 6.0,
            clip_format: ClipFormat::Wav,
            spectrograms: true,
        }
    }
}

impl StorageConfig {
    pub fn database_path(&self) -> PathBuf {
        self.data_dir.join("birdsong.sqlite")
    }
    pub fn clips_dir(&self) -> PathBuf {
        self.data_dir.join("clips")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RetentionConfig {
    /// Rolling window: clips older than this are deleted. `0` = no age limit.
    pub clip_max_age_days: u32,
    /// Delete oldest clips while the total exceeds this. `0` = unlimited.
    pub clip_max_total_mb: u64,
    /// Highest-confidence clips per species per day exempt from purging. `0` = none.
    pub keep_best_per_species_per_day: u32,
    /// `0` = keep detection rows forever.
    pub detection_rows_max_age_days: u32,
    pub purge_interval_minutes: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            clip_max_age_days: 14,
            clip_max_total_mb: 4096,
            keep_best_per_species_per_day: 1,
            detection_rows_max_age_days: 0,
            purge_interval_minutes: 30,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    /// Socket address to listen on.
    pub bind: String,
    pub cors_allow_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8080".into(),
            cors_allow_origins: vec!["*".into()],
        }
    }
}

impl ServerConfig {
    pub fn bind_addr(&self) -> Result<SocketAddr, ConfigError> {
        self.bind
            .parse()
            .map_err(|e| ConfigError::Invalid(format!("server.bind {:?}: {e}", self.bind)))
    }
}

impl Config {
    /// Load from an optional TOML file plus `BIRDSONG__*` environment overrides, then validate.
    pub fn load(file: Option<&Path>) -> Result<Self, ConfigError> {
        let mut builder = config::Config::builder();
        if let Some(path) = file {
            builder = builder.add_source(
                config::File::from(path)
                    .format(config::FileFormat::Toml)
                    .required(true),
            );
        }
        builder = builder.add_source(
            config::Environment::with_prefix(ENV_PREFIX)
                .separator("__")
                .try_parsing(true),
        );
        let cfg: Config = builder.build()?.try_deserialize()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse from a TOML string (no environment), then validate. Handy for tests.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let cfg: Config = toml::from_str(text)
            .map_err(|e| ConfigError::Invalid(format!("TOML parse error: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Render as TOML (for `birdsong check-config`).
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_else(|e| format!("# serialisation failed: {e}"))
    }

    /// Check every rule from `BUILD_PLAN.md` §4.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();
        let s = &self.station;
        if !(-90.0..=90.0).contains(&s.latitude) {
            errors.push(format!("station.latitude {} not in -90..=90", s.latitude));
        }
        if !(-180.0..=180.0).contains(&s.longitude) {
            errors.push(format!(
                "station.longitude {} not in -180..=180",
                s.longitude
            ));
        }

        let min_ring = (2.0 * self.storage.clip_seconds).max(30.0);
        if self.audio.ring_buffer_seconds < min_ring {
            errors.push(format!(
                "audio.ring_buffer_seconds {} must be >= {min_ring} (30 s, and twice storage.clip_seconds, \
                 so clips can still be cut after inference and privacy-filter latency)",
                self.audio.ring_buffer_seconds
            ));
        }
        if self.audio.sources.is_empty() {
            errors.push("audio.sources must contain at least one source".into());
        }
        let mut ids = std::collections::HashSet::new();
        for src in &self.audio.sources {
            if src.id.is_empty()
                || !src
                    .id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                errors.push(format!(
                    "audio source id {:?} must be non-empty [A-Za-z0-9_-]",
                    src.id
                ));
            }
            if !ids.insert(&src.id) {
                errors.push(format!("duplicate audio source id {:?}", src.id));
            }
            match src.kind {
                AudioSourceKind::Alsa if src.device.is_none() => errors.push(format!(
                    "audio source {:?}: kind=alsa needs `device`",
                    src.id
                )),
                AudioSourceKind::Rtsp if src.url.is_none() => {
                    errors.push(format!("audio source {:?}: kind=rtsp needs `url`", src.id))
                }
                AudioSourceKind::File if src.path.is_none() => {
                    errors.push(format!("audio source {:?}: kind=file needs `path`", src.id))
                }
                _ => {}
            }
        }

        let d = &self.detection;
        if !(0.0..1.0).contains(&d.min_confidence) {
            errors.push(format!(
                "detection.min_confidence {} not in 0..1",
                d.min_confidence
            ));
        }
        if !(0.5..=1.5).contains(&d.sensitivity) {
            errors.push(format!(
                "detection.sensitivity {} not in 0.5..=1.5",
                d.sensitivity
            ));
        }
        if !(0.0..3.0).contains(&d.overlap_seconds) {
            errors.push(format!(
                "detection.overlap_seconds {} not in 0..3",
                d.overlap_seconds
            ));
        }
        if !(0.0..=1.0).contains(&d.species_filter_threshold) {
            errors.push(format!(
                "detection.species_filter_threshold {} not in 0..=1",
                d.species_filter_threshold
            ));
        }
        if d.top_n_per_chunk == 0 {
            errors.push("detection.top_n_per_chunk must be at least 1".into());
        }
        if !(0.0..=100.0).contains(&d.privacy_threshold) {
            errors.push(format!(
                "detection.privacy_threshold {} not in 0..=100",
                d.privacy_threshold
            ));
        }

        if self.model.classifier.is_empty() || self.model.labels.is_empty() {
            errors.push("model.classifier and model.labels must be set".into());
        }

        if self.storage.clip_seconds < crate::CHUNK_SECONDS {
            errors.push(format!(
                "storage.clip_seconds {} must be >= 3",
                self.storage.clip_seconds
            ));
        }

        if self.retention.purge_interval_minutes == 0 {
            errors.push("retention.purge_interval_minutes must be at least 1".into());
        }

        if let Err(e) = self.server.bind_addr() {
            errors.push(e.to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(errors.join("; ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
[[audio.sources]]
id = "mic0"
kind = "alsa"
device = "hw:1,0"
"#;

    fn example_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../config/birdsong.example.toml")
    }

    #[test]
    fn minimal_file_gets_defaults() {
        let cfg = Config::from_toml(MINIMAL).unwrap();
        assert_eq!(cfg.detection.min_confidence, 0.7);
        assert_eq!(cfg.detection.sensitivity, 1.25);
        assert_eq!(cfg.storage.clip_seconds, 6.0);
        assert_eq!(cfg.retention.clip_max_age_days, 14);
        assert_eq!(cfg.station.timezone, Tz::UTC);
        assert!(!cfg.station.has_location());
        assert_eq!(cfg.server.bind_addr().unwrap().port(), 8080);
    }

    #[test]
    fn example_file_round_trips() {
        let text = std::fs::read_to_string(example_path()).expect("config/birdsong.example.toml");
        let cfg = Config::from_toml(&text).unwrap();
        assert_eq!(cfg.station.timezone, chrono_tz::America::New_York);
        assert_eq!(cfg.audio.sources.len(), 1);
        assert_eq!(cfg.audio.sources[0].kind, AudioSourceKind::Alsa);
        let again = Config::from_toml(&cfg.to_toml()).unwrap();
        assert_eq!(cfg, again);
    }

    #[test]
    fn env_overrides_file() {
        let dir = std::env::temp_dir().join(format!("birdsong-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("birdsong.toml");
        std::fs::write(&path, MINIMAL).unwrap();
        // Process-wide env var: keep the key unique to this test to avoid interference.
        std::env::set_var("BIRDSONG__DETECTION__MIN_CONFIDENCE", "0.85");
        std::env::set_var("BIRDSONG__STATION__NAME", "Roof");
        let cfg = Config::load(Some(&path)).unwrap();
        std::env::remove_var("BIRDSONG__DETECTION__MIN_CONFIDENCE");
        std::env::remove_var("BIRDSONG__STATION__NAME");
        assert_eq!(cfg.detection.min_confidence, 0.85);
        assert_eq!(cfg.station.name, "Roof");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_file_is_an_error() {
        let err = Config::load(Some(Path::new("/nonexistent/birdsong.toml"))).unwrap_err();
        assert!(matches!(err, ConfigError::Load(_)), "{err}");
    }

    #[test]
    fn validation_rejects_bad_values() {
        let bad = |extra: &str| {
            let text = format!("{MINIMAL}\n{extra}");
            Config::from_toml(&text).unwrap_err().to_string()
        };
        assert!(bad("[detection]\nsensitivity = 2.0").contains("sensitivity"));
        assert!(bad("[detection]\noverlap_seconds = 3.0").contains("overlap"));
        assert!(bad("[detection]\nmin_confidence = 1.0").contains("min_confidence"));
        assert!(bad("[detection]\ntop_n_per_chunk = 0").contains("top_n"));
        assert!(bad("[station]\nlatitude = 91.0").contains("latitude"));
        assert!(bad("[storage]\nclip_seconds = 2.0").contains("clip_seconds"));
        assert!(bad("[server]\nbind = \"nope\"").contains("server.bind"));
        assert!(bad("[detection]\nunknown_key = 1").contains("unknown"));
        assert!(bad("[audio]\nring_buffer_seconds = 10.0").contains("ring_buffer_seconds"));
        assert!(bad("[[audio.sources]]\nid = \"mic0\"\nkind = \"rtsp\"").contains("duplicate"));
        assert!(bad("[[audio.sources]]\nid = \"cam\"\nkind = \"rtsp\"").contains("needs `url`"));
    }

    #[test]
    fn bad_timezone_is_rejected() {
        let text = format!("{MINIMAL}\n[station]\ntimezone = \"Mars/Olympus\"");
        assert!(Config::from_toml(&text).is_err());
    }

    #[test]
    fn no_sources_is_rejected() {
        let err = Config::from_toml("[station]\nname = \"x\"")
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least one source"), "{err}");
    }
}
