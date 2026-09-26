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
    pub birdweather: BirdWeatherConfig,
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
    /// ALSA device (kind = alsa). Prefer the stable form `"plughw:CARD=Device,DEV=0"`
    /// (`birdsong devices` lists them): card numbers such as `"hw:1,0"` can change between boots.
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
    /// Detections of a species needed within `confirmation_window_seconds` before any are
    /// stored. `1` stores every detection immediately (BirdNET-Pi behaviour).
    pub min_detections: usize,
    /// How long detections of one species count towards `min_detections`.
    pub confirmation_window_seconds: f32,
    /// Scientific names stored without waiting for confirmation: species that call rarely (owls,
    /// loons) would otherwise be dropped by `min_detections`.
    pub confirmation_exempt_species: Vec<String>,
    /// Lower a species' threshold for a while after it has been heard clearly.
    pub dynamic_threshold: bool,
    /// Confidence that counts as "heard clearly".
    pub dynamic_threshold_trigger: f32,
    /// Lowest threshold a species can reach.
    pub dynamic_threshold_min: f32,
    /// How long the lowered threshold lasts, in hours.
    pub dynamic_threshold_hours: u32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            sensitivity: 1.25,
            overlap_seconds: 0.0,
            species_filter_threshold: 0.03,
            top_n_per_chunk: 3,
            include_species: Vec::new(),
            exclude_species: Vec::new(),
            privacy_filter: true,
            privacy_threshold: 0.0,
            min_detections: 2,
            confirmation_window_seconds: 30.0,
            confirmation_exempt_species: Vec::new(),
            dynamic_threshold: false,
            dynamic_threshold_trigger: 0.9,
            dynamic_threshold_min: 0.2,
            dynamic_threshold_hours: 24,
        }
    }
}

/// Which classifier to run. Choosing one also chooses its model files and detection defaults
/// (see [`ModelKind::defaults`]); anything set explicitly in the file still wins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    /// BirdNET V2.4: 3 s windows, sigmoid confidences, BirdNET's location filter.
    #[serde(rename = "birdnet-v2.4")]
    BirdnetV24,
    /// Google Perch v2: 5 s windows, softmax confidences, birds and other animals, filtered by
    /// the BirdNET Geomodel. BirdWeather receives its birds only.
    #[default]
    #[serde(rename = "perch-v2")]
    PerchV2,
}

/// A per-model default value, applied to keys the configuration leaves out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KindDefault {
    Text(&'static str),
    Number(f64),
    Count(i64),
}

/// Perch v2, full model (`scripts/fetch-perch.sh full`).
pub const PERCH_CLASSIFIER: &str = "perch/perch_v2_no_dft_fp32.onnx";
pub const PERCH_LABELS: &str = "perch/perch_v2_labels.txt";
/// BirdNET Geomodel v3.0.4 (`scripts/fetch-geomodel.sh`).
pub const GEOMODEL: &str = "geomodel/BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.onnx";
pub const GEOMODEL_LABELS: &str = "geomodel/BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt";

impl ModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BirdnetV24 => "birdnet-v2.4",
            Self::PerchV2 => "perch-v2",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "birdnet-v2.4" => Some(Self::BirdnetV24),
            "perch-v2" => Some(Self::PerchV2),
            _ => None,
        }
    }

    /// Model files and detection settings that go with this model, used for every key the
    /// configuration does not set. An empty string turns an optional file off.
    pub fn defaults(self) -> &'static [(&'static str, KindDefault)] {
        use KindDefault::{Count, Number, Text};
        match self {
            Self::BirdnetV24 => &[
                ("model.classifier", Text("birdnet-v2.4-headless.onnx")),
                ("model.labels", Text("labels/en_us.txt")),
                ("model.meta_model", Text("meta-model.onnx")),
                ("model.meta_model_labels", Text("")),
                ("detection.min_confidence", Number(0.7)),
                ("detection.min_detections", Count(1)),
                ("detection.confirmation_window_seconds", Number(15.0)),
            ],
            // Thresholds from comparing Perch with a BirdNET-Pi station (docs/MODEL.md).
            Self::PerchV2 => &[
                ("model.classifier", Text(PERCH_CLASSIFIER)),
                ("model.labels", Text(PERCH_LABELS)),
                ("model.meta_model", Text(GEOMODEL)),
                ("model.meta_model_labels", Text(GEOMODEL_LABELS)),
                ("detection.min_confidence", Number(0.5)),
                ("detection.min_detections", Count(2)),
                ("detection.confirmation_window_seconds", Number(30.0)),
            ],
        }
    }
}

/// Insert `kind`'s defaults into a TOML document for keys it does not set.
fn apply_kind_defaults(table: &mut toml::Table, kind: ModelKind) {
    // A location model chosen explicitly does not inherit the default model's label file.
    if let Some(toml::Value::Table(model)) = table.get_mut("model") {
        if model.contains_key("meta_model") {
            model
                .entry("meta_model_labels")
                .or_insert_with(|| toml::Value::String(String::new()));
        }
    }
    for (key, value) in kind.defaults() {
        let Some((section, field)) = key.split_once('.') else {
            continue;
        };
        let section = table
            .entry(section)
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        if let toml::Value::Table(section) = section {
            section.entry(field).or_insert_with(|| match *value {
                KindDefault::Text(s) => toml::Value::String(s.to_string()),
                KindDefault::Number(n) => toml::Value::Float(n),
                KindDefault::Count(n) => toml::Value::Integer(n),
            });
        }
    }
}

fn kind_of(table: &toml::Table) -> Result<ModelKind, ConfigError> {
    match table.get("model").and_then(|m| m.get("kind")) {
        None => Ok(ModelKind::default()),
        Some(value) => value.as_str().and_then(ModelKind::parse).ok_or_else(|| {
            ConfigError::Invalid(format!(
                "model.kind must be \"birdnet-v2.4\" or \"perch-v2\", got {value}"
            ))
        }),
    }
}

/// What the location filter does with Perch species that BirdNET's location model does not know.
///
/// Serialised as `"allow"` / `"block"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnmappedSpecies {
    /// Report them (amphibians, insects and mammals outside BirdNET's list stay detectable).
    #[default]
    Allow,
    /// Never report them.
    Block,
}

impl UnmappedSpecies {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Block => "block",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    pub dir: PathBuf,
    pub kind: ModelKind,
    /// Classifier ONNX, relative to `dir` (see `docs/MODEL.md`).
    pub classifier: String,
    /// Labels file, relative to `dir`.
    pub labels: String,
    /// Perch only: a BirdNET labels file (relative to `dir`) to take common names from. Also needed
    /// for the location filter, which maps Perch species to BirdNET's by scientific name.
    pub common_names: Option<String>,
    /// Perch only: species the location model cannot score (not in BirdNET's labels).
    pub location_filter_unmapped: UnmappedSpecies,
    /// Location/week model ONNX, relative to `dir`. `None` disables the location filter.
    pub meta_model: Option<String>,
    /// Species of the location model's outputs, relative to `dir`, for location models other than
    /// BirdNET's own (such as the BirdNET Geomodel). Classes are then matched by scientific name.
    pub meta_model_labels: Option<String>,
    /// Precomputed allowed-species list; when set, `meta_model` is ignored.
    pub species_list: Option<PathBuf>,
    /// Inference threads; `0` = number of CPUs minus one, minimum one.
    pub threads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("/models"),
            kind: ModelKind::PerchV2,
            classifier: PERCH_CLASSIFIER.into(),
            labels: PERCH_LABELS.into(),
            common_names: None,
            location_filter_unmapped: UnmappedSpecies::Allow,
            meta_model: Some(GEOMODEL.into()),
            meta_model_labels: Some(GEOMODEL_LABELS.into()),
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
        self.meta_model
            .as_deref()
            .filter(|m| !m.is_empty())
            .map(|m| self.dir.join(m))
    }
    pub fn meta_model_labels_path(&self) -> Option<PathBuf> {
        self.meta_model_labels
            .as_deref()
            .filter(|m| !m.is_empty())
            .map(|m| self.dir.join(m))
    }
    pub fn common_names_path(&self) -> Option<PathBuf> {
        self.common_names
            .as_deref()
            .filter(|m| !m.is_empty())
            .map(|m| self.dir.join(m))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClipFormat {
    /// Lossless and about half the size of WAV. BirdWeather uploads are always FLAC.
    Flac,
    /// Uncompressed 16-bit PCM.
    Wav,
}

impl ClipFormat {
    /// File extension without the dot.
    pub fn extension(&self) -> &'static str {
        match self {
            ClipFormat::Flac => "flac",
            ClipFormat::Wav => "wav",
        }
    }
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
            clip_format: ClipFormat::Flac,
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

/// Uploads to BirdWeather. Disabled while `token` is empty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BirdWeatherConfig {
    /// Station token from the BirdWeather app (keep it secret; `BIRDSONG__BIRDWEATHER__TOKEN` works).
    pub token: String,
    /// API base URL. Only change it for testing.
    pub api_url: String,
}

impl Default for BirdWeatherConfig {
    fn default() -> Self {
        Self {
            token: String::new(),
            api_url: "https://app.birdweather.com/api/v1".into(),
        }
    }
}

impl BirdWeatherConfig {
    pub fn enabled(&self) -> bool {
        !self.token.trim().is_empty()
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
        // The chosen model decides the defaults for its files and thresholds, so read it first.
        let probe = builder.build_cloned()?;
        let kind = match probe.get_string("model.kind") {
            Ok(value) => ModelKind::parse(&value).ok_or_else(|| {
                ConfigError::Invalid(format!(
                    "model.kind must be \"birdnet-v2.4\" or \"perch-v2\", got {value:?}"
                ))
            })?,
            Err(config::ConfigError::NotFound(_)) => ModelKind::default(),
            Err(e) => return Err(e.into()),
        };
        // A location model chosen explicitly does not inherit the default model's label file.
        if probe.get_string("model.meta_model").is_ok()
            && probe.get_string("model.meta_model_labels").is_err()
        {
            builder = builder.set_override("model.meta_model_labels", "")?;
        }
        for (key, value) in kind.defaults() {
            builder = match *value {
                KindDefault::Text(s) => builder.set_default(*key, s)?,
                KindDefault::Number(n) => builder.set_default(*key, n)?,
                KindDefault::Count(n) => builder.set_default(*key, n)?,
            };
        }
        let cfg: Config = builder.build()?.try_deserialize()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse from a TOML string (no environment), then validate. Handy for tests.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let mut table: toml::Table = text
            .parse()
            .map_err(|e| ConfigError::Invalid(format!("TOML parse error: {e}")))?;
        let kind = kind_of(&table)?;
        apply_kind_defaults(&mut table, kind);
        let cfg: Config = toml::Value::Table(table)
            .try_into()
            .map_err(|e| ConfigError::Invalid(format!("TOML parse error: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Every default for `kind`, with no file (the offline tools without `--config`).
    pub fn default_for(kind: ModelKind) -> Self {
        let mut table = toml::Table::new();
        let mut model = toml::Table::new();
        model.insert("kind".into(), toml::Value::String(kind.as_str().into()));
        table.insert("model".into(), toml::Value::Table(model));
        apply_kind_defaults(&mut table, kind);
        toml::Value::Table(table)
            .try_into()
            .unwrap_or_else(|_| Self::default())
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

        if !(1..=20).contains(&d.min_detections) {
            errors.push(format!(
                "detection.min_detections {} not in 1..=20",
                d.min_detections
            ));
        }
        if !(crate::CHUNK_SECONDS..=600.0).contains(&d.confirmation_window_seconds) {
            errors.push(format!(
                "detection.confirmation_window_seconds {} not in 3..=600",
                d.confirmation_window_seconds
            ));
        }
        if d.min_detections > 1
            && self.audio.ring_buffer_seconds
                < d.confirmation_window_seconds + self.storage.clip_seconds
        {
            errors.push(format!(
                "audio.ring_buffer_seconds {} must be at least detection.confirmation_window_seconds + \
                 storage.clip_seconds ({}) so clips can still be cut for confirmed detections",
                self.audio.ring_buffer_seconds,
                d.confirmation_window_seconds + self.storage.clip_seconds
            ));
        }
        if !(0.0..=1.0).contains(&d.dynamic_threshold_trigger) {
            errors.push(format!(
                "detection.dynamic_threshold_trigger {} not in 0..=1",
                d.dynamic_threshold_trigger
            ));
        }
        if !(0.0..=1.0).contains(&d.dynamic_threshold_min) {
            errors.push(format!(
                "detection.dynamic_threshold_min {} not in 0..=1",
                d.dynamic_threshold_min
            ));
        }
        if d.dynamic_threshold && d.dynamic_threshold_trigger <= d.min_confidence {
            errors.push(format!(
                "detection.dynamic_threshold_trigger {} must be above detection.min_confidence {}",
                d.dynamic_threshold_trigger, d.min_confidence
            ));
        }
        if d.dynamic_threshold_hours == 0 {
            errors.push("detection.dynamic_threshold_hours must be at least 1".into());
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

        let bw = &self.birdweather;
        if bw.enabled() {
            if !bw
                .token
                .trim()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                errors
                    .push("birdweather.token may only contain letters, digits, '-' and '_'".into());
            }
            if !self.station.has_location() {
                errors
                    .push("birdweather.token is set but station.latitude/longitude are not".into());
            }
        }
        if !(bw.api_url.starts_with("https://") || bw.api_url.starts_with("http://")) {
            errors.push(format!(
                "birdweather.api_url {:?} must start with https:// or http://",
                bw.api_url
            ));
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
    fn the_model_kind_chooses_file_and_threshold_defaults() {
        let birdnet =
            Config::from_toml(&format!("{MINIMAL}\n[model]\nkind = \"birdnet-v2.4\"")).unwrap();
        assert_eq!(birdnet.model.classifier, "birdnet-v2.4-headless.onnx");
        assert_eq!(birdnet.model.labels, "labels/en_us.txt");
        assert_eq!(
            birdnet.model.meta_model_path(),
            Some(PathBuf::from("/models/meta-model.onnx"))
        );
        assert_eq!(birdnet.model.meta_model_labels_path(), None);
        assert_eq!(
            (
                birdnet.detection.min_confidence,
                birdnet.detection.min_detections
            ),
            (0.7, 1)
        );

        // Explicit settings win over the model's defaults.
        let tuned = Config::from_toml(&format!(
            "{MINIMAL}\n[model]\nkind = \"perch-v2\"\nmeta_model = \"\"\n[detection]\nmin_confidence = 0.4"
        ))
        .unwrap();
        assert_eq!(tuned.detection.min_confidence, 0.4);
        assert_eq!(tuned.detection.min_detections, 2);
        assert_eq!(
            tuned.model.meta_model_path(),
            None,
            "an empty path turns the file off"
        );
        let own_location_model = Config::from_toml(&format!(
            "{MINIMAL}\n[model]\nkind = \"perch-v2\"\nmeta_model = \"meta-model.onnx\""
        ))
        .unwrap();
        assert_eq!(
            own_location_model.model.meta_model_labels_path(),
            None,
            "an explicit location model does not inherit the Geomodel's labels"
        );
        assert!(
            Config::from_toml(&format!("{MINIMAL}\n[model]\nkind = \"yamnet\""))
                .unwrap_err()
                .to_string()
                .contains("model.kind")
        );

        let tools = Config::default_for(ModelKind::BirdnetV24);
        assert_eq!(tools.model.classifier, "birdnet-v2.4-headless.onnx");
        assert_eq!(tools.detection.min_confidence, 0.7);
        assert_eq!(
            Config::default_for(ModelKind::PerchV2).model.classifier,
            PERCH_CLASSIFIER
        );
    }

    #[test]
    fn loading_a_file_applies_the_model_defaults() {
        let dir = std::env::temp_dir().join(format!("birdsong-kind-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("birdsong.toml");
        std::fs::write(
            &path,
            format!("{MINIMAL}\n[model]\nkind = \"birdnet-v2.4\"\n[detection]\nmin_detections = 3"),
        )
        .unwrap();
        let cfg = Config::load(Some(&path)).unwrap();
        // (min_confidence is not checked: another test sets it through the environment.)
        assert_eq!(cfg.model.classifier, "birdnet-v2.4-headless.onnx");
        assert_eq!(cfg.model.meta_model_labels_path(), None);
        assert_eq!(cfg.detection.min_detections, 3);
        std::fs::write(&path, MINIMAL).unwrap();
        let perch = Config::load(Some(&path)).unwrap();
        assert_eq!(perch.model.kind, ModelKind::PerchV2);
        assert_eq!(perch.detection.confirmation_window_seconds, 30.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn minimal_file_gets_defaults() {
        let cfg = Config::from_toml(MINIMAL).unwrap();
        assert_eq!(
            cfg.model.kind,
            ModelKind::PerchV2,
            "Perch is the default model"
        );
        assert_eq!(cfg.model.classifier, PERCH_CLASSIFIER);
        assert_eq!(
            cfg.model.meta_model_labels_path(),
            Some(PathBuf::from("/models").join(GEOMODEL_LABELS))
        );
        assert_eq!(cfg.detection.min_confidence, 0.5);
        assert_eq!(cfg.detection.min_detections, 2);
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
    fn clip_format_defaults_to_flac_and_accepts_wav() {
        assert_eq!(
            Config::from_toml(MINIMAL).unwrap().storage.clip_format,
            ClipFormat::Flac
        );
        let wav =
            Config::from_toml(&format!("{MINIMAL}\n[storage]\nclip_format = \"wav\"")).unwrap();
        assert_eq!(wav.storage.clip_format.extension(), "wav");
        assert!(
            Config::from_toml(&format!("{MINIMAL}\n[storage]\nclip_format = \"mp3\"")).is_err()
        );
    }

    #[test]
    fn birdweather_needs_a_location_and_a_clean_token() {
        let cfg = Config::from_toml(MINIMAL).unwrap();
        assert!(!cfg.birdweather.enabled());
        assert_eq!(cfg.model.kind, ModelKind::PerchV2);
        let located = format!("{MINIMAL}\n[station]\nlatitude = 42.36\nlongitude = -71.06");
        let ok =
            Config::from_toml(&format!("{located}\n[birdweather]\ntoken = \"abc_123-X\"")).unwrap();
        assert!(ok.birdweather.enabled());
        let err = Config::from_toml(&format!("{MINIMAL}\n[birdweather]\ntoken = \"abc\""))
            .unwrap_err()
            .to_string();
        assert!(err.contains("station.latitude"), "{err}");
        let err = Config::from_toml(&format!("{located}\n[birdweather]\ntoken = \"a/b\""))
            .unwrap_err()
            .to_string();
        assert!(err.contains("birdweather.token"), "{err}");
    }

    #[test]
    fn detection_quality_options_are_off_by_default_and_validated() {
        let cfg =
            Config::from_toml(&format!("{MINIMAL}\n[model]\nkind = \"birdnet-v2.4\"")).unwrap();
        assert_eq!(cfg.detection.min_detections, 1, "off for BirdNET");
        assert!(!cfg.detection.dynamic_threshold);

        let bad = |extra: &str| {
            Config::from_toml(&format!("{MINIMAL}\n{extra}"))
                .unwrap_err()
                .to_string()
        };
        assert!(bad("[detection]\nmin_detections = 0").contains("min_detections"));
        assert!(
            bad("[detection]\nconfirmation_window_seconds = 1.0").contains("confirmation_window")
        );
        assert!(
            bad("[detection]\ndynamic_threshold = true\ndynamic_threshold_trigger = 0.5")
                .contains("must be above detection.min_confidence")
        );
        // Repeat confirmation needs enough buffered audio to still cut a clip afterwards.
        assert!(
            bad("[detection]\nmin_detections = 2\nconfirmation_window_seconds = 90.0")
                .contains("ring_buffer_seconds")
        );
        let ok = Config::from_toml(&format!(
            "{MINIMAL}\n[detection]\nmin_detections = 2\n[audio]\nring_buffer_seconds = 90.0"
        ))
        .unwrap();
        assert_eq!(ok.detection.min_detections, 2);
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
