//! Uploads to BirdWeather (<https://app.birdweather.com>), matching BirdNET-Pi: for every saved
//! clip, POST the audio as a FLAC soundscape, then POST each detection of that window pointing at
//! the soundscape id.
//!
//! ```text
//! POST {api}/stations/{token}/soundscapes?timestamp=<RFC 3339>&type=flac   (FLAC body) -> {"success":true,"soundscape":{"id":N}}
//! POST {api}/stations/{token}/detections                                    (JSON body)
//! ```
//!
//! Shutdown: once the pipeline's cancellation token fires, queued uploads are skipped, retry waits
//! end early, and an upload in progress stops before its next request. A request already on the
//! wire is bounded by the client timeouts (5 s to connect, 15 s in total).

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use birdsong_core::SAMPLE_RATE_HZ;
use chrono::{DateTime, SecondsFormat, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::stats::PipelineStats;

/// BirdWeather's identifier for BirdNET V2.4 detections.
pub const ALGORITHM_V24: &str = "2p4";
/// Connection timeout for BirdWeather requests.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Total time allowed for one BirdWeather request, including the upload body and response.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Why an upload did not succeed.
#[derive(Debug)]
pub enum UploadError {
    /// Connection, TLS or timeout problem.
    Transport(String),
    /// A non-2xx response.
    Status { status: u16, body: String },
    /// A 2xx response whose body was not the expected JSON, or `success: false`.
    Rejected(String),
    /// The clip could not be encoded as FLAC.
    Encode(String),
    /// Shutdown started before the upload finished.
    Cancelled,
}

impl fmt::Display for UploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(m) => write!(f, "network error: {m}"),
            Self::Status { status, body } => write!(f, "HTTP {status}: {body}"),
            Self::Rejected(m) => write!(f, "rejected: {m}"),
            Self::Encode(m) => write!(f, "FLAC encoding: {m}"),
            Self::Cancelled => write!(f, "cancelled by shutdown"),
        }
    }
}

impl std::error::Error for UploadError {}

impl UploadError {
    /// Worth trying again: network trouble, rate limiting, or a server-side error.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::Status { status, .. } => *status == 429 || *status >= 500,
            _ => false,
        }
    }
}

/// One detection as BirdWeather expects it (same fields as BirdNET-Pi sends).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionUpload {
    /// Start of the 3 s window, station time zone, RFC 3339 with milliseconds.
    pub timestamp: String,
    pub lat: f64,
    pub lon: f64,
    pub soundscape_id: i64,
    /// Seconds from the start of the soundscape to the start of the detection window.
    pub soundscape_start_time: f64,
    pub soundscape_end_time: f64,
    pub common_name: String,
    pub scientific_name: String,
    /// The model that made the detection; left out for models BirdWeather has no code for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    pub confidence: f32,
}

/// What a station sends: BirdNET V2.4 detections as algorithm `2p4`; other models' detections
/// restricted to birds, with no algorithm claimed.
#[derive(Clone, Debug, Default)]
pub struct UploadPolicy {
    /// `algorithm` sent with each detection; `None` leaves the field out.
    pub algorithm: Option<String>,
    /// When set, only these species (scientific names) are uploaded.
    pub birds: Option<Arc<HashSet<String>>>,
}

impl UploadPolicy {
    pub fn birdnet_v24() -> Self {
        Self {
            algorithm: Some(ALGORITHM_V24.into()),
            birds: None,
        }
    }

    fn allows(&self, scientific: &str) -> bool {
        self.birds.as_ref().is_none_or(|b| b.contains(scientific))
    }
}

#[derive(Deserialize)]
struct SoundscapeResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    message: Option<String>,
    soundscape: Option<SoundscapeId>,
}

#[derive(Deserialize)]
struct SoundscapeId {
    id: i64,
}

/// Percent-encode a query value (everything except unreserved characters).
pub fn encode_query_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// RFC 3339 in the station time zone with milliseconds, e.g. `2026-05-15T06:00:00.000-04:00`.
pub fn station_timestamp(at: DateTime<Utc>, tz: Tz) -> String {
    at.with_timezone(&tz)
        .to_rfc3339_opts(SecondsFormat::Millis, false)
}

/// Sleep for `duration`, waking early if `cancel` fires. Returns false when cancelled.
fn sleep_unless_cancelled(duration: Duration, cancel: &CancellationToken) -> bool {
    let until = Instant::now() + duration;
    while Instant::now() < until {
        if cancel.is_cancelled() {
            return false;
        }
        std::thread::sleep((until - Instant::now()).min(Duration::from_millis(50)));
    }
    !cancel.is_cancelled()
}

/// Blocking HTTP client for the two BirdWeather endpoints.
pub struct BirdWeatherClient {
    agent: ureq::Agent,
    api_url: String,
    token: String,
    /// Waits before the second and third attempt of a transient failure.
    pub retry_delays: Vec<Duration>,
}

impl BirdWeatherClient {
    pub fn new(api_url: &str, token: &str) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_global(Some(REQUEST_TIMEOUT))
            .http_status_as_error(false)
            .user_agent(concat!("birdsong/", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        Self {
            agent,
            api_url: api_url.trim_end_matches('/').to_string(),
            token: token.trim().to_string(),
            retry_delays: vec![Duration::from_secs(2), Duration::from_secs(10)],
        }
    }

    fn station_url(&self, rest: &str) -> String {
        format!("{}/stations/{}/{rest}", self.api_url, self.token)
    }

    fn with_retries<T>(
        &self,
        what: &str,
        cancel: &CancellationToken,
        mut call: impl FnMut() -> Result<T, UploadError>,
    ) -> Result<T, UploadError> {
        let mut delays = self.retry_delays.iter();
        loop {
            if cancel.is_cancelled() {
                return Err(UploadError::Cancelled);
            }
            match call() {
                Err(e) if e.is_transient() => match delays.next() {
                    Some(delay) => {
                        tracing::warn!(error = %e, retry_in = ?delay, "BirdWeather {what} failed; retrying");
                        if !sleep_unless_cancelled(*delay, cancel) {
                            return Err(UploadError::Cancelled);
                        }
                    }
                    None => return Err(e),
                },
                other => return other,
            }
        }
    }

    fn read_response(
        mut response: ureq::http::Response<ureq::Body>,
    ) -> Result<String, UploadError> {
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .with_config()
            .limit(64 * 1024)
            .read_to_string()
            .unwrap_or_default();
        if (200..300).contains(&status) {
            Ok(body)
        } else {
            Err(UploadError::Status {
                status,
                body: body.chars().take(300).collect(),
            })
        }
    }

    /// Upload FLAC audio; returns the soundscape id.
    pub fn upload_soundscape(
        &self,
        flac: &[u8],
        timestamp: &str,
        cancel: &CancellationToken,
    ) -> Result<i64, UploadError> {
        let url = self.station_url(&format!(
            "soundscapes?timestamp={}&type=flac",
            encode_query_value(timestamp)
        ));
        self.with_retries("soundscape upload", cancel, || {
            let response = self
                .agent
                .post(&url)
                .header("Content-Type", "audio/flac")
                .send(flac)
                .map_err(|e| UploadError::Transport(e.to_string()))?;
            let body = Self::read_response(response)?;
            let parsed: SoundscapeResponse = serde_json::from_str(&body)
                .map_err(|e| UploadError::Rejected(format!("unexpected response {body:?}: {e}")))?;
            match (parsed.success, parsed.soundscape) {
                (true, Some(s)) => Ok(s.id),
                _ => Err(UploadError::Rejected(
                    parsed
                        .message
                        .unwrap_or_else(|| format!("unexpected response {body:?}")),
                )),
            }
        })
    }

    /// Post one detection that refers to an uploaded soundscape.
    pub fn post_detection(
        &self,
        detection: &DetectionUpload,
        cancel: &CancellationToken,
    ) -> Result<(), UploadError> {
        let url = self.station_url("detections");
        self.with_retries("detection upload", cancel, || {
            let response = self
                .agent
                .post(&url)
                .send_json(detection)
                .map_err(|e| UploadError::Transport(e.to_string()))?;
            Self::read_response(response).map(|_| ())
        })
    }
}

/// Everything needed to upload one saved clip and its detections.
#[derive(Clone, Debug)]
pub struct UploadJob {
    pub clip_samples: Vec<f32>,
    pub clip_start_at: DateTime<Utc>,
    pub chunk_start_at: DateTime<Utc>,
    /// Length of the analysis window inside the clip.
    pub window_seconds: f32,
    /// (scientific name, common name, confidence) for every detection of the window.
    pub detections: Vec<(String, String, f32)>,
}

/// Station details sent with each detection.
#[derive(Clone, Copy, Debug)]
pub struct Station {
    pub latitude: f64,
    pub longitude: f64,
    pub timezone: Tz,
}

/// Outcome of uploading one window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowResult {
    pub detections_posted: usize,
    /// Detections BirdWeather refused with 422, typically species it does not accept.
    pub detections_refused: usize,
    /// Detections not sent because the policy only uploads birds.
    pub not_birds: usize,
}

/// Upload one clip as a soundscape, then its detections. Blocking; stops early on `cancel`.
pub fn upload_window(
    client: &BirdWeatherClient,
    station: Station,
    job: &UploadJob,
    policy: &UploadPolicy,
    cancel: &CancellationToken,
) -> Result<WindowResult, UploadError> {
    let mut result = WindowResult::default();
    let detections: Vec<_> = job
        .detections
        .iter()
        .filter(|(scientific, _, _)| policy.allows(scientific))
        .collect();
    result.not_birds = job.detections.len() - detections.len();
    if detections.is_empty() {
        return Ok(result); // nothing to report: do not upload the soundscape either
    }
    let flac = birdsong_audio::flac::encode_flac(&job.clip_samples, SAMPLE_RATE_HZ)
        .map_err(|e| UploadError::Encode(e.to_string()))?;
    let soundscape_id = client.upload_soundscape(
        &flac,
        &station_timestamp(job.clip_start_at, station.timezone),
        cancel,
    )?;

    let clip_seconds = job.clip_samples.len() as f64 / f64::from(SAMPLE_RATE_HZ);
    let start = ((job.chunk_start_at - job.clip_start_at).num_milliseconds() as f64 / 1000.0)
        .clamp(0.0, clip_seconds);
    let end = (start + f64::from(job.window_seconds)).min(clip_seconds);

    for (scientific, common, confidence) in detections {
        let upload = DetectionUpload {
            timestamp: station_timestamp(job.chunk_start_at, station.timezone),
            lat: station.latitude,
            lon: station.longitude,
            soundscape_id,
            soundscape_start_time: start,
            soundscape_end_time: end,
            common_name: common.clone(),
            scientific_name: scientific.clone(),
            algorithm: policy.algorithm.clone(),
            confidence: *confidence,
        };
        match client.post_detection(&upload, cancel) {
            Ok(()) => result.detections_posted += 1,
            Err(UploadError::Status { status: 422, body }) => {
                tracing::debug!(species = %scientific, %body, "BirdWeather refused the detection");
                result.detections_refused += 1;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(result)
}

/// Background uploader. Jobs arrive from the clip writer through a bounded channel; the clip
/// writer never waits for the network. After `cancel` fires, remaining jobs are skipped.
pub async fn upload_task(
    client: Arc<BirdWeatherClient>,
    station: Station,
    mut jobs: mpsc::Receiver<UploadJob>,
    stats: Arc<PipelineStats>,
    cancel: CancellationToken,
    policy: UploadPolicy,
) {
    let policy = Arc::new(policy);
    let mut skipped = 0u64;
    while let Some(job) = jobs.recv().await {
        if cancel.is_cancelled() {
            skipped += 1;
            stats.birdweather_skipped();
            continue;
        }
        let client = Arc::clone(&client);
        let token = cancel.clone();
        let window_policy = Arc::clone(&policy);
        let outcome = tokio::task::spawn_blocking(move || {
            upload_window(&client, station, &job, &window_policy, &token)
        })
        .await;
        match outcome {
            Ok(Ok(result)) if result.detections_posted + result.detections_refused == 0 => {
                tracing::debug!(
                    not_birds = result.not_birds,
                    "nothing for BirdWeather in this clip"
                );
            }
            Ok(Ok(result)) => {
                stats.birdweather_uploaded(result.detections_posted as u64);
                tracing::info!(
                    posted = result.detections_posted,
                    refused = result.detections_refused,
                    not_birds = result.not_birds,
                    "uploaded to BirdWeather"
                );
            }
            Ok(Err(UploadError::Cancelled)) => {
                skipped += 1;
                stats.birdweather_skipped();
            }
            Ok(Err(e)) => {
                stats.birdweather_error();
                tracing::warn!(error = %e, "BirdWeather upload failed");
            }
            Err(e) => {
                stats.birdweather_error();
                tracing::error!(error = %e, "BirdWeather upload task panicked");
            }
        }
    }
    if skipped > 0 {
        tracing::info!(skipped, "BirdWeather uploads skipped because of shutdown");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn query_values_are_percent_encoded() {
        assert_eq!(
            encode_query_value("2026-05-15T06:00:00.000-04:00"),
            "2026-05-15T06%3A00%3A00.000-04%3A00"
        );
        assert_eq!(encode_query_value("+02:00 x"), "%2B02%3A00%20x");
    }

    #[test]
    fn timestamps_use_the_station_time_zone() {
        let at = Utc.with_ymd_and_hms(2026, 5, 15, 10, 0, 0).unwrap();
        assert_eq!(
            station_timestamp(at, chrono_tz::America::New_York),
            "2026-05-15T06:00:00.000-04:00"
        );
        assert_eq!(
            station_timestamp(at, chrono_tz::UTC),
            "2026-05-15T10:00:00.000+00:00"
        );
    }

    #[test]
    fn detection_json_matches_birdnet_pi_field_names() {
        let upload = DetectionUpload {
            timestamp: "2026-05-15T06:00:00.000-04:00".into(),
            lat: 42.36,
            lon: -71.06,
            soundscape_id: 42,
            soundscape_start_time: 1.5,
            soundscape_end_time: 4.5,
            common_name: "Black-capped Chickadee".into(),
            scientific_name: "Poecile atricapillus".into(),
            algorithm: Some(ALGORITHM_V24.into()),
            confidence: 0.75,
        };
        let v = serde_json::to_value(&upload).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "algorithm",
                "commonName",
                "confidence",
                "lat",
                "lon",
                "scientificName",
                "soundscapeEndTime",
                "soundscapeId",
                "soundscapeStartTime",
                "timestamp"
            ]
        );
        assert_eq!(v["algorithm"], "2p4");
        let perch = DetectionUpload {
            algorithm: None,
            ..upload.clone()
        };
        let v = serde_json::to_value(&perch).unwrap();
        assert!(v.get("algorithm").is_none(), "no algorithm is claimed: {v}");
    }

    #[test]
    fn transient_errors() {
        assert!(UploadError::Transport("x".into()).is_transient());
        assert!(UploadError::Status {
            status: 503,
            body: String::new()
        }
        .is_transient());
        assert!(UploadError::Status {
            status: 429,
            body: String::new()
        }
        .is_transient());
        assert!(!UploadError::Status {
            status: 422,
            body: String::new()
        }
        .is_transient());
        assert!(!UploadError::Rejected("no".into()).is_transient());
        assert!(!UploadError::Cancelled.is_transient());
    }

    #[test]
    fn cancellable_sleep() {
        let token = CancellationToken::new();
        assert!(sleep_unless_cancelled(Duration::from_millis(20), &token));
        token.cancel();
        let started = Instant::now();
        assert!(!sleep_unless_cancelled(Duration::from_secs(10), &token));
        assert!(started.elapsed() < Duration::from_millis(200));
    }
}
