//! Capture through an `ffmpeg` child process that writes raw `f32le` mono 48 kHz to stdout.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use birdsong_core::config::{AudioSourceConfig, AudioSourceKind};
use birdsong_core::SAMPLE_RATE_HZ;
use chrono::{DateTime, TimeDelta, Utc};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::frame::{apply_gain, db_to_gain, samples_to_delta};
use crate::{AudioError, AudioFrame, AudioSource};

/// Lines of ffmpeg stderr kept for error messages.
const STDERR_TAIL_LINES: usize = 20;

/// Tuning for [`FfmpegSource`]. `Default` is right for production.
#[derive(Clone, Debug)]
pub struct FfmpegOptions {
    pub ffmpeg_path: PathBuf,
    /// `kind = "file"`: replay at recording speed (`-re`) with wall-clock timestamps. When false
    /// the file is decoded as fast as possible and stamped from `fast_start_at`.
    pub realtime_files: bool,
    pub fast_start_at: DateTime<Utc>,
    /// Delay before the first restart of a live source; doubles up to `restart_backoff_max`.
    pub restart_backoff_min: Duration,
    pub restart_backoff_max: Duration,
    /// A run at least this long resets the backoff.
    pub healthy_run: Duration,
    /// Samples per emitted frame (4 800 = 100 ms).
    pub frame_samples: usize,
    /// Live sources: re-anchor timestamps to the wall clock when they drift further than this.
    pub drift_tolerance: TimeDelta,
}

impl Default for FfmpegOptions {
    fn default() -> Self {
        Self {
            ffmpeg_path: PathBuf::from("ffmpeg"),
            realtime_files: true,
            fast_start_at: Utc::now(),
            restart_backoff_min: Duration::from_secs(1),
            restart_backoff_max: Duration::from_secs(30),
            healthy_run: Duration::from_secs(60),
            frame_samples: 4_800,
            drift_tolerance: TimeDelta::seconds(2),
        }
    }
}

fn missing(src: &AudioSourceConfig, field: &str) -> AudioError {
    AudioError::Config {
        id: src.id.clone(),
        message: format!("kind={:?} needs `{field}`", src.kind),
    }
}

/// The ffmpeg command line (without the program name) for a source. Requires ffmpeg >= 5.0
/// (`-timeout` for RTSP replaced `-stimeout`).
pub fn ffmpeg_args(
    src: &AudioSourceConfig,
    realtime_files: bool,
) -> Result<Vec<String>, AudioError> {
    let mut a: Vec<String> = ["-hide_banner", "-nostdin", "-loglevel", "warning"]
        .map(String::from)
        .to_vec();
    match src.kind {
        AudioSourceKind::Alsa => {
            let device = src
                .device
                .as_deref()
                .ok_or_else(|| missing(src, "device"))?;
            a.extend(["-f", "alsa", "-i", device].map(String::from));
        }
        AudioSourceKind::Rtsp => {
            let url = src.url.as_deref().ok_or_else(|| missing(src, "url"))?;
            let scheme = url.split_once("://").map(|(s, _)| s.to_ascii_lowercase());
            match scheme.as_deref() {
                Some("rtsp" | "rtsps") => {
                    a.extend(["-rtsp_transport", "tcp", "-timeout", "10000000"].map(String::from))
                }
                Some("http" | "https" | "rtmp" | "rtmps" | "tcp" | "udp") => {
                    a.extend(["-rw_timeout", "10000000"].map(String::from))
                }
                _ => {}
            }
            a.extend(["-i".to_string(), url.to_string()]);
        }
        AudioSourceKind::File => {
            let path = src.path.as_ref().ok_or_else(|| missing(src, "path"))?;
            if realtime_files {
                a.push("-re".into());
            }
            a.extend(["-i".to_string(), path.display().to_string()]);
        }
    }
    a.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE_HZ.to_string(),
            "-f",
            "f32le",
            "pipe:1",
        ]
        .map(String::from),
    );
    Ok(a)
}

/// Hide credentials in a URL: `rtsp://user:pass@cam/x` → `rtsp://***@cam/x`.
pub fn redact_url(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    match rest[..authority_end].rfind('@') {
        Some(at) => format!("{scheme}://***{}", &rest[at..]),
        None => url.to_string(),
    }
}

/// [`redact_url`] applied to every whitespace-separated token containing `://`.
pub fn redact_text(text: &str) -> String {
    text.split(' ')
        .map(|tok| {
            if tok.contains("://") {
                redact_url(tok)
            } else {
                tok.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Assigns capture times to frames by sample count.
#[derive(Debug)]
struct FrameClock {
    fixed_start: Option<DateTime<Utc>>,
    anchor: Option<DateTime<Utc>>,
    samples: u64,
    drift_tolerance: TimeDelta,
    reanchors: u64,
}

impl FrameClock {
    fn fixed(start: DateTime<Utc>) -> Self {
        Self {
            fixed_start: Some(start),
            anchor: None,
            samples: 0,
            drift_tolerance: TimeDelta::zero(),
            reanchors: 0,
        }
    }

    fn wall(drift_tolerance: TimeDelta) -> Self {
        Self {
            fixed_start: None,
            anchor: None,
            samples: 0,
            drift_tolerance,
            reanchors: 0,
        }
    }

    /// Timestamp for the next `n` samples, which finished arriving at `now`.
    fn stamp(&mut self, n: usize, now: DateTime<Utc>) -> DateTime<Utc> {
        let at = match self.fixed_start {
            Some(start) => start + samples_to_delta(self.samples),
            None => {
                let end_offset = samples_to_delta(self.samples + n as u64);
                let anchor = match self.anchor {
                    Some(a) if (now - (a + end_offset)).abs() <= self.drift_tolerance => a,
                    Some(_) => {
                        self.reanchors += 1;
                        now - end_offset
                    }
                    None => now - end_offset,
                };
                self.anchor = Some(anchor);
                anchor + samples_to_delta(self.samples)
            }
        };
        self.samples += n as u64;
        at
    }
}

enum RunEnd {
    Cancelled,
    ReceiverGone,
    Exited {
        status: std::process::ExitStatus,
        stderr: String,
        samples: u64,
        ran_for: Duration,
    },
}

enum Sent {
    Ok,
    Cancelled,
    ReceiverGone,
}

/// Captures one configured input (ALSA device, network stream or file) through ffmpeg.
///
/// Live inputs (alsa, rtsp) are restarted with exponential backoff whenever ffmpeg exits; each
/// restart re-anchors timestamps, which the [`crate::Chunker`] reports as a gap. File inputs
/// end the source at end of file.
pub struct FfmpegSource {
    src: AudioSourceConfig,
    opts: FfmpegOptions,
    gain: f32,
}

impl FfmpegSource {
    /// Validates the source configuration up front.
    pub fn new(src: AudioSourceConfig, opts: FfmpegOptions) -> Result<Self, AudioError> {
        ffmpeg_args(&src, opts.realtime_files)?;
        let gain = db_to_gain(src.gain_db);
        Ok(Self { src, opts, gain })
    }

    fn describe(&self) -> String {
        match self.src.kind {
            AudioSourceKind::Alsa => self.src.device.clone().unwrap_or_default(),
            AudioSourceKind::Rtsp => redact_url(self.src.url.as_deref().unwrap_or_default()),
            AudioSourceKind::File => self
                .src
                .path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        }
    }

    fn is_file(&self) -> bool {
        self.src.kind == AudioSourceKind::File
    }

    async fn send(
        &self,
        bytes: &[u8],
        clock: &mut FrameClock,
        tx: &mpsc::Sender<AudioFrame>,
        cancel: &CancellationToken,
    ) -> Sent {
        let (whole, _) = bytes.as_chunks::<4>();
        let mut samples: Vec<f32> = whole.iter().map(|b| f32::from_le_bytes(*b)).collect();
        apply_gain(&mut samples, self.gain);
        let captured_at = clock.stamp(samples.len(), Utc::now());
        if clock.reanchors == 1 && clock.anchor.is_some() {
            tracing::warn!(source = %self.src.id, "capture clock drifted; timestamps re-anchored to wall clock");
            clock.reanchors += 1; // warn once per run
        }
        let frame = AudioFrame {
            samples,
            captured_at,
        };
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Sent::Cancelled,
            r = tx.send(frame) => if r.is_ok() { Sent::Ok } else { Sent::ReceiverGone },
        }
    }

    async fn run_once(
        &self,
        tx: &mpsc::Sender<AudioFrame>,
        cancel: &CancellationToken,
    ) -> Result<RunEnd, AudioError> {
        let args = ffmpeg_args(&self.src, self.opts.realtime_files)?;
        tracing::info!(source = %self.src.id, input = %self.describe(), "starting ffmpeg");
        let mut child = Command::new(&self.opts.ffmpeg_path)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    AudioError::FfmpegNotFound(self.opts.ffmpeg_path.clone())
                }
                _ => AudioError::Spawn(e),
            })?;
        let started = std::time::Instant::now();

        let tail: Arc<Mutex<VecDeque<String>>> = Arc::default();
        let stderr_task = child.stderr.take().map(|stderr| {
            let tail = Arc::clone(&tail);
            let id = self.src.id.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let line = redact_text(&line);
                    tracing::warn!(source = %id, "ffmpeg: {line}");
                    let mut t = tail.lock().unwrap_or_else(|p| p.into_inner());
                    if t.len() == STDERR_TAIL_LINES {
                        t.pop_front();
                    }
                    t.push_back(line);
                }
            })
        });
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| AudioError::Task("ffmpeg stdout was not captured".into()))?;

        let mut clock = if self.is_file() && !self.opts.realtime_files {
            FrameClock::fixed(self.opts.fast_start_at)
        } else {
            FrameClock::wall(self.opts.drift_tolerance)
        };
        let bytes_per_frame = self.opts.frame_samples.max(1) * 4;
        let mut buf = vec![0u8; bytes_per_frame];
        let mut filled = 0usize;

        loop {
            let read = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    let _ = child.kill().await;
                    return Ok(RunEnd::Cancelled);
                }
                r = stdout.read(&mut buf[filled..]) => r,
            };
            let n = read.map_err(|e| AudioError::Io {
                path: "<ffmpeg stdout>".into(),
                source: e,
            })?;
            let at_eof = n == 0;
            filled += n;
            if filled == bytes_per_frame || (at_eof && filled >= 4) {
                let whole = filled / 4 * 4;
                match self.send(&buf[..whole], &mut clock, tx, cancel).await {
                    Sent::Ok => {}
                    Sent::Cancelled => {
                        let _ = child.kill().await;
                        return Ok(RunEnd::Cancelled);
                    }
                    Sent::ReceiverGone => {
                        let _ = child.kill().await;
                        return Ok(RunEnd::ReceiverGone);
                    }
                }
                filled = 0;
            }
            if at_eof {
                break;
            }
        }

        let status = child.wait().await.map_err(AudioError::Spawn)?;
        if let Some(task) = stderr_task {
            let _ = task.await;
        }
        let stderr = tail
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(" | ");
        Ok(RunEnd::Exited {
            status,
            stderr,
            samples: clock.samples,
            ran_for: started.elapsed(),
        })
    }
}

#[async_trait::async_trait]
impl AudioSource for FfmpegSource {
    fn id(&self) -> &str {
        &self.src.id
    }

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<AudioFrame>,
        cancel: CancellationToken,
    ) -> Result<(), AudioError> {
        let mut backoff = self.opts.restart_backoff_min;
        loop {
            match self.run_once(&tx, &cancel).await {
                Ok(RunEnd::Cancelled | RunEnd::ReceiverGone) => return Ok(()),
                Ok(RunEnd::Exited {
                    status,
                    stderr,
                    samples,
                    ran_for,
                }) => {
                    if self.is_file() {
                        return if status.success() {
                            tracing::info!(source = %self.src.id, samples, "file finished");
                            Ok(())
                        } else {
                            Err(AudioError::FfmpegFailed {
                                status: status.to_string(),
                                stderr,
                            })
                        };
                    }
                    if ran_for >= self.opts.healthy_run {
                        backoff = self.opts.restart_backoff_min;
                    }
                    tracing::warn!(
                        source = %self.src.id, %status, samples, ?ran_for, %stderr,
                        "ffmpeg exited; restarting in {backoff:?}"
                    );
                }
                Err(e @ AudioError::FfmpegNotFound(_)) => return Err(e),
                Err(e) if self.is_file() => return Err(e),
                Err(e) => {
                    tracing::warn!(source = %self.src.id, error = %e, "capture failed; retrying in {backoff:?}")
                }
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Ok(()),
                _ = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(self.opts.restart_backoff_max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn src(kind: AudioSourceKind) -> AudioSourceConfig {
        AudioSourceConfig {
            id: "s".into(),
            kind,
            device: None,
            url: None,
            path: None,
            gain_db: 0.0,
        }
    }

    #[test]
    fn args_per_kind() {
        let mut alsa = src(AudioSourceKind::Alsa);
        assert!(ffmpeg_args(&alsa, true).is_err());
        alsa.device = Some("hw:1,0".into());
        let a = ffmpeg_args(&alsa, true).unwrap().join(" ");
        assert!(a.contains("-f alsa -i hw:1,0"), "{a}");
        assert!(a.ends_with("-vn -ac 1 -ar 48000 -f f32le pipe:1"), "{a}");
        assert!(!a.contains("-re"));

        let mut rtsp = src(AudioSourceKind::Rtsp);
        rtsp.url = Some("rtsp://u:p@cam/stream".into());
        let a = ffmpeg_args(&rtsp, true).unwrap().join(" ");
        assert!(
            a.contains("-rtsp_transport tcp -timeout 10000000 -i rtsp://u:p@cam/stream"),
            "{a}"
        );
        rtsp.url = Some("https://radio/x.mp3".into());
        assert!(ffmpeg_args(&rtsp, true)
            .unwrap()
            .join(" ")
            .contains("-rw_timeout 10000000 -i https://"));
        rtsp.url = Some("/tmp/plain.wav".into());
        let a = ffmpeg_args(&rtsp, true).unwrap().join(" ");
        assert!(
            !a.contains("timeout") && a.contains("-i /tmp/plain.wav"),
            "{a}"
        );

        let mut file = src(AudioSourceKind::File);
        file.path = Some("/data/t.wav".into());
        assert!(ffmpeg_args(&file, true)
            .unwrap()
            .join(" ")
            .contains("-re -i /data/t.wav"));
        assert!(!ffmpeg_args(&file, false).unwrap().join(" ").contains("-re"));
    }

    #[test]
    fn redaction() {
        assert_eq!(
            redact_url("rtsp://user:secret@10.0.0.2:554/live?x=1"),
            "rtsp://***@10.0.0.2:554/live?x=1"
        );
        assert_eq!(redact_url("rtsp://10.0.0.2/live"), "rtsp://10.0.0.2/live");
        assert_eq!(
            redact_url("http://host/path@notuser"),
            "http://host/path@notuser"
        );
        assert_eq!(redact_url("hw:1,0"), "hw:1,0");
        assert_eq!(
            redact_text("[rtsp @ 0x1] rtsp://a:b@cam/s: Connection refused"),
            "[rtsp @ 0x1] rtsp://***@cam/s: Connection refused"
        );
    }

    #[test]
    fn fixed_clock_counts_samples() {
        let t0 = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap();
        let mut c = FrameClock::fixed(t0);
        assert_eq!(c.stamp(4_800, t0 + TimeDelta::days(9)), t0);
        assert_eq!(c.stamp(4_800, t0), t0 + TimeDelta::milliseconds(100));
    }

    #[test]
    fn wall_clock_anchors_and_reanchors() {
        let t0 = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap();
        let tol = TimeDelta::seconds(2);
        let mut c = FrameClock::wall(tol);
        // First frame (100 ms) finished arriving at t0 → it started at t0 − 100 ms.
        assert_eq!(c.stamp(4_800, t0), t0 - TimeDelta::milliseconds(100));
        // Steady arrival with jitter stays contiguous.
        assert_eq!(c.stamp(4_800, t0 + TimeDelta::milliseconds(250)), t0);
        assert_eq!(
            c.stamp(4_800, t0 + TimeDelta::milliseconds(150)),
            t0 + TimeDelta::milliseconds(100)
        );
        assert_eq!(c.reanchors, 0);
        // Arrival 5 s late (e.g. stalled stream) → re-anchor to the wall clock.
        let late = t0 + TimeDelta::seconds(5) + TimeDelta::milliseconds(300);
        assert_eq!(c.stamp(4_800, late), late - TimeDelta::milliseconds(100));
        assert_eq!(c.reanchors, 1);
    }
}
