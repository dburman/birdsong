//! Offline tools: `birdsong analyze` (per-chunk scores for a recording) and `birdsong species-list`.
//!
//! `analyze` uses exactly the pipeline's code path (chunker, classifier, `analyze_chunk`,
//! `NeighbourMask`) so its "reported" column is what `birdsong run` would store.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use birdsong_audio::{
    wav, AudioFrame, AudioSource, Chunker, ChunkerEvent, FfmpegOptions, FfmpegSource,
};
use birdsong_core::config::{AudioSourceConfig, AudioSourceKind, ModelConfig};
use birdsong_core::{week_of_year, Config, SAMPLE_RATE_HZ, YEAR_ROUND_WEEK};
use birdsong_model::{
    analyze_chunk_with, top_scores, ChunkContext, Confirmer, DynamicThresholds, ModelBundle,
    NeighbourMask, SpeciesFilter, SpeciesFilterKind,
};
use chrono::{DateTime, NaiveDate, NaiveTime, TimeDelta, TimeZone, Utc};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

/// Configuration for the offline tools: a config file when given, otherwise defaults with the
/// model directory `models/`. `--models`, `--lat` and `--lon` override either.
pub fn tool_config(
    config: Option<&Path>,
    models: Option<PathBuf>,
    lat: Option<f64>,
    lon: Option<f64>,
) -> anyhow::Result<Config> {
    let mut cfg = match config {
        Some(path) => {
            Config::load(Some(path)).with_context(|| format!("loading {}", path.display()))?
        }
        None => Config {
            model: ModelConfig {
                dir: PathBuf::from("models"),
                ..ModelConfig::default()
            },
            ..Config::default()
        },
    };
    if let Some(dir) = models {
        cfg.model.dir = dir;
    }
    match (lat, lon) {
        (Some(la), Some(lo)) => {
            anyhow::ensure!(
                (-90.0..=90.0).contains(&la) && (-180.0..=180.0).contains(&lo),
                "--lat/--lon out of range"
            );
            cfg.station.latitude = la;
            cfg.station.longitude = lo;
        }
        (None, None) => {}
        _ => anyhow::bail!("--lat and --lon must be given together"),
    }
    Ok(cfg)
}

/// Today's date in the station time zone.
pub fn station_today(cfg: &Config) -> NaiveDate {
    Utc::now().with_timezone(&cfg.station.timezone).date_naive()
}

/// Read a recording as mono 48 kHz: 48 kHz WAV files directly, anything else through ffmpeg.
pub async fn load_audio(path: &Path, ffmpeg_path: &Path) -> anyhow::Result<Vec<f32>> {
    match wav::read_wav_48k_mono(path) {
        Ok(samples) => return Ok(samples),
        Err(e) => tracing::debug!(error = %e, "not a 48 kHz WAV; decoding with ffmpeg"),
    }
    let src = AudioSourceConfig {
        id: "analyze".into(),
        kind: AudioSourceKind::File,
        device: None,
        url: None,
        path: Some(path.to_path_buf()),
        gain_db: 0.0,
    };
    let opts = FfmpegOptions {
        ffmpeg_path: ffmpeg_path.to_path_buf(),
        realtime_files: false,
        ..FfmpegOptions::default()
    };
    let source: Box<dyn AudioSource> = Box::new(FfmpegSource::new(src, opts)?);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<AudioFrame>(256);
    let decoder = tokio::spawn(source.run(tx, CancellationToken::new()));
    let mut samples = Vec::new();
    while let Some(frame) = rx.recv().await {
        samples.extend_from_slice(&frame.samples);
    }
    decoder
        .await
        .context("decoder task")?
        .with_context(|| format!("decoding {}", path.display()))?;
    Ok(samples)
}

#[derive(Clone, Debug)]
pub struct AnalyzeOptions {
    /// Classes shown per chunk.
    pub top: usize,
    /// Recording date: selects the location-filter week.
    pub date: NaiveDate,
    /// Apply the configured species filter (location model or species list).
    pub apply_species_filter: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnalysisReport {
    pub file: String,
    pub model_id: String,
    pub date: NaiveDate,
    pub week: u32,
    pub species_filter: &'static str,
    pub allowed_species: usize,
    pub species_filter_threshold: f32,
    pub sensitivity: f32,
    pub sigmoid_slope: f32,
    pub min_confidence: f32,
    pub top_n_per_chunk: usize,
    pub overlap_seconds: f32,
    pub privacy_filter: bool,
    pub min_detections: usize,
    pub confirmation_window_seconds: f32,
    pub dynamic_threshold: bool,
    pub duration_seconds: f64,
    pub chunks: Vec<ChunkReport>,
    /// Species that would be reported, most detections first.
    pub summary: Vec<SpeciesCount>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkReport {
    pub index: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub padded: bool,
    /// A human class ranked within the privacy cutoff in this chunk.
    pub human_present: bool,
    /// Blanked by the privacy rule (this chunk or a neighbour had a human).
    pub masked: bool,
    pub top: Vec<ScoreReport>,
    /// What `birdsong run` would store for this chunk.
    pub detections: Vec<DetectionReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScoreReport {
    pub rank: usize,
    pub class_index: usize,
    pub scientific_name: String,
    pub common_name: String,
    pub logit: f32,
    pub confidence: f32,
    /// Allowed by the species filter.
    pub allowed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DetectionReport {
    pub scientific_name: String,
    pub common_name: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpeciesCount {
    pub scientific_name: String,
    pub common_name: String,
    pub count: usize,
    pub max_confidence: f32,
}

fn seconds(d: TimeDelta) -> f64 {
    d.num_microseconds().unwrap_or(i64::MAX) as f64 / 1e6
}

/// Analyse a whole recording (already mono 48 kHz).
pub fn analyze_samples(
    bundle: &mut ModelBundle,
    cfg: &Config,
    samples: &[f32],
    file_label: &str,
    opts: &AnalyzeOptions,
) -> anyhow::Result<AnalysisReport> {
    let tz = cfg.station.timezone;
    let noon = opts
        .date
        .and_time(NaiveTime::from_hms_opt(12, 0, 0).unwrap_or(NaiveTime::MIN));
    let base: DateTime<Utc> = tz
        .from_local_datetime(&noon)
        .earliest()
        .map_or_else(|| Utc.from_utc_datetime(&noon), |t| t.with_timezone(&Utc));
    let week = week_of_year(opts.date);

    let (filter, kind) = if opts.apply_species_filter {
        (
            bundle.species_filter_for_week(week as i32)?,
            bundle.species_filter_kind(),
        )
    } else {
        (
            SpeciesFilter::allow_all(bundle.labels.len()),
            SpeciesFilterKind::None,
        )
    };

    let mut chunker = Chunker::new(
        "analyze",
        cfg.detection.overlap_seconds,
        cfg.audio.ring_buffer_seconds,
    );
    let mut chunks: Vec<_> = chunker
        .push(AudioFrame {
            samples: samples.to_vec(),
            captured_at: base,
        })
        .into_iter()
        .filter_map(|e| match e {
            ChunkerEvent::Chunk(c) => Some(c),
            ChunkerEvent::Gap { .. } => None,
        })
        .collect();
    chunks.extend(chunker.finish());

    let duration = samples.len() as f64 / SAMPLE_RATE_HZ as f64;
    let sensitivity = bundle.postprocess.sensitivity;
    let privacy = bundle.postprocess.privacy_filter;
    let model_id = bundle.classifier.model_id().to_string();

    let mut dynamic = cfg.detection.dynamic_threshold.then(|| {
        DynamicThresholds::new(
            cfg.detection.min_confidence,
            cfg.detection.dynamic_threshold_trigger,
            cfg.detection.dynamic_threshold_min,
            TimeDelta::hours(i64::from(cfg.detection.dynamic_threshold_hours)),
        )
    });
    let mut rows = Vec::with_capacity(chunks.len());
    let mut released = Vec::with_capacity(chunks.len());
    let mut mask = NeighbourMask::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let logits = bundle
            .classifier
            .predict(&chunk.samples)
            .with_context(|| format!("chunk {index}"))?;
        let top = top_scores(&logits, sensitivity, opts.top)
            .into_iter()
            .enumerate()
            .map(|(rank, (i, confidence))| {
                let label = bundle.labels.get(i);
                ScoreReport {
                    rank: rank + 1,
                    class_index: i,
                    scientific_name: label.map(|l| l.scientific.clone()).unwrap_or_default(),
                    common_name: label.map(|l| l.common.clone()).unwrap_or_default(),
                    logit: logits[i],
                    confidence,
                    allowed: filter.is_allowed(i),
                }
            })
            .collect();
        let ctx = ChunkContext {
            start_at: chunk.start_at,
            source_id: "analyze".into(),
            model_id: model_id.clone(),
        };
        let start_at = chunk.start_at;
        let threshold_for = |i: usize| match (&dynamic, bundle.labels.get(i)) {
            (Some(d), Some(label)) => d.threshold_for(&label.scientific, start_at),
            _ => bundle.postprocess.min_confidence,
        };
        let analysis = analyze_chunk_with(
            &logits,
            &bundle.labels,
            &filter,
            &bundle.postprocess,
            &ctx,
            &threshold_for,
        );
        if let Some(d) = dynamic.as_mut() {
            if !analysis.human_present {
                for det in &analysis.detections {
                    d.observe(&det.scientific_name, det.confidence, start_at);
                }
            }
        }
        let start = seconds(chunk.start_at - base);
        rows.push(ChunkReport {
            index,
            start_seconds: start,
            end_seconds: (start + 3.0).min(duration),
            padded: chunk.padded,
            human_present: analysis.human_present,
            masked: false,
            top,
            detections: Vec::new(),
        });
        if privacy {
            released.extend(mask.push(analysis));
        } else {
            released.push(analysis);
        }
    }
    if privacy {
        released.extend(mask.flush());
    }
    // Confirmation releases every analysis exactly once and in order, so the rows still line up.
    let released = if cfg.detection.min_detections > 1 {
        let window = TimeDelta::milliseconds(
            (cfg.detection.confirmation_window_seconds * 1000.0).round() as i64,
        );
        let mut confirmer = Confirmer::new(cfg.detection.min_detections, window);
        let mut settled = Vec::with_capacity(released.len());
        for a in released {
            settled.extend(confirmer.push(a));
        }
        settled.extend(confirmer.flush());
        settled
    } else {
        released
    };

    let mut species: BTreeMap<String, SpeciesCount> = BTreeMap::new();
    for (row, analysis) in rows.iter_mut().zip(released) {
        row.masked = analysis.masked;
        for d in &analysis.detections {
            let entry = species
                .entry(d.scientific_name.clone())
                .or_insert_with(|| SpeciesCount {
                    scientific_name: d.scientific_name.clone(),
                    common_name: d.common_name.clone(),
                    count: 0,
                    max_confidence: 0.0,
                });
            entry.count += 1;
            entry.max_confidence = entry.max_confidence.max(d.confidence);
        }
        row.detections = analysis
            .detections
            .into_iter()
            .map(|d| DetectionReport {
                scientific_name: d.scientific_name,
                common_name: d.common_name,
                confidence: d.confidence,
            })
            .collect();
    }
    let mut summary: Vec<_> = species.into_values().collect();
    summary.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.common_name.cmp(&b.common_name))
    });

    Ok(AnalysisReport {
        file: file_label.to_string(),
        model_id,
        date: opts.date,
        week,
        species_filter: kind.as_str(),
        allowed_species: filter.num_allowed(),
        species_filter_threshold: bundle.species_filter_threshold(),
        sensitivity: cfg.detection.sensitivity,
        sigmoid_slope: sensitivity.slope(),
        min_confidence: bundle.postprocess.min_confidence,
        top_n_per_chunk: bundle.postprocess.top_n_per_chunk,
        overlap_seconds: cfg.detection.overlap_seconds,
        privacy_filter: privacy,
        min_detections: cfg.detection.min_detections,
        confirmation_window_seconds: cfg.detection.confirmation_window_seconds,
        dynamic_threshold: cfg.detection.dynamic_threshold,
        duration_seconds: duration,
        chunks: rows,
        summary,
    })
}

/// `mm:ss.s`
pub fn format_offset(seconds: f64) -> String {
    let tenths = (seconds.max(0.0) * 10.0).round() as u64;
    format!(
        "{:02}:{:02}.{}",
        tenths / 600,
        (tenths / 10) % 60,
        tenths % 10
    )
}

/// Human-readable report.
pub fn render_text(r: &AnalysisReport) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}  ({:.1} s)  model {}  date {} (week {})",
        r.file, r.duration_seconds, r.model_id, r.date, r.week
    );
    let filter = match r.species_filter {
        "none" => "species filter: off (all species allowed)".to_string(),
        kind => format!(
            "species filter: {kind} ({} species allowed)",
            r.allowed_species
        ),
    };
    let _ = writeln!(
        out,
        "{filter}  sensitivity {} (slope {:.2})  min confidence {:.2}  top {} per chunk  privacy filter {}",
        r.sensitivity,
        r.sigmoid_slope,
        r.min_confidence,
        r.top_n_per_chunk,
        if r.privacy_filter { "on" } else { "off" }
    );
    for c in &r.chunks {
        let _ = writeln!(
            out,
            "\n[{} - {}] chunk {}{}",
            format_offset(c.start_seconds),
            format_offset(c.end_seconds),
            c.index,
            if c.padded { " (padded)" } else { "" }
        );
        for s in &c.top {
            let reported = c
                .detections
                .iter()
                .any(|d| d.scientific_name == s.scientific_name);
            let mark = if reported { "REPORTED" } else { "        " };
            let note = if !s.allowed {
                "  [filtered: not expected here]"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "  {mark}  {:.3}  {} ({}){note}",
                s.confidence, s.common_name, s.scientific_name
            );
        }
        if c.masked {
            let why = if c.human_present {
                "human sound in this chunk"
            } else {
                "human sound in a neighbouring chunk"
            };
            let _ = writeln!(out, "  (privacy filter: nothing reported, {why})");
        }
    }
    let _ = writeln!(out, "\nsummary:");
    if r.summary.is_empty() {
        let _ = writeln!(
            out,
            "  no detections at min confidence {:.2}",
            r.min_confidence
        );
    }
    for s in &r.summary {
        let _ = writeln!(
            out,
            "  {:>3} x {} ({})  max {:.3}",
            s.count, s.common_name, s.scientific_name, s.max_confidence
        );
    }
    out
}

#[derive(Clone, Debug, Serialize)]
pub struct SpeciesListReport {
    pub week: i32,
    pub species_filter: &'static str,
    /// Location-model threshold (only for the location model).
    pub threshold: Option<f32>,
    pub count: usize,
    pub species: Vec<SpeciesEntry>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpeciesEntry {
    pub class_index: usize,
    pub scientific_name: String,
    pub common_name: String,
    /// Location-model occurrence score (only for the location model).
    pub score: Option<f32>,
}

/// Species the configured filter allows in a BirdNET week (`1..=48`, or `-1` for year-round).
/// Sorted by score (location model) or common name (species list).
pub fn species_list(bundle: &ModelBundle, week: i32) -> anyhow::Result<SpeciesListReport> {
    anyhow::ensure!(
        week == YEAR_ROUND_WEEK || (1..=48).contains(&week),
        "week must be 1..=48 or {YEAR_ROUND_WEEK} (year-round), got {week}"
    );
    let kind = bundle.species_filter_kind();
    if kind == SpeciesFilterKind::None {
        anyhow::bail!(
            "species filter disabled: set station latitude/longitude (or --lat/--lon) with model.meta_model, \
             or model.species_list"
        );
    }
    let filter = bundle.species_filter_for_week(week)?;
    let scores = bundle.location_scores_for_week(week)?;
    let mut species: Vec<SpeciesEntry> = bundle
        .labels
        .iter()
        .enumerate()
        .filter(|(i, _)| filter.is_allowed(*i))
        .map(|(i, l)| SpeciesEntry {
            class_index: i,
            scientific_name: l.scientific.clone(),
            common_name: l.common.clone(),
            score: scores.as_ref().and_then(|s| s.get(i).copied()),
        })
        .collect();
    species.sort_by(|a, b| {
        b.score
            .unwrap_or(0.0)
            .total_cmp(&a.score.unwrap_or(0.0))
            .then_with(|| a.common_name.cmp(&b.common_name))
    });
    Ok(SpeciesListReport {
        week,
        species_filter: kind.as_str(),
        threshold: (kind == SpeciesFilterKind::LocationModel)
            .then(|| bundle.species_filter_threshold()),
        count: species.len(),
        species,
    })
}

/// Tab-separated `score  Scientific name  Common name` lines.
pub fn render_species_list(r: &SpeciesListReport) -> String {
    let mut out = String::new();
    for s in &r.species {
        let score = s
            .score
            .map_or_else(|| "-".to_string(), |v| format!("{v:.3}"));
        let _ = writeln!(out, "{score}\t{}\t{}", s.scientific_name, s.common_name);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets() {
        assert_eq!(format_offset(0.0), "00:00.0");
        assert_eq!(format_offset(3.0), "00:03.0");
        assert_eq!(format_offset(61.25), "01:01.3");
        assert_eq!(format_offset(15.0187), "00:15.0");
    }

    #[test]
    fn tool_config_overrides() {
        let cfg = tool_config(None, Some("/m".into()), Some(1.0), Some(2.0)).unwrap();
        assert_eq!(cfg.model.dir, PathBuf::from("/m"));
        assert_eq!((cfg.station.latitude, cfg.station.longitude), (1.0, 2.0));
        assert_eq!(
            tool_config(None, None, None, None).unwrap().model.dir,
            PathBuf::from("models")
        );
        assert!(tool_config(None, None, Some(1.0), None).is_err());
        assert!(tool_config(None, None, Some(100.0), Some(0.0)).is_err());
    }
}
