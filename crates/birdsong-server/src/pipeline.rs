//! Capture → chunk → infer → store, as in `BUILD_PLAN.md` §2.1.
//!
//! ```text
//! per source:  AudioSource task ──frames──▶ chunker task ──▶ ┐
//!                                                             ├─ ChunkQueue ─▶ inference thread ─▶ storage task ─▶ broadcast
//! per source:  AudioSource task ──frames──▶ chunker task ──▶ ┘   (4 chunks)    (model, privacy)    (SQLite, logs)
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use birdsong_audio::{AudioSource, Chunker, ChunkerEvent, FfmpegOptions, FfmpegSource};
use birdsong_core::config::{AudioSourceKind, DetectionConfig};
use birdsong_core::{local_date_and_hour, week_of_year, Config, Detection};
use birdsong_model::{
    analyze_chunk_with, ChunkAnalysis, ChunkContext, Confirmer, DynamicThresholds, ModelBundle,
    NeighbourMask, SpeciesFilter, BIRDNET_V24_MODEL_ID,
};
use birdsong_store::DetectionStore;
use chrono::{TimeDelta, Utc};
use chrono_tz::Tz;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::birdweather::{upload_task, BirdWeatherClient, Station};
use crate::clips::{clip_task, ClipJob, ClipMsg, ClipSettings};
use crate::queue::{Backpressure, ChunkQueue, ConsumerGuard, PushOutcome, QueueItem};
use crate::stats::{PipelineStats, StatsSnapshot};

/// Chunks waiting for inference, across all sources.
pub const QUEUE_CAPACITY: usize = 4;
/// Frames buffered between a source and its chunker (100 ms each).
const FRAME_CHANNEL: usize = 64;
/// Released chunk analyses waiting for the storage task.
const RESULT_CHANNEL: usize = 64;
/// Clip jobs waiting for the clip writer.
const CLIP_CHANNEL: usize = 64;
/// Saved clips waiting to be uploaded to BirdWeather; when full, clips are skipped, never delayed.
const UPLOAD_CHANNEL: usize = 32;
/// Detections buffered for slow broadcast subscribers before they lag.
const BROADCAST_CAPACITY: usize = 256;

/// Run-time behaviour switches (CLI flags).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineOptions {
    /// Return once every source has ended, instead of waiting for a shutdown signal.
    pub exit_on_eof: bool,
    /// Decode `kind = "file"` sources as fast as possible (and never drop their chunks).
    pub fast_files: bool,
}

/// One audio source and what to do when inference falls behind it.
pub struct SourceSpec {
    pub source: Box<dyn AudioSource>,
    pub backpressure: Backpressure,
}

/// What the inference thread hands to the storage task.
enum WorkerOutput {
    Analysis(ChunkAnalysis),
    SourceEnded(Arc<str>),
}

/// Counters at the end of a run.
pub type PipelineSummary = StatsSnapshot;

/// The assembled detection pipeline. Create, grab [`Pipeline::stats`] / [`Pipeline::subscribe`]
/// handles, then [`Pipeline::run`].
pub struct Pipeline {
    cfg: Config,
    bundle: ModelBundle,
    store: Arc<dyn DetectionStore>,
    sources: Vec<SourceSpec>,
    opts: PipelineOptions,
    stats: Arc<PipelineStats>,
    detections: broadcast::Sender<Detection>,
}

impl Pipeline {
    /// Build ffmpeg capture sources for every configured input.
    pub fn from_config(
        cfg: Config,
        bundle: ModelBundle,
        store: Arc<dyn DetectionStore>,
        opts: PipelineOptions,
    ) -> anyhow::Result<Self> {
        let ffmpeg = FfmpegOptions {
            ffmpeg_path: cfg.audio.ffmpeg_path.clone(),
            realtime_files: !opts.fast_files,
            fast_start_at: Utc::now(),
            ..FfmpegOptions::default()
        };
        let mut sources = Vec::new();
        for src in &cfg.audio.sources {
            let backpressure = if src.kind == AudioSourceKind::File && opts.fast_files {
                Backpressure::Wait
            } else {
                Backpressure::DropOldest
            };
            let source = FfmpegSource::new(src.clone(), ffmpeg.clone())
                .with_context(|| format!("audio source {:?}", src.id))?;
            sources.push(SourceSpec {
                source: Box::new(source),
                backpressure,
            });
        }
        Ok(Self::with_sources(cfg, bundle, store, sources, opts))
    }

    /// Use explicit sources (tests, offline tools).
    pub fn with_sources(
        cfg: Config,
        bundle: ModelBundle,
        store: Arc<dyn DetectionStore>,
        sources: Vec<SourceSpec>,
        opts: PipelineOptions,
    ) -> Self {
        let (detections, _) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            cfg,
            bundle,
            store,
            sources,
            opts,
            stats: Arc::new(PipelineStats::new()),
            detections,
        }
    }

    pub fn stats(&self) -> Arc<PipelineStats> {
        Arc::clone(&self.stats)
    }

    /// Every stored detection (with its database id), as it is stored.
    pub fn subscribe(&self) -> broadcast::Receiver<Detection> {
        self.detections.subscribe()
    }

    pub fn detections_sender(&self) -> broadcast::Sender<Detection> {
        self.detections.clone()
    }

    /// Run until `cancel` fires, a source fails fatally, or (with `exit_on_eof`) every source ends.
    /// Chunks already queued are still analysed and stored before returning.
    pub async fn run(self, cancel: CancellationToken) -> anyhow::Result<PipelineSummary> {
        let Self {
            cfg,
            bundle,
            store,
            sources,
            opts,
            stats,
            detections,
        } = self;
        if sources.is_empty() {
            anyhow::bail!("no audio sources configured");
        }

        let queue = Arc::new(ChunkQueue::new(QUEUE_CAPACITY));
        let (results_tx, results_rx) = mpsc::channel::<WorkerOutput>(RESULT_CHANNEL);
        let (clip_tx, clip_rx) = mpsc::channel::<ClipMsg>(CLIP_CHANNEL);

        // Chunkers exist before any task starts so the clip writer can read every ring buffer.
        let window_seconds = bundle.classifier.window_seconds();
        let model_id = bundle.classifier.model_id().to_string();
        let mut rings = HashMap::new();
        let mut prepared = Vec::with_capacity(sources.len());
        for spec in sources {
            let chunker = Chunker::new(
                spec.source.id(),
                window_seconds,
                cfg.detection.overlap_seconds,
                cfg.audio.ring_buffer_seconds,
            );
            rings.insert(Arc::clone(chunker.source_id()), chunker.ring());
            prepared.push((spec, chunker));
        }

        let worker = {
            let queue = Arc::clone(&queue);
            let stats = Arc::clone(&stats);
            let tz = cfg.station.timezone;
            let detection = cfg.detection.clone();
            std::thread::Builder::new()
                .name("inference".into())
                .spawn(move || inference_worker(bundle, queue, results_tx, stats, detection, tz))
                .context("starting inference thread")?
        };
        // BirdWeather records detections as BirdNET V2.4 results, so other models do not upload.
        let birdweather = cfg.birdweather.enabled()
            && if model_id == BIRDNET_V24_MODEL_ID {
                true
            } else {
                tracing::warn!(model = %model_id, "BirdWeather uploads need BirdNET V2.4; uploads are off");
                false
            };
        let (upload_tx, uploads) = if birdweather {
            let (tx, rx) = mpsc::channel(UPLOAD_CHANNEL);
            let client = Arc::new(BirdWeatherClient::new(
                &cfg.birdweather.api_url,
                &cfg.birdweather.token,
            ));
            let station = Station {
                latitude: cfg.station.latitude,
                longitude: cfg.station.longitude,
                timezone: cfg.station.timezone,
            };
            tracing::info!("BirdWeather uploads enabled");
            (
                Some(tx),
                Some(tokio::spawn(upload_task(
                    client,
                    station,
                    rx,
                    Arc::clone(&stats),
                    cancel.clone(),
                ))),
            )
        } else {
            (None, None)
        };
        let clips = tokio::spawn(clip_task(
            ClipSettings::from_config(&cfg, window_seconds),
            rings,
            Arc::clone(&store),
            clip_rx,
            Arc::clone(&stats),
            upload_tx,
        ));
        let storage = tokio::spawn(storage_task(
            store,
            results_rx,
            detections,
            Arc::clone(&stats),
            clip_tx,
            cfg.detection.clone(),
        ));

        let sources_cancel = cancel.child_token();
        let mut tasks = JoinSet::new();
        for (spec, chunker) in prepared {
            tasks.spawn(source_task(
                spec,
                chunker,
                Arc::clone(&queue),
                Arc::clone(&stats),
                sources_cancel.child_token(),
            ));
        }

        let mut failure: Option<anyhow::Error> = None;
        while let Some(joined) = tasks.join_next().await {
            match joined {
                Ok((id, Ok(()))) => tracing::info!(source = %id, "audio source ended"),
                Ok((id, Err(e))) => {
                    tracing::error!(source = %id, error = %e, "audio source failed; stopping");
                    failure.get_or_insert_with(|| {
                        anyhow::Error::new(e).context(format!("audio source {id:?}"))
                    });
                    sources_cancel.cancel();
                }
                Err(join_error) => {
                    tracing::error!(error = %join_error, "audio task panicked; stopping");
                    failure.get_or_insert_with(|| {
                        anyhow::anyhow!("audio task panicked: {join_error}")
                    });
                    sources_cancel.cancel();
                }
            }
        }

        if failure.is_none() && !opts.exit_on_eof && !cancel.is_cancelled() {
            tracing::info!("all audio sources ended; waiting for a shutdown signal");
            cancel.cancelled().await;
        }

        queue.close_producers();
        let worker_result = tokio::task::spawn_blocking(move || worker.join()).await;
        let storage_result = storage.await;
        let clips_result = clips.await;
        let uploads_result = match uploads {
            Some(task) => Some(task.await),
            None => None,
        };

        let summary = stats.snapshot();
        tracing::info!(
            chunks = summary.chunks_processed,
            dropped = summary.chunks_dropped,
            masked = summary.masked_chunks,
            detections = summary.detections,
            mean_inference_ms = ?summary.mean_inference_ms,
            "pipeline stopped"
        );
        if let Some(e) = failure {
            return Err(e);
        }
        match worker_result {
            Ok(Ok(())) => {}
            Ok(Err(_)) => anyhow::bail!("inference thread panicked"),
            Err(e) => anyhow::bail!("waiting for inference thread: {e}"),
        }
        storage_result.context("storage task panicked")?;
        clips_result.context("clip writer task panicked")?;
        if let Some(result) = uploads_result {
            result.context("BirdWeather upload task panicked")?;
        }
        Ok(summary)
    }
}

/// Run one source and cut its frames into chunks for the queue.
async fn source_task(
    spec: SourceSpec,
    mut chunker: Chunker,
    queue: Arc<ChunkQueue>,
    stats: Arc<PipelineStats>,
    cancel: CancellationToken,
) -> (String, Result<(), birdsong_audio::AudioError>) {
    let SourceSpec {
        source,
        backpressure,
    } = spec;
    let id = source.id().to_string();
    let (frames_tx, mut frames_rx) = mpsc::channel(FRAME_CHANNEL);
    let capture = tokio::spawn(source.run(frames_tx, cancel.clone()));
    let source_id: Arc<str> = Arc::clone(chunker.source_id());

    let enqueue = |item: QueueItem| {
        let queue = Arc::clone(&queue);
        let stats = Arc::clone(&stats);
        async move {
            match queue.push(item, backpressure).await {
                PushOutcome::Queued => true,
                PushOutcome::DroppedOldest {
                    source_id,
                    start_at,
                } => {
                    stats.chunk_dropped();
                    tracing::warn!(source = %source_id, %start_at, "inference is behind; dropped the oldest queued chunk");
                    true
                }
                PushOutcome::ConsumerGone => false,
            }
        }
    };

    let mut consumer_alive = true;
    while let Some(frame) = frames_rx.recv().await {
        for event in chunker.push(frame) {
            let item = match event {
                ChunkerEvent::Chunk(c) => QueueItem::Chunk(c),
                ChunkerEvent::Gap { .. } => {
                    stats.gap();
                    QueueItem::Gap(Arc::clone(&source_id))
                }
            };
            consumer_alive = consumer_alive && enqueue(item).await;
        }
        if !consumer_alive {
            cancel.cancel();
        }
    }

    let result = match capture.await {
        Ok(r) => r,
        Err(e) => Err(birdsong_audio::AudioError::Task(e.to_string())),
    };
    if result.is_ok() && !cancel.is_cancelled() && consumer_alive {
        for tail in chunker.finish() {
            if !enqueue(QueueItem::Chunk(tail)).await {
                break;
            }
        }
    }
    enqueue(QueueItem::End(source_id)).await;
    (id, result)
}

/// The inference thread: model, species filter and per-source privacy state live here.
fn inference_worker(
    mut bundle: ModelBundle,
    queue: Arc<ChunkQueue>,
    results: mpsc::Sender<WorkerOutput>,
    stats: Arc<PipelineStats>,
    detection: DetectionConfig,
    tz: Tz,
) {
    let _guard = ConsumerGuard(Arc::clone(&queue));
    let privacy = bundle.postprocess.privacy_filter;
    // Off unless configured: every species keeps `detection.min_confidence`.
    let mut dynamic = detection.dynamic_threshold.then(|| {
        DynamicThresholds::new(
            detection.min_confidence,
            detection.dynamic_threshold_trigger,
            detection.dynamic_threshold_min,
            TimeDelta::hours(i64::from(detection.dynamic_threshold_hours)),
        )
    });
    let mut masks: HashMap<Arc<str>, NeighbourMask> = HashMap::new();
    let mut filter: Option<(i32, SpeciesFilter)> = None;

    let emit = |analysis: ChunkAnalysis| -> bool {
        if analysis.masked {
            stats.chunk_masked();
        }
        results
            .blocking_send(WorkerOutput::Analysis(analysis))
            .is_ok()
    };

    while let Some(item) = queue.pop() {
        match item {
            QueueItem::Chunk(chunk) => {
                let (local_date, _) = local_date_and_hour(chunk.start_at, tz);
                let week = week_of_year(local_date) as i32;
                if filter.as_ref().is_none_or(|(w, _)| *w != week) {
                    let f = match bundle.species_filter_for_week(week) {
                        Ok(f) => f,
                        Err(e) => {
                            tracing::error!(error = %e, week, "species filter failed; allowing all species this week");
                            SpeciesFilter::allow_all(bundle.labels.len())
                        }
                    };
                    tracing::info!(
                        week,
                        allowed_species = f.num_allowed(),
                        "species filter updated"
                    );
                    filter = Some((week, f));
                }
                let Some((_, species)) = filter.as_ref() else {
                    continue;
                };

                let started = Instant::now();
                let logits = match bundle.classifier.predict(&chunk.samples) {
                    Ok(l) => l,
                    Err(e) => {
                        stats.inference_error();
                        tracing::error!(source = %chunk.source_id, error = %e, "inference failed; chunk skipped");
                        continue;
                    }
                };
                stats.chunk_processed(chunk.start_at, started.elapsed());

                let ctx = ChunkContext {
                    start_at: chunk.start_at,
                    source_id: chunk.source_id.to_string(),
                    model_id: bundle.classifier.model_id().to_string(),
                };
                let start_at = chunk.start_at;
                let threshold_for = |i: usize| match (&dynamic, bundle.labels.get(i)) {
                    (Some(d), Some(label)) => d.threshold_for(&label.scientific, start_at),
                    _ => bundle.postprocess.min_confidence,
                };
                let analysis = analyze_chunk_with(
                    &logits,
                    &bundle.labels,
                    species,
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
                let released = if privacy {
                    masks
                        .entry(Arc::clone(&chunk.source_id))
                        .or_default()
                        .push(analysis)
                } else {
                    Some(analysis)
                };
                if let Some(a) = released {
                    if !emit(a) {
                        return;
                    }
                }
            }
            QueueItem::Missing(source_id) => {
                if privacy {
                    if let Some(a) = masks.entry(source_id).or_default().push_missing() {
                        if !emit(a) {
                            return;
                        }
                    }
                }
            }
            QueueItem::Gap(source_id) => {
                if let Some(a) = masks.get_mut(&source_id).and_then(NeighbourMask::reset) {
                    if !emit(a) {
                        return;
                    }
                }
            }
            QueueItem::End(source_id) => {
                if let Some(a) = masks.get_mut(&source_id).and_then(NeighbourMask::reset) {
                    if !emit(a) {
                        return;
                    }
                }
                if results
                    .blocking_send(WorkerOutput::SourceEnded(source_id))
                    .is_err()
                {
                    return;
                }
            }
        }
    }
    for (_, mut mask) in masks {
        if let Some(a) = mask.flush() {
            if !emit(a) {
                return;
            }
        }
    }
}

/// Persist released detections, log them, broadcast them with their ids, and queue their clip.
///
/// With `detection.min_detections > 1` each source keeps a [`Confirmer`], so a species is stored
/// only once it has been heard repeatedly inside the window; its earlier hits are then stored too.
async fn storage_task(
    store: Arc<dyn DetectionStore>,
    mut results: mpsc::Receiver<WorkerOutput>,
    detections: broadcast::Sender<Detection>,
    stats: Arc<PipelineStats>,
    clips: mpsc::Sender<ClipMsg>,
    detection: DetectionConfig,
) {
    let confirming = detection.min_detections > 1;
    let window =
        TimeDelta::milliseconds((detection.confirmation_window_seconds * 1000.0).round() as i64);
    // Per source: its confirmer and the discard count already reported to `stats`.
    let mut confirmers: HashMap<String, (Confirmer, u64)> = HashMap::new();

    while let Some(output) = results.recv().await {
        let analysis = match output {
            WorkerOutput::SourceEnded(source) => {
                if let Some((mut confirmer, reported)) = confirmers.remove(source.as_ref()) {
                    let flushed = confirmer.flush();
                    stats.detections_unconfirmed(confirmer.discarded() - reported);
                    for a in flushed {
                        store_analysis(&store, &detections, &stats, &clips, a).await;
                    }
                }
                let _ = clips.send(ClipMsg::SourceEnded(source)).await;
                continue;
            }
            WorkerOutput::Analysis(analysis) => analysis,
        };
        if !confirming {
            store_analysis(&store, &detections, &stats, &clips, analysis).await;
            continue;
        }
        let entry = confirmers
            .entry(analysis.source_id.clone())
            .or_insert_with(|| (Confirmer::new(detection.min_detections, window), 0));
        let released = entry.0.push(analysis);
        let discarded = entry.0.discarded();
        stats.detections_unconfirmed(discarded - entry.1);
        entry.1 = discarded;
        for a in released {
            store_analysis(&store, &detections, &stats, &clips, a).await;
        }
    }

    // The queue closed without an End marker (shutdown): settle what is still held.
    for (_, (mut confirmer, reported)) in std::mem::take(&mut confirmers) {
        let flushed = confirmer.flush();
        stats.detections_unconfirmed(confirmer.discarded() - reported);
        for a in flushed {
            store_analysis(&store, &detections, &stats, &clips, a).await;
        }
    }
}

/// Store one settled analysis. Confirmation can empty it, in which case nothing is stored.
async fn store_analysis(
    store: &Arc<dyn DetectionStore>,
    detections: &broadcast::Sender<Detection>,
    stats: &PipelineStats,
    clips: &mpsc::Sender<ClipMsg>,
    analysis: ChunkAnalysis,
) {
    let Some(best) = analysis.detections.first() else {
        return;
    };
    let mut job = ClipJob {
        ids: Vec::new(),
        source_id: analysis.source_id.clone(),
        start_at: analysis.start_at,
        common_name: best.common_name.clone(),
        confidence: best.confidence,
        detections: analysis
            .detections
            .iter()
            .map(|d| {
                (
                    d.scientific_name.clone(),
                    d.common_name.clone(),
                    d.confidence,
                )
            })
            .collect(),
    };
    match insert_with_retry(store.as_ref(), &analysis.detections).await {
        Ok(ids) => {
            stats.detections_stored(ids.len() as u64);
            job.ids.clone_from(&ids);
            if clips.send(ClipMsg::Job(job)).await.is_err() {
                tracing::warn!("clip writer stopped; clip not saved");
            }
            for (mut d, id) in analysis.detections.into_iter().zip(ids) {
                d.id = Some(id);
                tracing::info!(
                    species = ?d.common_name,
                    conf = %format_args!("{:.2}", d.confidence),
                    source = %d.source_id,
                    id,
                    "detection"
                );
                // No subscribers is fine.
                let _ = detections.send(d);
            }
        }
        Err(e) => {
            stats.store_error();
            tracing::error!(error = %e, count = analysis.detections.len(), "failed to store detections");
        }
    }
}

/// Insert detections, retrying twice (after 200 ms, then 1 s) so a briefly locked or busy database
/// does not lose them.
async fn insert_with_retry(
    store: &dyn DetectionStore,
    detections: &[Detection],
) -> Result<Vec<i64>, birdsong_store::StoreError> {
    let mut delay = std::time::Duration::from_millis(200);
    let mut attempt = 1;
    loop {
        match store.insert_many(detections).await {
            Ok(ids) => return Ok(ids),
            Err(e) if attempt < 3 => {
                tracing::warn!(attempt, error = %e, "storing detections failed; retrying");
                tokio::time::sleep(delay).await;
                delay *= 5;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use birdsong_store::{
        ClipInfo, DailyStats, DetectionQuery, DetectionRecord, SpeciesSummary, StoreError,
    };
    use chrono::{DateTime, NaiveDate, TimeZone, Utc};

    /// Fails the first `failures` inserts, then succeeds.
    struct FlakyStore {
        failures: usize,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl DetectionStore for FlakyStore {
        async fn insert_many(&self, detections: &[Detection]) -> Result<Vec<i64>, StoreError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call < self.failures {
                return Err(StoreError::Corrupt {
                    column: "test",
                    value: "database is locked".into(),
                });
            }
            Ok((1..=detections.len() as i64).collect())
        }
        async fn set_clip(&self, _: &[i64], _: Option<&ClipInfo>) -> Result<(), StoreError> {
            Ok(())
        }
        async fn get(&self, _: i64) -> Result<Option<DetectionRecord>, StoreError> {
            Ok(None)
        }
        async fn list(&self, _: &DetectionQuery) -> Result<Vec<DetectionRecord>, StoreError> {
            Ok(Vec::new())
        }
        async fn stats_daily(&self, date: NaiveDate) -> Result<DailyStats, StoreError> {
            Ok(DailyStats {
                date,
                species: Vec::new(),
            })
        }
        async fn species_summary(
            &self,
            _: Option<DateTime<Utc>>,
        ) -> Result<Vec<SpeciesSummary>, StoreError> {
            Ok(Vec::new())
        }
        async fn total_clip_bytes(&self) -> Result<u64, StoreError> {
            Ok(0)
        }
    }

    fn detection() -> Detection {
        Detection {
            id: None,
            detected_at: Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap(),
            scientific_name: "A".into(),
            common_name: "a".into(),
            confidence: 0.9,
            source_id: "mic0".into(),
            model_id: "test".into(),
            clip_path: None,
        }
    }

    #[tokio::test]
    async fn inserts_are_retried_then_given_up() {
        let flaky = FlakyStore {
            failures: 2,
            calls: AtomicUsize::new(0),
        };
        assert_eq!(
            insert_with_retry(&flaky, &[detection()]).await.unwrap(),
            [1]
        );
        assert_eq!(flaky.calls.load(Ordering::SeqCst), 3);

        let broken = FlakyStore {
            failures: usize::MAX,
            calls: AtomicUsize::new(0),
        };
        assert!(insert_with_retry(&broken, &[detection()]).await.is_err());
        assert_eq!(
            broken.calls.load(Ordering::SeqCst),
            3,
            "three attempts in total"
        );
    }
}
