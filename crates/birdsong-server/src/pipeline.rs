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
use birdsong_core::config::AudioSourceKind;
use birdsong_core::{local_date_and_hour, week_of_year, Config, Detection};
use birdsong_model::{
    analyze_chunk, ChunkAnalysis, ChunkContext, ModelBundle, NeighbourMask, SpeciesFilter,
};
use birdsong_store::DetectionStore;
use chrono::Utc;
use chrono_tz::Tz;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::queue::{Backpressure, ChunkQueue, ConsumerGuard, PushOutcome, QueueItem};
use crate::stats::{PipelineStats, StatsSnapshot};

/// Chunks waiting for inference, across all sources.
pub const QUEUE_CAPACITY: usize = 4;
/// Frames buffered between a source and its chunker (100 ms each).
const FRAME_CHANNEL: usize = 64;
/// Released chunk analyses waiting for the storage task.
const RESULT_CHANNEL: usize = 64;
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
        let (results_tx, results_rx) = mpsc::channel::<ChunkAnalysis>(RESULT_CHANNEL);

        let worker = {
            let queue = Arc::clone(&queue);
            let stats = Arc::clone(&stats);
            let tz = cfg.station.timezone;
            std::thread::Builder::new()
                .name("inference".into())
                .spawn(move || inference_worker(bundle, queue, results_tx, stats, tz))
                .context("starting inference thread")?
        };
        let storage = tokio::spawn(storage_task(
            store,
            results_rx,
            detections,
            Arc::clone(&stats),
        ));

        let sources_cancel = cancel.child_token();
        let mut tasks = JoinSet::new();
        for spec in sources {
            tasks.spawn(source_task(
                spec,
                Arc::clone(&queue),
                Arc::clone(&stats),
                sources_cancel.child_token(),
                cfg.detection.overlap_seconds,
                cfg.audio.ring_buffer_seconds,
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
        Ok(summary)
    }
}

/// Run one source and cut its frames into chunks for the queue.
async fn source_task(
    spec: SourceSpec,
    queue: Arc<ChunkQueue>,
    stats: Arc<PipelineStats>,
    cancel: CancellationToken,
    overlap_seconds: f32,
    ring_buffer_seconds: f32,
) -> (String, Result<(), birdsong_audio::AudioError>) {
    let SourceSpec {
        source,
        backpressure,
    } = spec;
    let id = source.id().to_string();
    let (frames_tx, mut frames_rx) = mpsc::channel(FRAME_CHANNEL);
    let capture = tokio::spawn(source.run(frames_tx, cancel.clone()));
    let mut chunker = Chunker::new(id.as_str(), overlap_seconds, ring_buffer_seconds);
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
    results: mpsc::Sender<ChunkAnalysis>,
    stats: Arc<PipelineStats>,
    tz: Tz,
) {
    let _guard = ConsumerGuard(Arc::clone(&queue));
    let privacy = bundle.postprocess.privacy_filter;
    let mut masks: HashMap<Arc<str>, NeighbourMask> = HashMap::new();
    let mut filter: Option<(i32, SpeciesFilter)> = None;

    let emit = |analysis: ChunkAnalysis| -> bool {
        if analysis.masked {
            stats.chunk_masked();
        }
        results.blocking_send(analysis).is_ok()
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
                let analysis =
                    analyze_chunk(&logits, &bundle.labels, species, &bundle.postprocess, &ctx);
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
            QueueItem::Gap(source_id) | QueueItem::End(source_id) => {
                if let Some(a) = masks.get_mut(&source_id).and_then(NeighbourMask::reset) {
                    if !emit(a) {
                        return;
                    }
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

/// Persist released detections, log them, and broadcast them with their ids.
async fn storage_task(
    store: Arc<dyn DetectionStore>,
    mut results: mpsc::Receiver<ChunkAnalysis>,
    detections: broadcast::Sender<Detection>,
    stats: Arc<PipelineStats>,
) {
    while let Some(analysis) = results.recv().await {
        if analysis.detections.is_empty() {
            continue;
        }
        match store.insert_many(&analysis.detections).await {
            Ok(ids) => {
                stats.detections_stored(ids.len() as u64);
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
}
