#![forbid(unsafe_code)]
//! `birdsong` command-line entry point.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use birdsong_core::Config;
use birdsong_model::ModelBundle;
use birdsong_server::analyze::{self, AnalyzeOptions};
use birdsong_server::api::{self, AppState};
use birdsong_server::{Pipeline, PipelineOptions};
use birdsong_store::{DetectionStore, Janitor, SqliteStore, StoreOptions};
use chrono::NaiveDate;
use clap::{Args, Parser, Subcommand};
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(
    name = "birdsong",
    version,
    about = "Bird sound detection for small computers"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List microphones and other capture devices by their stable ALSA names.
    Devices,
    /// Load and validate a configuration file, then print the effective configuration.
    CheckConfig {
        /// Path to the TOML configuration file.
        #[arg(long, default_value = "config/birdsong.toml")]
        config: PathBuf,
    },
    /// Capture audio, detect birds, store detections and serve the HTTP API until stopped.
    Run {
        /// Path to the TOML configuration file.
        #[arg(long, default_value = "config/birdsong.toml")]
        config: PathBuf,
        /// Exit once every audio source has ended (useful with `kind = "file"` sources).
        #[arg(long)]
        exit_on_eof: bool,
        /// Decode file sources as fast as possible instead of at recording speed.
        #[arg(long)]
        fast_files: bool,
    },
    /// Show per-chunk classifier scores for a recording and what `run` would report.
    Analyze {
        /// Audio file (48 kHz WAV is read directly; anything else is decoded with ffmpeg).
        file: PathBuf,
        #[command(flatten)]
        tool: ToolArgs,
        /// Classes to show per chunk.
        #[arg(long, default_value_t = 5)]
        top: usize,
        /// Recording date (YYYY-MM-DD) for the location filter; defaults to today.
        #[arg(long)]
        date: Option<NaiveDate>,
        /// Ignore the species filter (score every class as allowed).
        #[arg(long)]
        no_filter: bool,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// List the species the configured filter allows for a week.
    SpeciesList {
        #[command(flatten)]
        tool: ToolArgs,
        /// BirdNET week 1..=48, or -1 for year-round. Defaults to the week of --date or today.
        #[arg(long, allow_negative_numbers = true)]
        week: Option<i32>,
        /// Date (YYYY-MM-DD) whose week to use.
        #[arg(long, conflicts_with = "week")]
        date: Option<NaiveDate>,
        /// Print JSON instead of tab-separated text.
        #[arg(long)]
        json: bool,
    },
    /// Exit 0 if the HTTP API answers with 2xx (used by the Docker HEALTHCHECK).
    Healthcheck {
        /// Health endpoint to request.
        #[arg(long, default_value = "http://127.0.0.1:8080/api/v1/health")]
        url: String,
        /// Give up after this many seconds.
        #[arg(long, default_value_t = 5)]
        timeout_secs: u64,
    },
}

/// Shared options for the offline tools.
#[derive(Args)]
struct ToolArgs {
    /// Configuration file. Without one, defaults are used with models from ./models.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Model directory (overrides the configuration).
    #[arg(long)]
    models: Option<PathBuf>,
    /// Station latitude (overrides the configuration; needs --lon).
    #[arg(long, allow_negative_numbers = true)]
    lat: Option<f64>,
    /// Station longitude (overrides the configuration; needs --lat).
    #[arg(long, allow_negative_numbers = true)]
    lon: Option<f64>,
}

impl ToolArgs {
    fn config(self) -> anyhow::Result<Config> {
        analyze::tool_config(self.config.as_deref(), self.models, self.lat, self.lon)
    }
}

fn init_logging() {
    // Colour only on a terminal: systemd journals and log files would otherwise keep the escape
    // codes. `NO_COLOR` (https://no-color.org) turns it off everywhere.
    let ansi = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(ansi)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

/// Warn about ALSA sources named by card number, which can change between boots.
fn warn_unstable_devices(cfg: &Config) {
    let devices = birdsong_audio::alsa_names::capture_devices();
    for src in &cfg.audio.sources {
        if src.kind != birdsong_core::config::AudioSourceKind::Alsa {
            continue;
        }
        if let Some(device) = &src.device {
            if let Some(warning) =
                birdsong_audio::alsa_names::unstable_name_warning(device, &devices)
            {
                tracing::warn!(source = %src.id, "{warning}");
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();
    match Cli::parse().command {
        Command::Healthcheck { url, timeout_secs } => {
            let timeout = Duration::from_secs(timeout_secs.max(1));
            let status = birdsong_server::healthcheck::check(&url, timeout).await?;
            println!("ok: HTTP {status} from {url}");
            Ok(())
        }
        Command::Devices => {
            let devices = birdsong_audio::alsa_names::capture_devices();
            if devices.is_empty() {
                println!("no ALSA capture devices found (Linux only; is a microphone connected?)");
            }
            for d in devices {
                println!(
                    "{:<34} card {}, device {}: {}",
                    d.stable_name(),
                    d.card,
                    d.device,
                    d.card_name
                );
            }
            Ok(())
        }
        Command::CheckConfig { config } => {
            let cfg = Config::load(Some(&config))?;
            warn_unstable_devices(&cfg);
            print!("{}", cfg.to_toml());
            tracing::info!(path = %config.display(), "configuration is valid");
            Ok(())
        }
        Command::Run {
            config,
            exit_on_eof,
            fast_files,
        } => {
            run(
                config,
                PipelineOptions {
                    exit_on_eof,
                    fast_files,
                },
            )
            .await
        }
        Command::Analyze {
            file,
            tool,
            top,
            date,
            no_filter,
            json,
        } => {
            let cfg = tool.config()?;
            let mut bundle = load_bundle(&cfg).await?;
            let samples = analyze::load_audio(&file, &cfg.audio.ffmpeg_path).await?;
            let opts = AnalyzeOptions {
                top: top.max(1),
                date: date.unwrap_or_else(|| analyze::station_today(&cfg)),
                apply_species_filter: !no_filter,
            };
            let label = file.file_name().map_or_else(
                || file.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            let report = tokio::task::spawn_blocking(move || {
                analyze::analyze_samples(&mut bundle, &cfg, &samples, &label, &opts)
            })
            .await
            .context("analysis task")??;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", analyze::render_text(&report));
            }
            Ok(())
        }
        Command::SpeciesList {
            tool,
            week,
            date,
            json,
        } => {
            let cfg = tool.config()?;
            let week = match (week, date) {
                (Some(w), _) => w,
                (None, Some(d)) => birdsong_core::week_of_year(d) as i32,
                (None, None) => birdsong_core::week_of_year(analyze::station_today(&cfg)) as i32,
            };
            let bundle = load_bundle(&cfg).await?;
            let report = analyze::species_list(&bundle, week)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                tracing::info!(
                    week = report.week,
                    filter = report.species_filter,
                    threshold = ?report.threshold,
                    count = report.count,
                    "allowed species"
                );
                print!("{}", analyze::render_species_list(&report));
            }
            Ok(())
        }
    }
}

async fn load_bundle(cfg: &Config) -> anyhow::Result<ModelBundle> {
    let cfg = cfg.clone();
    tokio::task::spawn_blocking(move || ModelBundle::load(&cfg))
        .await
        .context("model loading task")?
        .context("loading models")
}

async fn run(config: PathBuf, opts: PipelineOptions) -> anyhow::Result<()> {
    let cfg =
        Config::load(Some(&config)).with_context(|| format!("loading {}", config.display()))?;
    warn_unstable_devices(&cfg);
    let bundle = load_bundle(&cfg).await?;
    let store = SqliteStore::open(
        &cfg.storage.database_path(),
        StoreOptions::from_config(&cfg),
    )
    .await
    .context("opening database")?;

    let janitor = Janitor::new(
        store.clone(),
        cfg.storage.clips_dir(),
        cfg.retention.clone(),
    );
    match janitor.reconcile().await {
        Ok(r) => tracing::info!(
            rows_cleared = r.rows_cleared,
            orphan_files_deleted = r.orphan_files_deleted,
            dirs_removed = r.dirs_removed,
            errors = r.errors,
            "clips directory reconciled"
        ),
        Err(e) => tracing::warn!(error = %e, "clips directory reconciliation failed"),
    }

    // Bind before capturing audio so a port conflict fails fast.
    let bind = cfg.server.bind_addr()?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding HTTP API to {bind}"))?;
    tracing::info!(addr = %listener.local_addr()?, "HTTP API listening");

    let cancel = CancellationToken::new();
    tokio::spawn(shutdown_on_signal(cancel.clone()));

    let config_for_api = Arc::new(cfg.clone());
    let clips_dir = cfg.storage.clips_dir();
    let model_id = bundle.classifier.model_id().to_string();
    let store_handle: Arc<dyn DetectionStore> = Arc::new(store.clone());
    let pipeline = Pipeline::from_config(cfg, bundle, Arc::clone(&store_handle), opts)?;

    let state = AppState {
        store: store_handle,
        clips_dir,
        config: config_for_api,
        stats: pipeline.stats(),
        detections: pipeline.detections_sender(),
        model_id,
        started: Instant::now(),
        shutdown: cancel.clone(),
    };
    let server = tokio::spawn(api::serve(state, listener));
    let janitor_task = tokio::spawn(janitor.run(cancel.clone()));

    let result = pipeline.run(cancel.clone()).await;
    cancel.cancel();
    let _ = janitor_task.await;
    match server.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(error = %e, "HTTP API stopped with an error"),
        Err(e) => tracing::warn!(error = %e, "HTTP API task panicked"),
    }
    store.close().await;
    result.map(|_| ())
}

/// Cancel on Ctrl-C or SIGTERM (what `docker stop` sends).
async fn shutdown_on_signal(cancel: CancellationToken) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "cannot listen for SIGTERM; only Ctrl-C will stop");
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
    tracing::info!("shutdown requested");
    cancel.cancel();
}
