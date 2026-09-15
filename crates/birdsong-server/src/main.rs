#![forbid(unsafe_code)]
//! `birdsong` command-line entry point.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use birdsong_core::Config;
use birdsong_model::ModelBundle;
use birdsong_server::analyze::{self, AnalyzeOptions};
use birdsong_server::{Pipeline, PipelineOptions};
use birdsong_store::{SqliteStore, StoreOptions};
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
    /// Load and validate a configuration file, then print the effective configuration.
    CheckConfig {
        /// Path to the TOML configuration file.
        #[arg(long, default_value = "config/birdsong.toml")]
        config: PathBuf,
    },
    /// Capture audio, detect birds and store detections until stopped (SIGINT/SIGTERM).
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
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();
    match Cli::parse().command {
        Command::CheckConfig { config } => {
            let cfg = Config::load(Some(&config))?;
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
    let bundle = load_bundle(&cfg).await?;
    let store = SqliteStore::open(
        &cfg.storage.database_path(),
        StoreOptions::from_config(&cfg),
    )
    .await
    .context("opening database")?;

    let pipeline = Pipeline::from_config(cfg, bundle, Arc::new(store.clone()), opts)?;
    let cancel = CancellationToken::new();
    tokio::spawn(shutdown_on_signal(cancel.clone()));

    let result = pipeline.run(cancel).await;
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
