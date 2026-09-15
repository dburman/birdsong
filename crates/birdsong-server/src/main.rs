#![forbid(unsafe_code)]
//! `birdsong` command-line entry point.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use birdsong_core::Config;
use birdsong_model::ModelBundle;
use birdsong_server::{Pipeline, PipelineOptions};
use birdsong_store::{SqliteStore, StoreOptions};
use clap::{Parser, Subcommand};
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
    }
}

async fn run(config: PathBuf, opts: PipelineOptions) -> anyhow::Result<()> {
    let cfg =
        Config::load(Some(&config)).with_context(|| format!("loading {}", config.display()))?;

    let bundle = {
        let cfg = cfg.clone();
        tokio::task::spawn_blocking(move || ModelBundle::load(&cfg))
            .await
            .context("model loading task")?
            .context("loading models")?
    };
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
