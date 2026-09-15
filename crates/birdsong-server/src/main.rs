#![forbid(unsafe_code)]
//! `birdsong` command-line entry point. Subcommands arrive step by step (see `BUILD_PLAN.md`).

use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::CheckConfig { config } => {
            let cfg = birdsong_core::Config::load(Some(&config))?;
            print!("{}", cfg.to_toml());
            tracing::info!(path = %config.display(), "configuration is valid");
        }
    }
    Ok(())
}
