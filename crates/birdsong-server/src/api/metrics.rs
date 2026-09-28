//! `GET /metrics` in the Prometheus text exposition format.

use std::fmt::Write as _;

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use chrono::Utc;

use super::error::ApiResult;
use super::AppState;

/// Escape a label value: backslash, double quote and newline.
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn metric(out: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
    );
}

pub async fn metrics(State(state): State<AppState>) -> ApiResult<Response> {
    let s = state.stats.snapshot();
    let clip_bytes = state.store.total_clip_bytes().await?;
    let species = state.store.species_summary(None, None).await?;
    let now = Utc::now();

    let mut out = String::with_capacity(4096);
    let _ = writeln!(
        out,
        "# HELP birdsong_build_info Build and station information.\n# TYPE birdsong_build_info gauge\nbirdsong_build_info{{version=\"{}\",model=\"{}\",station=\"{}\"}} 1",
        label(env!("CARGO_PKG_VERSION")),
        label(&state.model_id),
        label(&state.config.station.name)
    );
    metric(
        &mut out,
        "birdsong_uptime_seconds",
        "gauge",
        "Seconds since the process started.",
        state.started.elapsed().as_secs(),
    );
    metric(
        &mut out,
        "birdsong_chunks_processed_total",
        "counter",
        "Analysis windows run through the classifier.",
        s.chunks_processed,
    );
    metric(
        &mut out,
        "birdsong_chunks_dropped_total",
        "counter",
        "Windows dropped because inference fell behind.",
        s.chunks_dropped,
    );
    metric(
        &mut out,
        "birdsong_masked_chunks_total",
        "counter",
        "Windows blanked by the human-voice privacy filter.",
        s.masked_chunks,
    );
    metric(
        &mut out,
        "birdsong_audio_gaps_total",
        "counter",
        "Discontinuities in captured audio.",
        s.gaps,
    );
    metric(
        &mut out,
        "birdsong_clock_reanchors_total",
        "counter",
        "Times capture timestamps jumped to the wall clock (stalls, restarts, excess drift).",
        s.clock_reanchors,
    );
    metric(
        &mut out,
        "birdsong_clock_samples_inserted_total",
        "counter",
        "Samples repeated to keep a slow capture clock in step with the wall clock.",
        s.clock_samples_inserted,
    );
    metric(
        &mut out,
        "birdsong_clock_samples_dropped_total",
        "counter",
        "Samples skipped to keep a fast capture clock in step with the wall clock.",
        s.clock_samples_dropped,
    );
    if let Some(ppm) = s.clock_correction_ppm {
        metric(
            &mut out,
            "birdsong_clock_correction_ppm",
            "gauge",
            "Net capture clock correction in parts per million (positive: the microphone runs slow).",
            ppm,
        );
    }
    metric(
        &mut out,
        "birdsong_detections_total",
        "counter",
        "Detections stored since the process started.",
        s.detections,
    );
    metric(
        &mut out,
        "birdsong_inference_errors_total",
        "counter",
        "Classifier failures.",
        s.inference_errors,
    );
    metric(
        &mut out,
        "birdsong_store_errors_total",
        "counter",
        "Detections that could not be stored.",
        s.store_errors,
    );
    metric(
        &mut out,
        "birdsong_clips_written_total",
        "counter",
        "Audio clips saved.",
        s.clips_written,
    );
    metric(
        &mut out,
        "birdsong_clip_errors_total",
        "counter",
        "Clips that could not be saved.",
        s.clip_errors,
    );
    if let Some(ms) = s.mean_inference_ms {
        metric(
            &mut out,
            "birdsong_inference_seconds",
            "gauge",
            "Moving average of classifier time per window.",
            ms / 1000.0,
        );
    }
    if let Some(at) = s.last_processed_at {
        let age = (now - at).num_milliseconds().max(0) as f64 / 1000.0;
        metric(
            &mut out,
            "birdsong_seconds_since_last_chunk",
            "gauge",
            "Wall-clock seconds since a window was last analysed.",
            age,
        );
    }
    metric(
        &mut out,
        "birdsong_birdweather_soundscapes_total",
        "counter",
        "Clips uploaded to BirdWeather.",
        s.birdweather_soundscapes,
    );
    metric(
        &mut out,
        "birdsong_birdweather_detections_total",
        "counter",
        "Detections accepted by BirdWeather.",
        s.birdweather_detections,
    );
    metric(
        &mut out,
        "birdsong_birdweather_errors_total",
        "counter",
        "BirdWeather uploads that failed.",
        s.birdweather_errors,
    );
    metric(
        &mut out,
        "birdsong_birdweather_skipped_total",
        "counter",
        "BirdWeather uploads skipped at shutdown.",
        s.birdweather_skipped,
    );
    metric(
        &mut out,
        "birdsong_unconfirmed_detections_total",
        "counter",
        "Detections dropped because their species was never confirmed.",
        s.unconfirmed_detections,
    );
    metric(
        &mut out,
        "birdsong_clip_bytes",
        "gauge",
        "Disk used by saved clips and spectrograms.",
        clip_bytes,
    );
    let _ = writeln!(
        out,
        "# HELP birdsong_species_detections Stored detections per species or sound event (all time).\n# TYPE birdsong_species_detections gauge"
    );
    for sp in &species {
        let _ = writeln!(
            out,
            "birdsong_species_detections{{scientific_name=\"{}\",common_name=\"{}\",kind=\"{}\"}} {}",
            label(&sp.scientific_name),
            label(&sp.common_name),
            sp.kind.as_str(),
            sp.count
        );
    }
    Ok((
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        out,
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::label;

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(label(r#"a "b" \c"#), r#"a \"b\" \\c"#);
        assert_eq!(label("x\ny"), "x\\ny");
    }
}
