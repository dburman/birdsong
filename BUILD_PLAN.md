# Birdsong — Architectural Build Plan

A Rust re-implementation of the bird-detection feature of
[BirdNET-Pi](https://github.com/Nachtzuster/BirdNET-Pi), designed to run in
Docker on a Raspberry Pi.

This document is written to be executed **one step at a time by an AI coding
model** (Opus, Sonnet, or similar). Every step is a self-contained ticket with
an objective, exact deliverables, interfaces, and acceptance tests. Do not skip
ahead. Do not "improve" earlier steps while doing a later one unless the step
says so.

---

## 0. How to use this document

**Rules for the implementing model**

1. Work on exactly one step at a time, in order. Finish its acceptance tests
   before starting the next step.
2. **No `unsafe`.** Every crate in this workspace has `#![forbid(unsafe_code)]`
   at the top of `lib.rs` / `main.rs`. If a step appears to require `unsafe`
   or a crate that forces you to write `unsafe`, **stop and ask the project
   owner**. Do not work around it silently.
3. Dependencies that contain `unsafe` internally (tokio, sqlite bindings,
   tract's SIMD kernels) are acceptable. Dependencies whose *entire purpose* is
   wrapping a C/C++ library through FFI (`ort`, `tflitec`, `tensorflow`) are
   **not to be added without owner approval**. See the decision log in §3.
4. `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` must pass
   at the end of every step. `unwrap()`/`expect()` are allowed only in tests
   and in `main.rs` startup code.
5. Every public function on a crate boundary gets a doc comment. Every step
   adds tests. Prefer small, boring code over clever code.
6. When a step says "verify against the Python reference", that means running
   the original BirdNET-Analyzer Python code (in a throwaway venv or Docker
   container, dev machine only) and comparing numbers. Python is a **dev-time
   tool only**; it never ships in the runtime image.
7. Record every non-obvious decision in `docs/DECISIONS.md` (create it in
   Step 1). One dated entry per decision, three lines: *Decision / Why /
   Alternatives rejected*.
8. If something in this plan turns out to be wrong (an API changed, a crate is
   abandoned, a number is off), fix the plan **and** note it in
   `docs/DECISIONS.md`. The plan is a living document.
9. **Git:** commit messages, PR titles/descriptions, and code comments must
   never mention Claude, Anthropic, or any AI tool, and must not carry
   `Co-Authored-By` or "Generated with" attribution lines. Write commits as a
   human engineer would: imperative subject, short body explaining why.

**Notation**

- `[VERIFY]` marks a fact taken from memory or upstream docs that must be
  checked against the real code/model before relying on it.
- `[ASK OWNER]` marks a decision the owner must make.

---

## 1. What we are building

### 1.1 The feature being replicated

BirdNET-Pi does this, continuously:

1. Records audio from a USB microphone (or an RTSP stream) at 48 kHz.
2. Cuts the audio into 3-second windows (optionally overlapping).
3. Runs each window through the **BirdNET V2.4** neural network, which outputs
   a score for each of ~6,500 classes (birds plus a few non-bird classes like
   dog, siren, engine, human voice).
4. Applies a sigmoid with a "sensitivity" knob, then filters out species that
   are not expected at the configured latitude/longitude for the current week
   of the year (using a small second "meta" model).
5. Keeps results above a confidence threshold (default 0.7).
6. Writes each detection to SQLite, extracts a ~6-second audio clip around the
   detection, and generates a spectrogram image.
7. Serves a web UI with daily charts, a species list, and recording playback.
8. Periodically deletes old clips to bound disk usage.

### 1.2 Owner requirements (from the brief)

| # | Requirement | Where it lands |
|---|-------------|----------------|
| R1 | Rust wherever reasonable | whole workspace; Python only as a dev-time model-conversion tool |
| R2 | No `unsafe` in project code; ask before using | `#![forbid(unsafe_code)]`, dependency policy in §3 |
| R3 | Easy API to ingest latest birds with times | `GET /api/v1/detections?after_id=…` cursor + SSE stream (Step 8) |
| R4 | Bar charts of latest birds discovered | `/api/v1/stats/*` endpoints + built-in web page (Step 9) |
| R5 | Save sound for playback; configurable rolling deletion window | clip store + retention janitor (Step 7) |
| R6 | Runs on Raspberry Pi in Docker | aarch64 multi-stage Dockerfile + compose (Step 10) |
| R7 | Document other model candidates and improvements incl. other animals | §12 |
| R8 | Spectrograms need not look like BirdNET-Pi's | Step 7 |
| R9 | No AI/Claude mentions anywhere in git history | §0 rule 9 |

### 1.3 Non-goals for v1

- Training or fine-tuning models.
- Replicating BirdNET-Pi's extras: BirdWeather upload, Apprise notifications,
  FTP, terminal-in-browser, file manager, multi-language UI. (Several are
  listed as follow-ups in §12.)
- GPU/NPU acceleration.

---

## 2. System architecture

### 2.1 Runtime data flow

```
 USB mic / RTSP                 (mono f32, 48 kHz)
      │
      ▼
 ┌────────────┐   frames    ┌──────────────┐  3 s chunks  ┌──────────────┐
 │ AudioSource│ ──────────▶ │   Chunker    │ ───────────▶ │  Inference   │
 │ (ffmpeg or │  mpsc       │ + ring buffer│   mpsc       │  worker      │
 │  cpal)     │             │ (last ~90 s) │              │ (blocking    │
 └────────────┘             └──────┬───────┘              │  thread)     │
                                   │ clip requests        └──────┬───────┘
                                   │ (start,len) → samples        │ raw scores
                                   ▼                              ▼
                            ┌──────────────┐  Detection   ┌──────────────┐
                            │  ClipWriter  │ ◀─────────── │ PostProcess  │
                            │ (WAV files)  │              │ sigmoid,     │
                            └──────┬───────┘              │ species filt,│
                                   │                      │ threshold    │
                                   ▼                      └──────────────┘
                            ┌──────────────┐    broadcast  ┌──────────────┐
                            │   Store      │ ────────────▶ │  HTTP API    │
                            │ (SQLite)     │               │  + SSE + UI  │
                            └──────┬───────┘               └──────────────┘
                                   ▲
                            ┌──────┴───────┐
                            │   Janitor    │  (retention window, disk cap)
                            └──────────────┘
```

All boxes run inside **one binary** (`birdsong`) as tokio tasks, except the
inference worker which is a dedicated blocking thread (model inference is
CPU-bound and must not stall the async runtime).

### 2.2 Cargo workspace layout

```
birdsong/
├── Cargo.toml                 # workspace
├── BUILD_PLAN.md              # this file
├── docs/
│   ├── DECISIONS.md           # decision log (Step 1)
│   ├── API.md                 # generated/maintained API reference (Step 8)
│   └── MODEL.md               # model provenance, conversion notes (Step 2)
├── crates/
│   ├── birdsong-core/         # config, domain types, time/week helpers, errors
│   ├── birdsong-audio/        # AudioSource trait, ffmpeg source, chunker, ring buffer, WAV I/O
│   ├── birdsong-model/        # Classifier trait, tract backend, labels, sigmoid, species filter
│   ├── birdsong-store/        # SQLite (sqlx), migrations, clip file layout, retention
│   └── birdsong-server/       # axum API, SSE, static UI, and the `birdsong` binary
├── tools/
│   ├── convert_model/         # Python (dev only): tflite/keras → ONNX, species-list export
│   └── fixtures/              # small test WAVs + golden outputs
├── models/                    # NOT committed; downloaded/converted model files
├── static/                    # web UI (HTML/CSS/JS, vendored chart lib)
├── config/
│   └── birdsong.example.toml
├── Dockerfile
└── docker-compose.yml
```

Why a workspace: crate boundaries stop a model from tangling audio code with
HTTP code, and `birdsong-model` can be swapped for a different inference
backend without touching anything else.

### 2.3 Key interfaces (fixed up front so steps can proceed independently)

These are the contracts between crates. Implementing steps may add methods but
must not change these signatures without a decision-log entry.

```rust
// birdsong-core
pub const SAMPLE_RATE_HZ: u32 = 48_000;
pub const CHUNK_SECONDS: f32 = 3.0;
pub const CHUNK_SAMPLES: usize = 144_000; // 3 s × 48 kHz

pub struct Config { /* see §4 */ }

#[derive(Clone, Debug)]
pub struct Detection {
    pub id: Option<i64>,                     // None until stored
    pub detected_at: chrono::DateTime<chrono::Utc>, // start of the 3 s window
    pub scientific_name: String,
    pub common_name: String,
    pub confidence: f32,                     // 0.0..=1.0 after sigmoid
    pub source_id: String,                   // "mic0", "rtsp1", …
    pub model_id: String,                    // "birdnet-v2.4"
    pub clip_path: Option<String>,           // relative to clips dir
}

// birdsong-audio
pub struct AudioFrame { pub samples: Vec<f32>, pub captured_at: DateTime<Utc> }

#[async_trait::async_trait]
pub trait AudioSource: Send {
    /// Runs until cancelled; pushes mono f32 @ 48 kHz frames into `tx`.
    async fn run(self: Box<Self>, tx: mpsc::Sender<AudioFrame>, cancel: CancellationToken) -> anyhow::Result<()>;
}

pub struct Chunk { pub samples: Arc<[f32]>, /* len == CHUNK_SAMPLES */ pub start_at: DateTime<Utc>, pub source_id: String }

// birdsong-model
pub struct RawPrediction { pub class_index: usize, pub logit: f32 }

pub trait Classifier: Send {
    fn model_id(&self) -> &str;
    fn num_classes(&self) -> usize;
    /// `samples.len()` must equal CHUNK_SAMPLES. Returns one logit per class.
    fn predict(&mut self, samples: &[f32]) -> anyhow::Result<Vec<f32>>;
}

pub struct Labels { /* index → (scientific, common) */ }
pub struct SpeciesFilter { /* set of allowed class indices, or None = allow all */ }

// birdsong-store
#[async_trait::async_trait]
pub trait DetectionStore: Send + Sync {
    async fn insert(&self, d: &Detection) -> anyhow::Result<i64>;
    async fn set_clip_path(&self, id: i64, path: Option<&str>) -> anyhow::Result<()>;
    async fn list(&self, q: &DetectionQuery) -> anyhow::Result<Vec<Detection>>;
    async fn get(&self, id: i64) -> anyhow::Result<Option<Detection>>;
    async fn stats_daily(&self, day: chrono::NaiveDate, tz: chrono_tz::Tz) -> anyhow::Result<DailyStats>;
    async fn species_summary(&self, since: DateTime<Utc>) -> anyhow::Result<Vec<SpeciesSummary>>;
}
```

---

## 3. Technology decisions and dependency policy

| Concern | Choice | Why | Alternatives (and why not now) |
|---------|--------|-----|--------------------------------|
| Async runtime | `tokio` | standard | — |
| HTTP | `axum` + `tower-http` | ergonomic, well documented | actix (fine too, no reason to switch) |
| Config | `serde` + `toml` + env overrides (`config` crate) | simple layered config | figment |
| DB | SQLite via `sqlx` (sqlite feature, bundled) | async, migrations, compile-time query checks | `rusqlite` (sync, fine but needs spawn_blocking everywhere); `turso`/limbo pure-Rust SQLite (promising, not yet stable enough) |
| Audio capture v1 | spawn `ffmpeg` as a child process, read raw PCM from stdout | zero FFI in our tree, handles ALSA devices **and** RTSP identically, trivial in Docker | `cpal` (pure Rust API over ALSA; good v2 option to drop the ffmpeg dependency for USB mics) |
| WAV I/O | `hound` | pure Rust | — |
| DSP (spectrogram PNGs, optional mel frontend) | `rustfft` + `image` | pure Rust | — |
| Inference | **`tract-onnx`** + Rust mel frontend (decided in Step 0) | pure Rust, runs on aarch64, parity with TFLite speed on the dev machine | `tract-tflite` (unsupported ops), `ort` (FFI wrapper, not needed) |
| Charts | vendored `Chart.js` (single JS file in `static/`) driven by the JSON API | simplest possible; no build toolchain | server-side SVG via `plotters` (pure Rust, no JS; good v2 if JS is unwanted) |
| Time zones | `chrono` + `chrono-tz` | daily charts must be in local time | — |
| Logging | `tracing` + `tracing-subscriber` | structured, works in Docker logs | — |
| Errors | `anyhow` in binaries/boundaries, `thiserror` for typed errors inside crates | standard | — |

**Dependency policy (write this into `docs/DECISIONS.md` as entry #1):**

- Allowed without asking: pure-Rust crates, and widely-used crates whose
  `unsafe` is internal and audited (tokio, sqlx/libsqlite3-sys bundled,
  tract, hound, image, rustfft).
- Requires owner approval: any crate that is primarily an FFI binding
  (`ort`, `tflitec`, `tensorflow`, `alsa-sys` used directly, `ffmpeg-next`).
- Never: writing `unsafe` in this workspace.

---

## 4. Configuration schema

File: `config/birdsong.toml` (example committed as
`config/birdsong.example.toml`). Every key can be overridden by an environment
variable `BIRDSONG__SECTION__KEY` (double underscore), e.g.
`BIRDSONG__DETECTION__MIN_CONFIDENCE=0.8`.

```toml
[station]
name = "Backyard"
latitude = 42.36          # required for species filtering; set both to 0 to disable filter
longitude = -71.06
timezone = "America/New_York"   # IANA name; used for daily charts and clip folder dates

[audio]
# One or more sources. Each becomes an independent capture task.
[[audio.sources]]
id = "mic0"
kind = "alsa"             # "alsa" | "rtsp" | "file" (file = replay a WAV, for testing)
device = "hw:1,0"         # ALSA device (ignored for rtsp/file)
# url = "rtsp://…"        # for kind = "rtsp"
# path = "/data/test.wav" # for kind = "file"
gain_db = 0.0

[detection]
min_confidence = 0.7      # BirdNET-Pi default CONFIDENCE
sensitivity = 1.25        # BirdNET-Pi default SENSITIVITY; valid 0.5..=1.5
overlap_seconds = 0.0     # BirdNET-Pi default OVERLAP; valid 0.0..<3.0
species_filter_threshold = 0.03   # BirdNET-Analyzer SF_THRESH default [VERIFY]
top_n_per_chunk = 3       # how many classes per chunk may become detections
include_species = []      # scientific names always allowed (bypass location filter)
exclude_species = []      # scientific names never reported
privacy_filter = true     # BirdNET-Pi human-voice mask (rank based, see §7.6)
privacy_threshold = 0.0   # percent 0..100; higher = stricter (looks deeper into the ranking)

[model]
dir = "/models"
classifier = "birdnet-v2.4-headless.onnx"   # produced by tools/convert_model (docs/MODEL.md)
labels = "labels/en_us.txt"
meta_model = "meta-model.onnx"              # optional
species_list = ""         # optional precomputed list; if set, meta_model is ignored
threads = 0               # 0 = num_cpus - 1, min 1

[storage]
data_dir = "/data"                 # sqlite db + clips live here
clip_seconds = 6.0                 # BirdNET-Pi EXTRACTION_LENGTH; window is centred on the 3 s chunk
clip_format = "wav"                # v1: wav only
spectrograms = true                # generate PNG next to clip

[retention]
clip_max_age_days = 14             # rolling window; clips older than this are deleted
clip_max_total_mb = 4096           # if exceeded, delete oldest clips first (0 = unlimited)
keep_best_per_species_per_day = 1  # these clips are exempt from age/size purge (0 = none exempt)
detection_rows_max_age_days = 0    # 0 = keep detection rows forever (rows are tiny)
purge_interval_minutes = 30

[server]
bind = "0.0.0.0:8080"
cors_allow_origins = ["*"]
```

Validation rules (implemented in `birdsong-core::Config::validate`):
sensitivity ∈ [0.5, 1.5]; overlap ∈ [0, 3); clip_seconds ≥ 3; at least one
audio source; timezone parses; latitude ∈ [-90, 90]; longitude ∈ [-180, 180].

---

## 5. Data model

### 5.1 SQLite schema (migration `0001_init.sql`)

```sql
CREATE TABLE detections (
    id                INTEGER PRIMARY KEY,
    detected_at_utc   TEXT    NOT NULL,   -- RFC 3339, start of 3 s window
    local_date        TEXT    NOT NULL,   -- YYYY-MM-DD in station timezone (for daily charts)
    local_hour        INTEGER NOT NULL,   -- 0..23 in station timezone
    scientific_name   TEXT    NOT NULL,
    common_name       TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    source_id         TEXT    NOT NULL,
    model_id          TEXT    NOT NULL,
    latitude          REAL,
    longitude         REAL,
    week              INTEGER,            -- 1..48 BirdNET week
    sensitivity       REAL,
    overlap_seconds   REAL,
    min_confidence    REAL,
    clip_path         TEXT,               -- relative to <data_dir>/clips; NULL when purged or never saved
    clip_bytes        INTEGER,
    spectrogram_path  TEXT,
    created_at_utc    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_detections_time    ON detections(detected_at_utc DESC);
CREATE INDEX idx_detections_species ON detections(scientific_name, detected_at_utc DESC);
CREATE INDEX idx_detections_day     ON detections(local_date, local_hour);
CREATE INDEX idx_detections_clip    ON detections(clip_path) WHERE clip_path IS NOT NULL;
```

Rows are kept when clips are purged (`clip_path` → NULL), so history and
charts survive the rolling window. This mirrors BirdNET-Pi, whose
`detections` table has the same columns in spirit (Date, Time, Sci_Name,
Com_Name, Confidence, Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name).

### 5.2 Clip file layout

```
<data_dir>/
├── birdsong.sqlite
└── clips/
    └── 2026-09-15/                          # local_date
        └── Northern_Cardinal/               # common name, non-alphanumerics → '_'
            ├── 2026-09-15T13-04-21Z_mic0_0.87.wav
            └── 2026-09-15T13-04-21Z_mic0_0.87.png   # spectrogram
```

---

## 6. HTTP API (v1)

Base path `/api/v1`. All timestamps RFC 3339 UTC. Detection list endpoints return
`{"items": [...], "next_after_id": <int|null>, "next_before_id": <int|null>}`; see `docs/API.md`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{"status":"ok","uptime_s":…,"last_chunk_at":…,"model_id":…}` |
| GET | `/detections` | Query params: `after_id` (cursor, **the recommended way to ingest**), `since` (RFC 3339), `until`, `species` (scientific name), `min_confidence`, `limit` (default 100, max 1000), `order` (`asc`\|`desc`, default `desc`). |
| GET | `/detections/latest?limit=20` | Convenience: newest N. |
| GET | `/detections/{id}` | One detection. |
| GET | `/detections/{id}/audio` | `audio/wav`, supports `Range` (for `<audio>` seeking). 404 if purged. |
| GET | `/detections/{id}/spectrogram.png` | PNG. 404 if absent. |
| GET | `/species?since=…` | Per-species: count, first_seen, last_seen, max_confidence, best_detection_id. Sorted by count desc. |
| GET | `/stats/daily?date=YYYY-MM-DD` | `{date, species:[{scientific_name, common_name, total, by_hour:[24 ints]}]}` — the data for BirdNET-Pi's daily chart. Default = today (station tz). |
| GET | `/stats/recent?window=24h` | `{window, species:[{…, count}]}` — bar chart of latest birds. `window` ∈ `1h,6h,24h,7d,30d`. |
| GET | `/stream` | **Server-Sent Events.** Emits `event: detection` with the JSON `Detection` for every new detection. Supports `Last-Event-ID` = detection id to replay missed events. |
| GET | `/config` | Effective config with nothing secret (RTSP URLs redacted). |
| GET | `/` | Built-in web UI (static). |

Ingestion recipe for a consumer (document this in `docs/API.md`):

```
# Poll: remember the highest id you have seen
curl 'http://pi:8080/api/v1/detections?after_id=1234&order=asc&limit=500'
# Or push: keep one connection open
curl -N 'http://pi:8080/api/v1/stream'
```

---

## 7. Detection math (the part that must match BirdNET-Pi)

Implementers: put these in `birdsong-model/src/postprocess.rs` with unit tests.

1. **Chunking.** Windows of `CHUNK_SAMPLES` (144,000). Step =
   `(3.0 - overlap_seconds) × 48000` samples. Trailing partial windows at end
   of a *file* are zero-padded if ≥ 1.5 s, else dropped (BirdNET's
   `splitSignal` `minlen=1.5`). For live streams there is no trailing window.

2. **Sigmoid with sensitivity.** Verified against BirdNET-Pi `scripts/utils/models.py`:
   ```
   s = clamp(1.0 - (sensitivity - 1.0), 0.5, 1.5)   # user sensitivity 1.25 → s = 0.75
   conf = 1 / (1 + exp(-s * logit))                  # no clipping in BirdNET-Pi
   ```
   Higher user sensitivity → flatter sigmoid → more detections above threshold.
   (BirdNET-Analyzer's `flat_sigmoid` also clips logits to ±20; we follow BirdNET-Pi, see DECISIONS #6.)

3. **Week number (1–48).** Four "weeks" per month, the scheme the meta model was trained on:
   `week = (month - 1) * 4 + min(4, (day - 1) / 7 + 1)`; `-1` means year-round.
   Note: BirdNET-Pi's `species.py` feeds the ISO calendar week (1–53) instead, which is
   slightly wrong for weeks 49–53; we deliberately use the 48-week scheme (DECISIONS #7).

4. **Species filter.** Meta model input is `[lat, lon, week]` (f32×3, raw values), output
   is one probability per class (verified: `tests/golden_meta_v24.rs`; e.g. Boston week 20 →
   126 species at threshold 0.03). Allowed = `{i | p[i] >= species_filter_threshold}`
   ∪ `include_species` − `exclude_species`. If `latitude == 0 && longitude == 0`,
   or no meta model/species list is configured → allow all. Recomputed when
   the week changes (check once an hour).

5. **Selection per chunk.** Sort classes by conf desc; drop disallowed; take
   `top_n_per_chunk`; keep those with `conf >= min_confidence`. Each survivor
   becomes one `Detection` with `detected_at = chunk.start_at`.

6. **Privacy mask (optional).** BirdNET-Pi (`filter_humans` in `utils/analysis.py`) is
   **rank-based**: `cutoff = max(10, int(6000 * privacy_threshold / 100))`; if any class whose
   label contains `Human` appears within the top `cutoff` ranks of a chunk, that chunk **and both
   neighbouring chunks** are replaced by no detections (and no clip). `privacy_threshold` is a
   percentage 0–100; 0 still masks when a Human class is in the top 10. Replicate exactly; make
   the whole feature switchable with `privacy_filter = true|false` (BirdNET-Pi has it always on).

7. **Clip extraction.** Clip covers
   `[chunk.start − (clip_seconds − 3)/2, chunk.start + 3 + (clip_seconds − 3)/2]`,
   clamped to what the ring buffer holds. Default clip_seconds = 6 → 1.5 s of
   context either side. One clip per chunk (shared by all detections in that
   chunk; each detection row gets the same `clip_path`).

8. **Labels file.** One line per class, `Scientific name_Common name`
   (e.g. `Cardinalis cardinalis_Northern Cardinal`). Index = line number. Verified: 6 522 lines.
   Non-bird classes in V2.4 (0-based index) include `Dog_Dog` (1949), `Engine_Engine` (2143),
   `Human non-vocal`/`Human vocal`/`Human whistle` (2818–2820), `Noise_Noise` (3927), `Siren_Siren` (5560).

---

## 8. Step-by-step plan

Each step ends with a checklist. Estimated sizes are for orientation only.

### Step 0 — Feasibility spike: run BirdNET in pure Rust  *(do this first; it decides everything)*

> **STATUS: DONE (2026-09-15).** Outcome: path **(c)** works. BirdNET V2.4 runs in pure Rust as
> `tract-onnx` (headless CNN) + a Rust mel frontend, reproducing the TFLite reference logits to
> 0.0017 max abs error with identical top-1 on all 40 chunks, at ~24–27 ms per chunk on an
> M-series core (Pi numbers still pending hardware). The meta model converts with tf2onnx and runs
> in tract in ~0.6 ms. Paths (a) and (b) failed; (d) `ort` was not needed. Details and checksums:
> `docs/MODEL.md`; reasoning: `docs/DECISIONS.md` #2–#6. What exists now:
> `crates/birdsong-model/src/mel.rs` (frontend + tests), `tests/golden_v24.rs`,
> `tests/golden_meta_v24.rs`, `examples/load_onnx.rs`, `tools/convert_model/*.py`,
> `tools/fixtures/`, `docker/spike.Dockerfile`. **Step 2 should productionise these, not rewrite them.**

**Objective.** Prove that `tract` can load and run the BirdNET V2.4 classifier
on x86_64 **and** on aarch64, with acceptable speed, and that its outputs match
the Python reference. Nothing else in this plan is worth doing until this is
answered.

**Background.** BirdNET V2.4 ships as TFLite (FP32/FP16/INT8) plus a Keras/
SavedModel (`BirdNET_GLOBAL_6K_V2.4_Model`) from the
[BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) repo /
Zenodo. `[VERIFY]` current download locations. The model's first layer computes
a mel spectrogram inside the graph using an STFT (TFLite `RFFT2D` op). That op
is the likely blocker for pure-Rust runtimes.

**Tasks.**

1. Create `tools/convert_model/` with a `README.md`, `requirements.txt`
   (`tensorflow`, `tf2onnx`, `onnx`, `onnxruntime`, `numpy`, `librosa` or
   `soundfile`) and scripts:
   - `export_reference.py`: loads the TFLite FP32 model, runs it on
     `tools/fixtures/*.wav` (3 s chunks), writes `fixtures/golden/<name>.json`
     with the raw logits for every chunk. **This is the golden truth.**
   - `to_onnx.py`: converts the Keras/SavedModel to ONNX with `tf2onnx`.
     Try opset 13–15 first: tf2onnx lowers `tf.signal.stft` to matmuls by
     DFT matrix, which any runtime can execute. `[VERIFY]` Also export the
     MData meta model to ONNX the same way.
   - `species_list.py`: given lat/lon/week, writes the allowed-species list
     (use `birdnet_analyzer` package or run the meta model directly). This is
     the **fallback** if the meta model cannot run in Rust.
2. Create `crates/birdsong-model` skeleton with `TractOnnxClassifier` that
   loads the ONNX, takes `&[f32; 144000]`, returns `Vec<f32>` logits.
   Try in this order, stop at the first that works:
   - (a) `tract-tflite` loading the FP32 `.tflite` directly (no conversion).
   - (b) `tract-onnx` loading the tf2onnx export.
   - (c) **Split the model**: export the graph *after* the mel layer to ONNX
     and implement `MelSpecLayerSimple` in Rust with `rustfft`
     (`[VERIFY]` parameters from `birdnet_analyzer/model.py`: 48 kHz,
     frame_length 2048, frame_step 278, 96 mel bands, fmin 0, fmax 15000,
     Hann window, magnitude, learned magnitude-scaling exponent, output
     96×511). Must match the Python mel output to `max|Δ| < 1e-3`.
   - (d) If (a)–(c) all fail → **[ASK OWNER]** about using `ort` (ONNX
     Runtime FFI). Do not add it unasked.
3. Write `tests/golden.rs`: for each fixture chunk, `max|logit_rust − logit_py| < 1e-2`
   and identical top-5 class ordering. (Tolerance may need loosening for FP16
   weights; note it in DECISIONS.)
4. Benchmark: `cargo bench` or a simple `Instant` loop, 20 chunks, report
   mean ms/chunk on (i) the dev machine, (ii) a Pi 4 or Pi 5 via the aarch64
   Docker image from Step 10 (build a throwaway image early). **Budget:**
   < 2,500 ms/chunk on Pi 4, < 1,000 ms on Pi 5, single source, overlap 0.
   If over budget, try tract's `optimize()` / `into_runnable()` settings,
   FP16 weights, thread count, and note results.
5. Write `docs/MODEL.md`: where files came from, checksums, license note
   (**BirdNET models are CC BY-NC-SA 4.0** — non-commercial; the code is MIT),
   which conversion path worked, benchmark numbers.

**Done when.** Golden test passes on x86_64 and aarch64; benchmark numbers are
recorded; DECISIONS.md has the "inference backend" entry.

---

### Step 1 — Workspace scaffolding

> **STATUS: DONE (2026-09-15).** Five crates compile; `birdsong-core` has `Config` (TOML + env),
> validation, `Detection`, constants, `week_of_year`, `local_date_and_hour`, `sanitize_name`,
> with 10 tests. `birdsong check-config --config <file>` validates and prints the effective config.
> CI: fmt, clippy, tests on x86_64 plus an aarch64 cross build. `Makefile` targets: check, test, run, docker-build.

**Objective.** Empty but compiling workspace with tooling.

**Deliverables.**
- Root `Cargo.toml` workspace with the five crates; shared `[workspace.dependencies]`.
- Each crate: `lib.rs` (or `main.rs`) starting with `#![forbid(unsafe_code)]`.
- `rust-toolchain.toml` (stable), `rustfmt.toml`, `clippy.toml`, `.gitignore`
  (ignore `models/`, `data/`, `target/`).
- `docs/DECISIONS.md` with entry #1 (dependency policy) and the Step 0 outcome.
- `Makefile` or `justfile`: `check` (fmt+clippy+test), `run`, `docker-build`.
- CI: `.github/workflows/ci.yml` running `cargo fmt --check`, `clippy -D warnings`,
  `cargo test` on `ubuntu-latest` (x86_64) and a cross-check build for
  `aarch64-unknown-linux-gnu` (build only).
- `birdsong-core`: `Config` struct + loader (TOML + env), `validate()`,
  `Detection`, constants, `week_of_year()`, `sanitize_name()`, error type.

**Tests.** Config round-trip from `birdsong.example.toml`; validation
rejects bad sensitivity/overlap/timezone; `week_of_year` table test for
Jan 1, Jan 7, Jan 8, Jan 29, Feb 1, Dec 31.

---

### Step 2 — Model crate (productionise the spike)

> **STATUS: DONE (2026-09-15).** `birdsong-model` now has: `Classifier` trait + `TractClassifier`
> (frontend + headless ONNX), `MetaModel`, `Labels`, `SpeciesFilter` (meta model / list file /
> allow-all, with include/exclude), `PostprocessConfig` + `analyze_chunk` (sigmoid with
> sensitivity, rank-based human check, species filter, top-N, threshold), `NeighbourMask`
> (BirdNET-Pi's adjacent-chunk privacy rule, one chunk of latency on live audio), `top_scores`
> for the CLI, and `ModelBundle::load(&Config)`. 16 unit tests + 5 integration tests (golden
> classifier, golden meta model, bundle end-to-end). Library errors are typed (`ModelError`),
> see DECISIONS #9.

**Objective.** Turn the spike into the real `birdsong-model` crate. The mel frontend
(`src/mel.rs`) and the two golden tests already exist and pass; keep them, add the pieces below.

**Deliverables.**
- `Classifier` trait + `TractClassifier` (whichever path Step 0 chose).
- `Labels::load(path)` parsing `Sci_Common` lines; `Labels::get(idx)`.
- `postprocess.rs`: `sigmoid_with_sensitivity`, `select_detections(logits, labels, filter, cfg) -> Vec<Detection>`, privacy mask.
- `species_filter.rs`: `SpeciesFilter::from_meta_model(model, lat, lon, week, threshold)` **and**
  `SpeciesFilter::from_list_file(path)`; `SpeciesFilter::allow_all()`.
- `ModelBundle::load(&Config) -> Result<ModelBundle>` that wires all of the above.
- `TractClassifier` = `BirdnetV24Frontend` + headless ONNX; keep `examples/load_onnx.rs` as a debugging aid.

**Tests.**
- Golden test from Step 0 moved here.
- `sigmoid_with_sensitivity(0.0, 1.0) == 0.5`; monotonic; clamping at ±15.
- `select_detections` with synthetic logits: respects threshold, top_n,
  include/exclude, disallowed-species removal, privacy mask.
- Labels parse the real file: 6,522 lines `[VERIFY count]`, no empty names.

---

### Step 3 — Audio crate: sources, ring buffer, chunker

> **STATUS: DONE (2026-09-15).** `birdsong-audio` has: `AudioSource` trait; `FfmpegSource` for
> alsa/rtsp/file (restart with 1→30 s backoff for live inputs, fatal if ffmpeg is missing,
> credentials redacted from logs, sample-count timestamps re-anchored on >2 s drift);
> pure-Rust `WavFileSource` (fast or realtime pacing); `RingBuffer` (time-indexed, shared via
> `Arc<Mutex<_>>` as `SharedRingBuffer`); `Chunker` (overlap, gap detection with 1 s tolerance,
> `finish()` for ≥1.5 s zero-padded tails); `wav::{read_wav, read_wav_48k_mono, write_wav}`;
> `source_from_config`. Config gained `audio.ffmpeg_path` and `audio.ring_buffer_seconds`
> (default 90, minimum 30 and 2×clip length). 19 unit + 9 integration tests; ffmpeg tests skip
> when ffmpeg is absent and CI installs it. ALSA capture is **untested** until run on Linux
> hardware (Homebrew ffmpeg has no ALSA). See DECISIONS #10.

**Objective.** Continuous 48 kHz mono f32 audio from ffmpeg, cut into
3-second chunks, with a ring buffer for clip extraction.

**Deliverables.**
- `FfmpegSource` implementing `AudioSource`. Spawns:
  ```
  ffmpeg -hide_banner -loglevel warning -nostdin \
    -f alsa -i hw:1,0            # or: -rtsp_transport tcp -i rtsp://…   or: -re -i file.wav
    -ac 1 -ar 48000 -f f32le -   # mono, 48 kHz, raw float32 LE to stdout
  ```
  Reads stdout in 4,800-sample frames (100 ms) with `tokio::process`.
  Restarts ffmpeg with exponential backoff (1 s → 30 s) if it exits. Applies
  `gain_db`. Logs stderr at `warn`.
- `FileSource` (kind = "file"): same as above with `-re` for real-time
  replay, plus a `fast` mode used by tests/CLI that streams as fast as
  possible.
- `RingBuffer`: fixed capacity (default 90 s = 4,320,000 f32 ≈ 17 MB),
  `push(&[f32], at: DateTime)`, `extract(start: DateTime, len_s: f32) -> Option<Vec<f32>>`.
  Time-indexed by sample count from a known anchor; no `unsafe`.
- `Chunker`: consumes `AudioFrame`s, pushes to the ring buffer, emits `Chunk`
  every `step` samples. Handles overlap. Detects gaps (ffmpeg restart) and
  resets alignment, logging a `warn`.
- `wav.rs`: `write_wav(path, &[f32])` (16-bit PCM, 48 kHz mono) and
  `read_wav(path) -> Vec<f32>` (resample not supported; assert 48 kHz mono or
  downmix stereo) using `hound`.

**Tests.**
- Chunker with overlap 0 on 10 s synthetic input → 3 chunks, correct
  `start_at`s. With overlap 1.5 → 5 chunks.
- RingBuffer extract across the wrap boundary returns the right samples;
  extract for a time no longer buffered returns `None`.
- FileSource (fast mode) on a fixture WAV yields the expected sample count.
- WAV round-trip.

---

### Step 4 — Store crate: SQLite + migrations

> **STATUS: DONE (2026-09-15).** `birdsong-store` has `SqliteStore` (sqlx 0.9, bundled SQLite, WAL,
> one writer connection + 4 read-only), migration `0001_init.sql` (§5.1 schema, `AUTOINCREMENT`
> ids), the `DetectionStore` trait (`insert`, `insert_many`, `set_clip`, `get`, `list`,
> `stats_daily`, `species_summary`), `DetectionQuery` (`after_id`, `before_id`, `since`, `until`,
> `species`, `min_confidence`, `limit` ≤ 1000, `order`), `DetectionRecord`, `DailyStats`,
> `SpeciesSummary` (with best clip id), and retention: `stored_clips`, `exempt_clip_paths`,
> `clips_to_purge` (pure `plan_purge`), `total_clip_bytes`, `clear_clip`,
> `delete_detections_before`. 5 unit + 7 integration tests (incl. concurrent cursor paging and
> DST). Interface differences from §2.3 are listed in DECISIONS #11.

**Objective.** Persist and query detections.

**Deliverables.**
- `sqlx` with `sqlite` + `runtime-tokio` + `migrate`; migrations in
  `crates/birdsong-store/migrations/`. WAL mode, `synchronous=NORMAL`,
  busy timeout 5 s, single-writer pool of size 1 for writes + a read pool.
- `SqliteStore` implementing `DetectionStore` (§2.3), plus:
  `clips_to_purge(policy) -> Vec<(id, clip_path, bytes)>`, `total_clip_bytes()`,
  `best_clip_ids_per_species_per_day(n)`.
- `DetectionQuery { after_id, since, until, species, min_confidence, limit, order }`.
- `DailyStats`, `SpeciesSummary` types (serde-serialisable; they are the API
  response shapes).

**Tests** (use a temp-file DB per test):
- insert → get round-trip; list with each filter; cursor pagination is stable
  under concurrent inserts (`after_id` + `asc`).
- `stats_daily` buckets into local hours across a DST change date.
- `clips_to_purge` respects age, total-size cap, and the keep-best exemption.

---

### Step 5 — Pipeline assembly (the `run` command, headless)

> **STATUS: DONE (2026-09-15).** `birdsong-server` is now a library plus the `birdsong` binary.
> `Pipeline` (`from_config` builds ffmpeg sources; `with_sources` for tests) runs one task per
> source (capture + `Chunker`), a shared `ChunkQueue` of 4 chunks (drop-oldest for live sources,
> wait for fast file decoding; gap/end markers never dropped), one `inference` thread (model,
> week-cached species filter, per-source `NeighbourMask`, bypassed when the privacy filter is
> off), and a storage task (insert, one `detection species="…" conf=0.87 source=… id=…` log line,
> `broadcast::Sender<Detection>` with ids). `PipelineStats` counts chunks processed/dropped/masked,
> gaps, detections, errors, EWMA inference ms and last chunk time. `birdsong run --config
> [--exit-on-eof] [--fast-files]` handles SIGINT/SIGTERM and drains queued chunks before exiting;
> logs go to stderr. A fatal source error (e.g. ffmpeg missing) stops the run with a non-zero exit.
> Tests: 5 unit, 3 pipeline integration, 1 CLI end to end. See DECISIONS #12.

**Objective.** One binary that captures, detects, stores. No HTTP yet.

**Deliverables** in `birdsong-server` (binary `birdsong`, use `clap`):
- `birdsong run --config <path>`: builds `Config`, `ModelBundle`, `SqliteStore`,
  one `FfmpegSource` + `Chunker` per configured source, one inference worker
  thread (`std::thread` + `crossbeam`/`std::sync::mpsc` or tokio `spawn_blocking`
  fed by a bounded channel of capacity 4 — **drop the oldest chunk with a
  `warn` counter if inference falls behind**, never grow unbounded).
- `pipeline.rs`: `run_pipeline(cfg, bundle, store, cancel) -> Result<()>`.
  Structure the tasks exactly as in §2.1. Use `tokio_util::sync::CancellationToken`
  and handle SIGTERM/SIGINT for clean shutdown (Docker sends SIGTERM).
- `tokio::sync::broadcast::Sender<Detection>` published for Step 8.
- Metrics counters (plain `AtomicU64`s in a `Stats` struct): chunks processed,
  chunks dropped, detections, last chunk time, mean inference ms (EWMA).
- Structured logs: one `info` line per detection
  `detection species="Northern Cardinal" conf=0.87 source=mic0`.

**Tests.**
- Integration test: `birdsong run` with a `file` source (fast mode) on a
  fixture with a known bird → the DB contains ≥ 1 row for that species and
  the process exits at end-of-file (`--exit-on-eof` flag).

---

### Step 6 — `analyze` CLI (offline tool and debugging aid)

> **STATUS: DONE (2026-09-15).** `birdsong analyze <file> [--config|--models] [--lat --lon]
> [--date YYYY-MM-DD] [--top N] [--no-filter] [--json]` runs the pipeline's own code path
> (chunker incl. padded tail, classifier, `analyze_chunk`, `NeighbourMask`) and prints per chunk
> the top N classes with logit, confidence and whether the species filter allows them, what
> `run` would report, and privacy masking; plus a species summary. 48 kHz WAV is read directly,
> other formats are decoded with ffmpeg. `birdsong species-list [--week N | --date D] [--json]`
> prints allowed species sorted by location score, and fails clearly when no filter is configured.
> Both work without a config file (models from `./models`). `ModelBundle` gained
> `species_filter_kind`, `species_filter_threshold` and `location_scores_for_week`.
> Tests: 2 unit, 4 CLI (golden logits via `--json`, text with location, week-20 count 126 and
> year-round 236 matching the meta golden, error cases). See DECISIONS #13.

**Objective.** `birdsong analyze <file.wav> [--json]` prints per-chunk top-5
with confidences, using the same code path as `run`. This is how humans and
models debug "why didn't it detect X". Also `birdsong species-list` prints
the currently allowed species for the configured lat/lon/week.

**Tests.** Golden fixture via CLI matches the Step 2 golden output.

---

### Step 7 — Clips, spectrograms, retention janitor

> **STATUS: DONE (2026-09-15).** `birdsong-audio::spectrogram` renders an 800×300 PNG (STFT 1024/256,
> dB, 0–12 kHz, dark-to-bright ramp; `png` crate). The pipeline's clip writer
> (`birdsong-server::clips`) receives one job per stored window, waits until the source's ring
> buffer holds the end of the §7.7 window (or the source ended), cuts it (clamped when not
> available), writes `<local date>/<Species>/<UTC time>_<source>_<conf>.wav` and `.png` via
> `.tmp` + rename, and attaches both to every detection of the window. `birdsong-store::Janitor`
> runs `run_once` every `purge_interval_minutes` (age, size cap, best-per-species exemption, row
> age limit that also removes those rows' clips, empty directories; files deleted before rows are
> detached) and `reconcile` at startup (missing files detached, unreferenced files deleted,
> never outside the clips directory via `safe_clip_path`). `birdsong run` reconciles, then runs the
> janitor alongside the pipeline. Stats gained `clips_written`/`clip_errors`.
> Tests: spectrogram size, tone position, silence; clip window, path layout, exact/clamped
> extraction, atomic write; janitor rules, row age, reconcile both ways incl. path traversal;
> pipeline fixture writes the expected clip (216 000 samples, clamped) and PNG. See DECISIONS #14.

**Objective.** Save audio for playback; enforce the rolling window.

**Deliverables.**
- `ClipWriter` task: receives `(chunk, detections)`; on ≥ 1 detection,
  extracts from the ring buffer per §7.7, writes WAV to the §5.2 layout
  (atomic: write to `.tmp` then rename), updates each row's `clip_path`/`clip_bytes`.
- `spectrogram.rs` (in `birdsong-audio`): STFT (`rustfft`, 1024-pt Hann, hop
  256), log-magnitude, map to a grayscale or simple colour map, 800×300 PNG via
  `image`. **The image does not need to resemble BirdNET-Pi's sox-generated
  spectrograms**; any legible time-vs-frequency rendering is acceptable, so
  pick whatever is simplest to implement. Off by default in tests (slow-ish),
  on by default in config.
- `Janitor` task (in `birdsong-store`): every `purge_interval_minutes`:
  1. Compute the exempt set (keep-best per species per day).
  2. Delete clips with `age > clip_max_age_days` (not exempt).
  3. While `total_clip_bytes > clip_max_total_mb`, delete oldest non-exempt clips.
  4. Set `clip_path = NULL` for each deleted file (delete file first, then
     update row; a missing file on startup is also nulled).
  5. If `detection_rows_max_age_days > 0`, delete old rows.
  6. Remove empty date/species directories.
  Log a one-line summary.
- Startup reconciliation: rows whose `clip_path` file is missing → NULL;
  files with no row → deleted (log count).

**Tests.**
- Janitor on a synthetic tree of files + rows: exercises each rule; exempt
  clips survive; the DB and filesystem agree afterwards.
- Clip extraction produces exactly `clip_seconds × 48000` samples when the
  buffer holds enough, fewer (clamped) otherwise.

---

### Step 8 — HTTP API + SSE

> **STATUS: DONE (2026-09-15).** `birdsong-server::api` provides `AppState`, `router(state)` and
> `serve(state, listener)` with every §6 endpoint except `/` (Step 9). JSON errors for bad
> parameters (hand-parsed, never axum's plain-text rejections), unknown ids, missing files and
> unknown routes. Pages carry `next_after_id` and `next_before_id`. Audio and spectrograms are
> served with `ServeFile` (Range support; audio never compressed). `/stream` subscribes before
> replaying from `Last-Event-ID`, never repeats an id, re-replays after a lag, sends keep-alives,
> and ends on shutdown so graceful stop completes. `/config` redacts stream credentials.
> gzip compression, CORS from config, request tracing. `birdsong run` binds the port before
> starting capture and stops the server with the pipeline. `docs/API.md` documents every
> endpoint with curl examples and the ingestion recipe. Tests: 11 API tests on a 50-row database
> (shapes, cursor walk both ways, filters, bad parameters, 404s incl. purged files, Range, PNG,
> species and stats, config redaction, CORS, live event within 1 s, replay without duplicates,
> shutdown, real socket). See DECISIONS #15.

**Objective.** Implement §6 with axum.

**Deliverables.**
- `api/mod.rs` with a router builder `fn router(state: AppState) -> Router`.
- Handlers per §6; JSON via `serde`; errors as `{"error": "..."}` with proper
  status codes; `tower-http` CORS, compression, request tracing.
- `/detections/{id}/audio` via `tower-http::services::ServeFile` (Range support).
- `/stream` SSE from the broadcast channel; on `Last-Event-ID`, first replay
  from the store (`after_id`), then live.
- `docs/API.md` written from §6 with one `curl` example per endpoint and the
  ingestion recipe.

**Tests** (axum `oneshot` or `reqwest` against an in-process server with a
temp DB seeded with 50 rows):
- each endpoint returns the documented shape; cursor pagination walks all rows
  exactly once; `min_confidence` filter; 404s; SSE delivers a detection pushed
  through the broadcast channel within 1 s.

---

### Step 9 — Web UI (charts and playback)

> **STATUS: DONE (2026-09-15).** `static/index.html`, `app.js`, `style.css` and `favicon.svg`,
> compiled into the binary (`include_str!`) and served at `/` by the API router. Plain JavaScript
> with no build step and **no third-party code**: the "latest birds" horizontal bar chart is HTML,
> the "by hour" stacked column chart is inline SVG (top 7 species plus "Other", gridlines, legend,
> per-segment tooltips). Sections: status strip (`/health`, warns when audio stops or chunks
> drop), latest birds with 1h/6h/24h/7d/30d buttons, by-hour chart with a date picker, recent
> detections (live via `EventSource`, flash on arrival, play button, spectrogram thumbnail opening
> a viewer dialog, clip filled in after it is saved), species table with best-recording playback.
> Times are shown in the station time zone from `/config`. Light and dark themes, usable at phone
> width. Tests: dashboard assets served with correct types and gzip; `docs/UI_CHECKLIST.md` for
> manual checks. Deviation: no vendored Chart.js (DECISIONS #17).

**Objective.** A single static page good enough to replace BirdNET-Pi's
dashboard for the core feature.

**Deliverables** in `static/` (served by axum at `/`):
- `index.html`, `app.js`, `style.css`, vendored `chart.umd.js` (pinned
  version, license file next to it).
- Sections:
  1. **Recent detections** table (live via SSE; newest on top; play button
     using `<audio src="/api/v1/detections/{id}/audio">`; spectrogram thumbnail).
  2. **Latest birds** horizontal bar chart from `/stats/recent?window=…` with
     a window selector (1h/6h/24h/7d/30d).
  3. **Today by hour** stacked bar chart from `/stats/daily` (species × hour),
     date picker.
  4. **Species** list from `/species` with count, last seen, best clip link.
  5. **Status** strip from `/health` (last chunk age, inference ms, dropped chunks).
- Works with no build step; no framework. Mobile-friendly width.

**Tests.** Manual checklist in `docs/UI_CHECKLIST.md`; plus one axum test that
`GET /` returns 200 HTML and `GET /app.js` returns JS.

---

### Step 10 — Docker for Raspberry Pi

**Objective.** `docker compose up` on a 64-bit Pi OS works.

**Deliverables.**
- `Dockerfile`, multi-stage:
  ```
  # stage 1: build (runs natively on the builder's arch, or under buildx/QEMU)
  FROM rust:1-bookworm AS build
  … cargo build --release --locked
  # stage 2: runtime
  FROM debian:bookworm-slim
  apt-get install -y --no-install-recommends ffmpeg ca-certificates tzdata && clean
  COPY --from=build /app/target/release/birdsong /usr/local/bin/
  COPY static /app/static
  USER 1000, VOLUME /data /models, EXPOSE 8080
  ENTRYPOINT ["birdsong"] CMD ["run","--config","/config/birdsong.toml"]
  ```
  Prefer `cargo-chef` for layer caching. Build for `linux/arm64` with
  `docker buildx build --platform linux/arm64` from the dev machine (building
  on the Pi itself works but is slow; document both).
- `docker-compose.yml`: `devices: ["/dev/snd:/dev/snd"]`, `group_add: [audio]`,
  volumes `./data:/data`, `./models:/models`, `./config:/config:ro`, `TZ`,
  `restart: unless-stopped`, healthcheck hitting `/api/v1/health`.
- `scripts/fetch-models.sh`: downloads model + labels into `./models`,
  verifies checksums from `docs/MODEL.md`, prints the CC BY-NC-SA notice.
  (Models are **not** baked into the image — license and size.)
- `docs/RASPBERRY_PI.md`: find the ALSA device (`arecord -l`), set
  `device = "hw:X,0"`, USB mic tips, SD-card wear note (WAL + clips on USB
  SSD if possible), expected CPU usage.

**Tests.** `docker build` succeeds for `linux/amd64` and `linux/arm64` in CI;
on a real Pi: `curl /api/v1/health` returns ok and a detection appears within
a few minutes outdoors.

---

### Step 11 — Hardening and observability

- `GET /metrics` Prometheus text (chunks, drops, inference ms, detections by
  species, clip bytes) via `metrics` + `metrics-exporter-prometheus`.
- Backpressure review: confirm memory is bounded (ring buffer + channels).
- Graceful degradation: model load failure → exit non-zero with clear
  message; ffmpeg missing → clear message; DB locked → retries.
- Load test: two sources on a Pi 4 with overlap 1.5 — document whether it
  keeps up; if not, document the drop behaviour.
- `cargo audit` / `cargo deny` in CI (license + advisories); `cargo deny`
  can also **ban FFI crates** (`ort`, `tflitec`) to enforce §3 mechanically.

---

### Step 12 — Nice-to-haves (each is optional, independent, pick by owner priority)

- `cpal` ALSA source (drops the ffmpeg dependency for USB mics).
- FLAC clips (`flacenc`, pure Rust) to cut disk use ~50 %.
- Webhook / ntfy / MQTT notifications on new species or configurable species.
- Home Assistant MQTT discovery.
- BirdWeather upload (BirdNET-Pi feature).
- Localised common names (BirdNET-Analyzer ships per-language label files).
- Live audio stream + live spectrogram over WebSocket.
- Species thumbnails via Wikipedia/Wikimedia lookup (cached on disk).
- Energy-based gate: skip inference when the chunk is near-silent (saves CPU at night).

---

## 9. Testing strategy summary

| Layer | What | Where |
|-------|------|-------|
| Golden | Rust logits == Python logits on fixture WAVs | `birdsong-model/tests/golden.rs` |
| Unit | sigmoid, week, chunker, ring buffer, filter, janitor rules, queries | each crate |
| Integration | `birdsong run` on a file source end-to-end into SQLite | `birdsong-server/tests/` |
| API | every endpoint's shape + pagination + SSE | `birdsong-server/tests/api.rs` |
| Bench | ms/chunk on x86_64 and aarch64 | `docs/MODEL.md` table |
| Hardware | real Pi + mic smoke test | `docs/RASPBERRY_PI.md` checklist |

Fixtures: keep them small (≤ 10 s each, ≤ 2 MB total). Suggested sources:
the `soundscape.wav` example from BirdNET-Analyzer (`[VERIFY]` license), or
Xeno-canto CC-BY recordings (record attribution in `tools/fixtures/README.md`).

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| tract cannot run the in-graph STFT | **happened** | Resolved in Step 0 with path (c): Rust mel frontend + headless model |
| Too slow on Pi 4 | medium | measure in Step 0; options: FP16, fewer threads contention, larger step (overlap 0), energy gate, or recommend Pi 5 |
| Meta (location) model conversion fails | low-medium | precomputed species-list file via Python at setup time (`species_list = …`) |
| SD-card wear from clips | medium | WAL, batched writes, retention caps, recommend USB SSD |
| Model license (CC BY-NC-SA) | certain | non-commercial notice in `docs/MODEL.md` and the fetch script; not baked into image |
| ffmpeg ALSA device naming differs per Pi/mic | high | `docs/RASPBERRY_PI.md`; `birdsong devices` subcommand that shells out to `arecord -l` |

---

## 11. Milestone summary

| Milestone | Steps | You can… |
|-----------|-------|----------|
| M0 Spike | 0 | run BirdNET on a WAV in Rust and trust the numbers |
| M1 Headless detector | 1–6 | run on a Pi (bare) and see detections in SQLite/logs |
| M2 Product | 7–9 | play clips, see charts, ingest via API/SSE, clips roll off |
| M3 Ship | 10–11 | `docker compose up` on a Pi; metrics; CI enforces the no-FFI rule |

---

## 12. Model candidates and improvement opportunities

### 12.1 Bird models (drop-in or near drop-in)

| Model | Notes for this project |
|-------|------------------------|
| **BirdNET V2.4** (default) | ~6.5k classes, 3 s @ 48 kHz, TFLite/Keras, CC BY-NC-SA. Well understood; BirdNET-Pi parity. |
| **BirdNET V3.0 (preview, 2026)** | Evaluated in Step 0: 11 560 species, 3 s at **32 kHz**, sigmoid inside the graph (no sensitivity knob), 1 280-dim embeddings, ONNX FP32 is 516 MB. tract cannot run its ONNX `STFT` op, so it would need the same frontend split as V2.4. Pi 5 territory. Details in `docs/MODEL.md`. Also note BirdNET-Pi itself now offers a `BirdNETGo20250916` model option. |
| BirdNET custom classifiers | BirdNET-Analyzer can train a small head on top of BirdNET embeddings for local species/dialects or new classes. Our `Classifier` trait can expose embeddings to support this later. |
| **Google Perch** (Perch 2.0, 2025) | Apache-2.0 model, ~10k+ bird species and, in 2.0, additional non-bird taxa (mammals, amphibians, insects) `[VERIFY scope]`. 5 s @ 32 kHz input. Larger than BirdNET; likely heavy for a Pi 4, plausible on a Pi 5. Strong embeddings for few-shot / nearest-neighbour search. Best second backend. |
| Merlin Sound ID (Cornell) | closed; not usable. |

### 12.2 Beyond birds

| Target | Candidate | Notes |
|--------|-----------|-------|
| Dogs, sirens, engines, human voice, fireworks | **already in BirdNET V2.4** | just don't filter them out; add a `report_non_bird = true` config and tag them |
| General environmental sounds (cats, cows, frogs, insects, vehicles, alarms, rain) | **YAMNet** (AudioSet, 521 classes, Apache-2.0, ~4 MB, 16 kHz) | tiny and fast; converts to ONNX easily; good "second opinion" model running in parallel |
| Bats | **BatDetect2** (PyTorch) | needs an ultrasonic mic (≥ 256 kHz sampling, e.g. AudioMoth/Pettersson); different capture path; convert to ONNX |
| Frogs / anurans | AnuraSet-trained models; Perch 2.0 | regional; check availability |
| Insects (orthoptera) | InsectSet models; Perch 2.0 | high-frequency, needs ≥ 44.1 kHz; partially covered by Perch 2.0 |
| Zero-shot / "describe what to listen for" | **BioLingual** (CLAP trained on bioacoustics), LAION-CLAP | text-prompted detection; heavier; Pi 5 territory; great for exploration |
| Multi-taxa foundation | Perch 2.0, NatureLM-audio | NatureLM is an LLM-scale model, server only |

Architecture hook: `Classifier` is a trait and detections carry `model_id`, so
running two models on the same chunk stream (e.g. BirdNET + YAMNet) is a
matter of spawning a second inference worker and tagging rows. Add a
`[[model]]` array to config when that day comes.

### 12.3 Improvement opportunities

**Accuracy**
- Overlap 1.5 s (halves the chance a call straddles a window boundary) at the
  cost of 2× inference.
- Temporal smoothing: require a species to appear in 2 of the last 3 windows
  before reporting (cuts one-off false positives; BirdNET-Pi does not do this).
- Per-species thresholds (owls at night vs. common confusions).
- Embedding store + nearest-neighbour review UI ("show me clips that sound
  like this one") using BirdNET/Perch embeddings; enables local re-labelling.
- Confusion pairs list in the UI (BirdNET's known confusions) for human review.

**Performance / Pi**
- INT8 weights if the chosen runtime supports them well (tract's quantised
  support is limited; `[VERIFY]`).
- Energy/VAD gate before inference; sleep schedule (e.g. don't run at 3 a.m.
  if you only care about songbirds, or *only* run at night for owls).
- NPU add-ons: Raspberry Pi AI Kit/HAT (Hailo-8) needs Hailo's compiler and
  FFI runtime — conflicts with the no-FFI policy; Coral TPU needs TFLite —
  same. Both are **owner decisions** if ever wanted.

**Product**
- Notifications on first-of-season / rare species (ntfy, webhook, MQTT).
- Weekly email/markdown summary; export CSV.
- Multi-station aggregation: the `after_id` cursor API makes a central
  collector trivial (poll N Pis, merge by `station.name`).
- BirdWeather / eBird-style data sharing.
- Privacy: on-device only by default; the human-voice mask (§7.6) already exists.

---

## Appendix A — Quick reference of BirdNET-Pi defaults being matched

| BirdNET-Pi key | Default | Ours |
|----------------|---------|------|
| RECORDING_LENGTH | 15 s files | n/a (streaming; no intermediate files) |
| CONFIDENCE | 0.7 | `detection.min_confidence` |
| SENSITIVITY | 1.25 | `detection.sensitivity` |
| OVERLAP | 0.0 | `detection.overlap_seconds` |
| EXTRACTION_LENGTH | 6 s | `storage.clip_seconds` |
| PRIVACY_THRESHOLD | 0 | `detection.privacy_threshold` |
| SF_THRESH (location filter) | 0.03 | `detection.species_filter_threshold` |
| LATITUDE / LONGITUDE | — | `station.latitude/longitude` |
| Audio | 48 kHz s16 stereo WAV via arecord/ffmpeg | 48 kHz f32 mono via ffmpeg pipe |
| DB | SQLite `detections` | SQLite `detections` (§5.1) |
| Purge | `keep best` + age-based | `[retention]` (§4) |

## Appendix B — Glossary

- **Chunk / window**: 3 s of audio = one model input.
- **Logit**: raw model output before sigmoid.
- **Confidence**: sigmoid(logit) with sensitivity applied.
- **Meta model**: tiny BirdNET model mapping (lat, lon, week) → per-species occurrence probability.
- **Week**: BirdNET's 48-week year (4 per month).
- **Clip**: extracted audio around a detection, saved for playback.
- **Rolling window**: age/size-based deletion of clips; rows are kept.
