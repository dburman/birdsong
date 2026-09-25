# Decision log

One entry per non-obvious decision. Newest at the bottom. Format: Decision / Why / Alternatives rejected.

## 1. 2026-09-15 — Dependency policy: no `unsafe`, no FFI-wrapper crates without approval

- **Decision.** Every crate has `#![forbid(unsafe_code)]`. Pure-Rust crates and widely used crates
  with internal, audited `unsafe` (tokio, sqlx/bundled SQLite, tract, hound, rustfft, image) are
  allowed. Crates whose purpose is wrapping a C/C++ library (`ort`, `tflitec`, `tensorflow`,
  `ffmpeg-next`, direct `alsa-sys`) need the owner's explicit approval first.
- **Why.** Owner requirement: no `unsafe` in project code, ask before using it.
- **Rejected.** Blanket ban on any `unsafe` in the dependency tree — impossible (even `std` and
  tokio contain it) and would forbid tract's SIMD kernels.

## 2. 2026-09-15 — Inference backend: tract-onnx + headless BirdNET V2.4 + Rust mel frontend

- **Decision.** Run the BirdNET V2.4 CNN from its `concatenate` layer onward as an ONNX model in
  `tract-onnx` (pure Rust), and compute the two mel-spectrogram layers in Rust (`rustfft`).
- **Why.** Golden test reproduces the TFLite reference logits to 0.0017 max abs error over 40
  chunks with identical top-1 classes, at 24–27 ms per chunk on an M-series core (parity with
  TFLite+XNNPACK). No FFI, no `unsafe` in our tree.
- **Rejected.**
  - (a) `tract-tflite` on the stock `.tflite`: unsupported ops (`SPLIT_V`; `STRIDED_SLICE` with shrink axis).
  - (b) `tf2onnx --tflite` whole model: converter refuses `RFFT2D` not followed by `ComplexAbs`.
  - (d) `ort` (ONNX Runtime): not needed; would require FFI approval.
  - BirdNET V3.0 ONNX: preview, 516 MB, 32 kHz, uses ONNX `STFT` which tract cannot run. See `MODEL.md`.

## 3. 2026-09-15 — Golden tolerances

- **Decision.** Logits: max abs error < 0.05 and identical top-1 per chunk. Mel filterbank vs
  TensorFlow dump: < 1e-4. Spectrogram: < 1 % of peak value. Meta model: < 1e-3 per probability
  and identical allowed-species count at threshold 0.03.
- **Why.** Observed errors are 0.0017, 1.2e-5, and ≪1 % respectively; tolerances leave headroom for
  aarch64 float differences without hiding real bugs.
- **Rejected.** Bit-exactness — FFT ordering and fused ops differ between runtimes.

## 4. 2026-09-15 — Fixtures: commit a 15 s clip, keep the 2-minute original local

- **Decision.** `tools/fixtures/soundscape_15s.wav` (1.4 MB, 5 chunks) plus its golden logits
  (676 KB) and one binary reference spectrogram (392 KB) are committed. The full `soundscape.wav`
  and its golden JSON are git-ignored; tests use them when present.
- **Why.** Keeps the repo under ~3 MB of fixtures while still giving CI a real end-to-end check.
- **Rejected.** Committing the 12 MB WAV and 5.4 MB JSON; downloading fixtures in CI (adds a
  network dependency to every test run).

## 5. 2026-09-15 — Python is a dev-time tool only

- **Decision.** `tools/convert_model/` (TensorFlow, tf2onnx, `birdnet`) produces the ONNX files,
  filterbank dumps and golden outputs. Nothing Python ships in the runtime image.
- **Why.** Owner wants Rust; the model conversion is a one-off per model version.
- **Rejected.** Converting at container start (would pull TensorFlow onto the Pi).

## 6. 2026-09-15 — Sigmoid/sensitivity follows BirdNET-Pi, not BirdNET-Analyzer

- **Decision.** `conf = 1 / (1 + exp(-s * logit))` with `s = clamp(1 - (sensitivity - 1), 0.5, 1.5)`,
  no clipping of the logit (BirdNET-Pi `scripts/utils/models.py`). Default `sensitivity = 1.25` → `s = 0.75`.
- **Why.** We replicate BirdNET-Pi's numbers. BirdNET-Analyzer's `flat_sigmoid` additionally clips
  logits to ±20 and has a `bias` term; the difference is negligible in practice but we pick one.
- **Rejected.** BirdNET-Analyzer's variant.

## 7. 2026-09-15 — Week numbering: BirdNET's 48-week scheme, not ISO weeks

- **Decision.** `week = (month - 1) * 4 + min(4, (day - 1) / 7 + 1)`, range 1–48; `-1` = year-round.
- **Why.** The meta model was trained on 4 "weeks" per month. BirdNET-Pi passes the ISO calendar
  week (1–53), which is out of range for the last weeks of the year; we prefer correctness over
  bug-for-bug parity here.
- **Rejected.** ISO week (BirdNET-Pi behaviour).

## 8. 2026-09-15 — Privacy mask replicates BirdNET-Pi's rank-based rule

- **Decision.** Implement `filter_humans` exactly as BirdNET-Pi: a chunk is masked when a
  `Human*` class is within the top `max(10, 6000 × threshold% / 100)` ranks; neighbouring chunks
  are masked too. Add an on/off switch (`privacy_filter`) since BirdNET-Pi cannot disable it.
- **Why.** Parity with the feature being replicated; the rank-based rule is cheap.
- **Rejected.** The confidence-threshold variant first drafted in the plan.

## 9. 2026-09-15 — Typed errors inside library crates; one-chunk latency for the privacy rule

- **Decision.** Library crates return their own `thiserror` enums (`ModelError`, later
  `AudioError`, `StoreError`); `anyhow` is used only in the binary and tests. The §2.3 interface
  sketch said `anyhow::Result`; the typed form converts into it transparently.
  BirdNET-Pi's rule that blanks the chunks *adjacent* to a human-voice chunk needs the next chunk,
  so on live audio `NeighbourMask` holds each analysis back by one chunk (3 s) before releasing it.
- **Why.** Typed errors let callers distinguish "model missing" from "bad input" without string
  matching. The latency is the only way to honour the neighbour rule on a stream; 3 s is
  irrelevant for a bird logger.
- **Rejected.** `anyhow` everywhere (loses error kinds); applying the neighbour rule only backwards
  (would leave the chunk *after* a human unmasked, unlike BirdNET-Pi).

## 10. 2026-09-15 — Audio timing, capture path and buffer sharing

- **Decision.**
  - Frames are timestamped by **sample count** from an anchor, not by read time. Live sources
    anchor to the wall clock when the first frame arrives and re-anchor only when drift exceeds
    2 s (a stalled stream or restart). The chunker treats a >1 s mismatch as a gap: it realigns
    chunks and emits `ChunkerEvent::Gap` so the pipeline can flush the privacy `NeighbourMask`.
    A forward jump is filled with silence so buffered audio survives (decision #28); only a
    backward jump or one longer than the buffer resets the ring buffer.
  - All configured inputs (alsa, rtsp, file) go through one `FfmpegSource`. A separate pure-Rust
    `WavFileSource` exists for tests and offline analysis so neither needs ffmpeg.
  - The ring buffer is shared between the chunker and the clip writer as `Arc<Mutex<RingBuffer>>`.
  - Requires ffmpeg >= 5.0 (`-timeout` for RTSP; Debian bookworm ships 5.1).
- **Why.** Read-time stamps jitter with pipe buffering and scheduling, which would misalign chunks
  and clips; sample counts are exact and gaps become explicit. One ffmpeg path keeps capture code
  small and handles resampling and every container format. The mutex is held only for memory
  copies (a 3 s chunk is 576 KB), so contention is negligible.
- **Rejected.** `cpal` for ALSA (kept as a later option, BUILD_PLAN Step 12); a channel-based
  request/response protocol to the chunker task for clip extraction (more code, same result).

## 11. 2026-09-15 — Store interface and storage details

- **Decision.**
  - Reads return `DetectionRecord` (all stored columns: local date/hour, week, clip bytes,
    spectrogram path) rather than the core `Detection`; `to_detection()` converts.
  - `set_clip(ids, Option<&ClipInfo>)` takes several ids, because one clip covers every detection
    in its 3 s window. Clip size totals and purge plans count each clip file once.
  - The time zone is fixed when the store opens; `local_date` and `local_hour` are computed at
    insert, so `stats_daily(date)` needs no time zone argument and DST is handled once.
  - Ids use `AUTOINCREMENT` and one writer connection: assigned in commit order and never reused
    after deletes, so `after_id` cursors cannot skip or repeat rows.
  - Timestamps are RFC 3339 UTC text with exactly six fractional digits (sortable as text;
    sub-microsecond precision dropped).
  - Retention planning is a pure function (`plan_purge`) fed by two queries; the exemption query
    uses a window function (bundled SQLite supports them).
  - SQLite is bundled through sqlx (C code behind FFI inside sqlx, allowed by #1). sqlx 0.9 needs
    Rust 1.94, so the workspace MSRV moved from 1.85 to 1.94.
- **Why.** Each point removes a class of bug for API consumers (cursors), charts (DST), or the
  janitor (double-counted shared clips).
- **Rejected.** `rusqlite` with `spawn_blocking` (works, but more glue); integer epoch timestamps
  (less readable in database tools, no real gain at this scale); compile-time checked `query!`
  macros (need a live database or offline metadata in CI).

## 12. 2026-09-15 — Pipeline shape, backpressure and shutdown

- **Decision.**
  - A `Pipeline` struct (`from_config`/`with_sources`, then `run(cancel)`) instead of a bare
    `run_pipeline` function, so the API (Step 8) can take `stats()` and `subscribe()` handles
    before the pipeline starts.
  - One inference thread for all sources. The queue holds 4 chunks; live sources drop the oldest
    queued chunk when full (counted and logged), file sources decoded with `--fast-files` wait
    instead. Control markers (gap, end of stream) bypass the capacity and are never dropped.
  - Shutdown drains: on SIGTERM sources stop, already queued chunks are still analysed and stored,
    then the process exits. A source returning an error stops everything and exits non-zero;
    a file source ending normally keeps the process (and later the API) up unless `--exit-on-eof`.
  - Logs go to stderr so `analyze --json` (Step 6) can own stdout.
- **Why.** Dropping old audio is the only way a slow Pi keeps detections current; waiting is the
  only way offline analysis stays complete. Draining at most 4 chunks costs about a second.
- **Known limitation.** If a *dropped* chunk contained human speech, its neighbours are not
  masked. Drops only happen when inference is badly overloaded; Step 11 revisits it. Resolved by #19.
- **Rejected.** A tokio `mpsc` channel (cannot drop oldest); one inference thread per source
  (doubles model memory, no gain on a 4-core Pi running a single-threaded model).

## 13. 2026-09-15 — Offline tools share the pipeline's code and need no config

- **Decision.** `analyze` reuses `Chunker`, `analyze_chunk` and `NeighbourMask` rather than a
  simplified loop, and reports both the raw ranking (ignoring filters) and what `run` would store.
  Both tools run without a config file (model directory `./models`, no location unless
  `--lat/--lon`), and `species-list` exits non-zero when no filter is active instead of printing
  all 6 522 classes. The recording date (`--date`, default today) picks the location-filter week.
- **Why.** The tool exists to answer "why didn't `run` detect X"; a different code path would give
  different answers. Config-free use makes it handy on a laptop with a downloaded recording.
- **Rejected.** Reading the date from WAV metadata or file names (unreliable across recorders).

## 14. 2026-09-15 — Clip writing and retention mechanics

- **Decision.**
  - A separate clip task, fed by the storage task after the insert, so waiting for the audio
    after a window (up to 1.5 s by default) never delays inserts, logs or the live stream.
    A job waits until the ring buffer covers the window's end, the source ended (signalled
    through the pipeline), or clip length + 5 s passed; then the window is cut, clamped to
    what is buffered.
  - One clip per window, filed under the best detection's species; every detection of the
    window points at it (DECISIONS #11).
  - Files are written as `.tmp` and renamed. Deletion order is file first, then row; a crash
    between the two leaves an orphan file that the startup reconcile removes, never a row that
    points at nothing.
  - Clip paths from the database are only joined onto the clips directory when every component
    is a plain name; anything else is detached and never touched on disk.
  - PNG encoding uses the `png` crate directly rather than `image` (smaller dependency, same
    pure-Rust encoder underneath).
  - The row age limit deletes the clips of the rows it removes, including otherwise exempt
    "best of day" clips, so no files are orphaned.
- **Why.** Keeps the database and disk consistent under crashes, keeps playback correct for
  multi-species windows, and bounds disk use as configured.
- **Rejected.** Writing clips inside the storage task (adds latency); per-detection copies of
  the same audio (wastes disk and double-counts the size cap).

## 15. 2026-09-15 — HTTP API details

- **Decision.**
  - Query parameters are read as strings and parsed by hand, so every bad value produces the
    documented `{"error": ...}` JSON with a message naming the parameter.
  - Detection pages return both `next_after_id` (highest id, for polling forward) and
    `next_before_id` (lowest id of a full page, for paging back).
  - The event stream sends full `DetectionRecord`s (same shape as the list endpoints, looked up by
    id) rather than the pipeline's smaller `Detection`, and uses the id as the SSE event id.
    It subscribes before replaying, skips ids at or below the last one sent, and replays from the
    database after a broadcast lag instead of silently losing events.
  - Streams end when the shutdown token is cancelled; otherwise axum's graceful shutdown would
    wait forever on open event streams.
  - gzip only (the zstd option pulls in C code); audio is excluded from compression so byte
    ranges stay valid.
  - The listener is bound before audio capture starts, so a port conflict fails immediately.
  - `/species` and `/stats/*` are not paginated: at most a few hundred species exist.
- **Why.** Consumers get one object shape, reliable cursors and reconnection, and errors they can
  act on. Shutdown stays prompt for `docker stop`.
- **Rejected.** axum's typed `Query<T>` extractor (plain-text rejections); sending events
  straight from the broadcast without a lookup (inconsistent shape).

## 16. 2026-09-15 — Spectrograms count towards the clip size cap

- **Decision.** `clip_bytes` records the audio and spectrogram files together, so
  `retention.clip_max_total_mb` bounds real disk use. Spectrograms are 8-bit indexed PNGs
  (256-colour palette) instead of RGB. Health liveness uses the wall-clock time the newest chunk
  was processed (`last_processed_at`), not its audio timestamp.
- **Why.** A smoke run of the real binary showed an RGB spectrogram (446 KB) larger than its 6 s
  WAV (432 KB) while only the WAV was counted, so disk use could reach about twice the cap. The
  same run showed a negative `seconds_since_last_chunk` when files are decoded faster than real
  time.
- **Rejected.** A separate `spectrogram_bytes` column (needs a migration for no practical gain);
  grayscale PNGs (smaller still, but much harder to read).

## 17. 2026-09-15 — Dashboard without third-party JavaScript, compiled into the binary

- **Decision.** Draw the two charts by hand (HTML bars and a small inline SVG) instead of
  vendoring Chart.js, and embed the four static files in the binary with `include_str!`.
  API calls use the relative base `api/v1`, and times are formatted in the station time zone.
- **Why.** The charts are simple enough that a library adds more than it saves; no vendored file
  means no licence file, no pinned download to refresh, and nothing to fetch at build time. An
  embedded UI keeps the Docker image to a single executable and cannot drift from the API it was
  built with. Relative URLs keep the page working behind a reverse-proxy path prefix.
- **Rejected.** Chart.js (plan default, ~200 KB for two charts); server-side SVG with `plotters`
  (a new dependency and less interactive); serving `static/` from disk (another thing to mount).

## 18. 2026-09-15 — Docker build, image contents and model provisioning

- **Decision.**
  - The build stage runs on `$BUILDPLATFORM` and cross-compiles with Debian's cross GCC for the
    target triple (native GCC when the architectures match). Only the small runtime stage runs
    under emulation, for `apt-get install`.
  - The image contains the binary, ffmpeg and CA certificates, and runs as uid 1000 in the
    `audio` group. The dashboard is inside the binary (DECISIONS #17); models, configuration and
    data are mounts.
  - `birdsong healthcheck` is a built-in HTTP GET, so the image needs no curl.
  - Models are never redistributed: users run `scripts/fetch-models.sh`, which downloads the
    official archives and converts them in a pinned container. Converted ONNX files are compared
    against `docs/MODEL.md` checksums as a warning only, because a different TensorFlow build
    produces byte-different but numerically equivalent graphs; the conversion itself enforces
    equivalence against the golden logits.
  - `plughw:` device names are recommended over `hw:` so ALSA converts rate and channels.
- **Why.** Cross-compiling keeps arm64 image builds to minutes on x86-64 CI runners instead of the
  hour or more a QEMU-emulated Rust build takes. Not redistributing respects the CC BY-NC-SA terms
  without us deciding how others may use the files.
- **Rejected.** Baking models into the image (licence, 80 MB, and every model change rebuilds the
  image); publishing converted models as release assets (redistribution the owner has not
  decided on); `cargo-chef` (BuildKit cache mounts do the same job without another tool).

## 19. 2026-09-15 — Observability and hardening choices

- **Decision.**
  - `/metrics` is rendered by hand from `PipelineStats` plus two store queries, instead of the
    `metrics` crate with a global recorder. Per-species counts come from the database, so they
    survive restarts; the other counters are per process.
  - A chunk dropped by the drop-oldest queue is replaced in place by a `Missing` marker. The
    inference thread treats it as possibly containing human speech and blanks both neighbours.
    Consecutive markers for one source merge, so a stalled consumer cannot grow the queue.
  - Detection inserts are retried twice with backoff before being counted as store errors.
  - `cargo-deny` enforces the dependency policy mechanically: FFI-wrapper crates are banned by
    name, licences are allow-listed, and only crates.io is an accepted source. All workspace crates
    are `publish = false` so their path dependencies are not treated as wildcards.
- **Why.** A hand-rendered endpoint is about 80 lines with no new dependency and no global state.
  Treating a lost chunk as possibly human errs on the side of privacy, which is the purpose of the
  rule. Checking the policy in CI means a future contributor cannot add an FFI crate by accident.
- **Rejected.** `metrics` + `metrics-exporter-prometheus` (its default features pull in an HTTP
  server and push-gateway client); ignoring dropped chunks for privacy (the #12 limitation).

## 20. 2026-09-15 — FLAC clips by default

- **Decision.** `storage.clip_format` accepts `flac` (new default) and `wav`. FLAC is encoded in
  pure Rust with `flacenc` (default features off: no thread pool, no serde), 16-bit mono, scaled
  exactly like the WAV writer so the two formats hold identical samples. Tests decode with `claxon`.
  A final frame shorter than 16 samples is padded with silence (at most 0.3 ms), because some
  decoders reject shorter frames even though the format allows them at the end of a stream.
- **Why.** Lossless at well under half the size (a 6 s fixture clip: 163 KB FLAC vs 576 KB WAV),
  which multiplies how much audio fits under `clip_max_total_mb` on an SD card, and BirdWeather
  accepts only FLAC soundscapes, so the same encoder serves both. Current browsers play FLAC.
- **Rejected.** Opus or MP3 (lossy, and BirdNET-Pi keeps full-quality audio); calling ffmpeg to
  encode (a process per clip, and ffmpeg is otherwise only needed for capture).

## 21. 2026-09-15 — BirdWeather uploads

- **Decision.**
  - Follow BirdNET-Pi's protocol: POST the clip as FLAC to
    `/stations/{token}/soundscapes?timestamp=…&type=flac` with `Content-Type: audio/flac`, then POST
    one JSON body per detection (`timestamp`, `lat`, `lon`, `soundscapeId`, `soundscapeStartTime`,
    `soundscapeEndTime` in seconds, `commonName`, `scientificName`, `algorithm = "2p4"`,
    `confidence`). `type=flac` is what BirdNET-Go sends; including both conventions is harmless.
  - The soundscape is the saved clip (6 s by default, centred on the window), so the start and end
    offsets point at the 3 s detection window inside it. Timestamps are RFC 3339 with milliseconds
    in the station time zone.
  - HTTP uses `ureq` with rustls on the `ring` provider and bundled Mozilla roots. `reqwest` 0.13
    selects `aws-lc-rs` (an FFI binding to AWS-LC) for rustls, which the policy (#1, enforced by
    `deny.toml`) does not allow without approval.
  - Uploads run in their own task, fed with `try_send` from the clip writer: a slow or unreachable
    BirdWeather never delays clips. Transport errors, 429 and 5xx are retried after 2 s and 10 s;
    `success: false` is not retried; 422 on a detection (a species BirdWeather refuses, such as
    `Dog`) is logged at debug level and does not count as a failure.
  - The token only appears in request URLs, never in logs, and `/api/v1/config` redacts it.
- **Why.** Matches the replicated project, keeps capture and storage independent of the network,
  and stays within the dependency policy.
- **Rejected.** Uploading the whole 3 s window as a separate recording (BirdNET-Pi uploads the file
  it analysed; the saved clip is our equivalent and is already on disk); location fuzzing as in
  BirdNET-Go (BirdNET-Pi sends the configured coordinates; users can configure coarser ones).

## 22. 2026-09-15 — BirdWeather uploads stop promptly at shutdown

- **Decision.** The pipeline's cancellation token reaches the upload task. After it fires, queued
  uploads are skipped and counted (`birdweather_skipped`, also in `/metrics`), retry waits end
  early, and an upload in progress stops before its next request. Request timeouts are 5 s to
  connect and 15 s in total, and the compose file allows 30 s to stop.
- **Why.** ureq is a blocking client, so a request already on the wire cannot be cancelled; it can
  only be bounded by timeouts. Without this, draining a queue of uploads while BirdWeather was
  unreachable could take minutes, and Docker would kill the container mid-upload. Detections and
  clips are already on disk, so skipping uploads loses only the upload.
- **Rejected.** An async HTTP client (would cancel mid-request, but reqwest's rustls now pulls in
  the FFI crypto library the policy excludes); uploading from the clip writer itself (network
  trouble would delay clips).

## 23. 2026-09-16 — Optional detection-quality filters

- **Decision.** Two switches in `[detection]`, both off by default, so the stock pipeline keeps
  BirdNET-Pi's behaviour exactly.
  - **Repeat confirmation.** `min_detections` (default `1`) and `confirmation_window_seconds`
    (default `15`): a species is stored only once it has been detected `min_detections` times
    within the window. The earlier hits that led to the confirmation are stored too, so nothing is
    lost, and once a species is confirmed further detections pass straight through while the window
    keeps rolling. Each source keeps its own `Confirmer`; a held analysis delays the ones behind it
    by at most one window, and everything still held is settled when the source ends and at
    shutdown. Detections dropped unconfirmed are counted (`unconfirmed_detections` in
    `/api/v1/stats`, `birdsong_unconfirmed_detections_total` in `/metrics`).
  - **Dynamic thresholds.** `dynamic_threshold` (default `false`) with
    `dynamic_threshold_trigger` (`0.9`), `dynamic_threshold_min` (`0.2`) and
    `dynamic_threshold_hours` (`24`): after a detection above the trigger, that species' threshold
    steps to 75 %, 50 % then 25 % of `min_confidence`, never below the floor, and expires. Only the
    species heard clearly is affected; every other class keeps `min_confidence`.
  - When `min_detections > 1`, `audio.ring_buffer_seconds` must cover the confirmation window plus
    the clip length, validated when the config loads.
  - `birdsong analyze` applies both, so the offline tool still reports what `run` would store.
- **Why.** The two commonest complaints about a BirdNET-Pi station are one-off false positives and
  missed quiet calls of a bird that is obviously present. Confirmation addresses the first, dynamic
  thresholds the second, and they are the mechanisms BirdNET-Go uses, so the behaviour is familiar.
  Both are off by default because they trade latency (confirmation delays storage by up to a
  window) and precision (lowered thresholds admit more) for recall, which is a choice for the
  station owner, not a default.
- **Rejected.** Making either unconditional (changes stored results for existing stations);
  confirming across sources (two microphones in different places are independent evidence, and
  merging them would let one noisy source confirm another's false positive); dropping the hits that
  preceded a confirmation (they are real detections, and losing them would leave gaps in the
  history and in the clip record).

## 24. 2026-09-16 — Perch v2 as an optional classifier

- **Decision.**
  - `model.kind = "perch-v2"` selects Google Perch v2 (full model or a regional slice) instead of
    BirdNET V2.4; one classifier runs at a time. It uses the ONNX export with the in-graph DFT
    removed, which tract runs without custom operators, fetched from a pinned Hugging Face revision
    and verified against its checksums (`scripts/fetch-perch.sh`).
  - The window length comes from the classifier (`Classifier::window_seconds`: 3 s for BirdNET,
    5 s for Perch) and drives the chunker, clip centring, BirdWeather offsets and `analyze`.
    Capture stays at 48 kHz everywhere, so clips, spectrograms and the ring buffer are unchanged;
    Perch resamples its own window to 32 kHz with an in-crate windowed-sinc resampler.
  - Perch confidences are the softmax of its logits over all of the model's classes, as the model
    card specifies. `detection.sensitivity` does not apply, and thresholds need recalibrating
    (softmax scores are lower than BirdNET's sigmoid, and differ between the full model and a
    regional slice).
  - Perch labels have no common names; they are taken from a BirdNET labels file where the
    scientific name matches (`model.common_names`), otherwise the scientific name is shown.
  - The privacy filter uses a fixed list of Perch's human sound-event classes
    (`PERCH_HUMAN_CLASSES`), with BirdNET-Pi's rank cutoff and neighbour masking.
  - With Perch the BirdNET location model is not used, and BirdWeather uploads are turned off with
    a warning even when a token is configured.
- **Why.** Perch is Apache-2.0 (BirdNET is non-commercial), covers amphibians, insects and mammals
  as well as birds, and ran at about 100 ms per 5 s window in tract with 265 MB peak memory on the
  regional slice, which makes it practical on a Raspberry Pi. On the fixture it independently
  reports the same Black-capped Chickadee as BirdNET. Keeping capture at 48 kHz limited the change
  to the window length. BirdWeather labels every detection with the BirdNET version, so uploading
  Perch results would misattribute them; the location model's outputs are indices into BirdNET's
  label list and do not correspond to Perch's classes.
- **Rejected.** Running BirdNET and Perch side by side (doubles CPU on a Pi; possible later with a
  second inference thread); the `*_int8_arm` Perch builds (quantised from the graph that contains
  the DFT, which tract cannot run); the `rubato` crate for resampling (a fixed 2:3 ratio needs two
  kernels, about 100 lines with tests, and no new dependency); matching the privacy classes by
  keyword (`Car_passing_by` contains "sing"); failing config validation when BirdWeather and Perch
  are both configured (switching models would then require editing the BirdWeather section too).

## 25. 2026-09-16 — Sound events are stored separately from animals

- **Decision.** Every detection has a `kind`: `animal` or `sound_event`, set from the label. For
  BirdNET V2.4 the sound events are `Engine`, `Environmental`, `Fireworks`, `Gun`, `Noise`,
  `Siren` and the three `Human` classes; `Dog` is an animal. For Perch v2 every FSD50K sound-event
  class is a sound event except those that are animals (`PERCH_ANIMAL_EVENTS`: dog, cat, bark,
  meow, purr, growling, frog, cricket, insect, fowl, chicken, crow, gull, chirp, and the generic
  animal groups); species are always animals. Ambiguous classes (`Buzz`, `Hiss`, `Squeak`,
  `Screech`, `Rattle`) are sound events. Human classes are sound events when the privacy filter
  lets them through. Sound events are still stored, clipped and streamed. `/species` and the two
  `/stats` endpoints return animals unless `kind` asks for more; `/detections` and the stream
  return both, each record carrying its `kind`. The dashboard shows sound events in their own card.
  Migration 0002 adds the column and classifies rows stored before it.
- **Why.** Perch recognises about 200 non-animal sounds, and without this rain, traffic or music
  would fill the "latest birds" chart. Keeping them (rather than dropping them) is useful for
  explaining missed birds and for noise monitoring. A dog, cat or frog is an animal and belongs
  with the other animals. Ingestion through `/detections` keeps seeing every row, so no consumer
  loses data.
- **Rejected.** Dropping sound events before storage (loses information and their clips);
  a third `human` kind (people are masked by the privacy filter in the default configuration);
  treating the ambiguous classes as animals (they are usually mechanical); defaulting
  `/detections` to animals (would silently change what existing ingestion receives).

## 26. 2026-09-16 — Perch uses BirdNET's location model, matched by scientific name

- **Decision.** With `model.kind = "perch-v2"`, a station location, `model.meta_model` and
  `model.common_names` (BirdNET's labels), each Perch species is matched to BirdNET's label with
  the same scientific name and allowed when BirdNET's location model scores it at least
  `detection.species_filter_threshold` for the week. Species BirdNET does not know follow
  `model.location_filter_unmapped` (`allow`, the default, or `block`); sound events are never
  filtered. `birdsong species-list` shows no score for unmatched species.
- **Why.** On 317 clips from a BirdNET-Pi station the full Perch model, unfiltered, sometimes chose
  species from other continents, and the regional slice lacked a common local woodpecker. With the
  location filter the full model agreed with BirdNET-Pi on 82–83 % of clips instead of 80 %.
  Allowing unmatched species cost two clips while keeping the amphibians, insects and mammals that
  BirdNET does not cover, so it is the default.
- **Rejected.** A taxonomy synonym table (would fix names like `Coloeus monedula` / `Corvus
  monedula`, but needs a maintained source; revisit if unmatched birds prove a problem); requiring
  a hand-made `model.species_list` (easy to get wrong, and it does not follow the season);
  Perch-specific geographic models (none published).

## 27. 2026-09-18 — BirdNET Geomodel as a location model; confirmation exemptions

- **Decision.**
  - `model.meta_model_labels` names the species of a location model other than BirdNET's own.
    When set, classifier classes are matched to it by scientific name (for BirdNET and Perch
    alike), year-round lists take the maximum over the 48 weeks, and Perch species without a common
    name take the location model's. The BirdNET Geomodel v3.0.4 (`scripts/fetch-geomodel.sh`) is
    the intended use. BirdNET's own location model, with no labels file, behaves as before.
  - `detection.confirmation_exempt_species` lists species stored without waiting for
    `min_detections`. Names are checked against the classifier's labels when the model loads, so
    a typo fails at start-up.
- **Why.** In a 39-hour live run the full Perch model, filtered with BirdNET's location model,
  reported a koala, a European roe deer and a European grasshopper in Minnesota: BirdNET's model
  only knows BirdNET's 6 522 classes, so Perch's other species could not be filtered. The Geomodel
  covers 14 082 species including mammals, amphibians and insects, matches 12 223 of Perch's
  species by name, and rejects those three while keeping a red fox. Repeat confirmation
  (`min_detections = 2`) removed the same false positives but also real one-off callers (a red fox,
  a Common Loon, a Trumpeter Swan); BirdNET-Go users report the same with its Deep Detection and
  have asked for per-species exemptions.
- **Rejected.** A built-in exemption list (what calls rarely depends on the region and the
  station); replacing BirdNET's location model by default (the BirdNET-Pi comparison and parity
  rely on it; the Geomodel is opt-in); a taxonomy synonym table for the remaining unmatched names.

## 28. 2026-09-24 — A forward jump in the audio timeline keeps the buffered audio

- **Decision.** When a frame's timestamp is more than 1 s ahead of where the ring buffer ends, the
  chunker fills the difference with silence instead of emptying the buffer, then realigns its
  windows to the new audio. Samples before the jump keep their times and stay available for clips.
  Backward jumps, and jumps longer than the buffer, still reset it. The gap warning now says
  whether buffered audio was kept.
- **Why.** On a Raspberry Pi with a USB microphone the capture clock ran about 140 ppm slow, so
  every ~4 hours its timestamps fell 2 s behind the wall clock and the source re-anchored them
  (decision #10). The chunker read each re-anchor as a gap and emptied the ring buffer. In 4.8
  days that happened 29 times; the Perch instance, which holds detections for up to 30 s while
  confirming them (`min_detections = 2`), lost 5 clips, all within 13 s of a re-anchor. The BirdNET
  instance, which stores immediately, lost none. The re-anchor warning is logged once per run,
  which hid how often it happened.
- **Rejected.** Re-timing the buffered samples to the new clock (their clips would show the wrong
  audio for their timestamps); keeping the partial window before the jump for analysis (it would
  contain silence); estimating the capture sample rate to avoid re-anchoring at all (worth doing,
  but a larger change; the silence fill protects clips either way).

## 29. 2026-09-24 — Follow the capture clock gradually; count clock corrections

- **Decision.**
  - A live source keeps its timeline at exactly 48 000 samples per second and follows the wall
    clock by repeating or skipping single samples, spread evenly through a frame and at most 1 in
    1 000. The offset between sample count and wall clock is smoothed over about 60 s, which
    averages out read-time jitter; the correction removes it over a similar time. Re-anchoring
    (decision #10) remains for deviations beyond 2 s: stalls and restarts.
  - Every re-anchor is logged with the size of the jump, not only the first per run. `/health` and
    `/metrics` report re-anchors, samples inserted and dropped, and the net correction in ppm.
    `gaps` still counts every timeline discontinuity, re-anchors included.
  - The by-hour chart gives each species a remembered colour slot (stored in the browser) rather
    than its rank of the day; the busier species wins when two of a day's species remember the
    same slot.
- **Why.** A Raspberry Pi's USB microphone ran 140 ppm slow, so every ~4 hours the timestamps fell
  2 s behind and jumped, skipping the analysis window around the jump and, before decision #28,
  losing clips. The warning was logged once per run and the jumps were counted as ordinary gaps,
  so it took a five-day soak to notice. In a four-hour simulation at 140 ppm with 0–80 ms arrival
  jitter, the correction keeps the timeline within 26 ms of the wall clock with no re-anchors, from
  500 ppm slow to 400 ppm fast. Everything downstream (ring buffer, chunker, clip timing) assumes
  exactly 48 kHz, so stamping frames at a measured rate would only move the jump into the chunker.
  One repeated sample in ~7 000 is inaudible and does not affect classification. Colouring by rank
  repainted species whenever the day changed, which the chart guidance rules out: colour follows
  the entity.
- **Rejected.** Resampling with interpolation (same result for classification at far more cost);
  stamping frames at an estimated sample rate (see above); a smaller re-anchor tolerance (more
  frequent, smaller jumps, each still realigning the windows); hashing species names to colours
  (two of a day's species could share a colour).
