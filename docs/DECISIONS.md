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
