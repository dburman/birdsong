# Model provenance and conversion

## BirdNET V2.4 (default classifier)

| Item | Value |
|------|-------|
| Source | BirdNET team, Zenodo record [15050749](https://zenodo.org/records/15050749) ("BirdNET Model V2.4", 2025-03-18) |
| License | **CC BY-NC-SA 4.0** — non-commercial. Educational and research use is explicitly permitted. Not bundled in the Docker image; fetched into a volume by the operator. |
| Input | 3.0 s of mono audio at 48 000 Hz = 144 000 float32 samples (any scale; the frontend normalises) |
| Output | 6 522 logits (one per class); sigmoid applied afterwards |
| Labels | `labels/en_us.txt` (and 20+ other languages), one `Scientific name_Common name` per line, index = line number |
| Non-bird classes (0-based index) | `Dog_Dog` (1949), `Engine_Engine` (2143), `Human non-vocal` (2818), `Human vocal` (2819), `Human whistle` (2820), `Noise_Noise` (3927), `Siren_Siren` (5560), plus a few more |

### Files (all under `models/`, git-ignored)

| File | Size | SHA-256 | Origin |
|------|------|---------|--------|
| `BirdNET_v2.4_tflite.zip` | 76 822 925 | `31377e128d86fe7b65fa91b206b8804bad1cd934e624bbe6e7c1788170c95e57` | Zenodo |
| `BirdNET_v2.4_keras.zip` | 123 874 754 | `893469ec2780d74f1f2bed5957071c97db26e75c97d9f55e6f664973b99e88ce` | Zenodo |
| `audio-model.tflite` | 51 726 412 | `55f3e4055b1a13bfa9a2452731d0d34f6a02d6b775a334362665892794165e4c` | in both zips (FP32) |
| `audio-model.h5` | 51 893 472 | `89f4aacb2d821d8d13351ccb131bc4f0732f0b4fe1d203d6d28e8c7ab295c50d` | keras zip |
| `meta-model.tflite` | 29 526 096 | `33aea6d21cc887d2414e9596d2531a480a3e5f4770c22aa257f217fb757d4653` | in both zips |
| `labels/en_us.txt` | 259 740 | `b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16` | in both zips |
| `birdnet-v2.4-headless.onnx` | 51 121 540 | `f47af29aa6665713e676843c26a906f8e58e075359b8ce6725d4d98c7e5621a6` | **generated** by `export_headless_v24.py` |
| `meta-model.onnx` | 29 526 215 | `3f7462decdbb0330a4d54245d7a09dd160a67acdb00ef2017aaa932da4dfe84c` | **generated** by tf2onnx |
| `MEL_SPEC1_melfb.f32`, `MEL_SPEC2_melfb.f32`, `birdnet-v2.4-frontend.json` | small | see script | dumped by `export_headless_v24.py`, used only by tests |

Regenerate the generated files with (dev machine, Python 3.12):

```bash
cd tools/convert_model
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python -m tf2onnx.convert --tflite ../../models/meta-model.tflite --output ../../models/meta-model.onnx --opset 13
.venv/bin/python export_headless_v24.py --wav ../fixtures/soundscape_15s.wav --golden ../fixtures/golden/soundscape_15s.json
```

### Producing the files on another machine

`scripts/fetch-models.sh` downloads both Zenodo archives (the Keras one holds the classifier to
convert, the TFLite one holds the location model), verifies their checksums, and converts them in
`docker/convert-models.Dockerfile` with pinned versions (TensorFlow 2.21.0, tf2onnx 1.17.0,
Python 3.12). The conversion fails if the rebuilt network does not reproduce the TFLite reference
logits.

The ONNX files it produces are **numerically equivalent but not byte-identical** to the checksums
above, because a different TensorFlow build serialises the graph differently. Verified 2026-09-15
on linux/arm64:

| Check with the container-converted files | Result |
|------------------------------------------|--------|
| `birdsong analyze --json` on `soundscape_15s.wav` vs golden logits | identical top-5 on all chunks, worst logit error 0.0003 |
| `birdsong species-list`, Boston week 20 / year-round | 126 / 236, same as the TFLite reference |
| Mel filterbank dumps | byte-identical |

The fetch script therefore reports a checksum difference on the ONNX files as a warning, not an error.

### Why the model is split ("headless" + Rust frontend)

The Keras graph is `INPUT(144000) → MEL_SPEC1, MEL_SPEC2 → concatenate(96,511,2) → CNN → 6522`.
The two `MelSpecLayerSimple` layers compute an STFT inside the graph. No pure-Rust runtime
could execute that part (see `DECISIONS.md` #2), so:

1. `tools/convert_model/export_headless_v24.py` rebuilds the network from `concatenate` to
   `CLASS_DENSE_LAYER` (logits; the TFLite export also stops before the final sigmoid) and exports
   it to ONNX opset 13. The resulting ops are all plain CNN ops: Conv, BatchNormalization, Add, Mul,
   Sigmoid, AveragePool, MaxPool, GlobalAveragePool, Pad, Concat, Reshape, Squeeze, Transpose, Gemm.
2. `crates/birdsong-model/src/mel.rs` re-implements both mel layers in Rust (`rustfft`).
   Reference algorithm, per layer (from `MelSpecLayerSimple.py` shipped in the Keras zip):

   ```text
   x = x - min(x); x = x / (max(x) + 1e-6); x = (x - 0.5) * 2          # per 3 s chunk
   S = stft(x, frame_length, frame_step, hann(periodic), pad_end=False)
   S = real(S)                        # tf.cast(complex64 -> float32) keeps the REAL part only
   M = S @ mel_filterbank             # tf.signal.linear_to_mel_weight_matrix (HTK mel, DC bin zeroed)
   M = M ** 2
   M = M ** (1 / (1 + exp(magnitude_scaling)))
   M = reverse(M, mel axis); M = transpose -> (n_mels, n_frames)
   ```

   | Layer | frame_length | frame_step | n_mels | fmin | fmax | magnitude_scaling (trained) | frames |
   |-------|-------------:|-----------:|-------:|-----:|-----:|----------------------------:|-------:|
   | MEL_SPEC1 | 2048 | 278 | 96 | 0 Hz | 3 000 Hz | 1.2110004 | 511 |
   | MEL_SPEC2 | 1024 | 280 | 96 | 500 Hz | 15 000 Hz | 1.4465874 | 511 |

   Outputs are interleaved on a trailing channel axis: NHWC `(1, 96, 511, 2)`, channel 0 = MEL_SPEC1.

## Golden verification (Step 0 result)

Reference: `audio-model.tflite` run with the TensorFlow Lite interpreter on `soundscape.wav`
(2 min, 40 chunks). Rust = mel frontend + headless ONNX in tract 0.23.7.

| Check | Result |
|-------|--------|
| Mel filterbank vs TF dump | max abs error 1.2e-5 (float32 rounding) |
| Spectrogram (96×511×2) vs TF | within 1 % of peak on chunk 0 |
| Logits, 40 chunks | worst max abs error **0.0017**; top-1 class identical on all 40 |
| Meta model, 4 (lat, lon, week) cases | see `tests/golden_meta_v24.rs` |

## Benchmarks (per 3 s chunk, single thread, tract 0.23.7, release build)

| Machine | Frontend | Network | Total | Real-time budget |
|---------|---------:|--------:|------:|-----------------:|
| Apple M-series (macOS, aarch64) | 5.0 ms | 19.0 ms | 24 ms | 3 000 ms |
| Same chip, linux/arm64 Docker | 4.0 ms | 23.0 ms | 27 ms | 3 000 ms |
| Raspberry Pi 5 | — | — | **pending hardware** | 1 000 ms target |
| Raspberry Pi 4 | — | — | **pending hardware** | 2 500 ms target |

For comparison the TFLite interpreter (XNNPACK) needs 18.8 ms per chunk on the same Mac, so tract
is at parity. A Pi 4 core is very roughly 10–20× slower than an M-series core; the expected
0.3–0.6 s per chunk is comfortably inside budget, but this **must be measured** (Step 10).

## BirdNET V3.0 (evaluated, not adopted yet)

| Item | Value |
|------|-------|
| Source | `birdnet` PyPI package 1.1.x; files on Zenodo record 20703646 (`BirdNET+_V3.0-preview3.1_Global_11K_*`) |
| Status | **preview** (as of 2026-09) |
| Input | 3.0 s at **32 000 Hz** (96 000 samples), bandpass 0–15 kHz |
| Output | 11 560 species probabilities (sigmoid inside the graph, so no sensitivity knob) + 1 280-dim embeddings |
| Formats | ONNX (FP32 516 MB, FP16), TFLite, PyTorch, protobuf |
| tract | **fails**: the graph uses the ONNX `STFT` op (opset 17+), which tract 0.23 rejects (`rank == 3` unification error). Splitting off the frontend as done for V2.4 would be needed. |
| Geo model | separate v3.0.4 geomodel, 14K labels, GitHub releases |

Revisit once V3.0 leaves preview and/or tract gains `STFT`. Size alone makes it a Pi 5 candidate only.

## Other models tried

- `tract-tflite` loading `audio-model.tflite` directly: fails (`SPLIT_V` unsupported); the meta
  TFLite fails on `STRIDED_SLICE` with `shrink_axis_mask`.
- `tf2onnx --tflite audio-model.tflite`: fails (`RFFT2D` may only feed `ComplexAbs`, but BirdNET
  feeds it to `Squeeze`/`Cast`).
- `tf2onnx --tflite meta-model.tflite`: works (opset 13), loads and runs in tract in ~0.6 ms.
