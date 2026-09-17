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
| Raspberry Pi 4 Model B (8 GB), one core, 2026-09-16 | — | — | **224 ms** | 2 500 ms target |

For comparison the TFLite interpreter (XNNPACK) needs 18.8 ms per chunk on the same Mac, so tract
is at parity. The Pi 4 figure was measured with `birdsong analyze` pinned to one core (`taskset`)
on 3 minutes of recorded audio, while the Pi was also running BirdNET-Pi: 13× faster than real
time, 209 MiB peak memory, 1.2 s to load the model and analyse the first window. A Pi 4 core is
about 9× slower than an M-series core here.

## Perch v2 (optional classifier, `model.kind = "perch-v2"`)

| Item | Value |
|------|-------|
| Source | Google Research, [bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier/) v2. ONNX conversion with the DFT replaced by matrix multiplication by justinchuby; regional slices by tphakala: [tphakala/Perch-v2-Models](https://huggingface.co/tphakala/Perch-v2-Models), pinned to revision `1214b70a9c14a855e366fe285f8df0e031d7b137` |
| License | **Apache-2.0** (commercial use allowed) |
| Input | 5.0 s at **32 000 Hz** = 160 000 float32 samples, `[1, 160000]`. Birdsong captures at 48 kHz and resamples each 5 s window (240 000 samples) inside the classifier: windowed-sinc, Blackman window, 128 taps, cutoff 14.5 kHz (`crates/birdsong-model/src/resample.rs`) |
| Outputs | 0: embedding `[1, 1536]`; 1: spatial embedding `[1, 16, 4, 1536]`; 2: spectrogram `[1, 500, 128]`; **3: class logits** `[1, N]`. Birdsong uses output 3 and applies a softmax over all N classes, so `detection.sensitivity` has no effect |
| Classes | full model: 14 795 = 14 597 species (birds, amphibians, insects, mammals) + 198 FSD50K sound events. Regional slices keep the species expected in a region plus the sound events and are bit-exact to the full model on those classes (`north-america-east`: 999 = 801 species + 198 events). Because the softmax runs over fewer classes, the same sound scores higher on a regional slice: recalibrate `detection.min_confidence` per model |
| Labels | one class per line: species as `Genus species` (no common name), sound events as `Words_with_underscores`. The full file starts with a header line `inat2024_fsd50k`, which is skipped. Common names come from a BirdNET labels file (`model.common_names`) where the scientific name matches: 635 of the 801 `north-america-east` species, 6 262 of the 14 597 in the full model; the rest show the scientific name |
| Privacy filter | the 35 FSD50K classes in `PERCH_HUMAN_CLASSES` (`crates/birdsong-model/src/labels.rs`): voices (speech, conversation, singing, laughter, shouting, whispering, crying) and body or activity sounds (footsteps, coughs, sneezes, breathing, clapping), matching BirdNET's `Human vocal` / `Human non-vocal`. `Speech_synthesizer` is excluded. The rank cutoff is the same as for BirdNET |
| Sound events | species and the animal sound events (`PERCH_ANIMAL_EVENTS`: dog, cat, frog, cricket, insect, …) are stored as `kind = "animal"`; the other 180 sound events (engines, rain, music, people) as `kind = "sound_event"`, kept out of the charts (decision #25) |
| Location filter | BirdNET's location/week model, mapped to Perch classes by scientific name (needs `model.common_names` and `model.meta_model`). Of the full model's 14 597 species, 6 262 have a BirdNET name and are filtered by location; the other 8 335 (other taxa, and birds under a different scientific name, such as `Coloeus monedula` for BirdNET's `Corvus monedula`) are allowed by default or blocked with `model.location_filter_unmapped = "block"`. Sound events are never filtered |
| Not available with Perch | BirdWeather uploads (BirdWeather records detections as BirdNET V2.4 results) |

### Files (`scripts/fetch-perch.sh [REGION]`, under `models/perch/`, git-ignored)

| File | Size | SHA-256 |
|------|-----:|---------|
| `perch_v2_north-america-east_no_dft_fp32.onnx` | 73 901 012 | `80d640c44e0775a7ef68967e5d9f073970d918486c7686ac4a14c77f3bab5d71` |
| `perch_v2_north-america-east_labels.txt` | 17 860 | `fc43269a24c11481b4bccd1564e80a9b1974f704601fa520ef467374fe71d71f` |
| `perch_v2_no_dft_fp32.onnx` (`full`) | 413 350 933 | `4dcf71c18a147198545944bb5149697e89e3ad2e16637fa8f0edf6d13035a017` |
| `perch_v2_labels.txt` (`full`) | 312 716 | `e4d5c0397d8fb08bf90c6b13a34810af53504faad927e472fcc567793c9de057` |

The script verifies every file against the repository's `SHA256SUMS` at the pinned revision. The
`*_int8_arm.onnx` variants were quantised from the graph that still contains the DFT and are not
supported.

### Verification and benchmarks (Apple M-series, single thread, tract 0.23.7, release build)

| Check | Result |
|-------|--------|
| tract loads and optimises the no-DFT graph | yes, no custom ops |
| Fixture window 0–5 s, `north-america-east` | Black-capped Chickadee 0.649 (0.698 when the file is resampled with ffmpeg instead), the species BirdNET reports; windows 2 and 3: House Finch 0.587 / 0.447 (ffmpeg: 0.585 / 0.436); same top-3 order in every window |
| Resampling one window | 12.6 ms |
| Inference one window | ~100 ms (5 s of audio: about 50× faster than real time) |
| `birdsong analyze` on the 15 s fixture | 1.07 s wall clock, 265 MB peak memory |

Per second of audio Perch costs about 2.5× BirdNET V2.4 on the Mac.

**Raspberry Pi 4 Model B (8 GB)**, one core, same method as the BirdNET row above:

| Model | Per 5 s window | Faster than real time | Peak memory | Load + first window |
|-------|---------------:|----------------------:|------------:|--------------------:|
| `north-america-east` (999 classes) | 1 698 ms | 2.9× | 245 MiB | 7.1 s |
| `full` (14 795 classes) | 2 660 ms | 1.9× | 1 142 MiB | 14.5 s |

Both keep up with one source on one core. The full model uses about a gigabyte, so it needs a Pi
with at least 2 GB; the regional slice fits in 1 GB. The Pi reached 70 °C without throttling.

The full model's graph also uses the symbolic batch size inside internal reshapes, so the loader
substitutes `batch = 1` throughout the model rather than only fixing the input shape.

### Comparison with a BirdNET-Pi station (2026-09-16)

317 saved 6 s clips (up to 6 per species, 67 species) from a BirdNET-Pi station in the northern
United States running BirdNET V2.4 FP16 at confidence 0.7 and sensitivity 1.25. For BirdNET, the
exact 3 s window BirdNET-Pi scored was analysed; for Perch, the 5 s centred on it. BirdNET-Pi's
detections are not ground truth: a disagreement can be either model's mistake.

| Classifier | BirdNET-Pi's species ranked 1st | In top 3 | Confidence for that species |
|------------|--------------------------------:|---------:|-----------------------------|
| Birdsong, BirdNET V2.4 (FP32) | **317 / 317** | 317 / 317 | median difference from BirdNET-Pi 0.0004, all but one within 0.01, max 0.036 (FP16 vs FP32) |
| Perch v2 `north-america-east` | 255 (80 %) | 289 (91 %) | median 0.79 |
| Perch v2 `full`, no species filter | 255 (80 %) | 291 (92 %) | median 0.68 |
| Perch v2 `full` + the 227 species BirdNET's location model allows there year-round, other species blocked | 262 (83 %) | 298 (94 %) | median 0.68 |
| Perch v2 `full` + the same list, species BirdNET cannot score allowed (the default location filter) | 260 (82 %) | 298 (94 %) | median 0.68 |

- **Regional slices can miss local species.** Hairy Woodpecker (*Dryobates villosus*), common at
  the station, is not in the `north-america-east` slice, so that model can never report it. The
  full model with the location filter avoids this.
- Without a species filter the full model's top pick was sometimes a species that does not occur
  there (Goldcrest, Australian King-Parrot, Jackdaw, Eurasian Dotterel); the location filter removes
  those. Allowing the species BirdNET cannot score cost two clips of agreement (one Jackdaw under
  its other scientific name), so allowing them is the default.
- The remaining disagreements are mostly between similar species: Yellow-throated Vireo,
  Yellow-bellied Flycatcher and American Redstart heard as Red-eyed Vireo; American Crow as Common
  Raven; Black-throated Green Warbler as Eastern Wood-Pewee. Listening to those clips is the only
  way to tell which model is right.
### Live side by side with a BirdNET-Pi station (24 h, Raspberry Pi 4)

Birdsong (BirdNET V2.4, same settings) and Perch v2 (full model, location filter, everything from
0.3 stored) ran on the same Pi as the station, reading the same microphone through the PulseAudio
daemon BirdNET-Pi starts. In 23.7 hours: BirdNET-Pi 474 detections, Birdsong 482, Perch 1 720 at
≥ 0.5.

- **Per-species counts agree; timestamps do not line up.** Red-breasted Nuthatch 251 vs 233, Hairy
  Woodpecker 84 vs 82, Common Raven 58 vs 59, but matching detections by species within ±4 s pairs
  only ~72 % (±30 s: 89 %). Time differences are centred on zero (median −0.3 s), so this is
  window alignment, not a clock offset: BirdNET-Pi cuts windows inside fixed 15 s recordings while
  Birdsong cuts a continuous stream. Compare counts per species rather than matching timestamps.
- **The location filters differ in two ways**, both making Birdsong more permissive (109 species
  allowed vs 86 for the station's week):
  - Week numbering (decision #7): BirdNET-Pi passes the ISO week (38), Birdsong BirdNET's 48-week
    number (35).
  - Model version: the meta model in the Zenodo archives, which Birdsong converts, is BirdNET-Pi's
    `MData_Model_V2`; BirdNET-Pi defaults to `DATA_MODEL_VERSION=1`, an older model that scores the
    same species lower (Blue-headed Vireo at week 35: 0.027 in V1, 0.068 in V2, Birdsong 0.071 —
    the gap to V2 is FP16 vs FP32).
  - Together these let Birdsong report Blue-headed Vireo (25×, up to 0.94), Barred Owl and Pine
    Warbler, which BirdNET-Pi hears but discards on the location filter ("Excluded as below
    Species Occurrence Frequency Threshold"). All three are plausible there in September.
- **Perch reports far more**: 1 720 detections at ≥ 0.5 against 474, including 465 Black-capped
  Chickadees where BirdNET found 9–14. Whether those are quiet calls BirdNET misses or false
  positives is unresolved; it needs listening.

- Thresholds: of the clips BirdNET-Pi reported at ≥ 0.7, Perch `full` gives the same species at
  least 0.3 for 75 % and at least 0.5 for 62 %; the regional slice, whose softmax covers fewer
  classes, 79 % and 69 %. This measures agreement with BirdNET, not precision; choosing a
  threshold for precision needs labelled recordings.

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
