# Model conversion tools (dev-time only, Python)

These scripts turn the upstream BirdNET V2.4 release into the files the Rust runtime and its tests
need. They are never run on the Pi and are not part of the Docker image. See `docs/MODEL.md` for
what each output is and its checksum.

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt

# 1. Golden logits from the reference TFLite model (ground truth for the Rust golden test)
.venv/bin/python export_reference.py --model ../../models/audio-model.tflite \
    --wav ../fixtures/soundscape_15s.wav --out ../fixtures/golden/soundscape_15s.json

# 2. Headless ONNX (CNN without the in-graph STFT) + mel frontend parameters + reference spectrogram
.venv/bin/python export_headless_v24.py --wav ../fixtures/soundscape_15s.wav \
    --golden ../fixtures/golden/soundscape_15s.json

# 3. Meta (location/week) model to ONNX, and its golden outputs
.venv/bin/python -m tf2onnx.convert --tflite ../../models/meta-model.tflite \
    --output ../../models/meta-model.onnx --opset 13
.venv/bin/python export_meta_reference.py
```

Inputs expected under `models/`: unzip `BirdNET_v2.4_keras.zip` from
<https://zenodo.org/records/15050749> there (it contains `audio-model.h5`, `audio-model.tflite`,
`meta-model.tflite`, `MelSpecLayerSimple.py` and `labels/`).
