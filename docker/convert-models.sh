#!/bin/sh
set -eu
cd /work
for f in audio-model.h5 MelSpecLayerSimple.py meta-model.tflite labels/en_us.txt; do
  if [ ! -f "/models/$f" ]; then
    echo "missing /models/$f: unzip BirdNET_v2.4_keras.zip and BirdNET_v2.4_tflite.zip into the models directory" >&2
    exit 1
  fi
done
echo "== location model: TFLite -> ONNX"
python -m tf2onnx.convert --tflite /models/meta-model.tflite --output /models/meta-model.onnx --opset 13
echo "== classifier: Keras -> headless ONNX (checked against the TFLite golden logits)"
python export_headless_v24.py \
  --models-dir /models --out-dir /models \
  --wav /fixtures/soundscape_15s.wav --golden /fixtures/golden/soundscape_15s.json \
  --no-reference
echo "== done"
