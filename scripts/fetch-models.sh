#!/usr/bin/env bash
# Download BirdNET V2.4 from Zenodo, verify it, and convert it into the files Birdsong runs.
# Needs curl, unzip and Docker. Run from anywhere; files land in ./models (override with MODELS_DIR).
set -euo pipefail

cd "$(dirname "$0")/.."
MODELS_DIR=${MODELS_DIR:-models}
RECORD="https://zenodo.org/api/records/15050749/files"

# name, SHA-256, size shown to the user. The Keras archive has the classifier to convert; the
# TFLite archive has the location (meta) model. Both carry the label files.
ARCHIVES=(
  "BirdNET_v2.4_keras.zip 893469ec2780d74f1f2bed5957071c97db26e75c97d9f55e6f664973b99e88ce 124MB"
  "BirdNET_v2.4_tflite.zip 31377e128d86fe7b65fa91b206b8804bad1cd934e624bbe6e7c1788170c95e57 77MB"
)

# Checksums of the converted files from docs/MODEL.md. A different TensorFlow build can produce a
# byte-different but numerically equivalent file, so a mismatch is a warning; the conversion itself
# fails if the result does not reproduce the reference logits.
declare -A EXPECTED=(
  [birdnet-v2.4-headless.onnx]=f47af29aa6665713e676843c26a906f8e58e075359b8ce6725d4d98c7e5621a6
  [meta-model.onnx]=3f7462decdbb0330a4d54245d7a09dd160a67acdb00ef2017aaa932da4dfe84c
  [labels/en_us.txt]=b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16
)

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

cat <<'NOTICE'
BirdNET V2.4 models are (c) the BirdNET team (Cornell Lab of Ornithology and Chemnitz University
of Technology) and licensed CC BY-NC-SA 4.0: non-commercial use only, share alike, with attribution.
https://zenodo.org/records/15050749
NOTICE

for tool in curl unzip docker; do
  command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done
mkdir -p "$MODELS_DIR"

for entry in "${ARCHIVES[@]}"; do
  read -r zip sum size <<<"$entry"
  if [ -f "$MODELS_DIR/$zip" ] && [ "$(sha256 "$MODELS_DIR/$zip")" = "$sum" ]; then
    echo "== $zip already downloaded"
  else
    echo "== downloading $zip ($size)"
    curl -fL --retry 3 -o "$MODELS_DIR/$zip.part" "$RECORD/$zip/content"
    mv "$MODELS_DIR/$zip.part" "$MODELS_DIR/$zip"
  fi
  actual=$(sha256 "$MODELS_DIR/$zip")
  if [ "$actual" != "$sum" ]; then
    echo "error: $zip checksum mismatch (got $actual)" >&2
    exit 1
  fi
  echo "== unpacking $zip"
  unzip -o -q "$MODELS_DIR/$zip" -d "$MODELS_DIR"
done

echo "== building the conversion image (first run downloads TensorFlow, several minutes)"
docker build -f docker/convert-models.Dockerfile -t birdsong-convert-models .

echo "== converting"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/$MODELS_DIR:/models" \
  -v "$PWD/tools/fixtures:/fixtures:ro" \
  birdsong-convert-models

echo "== checking results"
status=0
for f in "${!EXPECTED[@]}"; do
  if [ ! -f "$MODELS_DIR/$f" ]; then
    echo "error: $MODELS_DIR/$f was not produced" >&2
    status=1
  elif [ "$(sha256 "$MODELS_DIR/$f")" = "${EXPECTED[$f]}" ]; then
    echo "ok       $f"
  else
    echo "warning  $f differs from docs/MODEL.md (conversion check passed, so it is equivalent)"
  fi
done
[ $status -eq 0 ] || exit $status

cat <<EOF
Models are ready in $MODELS_DIR. On the Pi, put that directory next to docker-compose.yml, then:
  docker compose up -d
To double-check detection on the bundled test recording (expects a Black-capped Chickadee):
  docker compose run --rm -v "\$PWD/tools/fixtures:/fixtures:ro" birdsong analyze /fixtures/soundscape_15s.wav --models /models --lat 42.36 --lon -71.06 --date 2026-05-15
EOF
