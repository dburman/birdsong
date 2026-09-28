#!/usr/bin/env bash
# Download the BirdNET Geomodel (location/week model for 14 082 species: birds plus mammals,
# amphibians and insects) and verify it. Files land in ./models/geomodel (override the models
# directory with MODELS_DIR). Needs curl.
set -euo pipefail

cd "$(dirname "$0")/.."
MODELS_DIR=${MODELS_DIR:-models}
VERSION=v3.0.4
BASE="https://github.com/birdnet-team/geomodel/releases/download/$VERSION"
DEST="$MODELS_DIR/geomodel"

# name, SHA-256 (from docs/MODEL.md)
FILES=(
  "BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.onnx 0de81d222c23dcb6fa428e958b4dac978783191357e01b7268a103fc6f08e61a"
  "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt 8250b457e45d43fc3e77b5cbd06a1d311baf585ab9c51ed8d42e011d98534835"
)

cat <<'NOTICE'
BirdNET Geomodel by the BirdNET team (https://github.com/birdnet-team/geomodel). Model weights and
label files are licensed Apache-2.0; see its ACCEPTABLE_USE.md for the project's guidance on use.
NOTICE

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
mkdir -p "$DEST"
for entry in "${FILES[@]}"; do
  read -r name expected <<<"$entry"
  if [ -f "$DEST/$name" ] && [ "$(sha256 "$DEST/$name")" = "$expected" ]; then
    echo "== $name already downloaded"
    continue
  fi
  echo "== downloading $name"
  curl -fL --retry 3 -o "$DEST/$name.part" "$BASE/$name"
  actual=$(sha256 "$DEST/$name.part")
  if [ "$actual" != "$expected" ]; then
    rm -f "$DEST/$name.part"
    echo "error: $name checksum mismatch (got $actual)" >&2
    exit 1
  fi
  mv "$DEST/$name.part" "$DEST/$name"
done

cat <<'DONE'

To use it as the location filter, set in birdsong.toml:

[model]
meta_model = "geomodel/BirdNET+_Geomodel_V3.0.4_Global_14K_FP32.onnx"
meta_model_labels = "geomodel/BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"
DONE
