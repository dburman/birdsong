#!/usr/bin/env bash
# Download Google Perch v2 (ONNX, DFT removed so tract can run it) and verify it.
#
#   scripts/fetch-perch.sh [REGION]
#
# REGION is "full" (the default: the complete 14 795-class model the default configuration uses,
# 413 MB, about 1.1 GB of memory) or a smaller regional slice such as north-america-east,
# central-europe or british-isles.
# The list of regions is at https://huggingface.co/tphakala/Perch-v2-Models (regional/).
# Files land in ./models/perch (override the models directory with MODELS_DIR). Needs curl.
set -euo pipefail

cd "$(dirname "$0")/.."
MODELS_DIR=${MODELS_DIR:-models}
REGION=${1:-full}
# Pinned repository revision: files and checksums cannot change underneath us.
REVISION=1214b70a9c14a855e366fe285f8df0e031d7b137
BASE="https://huggingface.co/tphakala/Perch-v2-Models/resolve/$REVISION"
DEST="$MODELS_DIR/perch"

cat <<'NOTICE'
Perch v2 is by Google Research, licensed Apache-2.0
(https://www.kaggle.com/models/google/bird-vocalization-classifier). ONNX conversion and DFT removal
by justinchuby; regional slices by tphakala (https://huggingface.co/tphakala/Perch-v2-Models).
NOTICE

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
case "$REGION" in
  *[!a-z-]*) echo "error: region must be lowercase letters and dashes, got '$REGION'" >&2; exit 1 ;;
esac

if [ "$REGION" = full ]; then
  FILES=(full/perch_v2_no_dft_fp32.onnx full/perch_v2_labels.txt)
else
  FILES=(
    "regional/$REGION/perch_v2_${REGION}_no_dft_fp32.onnx"
    "regional/$REGION/perch_v2_${REGION}_labels.txt"
  )
fi

mkdir -p "$DEST"
curl -fsSL --retry 3 -o "$DEST/SHA256SUMS" "$BASE/SHA256SUMS"

for remote in "${FILES[@]}"; do
  name=$(basename "$remote")
  expected=$(awk -v f="$remote" '$2 == f { print $1 }' "$DEST/SHA256SUMS")
  if [ -z "$expected" ]; then
    echo "error: $remote is not in the repository (unknown region '$REGION'?)" >&2
    exit 1
  fi
  if [ -f "$DEST/$name" ] && [ "$(sha256 "$DEST/$name")" = "$expected" ]; then
    echo "== $name already downloaded"
    continue
  fi
  echo "== downloading $name"
  curl -fL --retry 3 -o "$DEST/$name.part" "$BASE/$remote"
  actual=$(sha256 "$DEST/$name.part")
  if [ "$actual" != "$expected" ]; then
    rm -f "$DEST/$name.part"
    echo "error: $name checksum mismatch (got $actual, expected $expected)" >&2
    exit 1
  fi
  mv "$DEST/$name.part" "$DEST/$name"
done

onnx=$(basename "${FILES[0]}")
labels=$(basename "${FILES[1]}")
cat <<DONE

Perch v2 ($REGION) is in $DEST. To use it, set in birdsong.toml:

[model]
kind = "perch-v2"
classifier = "perch/$onnx"
labels = "perch/$labels"
common_names = "labels/en_us.txt"   # optional: BirdNET labels, for common names

and lower detection.min_confidence (Perch confidences are a softmax; start around 0.3).
DONE
