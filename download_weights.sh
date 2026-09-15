#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WEIGHTS_DIR="$SCRIPT_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

model_ready() {
  dir=$1
  [ -f "$WEIGHTS_DIR/$dir/inference.json" ] &&
  [ -f "$WEIGHTS_DIR/$dir/inference.pdiparams" ] &&
  [ -f "$WEIGHTS_DIR/$dir/inference.yml" ]
}

required_ready=true
for model in det_single rec_v5 rec_v6; do
  if ! model_ready "$model"; then required_ready=false; fi
done

if [ "$required_ready" = true ]; then
  printf 'bundled: det_single + rec_v5 + rec_v6\n'
else
  : "${ITDA_PP_OCR_WEIGHTS_URL:?Set ITDA_PP_OCR_WEIGHTS_URL to the release URL of itda-ocr-weights.tar.gz}"
  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' EXIT INT TERM
  curl --fail --location "$ITDA_PP_OCR_WEIGHTS_URL" \
    --output "$tmp_dir/itda-rec-weights.tar.gz"
  tar -xzf "$tmp_dir/itda-rec-weights.tar.gz" -C "$WEIGHTS_DIR"
  for model in det_single rec_v5 rec_v6; do
    model_ready "$model" || { printf '%s files are missing after download\n' "$model" >&2; exit 1; }
  done
  printf 'downloaded: det_single + rec_v5 + rec_v6\n'
fi
