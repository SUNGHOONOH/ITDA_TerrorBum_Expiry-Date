#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WEIGHTS_DIR="$SCRIPT_DIR/weights"
WEIGHTS_URL='https://github.com/SUNGHOONOH/ITDA_TerrorBum_Expiry-Date/releases/download/ft5/itda-ocr-weights-v5.tar.gz'
MODELS="det_single rec_v5 rec_v6 textline_ori uvdoc yolo_field refiner_det rec_v5_med rec_v6_med"
mkdir -p "$WEIGHTS_DIR"

model_ready() {
  dir=$1
  if [ "$dir" = "yolo_field" ]; then
    [ -f "$WEIGHTS_DIR/$dir/best.onnx" ]
  elif [ "$dir" = "uvdoc" ]; then
    [ -f "$WEIGHTS_DIR/$dir/model.safetensors" ] &&
    [ -f "$WEIGHTS_DIR/$dir/config.json" ] &&
    [ -f "$WEIGHTS_DIR/$dir/preprocessor_config.json" ] &&
    [ -f "$WEIGHTS_DIR/$dir/inference.yml" ]
  else
    [ -f "$WEIGHTS_DIR/$dir/inference.json" ] &&
    [ -f "$WEIGHTS_DIR/$dir/inference.pdiparams" ] &&
    [ -f "$WEIGHTS_DIR/$dir/inference.yml" ]
  fi
}

required_ready=true
for model in $MODELS; do
  if ! model_ready "$model"; then required_ready=false; fi
done

if [ "$required_ready" = true ]; then
  printf 'bundled: %s\n' "$MODELS"
else
  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' EXIT INT TERM
  curl --fail --location "$WEIGHTS_URL" \
    --output "$tmp_dir/itda-rec-weights.tar.gz"
  tar -xzf "$tmp_dir/itda-rec-weights.tar.gz" -C "$WEIGHTS_DIR"
  for model in $MODELS; do
    model_ready "$model" || { printf '%s files are missing after download\n' "$model" >&2; exit 1; }
  done
  printf 'downloaded: %s\n' "$MODELS"
fi
