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

DET_MODE=${ITDA_DET_MODE:-single}
case "$DET_MODE" in
  single) REQUIRED_MODELS='det_single' ;;
  full) REQUIRED_MODELS='det_full' ;;
  date) REQUIRED_MODELS='det_date' ;;
  cascade) REQUIRED_MODELS='det_full det_date' ;;
  *) printf 'unknown ITDA_DET_MODE: %s\n' "$DET_MODE" >&2; exit 2 ;;
esac

required_ready=true
for model in rec_v5 rec_v6 $REQUIRED_MODELS; do
  if ! model_ready "$model"; then required_ready=false; fi
done

if [ "$required_ready" = true ]; then
  printf 'bundled: rec_v5 + rec_v6 + %s\n' "$REQUIRED_MODELS"
else
  : "${ITDA_PP_OCR_WEIGHTS_URL:?Set ITDA_PP_OCR_WEIGHTS_URL to the release URL of itda-ocr-weights.tar.gz}"
  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' EXIT INT TERM
  curl --fail --location "$ITDA_PP_OCR_WEIGHTS_URL" \
    --output "$tmp_dir/itda-rec-weights.tar.gz"
  tar -xzf "$tmp_dir/itda-rec-weights.tar.gz" -C "$WEIGHTS_DIR"
  for model in rec_v5 rec_v6 $REQUIRED_MODELS; do
    model_ready "$model" || { printf '%s files are missing after download\n' "$model" >&2; exit 1; }
  done
  printf 'downloaded: rec_v5 + rec_v6 + %s\n' "$REQUIRED_MODELS"
fi
