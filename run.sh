#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run.sh <entry> [args...]"
  echo "Examples:"
  echo "  ./run.sh tta --pipeline-cfg configs/pipeline.json --tta-cfg configs/tta_config.json"
  echo "  ./run.sh pipeline --pipeline-cfg configs/pipeline.json"
  echo "  ./run.sh debug"
  echo "  ./run.sh experiments/sweep_tta.py --sweep configs/sweep_tta.json"
  exit 1
fi

ENTRY="$1"
shift

if [[ "$ENTRY" == "tta" ]]; then
  ENTRY="experiments/run_pipeline_tta.py"
  OUT_PREFIX="tta_experiment"
elif [[ "$ENTRY" == "pipeline" ]]; then
  ENTRY="experiments/run_pipeline.py"
  OUT_PREFIX="pipeline_experiment"
elif [[ "$ENTRY" == "debug" ]]; then
  ENTRY="experiments/debug.py"
  OUT_PREFIX="debug_experiment"
else
  OUT_PREFIX="$(basename "$ENTRY" .py)"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
out_dir="$ROOT_DIR/assets/${OUT_PREFIX}_${timestamp}"
mkdir -p "$out_dir"

echo "Starting run: $ENTRY"
echo "Output dir: $out_dir"

EXTRA_ARGS=()
if [[ " $* " != *" --output-dir "* ]]; then
  EXTRA_ARGS+=(--output-dir "$out_dir")
fi

PIPELINE_OUTPUT_DIR="$out_dir" TTA_OUTPUT_DIR="$out_dir" \
  nohup python "$ENTRY" "${EXTRA_ARGS[@]}" "$@" > "$out_dir/nohup.log" 2>&1 &
echo "PID: $!"
