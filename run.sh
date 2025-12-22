#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
out_dir="$ROOT_DIR/assets/tta_experiment_${timestamp}"
mkdir -p "$out_dir"

echo "Starting TTA run in: $out_dir"
TTA_OUTPUT_DIR="$out_dir" nohup python debug_tests/run_tta.py > "$out_dir/nohup.log" 2>&1 &
echo "PID: $!"
