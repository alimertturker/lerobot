#!/usr/bin/env bash
# Push the demos or the latest policy checkpoint to $HF_USER on the Hugging Face Hub (private repos).
#   scripts/push_hub.sh demos | policy [--public]
set -euo pipefail
source "$(dirname "$0")/common.sh"

HF_USER="${HF_USER:-}" "$PY" "$PROJECT_DIR/scripts/push_hub.py" "$@"
