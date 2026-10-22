#!/usr/bin/env bash
# Turn the xyz-only demos into wrist demos (see scripts/convert_demos_wrist.py). Nothing moves.
set -euo pipefail
source "$(dirname "$0")/common.sh"

"$PY" "$PROJECT_DIR/scripts/convert_demos_wrist.py" --dst "$DATASET_ROOT" "$@"
