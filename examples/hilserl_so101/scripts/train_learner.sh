#!/usr/bin/env bash
# Step 4a: start the learner (terminal 1). Start it before the actor. Extra arguments are passed through.
set -euo pipefail
source "$(dirname "$0")/common.sh"

PYTHONUNBUFFERED=1 "$PY" -m lerobot.rl.learner --config_path "$PROJECT_DIR/configs/train_hilserl_leader.json" \
  "${SETUP_ARGS[@]}" --output_dir="$TRAIN_DIR" --policy.device="$LEARNER_DEVICE" "$@" \
  2>&1 | tee "$PROJECT_DIR/outputs/learner.log"
