#!/usr/bin/env bash
# Step 4b: start the actor (terminal 2, keep it focused for the s/q/r keys). Extra arguments are passed through.
# The policy drives the follower; space = pause, tab = take over / hold, space = resume.
set -euo pipefail
source "$(dirname "$0")/common.sh"

# The learner owns outputs/train/...; the actor only writes logs, into its own folder per run.
ACTOR_DIR="$PROJECT_DIR/outputs/actor/$(date +%Y%m%d_%H%M%S)"
PYTHONUNBUFFERED=1 "$PY" -m lerobot.rl.actor --config_path "$PROJECT_DIR/configs/train_hilserl_leader.json" \
  "${SETUP_ARGS[@]}" --output_dir="$ACTOR_DIR" --policy.device="$ACTOR_DEVICE" "$@" \
  2>&1 | tee -i "$PROJECT_DIR/outputs/actor.log"  # -i: Ctrl+C must not kill tee before the shutdown is done
