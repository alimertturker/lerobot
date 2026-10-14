#!/usr/bin/env bash
# Stage 3 test: no dataset, no policy. The follower goes to the reset pose and holds still,
# the leader stays parked. space = pause (leader moves onto the follower), tab (hand on leader) = take
# over / hold again, space = resume. s = success, q/esc = failure. Ctrl+C = both arms go home and stop.
set -euo pipefail
source "$(dirname "$0")/common.sh"

PYTHONUNBUFFERED=1 "$PY" -m lerobot.rl.gym_manipulator --config_path "$PROJECT_DIR/configs/env_test_leader.json" \
  "${SETUP_ARGS[@]}" "$@" \
  2>&1 | tee "$PROJECT_DIR/outputs/test_leader.log"
