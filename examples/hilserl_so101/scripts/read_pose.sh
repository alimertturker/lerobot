#!/usr/bin/env bash
# Read the leader's current pose (torque off, nothing moves) and check it is inside the workspace bounds.
set -euo pipefail
source "$(dirname "$0")/common.sh"

LEADER_PORT="$LEADER_PORT" LEADER_ID="$LEADER_ID" URDF="$URDF" EE_BOUNDS_MIN="$EE_BOUNDS_MIN" \
  EE_BOUNDS_MAX="$EE_BOUNDS_MAX" "$PY" "$PROJECT_DIR/scripts/read_pose.py" \
  2>&1 | tee "$PROJECT_DIR/outputs/read_pose.log"
