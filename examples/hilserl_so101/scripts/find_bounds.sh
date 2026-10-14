#!/usr/bin/env bash
# Step 1: the follower mirrors the leader. Sweep the leader over the whole task workspace
# (every place the gripper must reach) and close/open the gripper. Prints joint limits and
# end-effector bounds to copy into local.env (EE_BOUNDS_MIN / EE_BOUNDS_MAX).
set -euo pipefail
source "$(dirname "$0")/common.sh"

lerobot-find-joint-limits \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
  --urdf_path="$URDF" --target_frame_name=gripper_frame_link \
  --teleop_time_s="${1:-40}" --warmup_time_s=5 --control_loop_fps=30 \
  2>&1 | tee "$PROJECT_DIR/outputs/find_bounds.log"
