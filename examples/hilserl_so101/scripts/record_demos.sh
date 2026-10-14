#!/usr/bin/env bash
# Step 2: record demonstrations with the leader arm.
# space = pause (leader moves onto the follower), tab (hand on leader) = take over / hold again,
# space = resume. s = success, q/esc = failure, r = re-record.
# Extra arguments are passed through, e.g. --dataset.num_episodes_to_record=2
set -euo pipefail
source "$(dirname "$0")/common.sh"

PYTHONUNBUFFERED=1 "$PY" -m lerobot.rl.gym_manipulator --config_path "$PROJECT_DIR/configs/env_record_leader.json" \
  "${SETUP_ARGS[@]}" "$@" \
  2>&1 | tee "$PROJECT_DIR/outputs/record_demos.log"

# Upload the recorded episodes to $HF_USER/so101_leader_demos (private). Skipped when HF_USER is empty
# or with NO_PUSH=1.
if [ -n "${HF_USER:-}" ] && [ -z "${NO_PUSH:-}" ]; then
  "$PROJECT_DIR/scripts/push_hub.sh" demos
fi
