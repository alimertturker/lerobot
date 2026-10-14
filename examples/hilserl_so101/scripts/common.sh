# Shared settings for the HIL-SERL SO-101 leader-arm project. Sourced by the other scripts.
# Machine and setup values come from local.env (see local.env.example); they are passed to lerobot as
# command-line overrides of the JSON configs in configs/.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$PROJECT_DIR/local.env" ]; then
  echo "Missing $PROJECT_DIR/local.env. Create it from the template and fill in your setup:" >&2
  echo "  cp $PROJECT_DIR/local.env.example $PROJECT_DIR/local.env" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$PROJECT_DIR/local.env"

PY="${LEROBOT_PYTHON:?set LEROBOT_PYTHON in local.env}"
if [ ! -x "$PY" ]; then
  echo "LEROBOT_PYTHON=$PY is not an executable Python (check local.env)" >&2
  exit 1
fi
export PATH="$(dirname "$PY"):$PATH"

URDF="$PROJECT_DIR/assets/SO101/so101_new_calib.urdf"
DATASET_NAME="so101_leader_demos"
DATASET_ROOT="$PROJECT_DIR/outputs/datasets/$DATASET_NAME"
DATASET_REPO_ID="${HF_USER:-local}/$DATASET_NAME"
TRAIN_DIR="$PROJECT_DIR/outputs/train/hilserl_so101_leader"
mkdir -p "$PROJECT_DIR/outputs"

camera_json() {
  printf '{"type": "opencv", "index_or_path": %s, "width": %s, "height": %s, "fps": %s}' \
    "$1" "${CAMERA_WIDTH:-640}" "${CAMERA_HEIGHT:-480}" "${CAMERA_FPS:-30}"
}

# Overrides applied to every config (configs/*.json).
SETUP_ARGS=(
  --env.robot.port="$FOLLOWER_PORT"
  --env.robot.id="$FOLLOWER_ID"
  --env.teleop.port="$LEADER_PORT"
  --env.teleop.id="$LEADER_ID"
  --env.robot.cameras="{\"wrist\": $(camera_json "$WRIST_CAMERA"), \"top\": $(camera_json "$TOP_CAMERA")}"
  --env.processor.inverse_kinematics.urdf_path="$URDF"
  --env.processor.inverse_kinematics.end_effector_bounds="{\"min\": $EE_BOUNDS_MIN, \"max\": $EE_BOUNDS_MAX}"
  --env.processor.reset.fixed_reset_joint_positions="$RESET_POSE"
  --env.processor.reset.home_joint_positions="$HOME_POSE"
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
)

# Keys pressed for the robot (space/tab/s/q) are also queued as terminal input; drop them on exit so
# bash doesn't run them (e.g. tab completion) after the script ends.
flush_tty_input() {
  if [ -t 0 ]; then
    "$PY" -c "import sys, termios; termios.tcflush(sys.stdin, termios.TCIFLUSH)" 2>/dev/null || true
  fi
}
trap flush_tty_input EXIT
