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
DATASET_NAME="so101_leader_demos_wrist"  # xyz + wrist actions (6-dim); the xyz-only demos stay in so101_leader_demos
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

# Optional wrist limits for the policy, [min, max] degrees each (see local.env.example).
if [ -n "${WRIST_FLEX_RANGE:-}" ] && [ -n "${WRIST_ROLL_RANGE:-}" ]; then
  SETUP_ARGS+=(
    --env.processor.inverse_kinematics.joint_bounds="{\"wrist_flex\": $WRIST_FLEX_RANGE, \"wrist_roll\": $WRIST_ROLL_RANGE}"
  )
fi

if [ -n "${CALIBRATION_DIR:-}" ]; then
  SETUP_ARGS+=(
    --env.robot.calibration_dir="$CALIBRATION_DIR"
    --env.teleop.calibration_dir="$CALIBRATION_DIR"
  )
fi

# Remote learner (scripts/ec2_*.sh). LEARNER_PORT must match actor_learner_config.learner_port.
LEARNER_PORT=50051
REMOTE_PROJECT="${EC2_DIR:-hilserl}/lerobot/examples/hilserl_so101"  # relative to the remote home
require_ec2() {
  if [ -z "${EC2_HOST:-}" ] || [ -z "${EC2_KEY:-}" ]; then
    echo "Set EC2_HOST and EC2_KEY in local.env to use the remote learner." >&2
    exit 1
  fi
  if [ ! -f "$EC2_KEY" ]; then
    echo "EC2_KEY=$EC2_KEY not found (check local.env)" >&2
    exit 1
  fi
}
ec2_ssh() {
  ssh -i "$EC2_KEY" -o ServerAliveInterval=30 -o ConnectTimeout=15 "$EC2_HOST" "$@"
}

# Keys pressed for the robot (space/tab/s/q/r) are also queued as terminal input; drop them on exit so
# bash doesn't run them (e.g. tab completion) after the script ends.
flush_tty_input() {
  if [ -t 0 ]; then
    "$PY" -c "import sys, termios; termios.tcflush(sys.stdin, termios.TCIFLUSH)" 2>/dev/null || true
  fi
}
trap flush_tty_input EXIT
