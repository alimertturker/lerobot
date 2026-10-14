#!/usr/bin/env bash
# Check local.env and the configs without moving anything: resolves every config with this machine's
# settings, and checks that the Python environment, the arm ports and the URDF exist.
set -euo pipefail
source "$(dirname "$0")/common.sh"

ok=1
echo "Python:   $PY ($("$PY" -c 'import lerobot, sys; print(f"lerobot {lerobot.__version__}, Python {sys.version.split()[0]}")'))"
for name in FOLLOWER_PORT LEADER_PORT; do
  if [ -e "${!name}" ]; then echo "$name: ${!name} found"; else echo "$name: ${!name} NOT FOUND (arm unplugged or wrong port?)"; ok=0; fi
done
[ -f "$URDF" ] && echo "URDF:     found" || { echo "URDF:     NOT FOUND, run scripts/get_urdf.sh"; ok=0; }
echo "Dataset:  $DATASET_REPO_ID at $DATASET_ROOT"

"$PY" - "$PROJECT_DIR" "${SETUP_ARGS[@]}" --output_dir="$TRAIN_DIR" <<'PYEOF'
import sys
import draccus
from lerobot.rl.gym_manipulator import GymManipulatorConfig
from lerobot.rl.train_rl import TrainRLServerPipelineConfig

project, *overrides = sys.argv[1:]
env_overrides = [o for o in overrides if not o.startswith("--output_dir")]
for cls, name, extra in [
    (GymManipulatorConfig, "env_record_leader", env_overrides),
    (GymManipulatorConfig, "env_test_leader", env_overrides),
    (TrainRLServerPipelineConfig, "train_hilserl_leader", overrides),
]:
    cfg = draccus.parse(cls, args=["--config_path", f"{project}/configs/{name}.json", *extra])
    env = cfg.env
    cams = {k: v.index_or_path for k, v in env.robot.cameras.items()}
    print(f"{name}: follower {env.robot.port} ({env.robot.id}), leader {env.teleop.port} ({env.teleop.id}), "
          f"cameras {cams}, reset {env.processor.reset.fixed_reset_joint_positions}")
print("All configs resolve.")
PYEOF
[ "$ok" = 1 ] && echo "Setup OK." || { echo "Setup has problems (see above)."; exit 1; }
