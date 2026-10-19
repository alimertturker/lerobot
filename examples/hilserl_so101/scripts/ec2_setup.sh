#!/usr/bin/env bash
# Remote learner, step 1 (no robot): copy this lerobot checkout, the URDF and the demos to the remote GPU
# machine and install lerobot there. Run again after changing code or recording new demos.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_ec2
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
RSYNC_SSH="ssh -i $EC2_KEY -o ServerAliveInterval=30"
EC2_ROOT="${EC2_DIR:-hilserl}"

echo "Copying the code to $EC2_HOST:$EC2_ROOT/lerobot"
ec2_ssh "mkdir -p $REMOTE_PROJECT/outputs/datasets"
git -C "$REPO_DIR" ls-files -co --exclude-standard \
  | rsync -az --files-from=- -e "$RSYNC_SSH" "$REPO_DIR/" "$EC2_HOST:$EC2_ROOT/lerobot/"

[ -f "$URDF" ] || bash "$PROJECT_DIR/scripts/get_urdf.sh"
rsync -az -e "$RSYNC_SSH" "$PROJECT_DIR/assets" "$EC2_HOST:$REMOTE_PROJECT/"
if [ -d "$DATASET_ROOT" ]; then
  echo "Copying the demos ($DATASET_ROOT)"
  rsync -az --delete -e "$RSYNC_SSH" "$DATASET_ROOT" "$EC2_HOST:$REMOTE_PROJECT/outputs/datasets/"
else
  echo "No demos at $DATASET_ROOT; record them first (scripts/record_demos.sh)" >&2
fi

# Same settings as here, with the remote Python and the GPU for the learner.
{
  cat "$PROJECT_DIR/local.env"
  echo ''
  echo '# Added by ec2_setup.sh for the remote learner'
  echo "LEROBOT_PYTHON=\"\$HOME/$EC2_ROOT/.venv/bin/python\""
  echo 'LEARNER_DEVICE="cuda"'
} | ec2_ssh "cat > $REMOTE_PROJECT/local.env"

echo "Installing lerobot[hilserl] (first run takes a few minutes)"
ec2_ssh "cd $EC2_ROOT && { [ -x .venv/bin/python ] || python3 -m venv .venv; } \
  && .venv/bin/pip install -q -e './lerobot[hilserl]' \
  && .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA GPU\"; print(\"GPU:\", torch.cuda.get_device_name(0))'"
echo "Remote setup OK. Next: scripts/ec2_learner.sh start"
