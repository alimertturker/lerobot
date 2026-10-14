#!/usr/bin/env bash
# Download the SO-101 URDF and meshes (used for kinematics) from TheRobotStudio's SO-ARM100 repo into
# assets/SO101/, as the lerobot docs recommend. Needs git; nothing else is installed.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100 "$TMP/SO-ARM100"
git -C "$TMP/SO-ARM100" sparse-checkout set Simulation/SO101

mkdir -p "$PROJECT_DIR/assets/SO101/assets"
cp "$TMP/SO-ARM100/Simulation/SO101/so101_new_calib.urdf" "$PROJECT_DIR/assets/SO101/"
cp "$TMP/SO-ARM100/Simulation/SO101/assets/"*.stl "$PROJECT_DIR/assets/SO101/assets/"
echo "URDF and meshes in $PROJECT_DIR/assets/SO101"
