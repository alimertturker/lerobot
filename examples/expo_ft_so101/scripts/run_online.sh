#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -n "${LEROBOT_PYTHON:-}" ]]; then
  exec "$LEROBOT_PYTHON" -m lerobot.rl.expo_ft.train \
    --config_path examples/expo_ft_so101/configs/train.json "$@"
fi
exec uv run python -m lerobot.rl.expo_ft.train \
  --config_path examples/expo_ft_so101/configs/train.json "$@"
