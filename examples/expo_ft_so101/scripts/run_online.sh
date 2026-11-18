#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_PATH="$REPO_ROOT/examples/expo_ft_so101/configs/train.json"
OVERRIDES=()
while (( $# )); do
  case "$1" in
    --config-path|--config_path)
      if (( $# < 2 )) || [[ "$2" == --* ]]; then
        echo "Missing path after $1" >&2
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --config-path=*|--config_path=*)
      CONFIG_PATH="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: run_online.sh [--config-path PATH] [--field=value ...]

Default config: examples/expo_ft_so101/configs/train.json
Custom config paths are relative to the caller's current directory.
Paths inside the JSON and CLI overrides are relative to the LeRobot repository root.

Ordinary EXPO-FT: --real_time=false
Real-time SmolVLA: --real_time=true --chunk_steps=10 --real_time_delay_steps=4
Evaluation: --resume=CHECKPOINT --eval_only=true --eval_record=true
Use a fresh output_dir when starting a new mode or execution window.
Set LEROBOT_PYTHON to use an installed Python environment; default: uv run python.
See examples/expo_ft_so101/README.md for complete commands.
HELP
      exit 0
      ;;
    *)
      OVERRIDES+=("$1")
      shift
      ;;
  esac
done
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$PWD/$CONFIG_PATH"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Configuration not found: $CONFIG_PATH" >&2
  exit 2
fi
cd "$REPO_ROOT"
if [[ -n "${LEROBOT_PYTHON:-}" ]]; then
  exec "$LEROBOT_PYTHON" -m lerobot.rl.expo_ft.train \
    --config_path "$CONFIG_PATH" "${OVERRIDES[@]}"
fi
exec uv run python -m lerobot.rl.expo_ft.train \
  --config_path "$CONFIG_PATH" "${OVERRIDES[@]}"
